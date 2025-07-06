
import os, gc, math
import json
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn
from comb_semisupervised_datasets import *
import clip

from swin_backbone import SwinTransformer3D as VideoBackbone

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device_ids = [0,1,2]

##############################3 Configuration ##################################3

depth = 32
bs = 6
ps = 224
fps = 4
num_clips = 1
num_label = 500

t_frag = 8
test_clip = 1
##############################################################################
class VQAHead(nn.Module):
    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score
    
class TextCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

class projection(nn.Module):

    def __init__(
        self, in_channels=1, hidden_channels=1, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.fc_hid = nn.Linear(self.in_channels, self.hidden_channels)
        # self.fc_hid = nn.Conv1d(self.in_channels, self.hidden_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, rois=None):
        # x = self.dropout(x)
        qlt_score = self.relu(self.fc_hid(x))
        return qlt_score

class BaseEvaluator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.backbone = VideoBackbone()#backbone
        self.vqa_head = VQAHead()

    def forward(self, vclip, get_feat=False, **kwargs):
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            if get_feat:
                return score, feat
            else:
                return score
                
def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def save_model(primary, auxillary, epoch):
    # savefolder = '/home/mitra/representation_learning/st_ablation/ablation_loss/'

    savefolder = '/home/mitra/CLIP_test/KD_clip_stvqrl/models/'
    state = {
        'primary': primary.module.state_dict(),
        'auxillary': auxillary.module.state_dict(),
        'epoch': epoch,
     }
    torch.save(state, os.path.join(savefolder, 'clip2stvqrl500_%d.pth'%(epoch)))
    # torch.save(state, os.path.join(savefolder, 'no_cons_mask%d.pth'%(epoch)))
    return()

########################## Contrastive pretrained #################################

loadfolder = 'path_to_pretrained_stvqrl_model'
cons_state_dict = torch.load(loadfolder+ 'shift_niqe_lsvd_model_30' + '.pth',map_location='cpu')['model']

primary = BaseEvaluator()
primary.backbone.load_state_dict(cons_state_dict, strict=False)

# primary= nn.DataParallel(primary,device_ids=device_ids)
primary = primary.to(device=device)

model,_ = clip.load('ViT-B/32', device) #VideoBackbone()  # RN50
model.float()

for name, param in model.named_parameters():
    if name.startswith('visual'):
        param.requires_grad_(True)
    else:
        param.requires_grad_(False)

model_text = TextCLIP(model)
auxillary = model.visual

# auxillary = nn.DataParallel(auxillary,device_ids=device_ids)
auxillary = auxillary.to(device=device)
auxillary.train()
model_text.eval()

classes = ['Good', 'Bad', 'High quality', 'Low quality']#, 'High definition','Low definition']
text_inputs = torch.cat([clip.tokenize(f"a {c} photo.") for c in classes]).to(device)
with torch.no_grad():
    text_features = model_text(text_inputs).detach()

p_opt = optim.AdamW([
                {'params': primary.module.vqa_head.parameters()},
                {'params': primary.module.backbone.parameters(), 'lr': 1e-4}
            ], lr=1e-3, weight_decay= 0.05)
n_opt = optim.AdamW(auxillary.module.parameters(), lr = 5e-6, weight_decay= 0.001)

#########################################################################################

sampler = FragmentSampleFrames(depth//t_frag, t_frag, frame_interval = 2, num_clips=num_clips)
# sampler =  SampleFrames(clip_len = 32,frame_interval = 2, num_clips=test_clip)
mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])

lsvd_file = 'path_to_json_file'

lbl_dataset = SemiSupervisedDataset(lsvd_file,depth, num_clips, 'sup', num_label, t_frag, fps)
lbl_dataloader = DataLoader(lbl_dataset, batch_size=bs, shuffle=True, drop_last=True)

unlbl_dataset = SemiSupervisedDataset(lsvd_file,depth, num_clips, 'unsup',num_label, t_frag, fps)
unlbl_dataloader = DataLoader(unlbl_dataset, batch_size=bs, shuffle=True, drop_last=True)

def cyclic(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

unlbl_dataloader = cyclic(unlbl_dataloader)

num_epochs = 40
warmup_iter = int(2.5 * len(lbl_dataloader))
max_iter = int(num_epochs * len(lbl_dataloader))
p_lambda = (
    lambda cur_iter: cur_iter / warmup_iter
    if cur_iter <= warmup_iter
    else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
)

n_lambda = (
    lambda cur_iter: cur_iter / warmup_iter
    if cur_iter <= warmup_iter
    else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
)

p_scheduler = torch.optim.lr_scheduler.LambdaLR(
    p_opt, lr_lambda=p_lambda,
)
n_scheduler = torch.optim.lr_scheduler.LambdaLR(
    n_opt, lr_lambda=n_lambda,
)


#############################################################################


wt_cons = 1     # Hyperparameter for Loss function
wt_stab = 1
tau = 0.1

start_time = time.time()
flag = True

mdic = {}
out_file = open("/home/mitra/logs/results.json", "w")
for epoch in range(num_epochs+1):
    epoch_loss, reg_loss, niqe_loss = 0, 0, 0
    gt_loss, cons_loss, stab_loss = 0, 0, 0
    
    for n_count, lbl_batch in enumerate(lbl_dataloader):   

        frag_vclip, frag_aclip, scale_vclip, scale_aclip, y = lbl_batch[0].to(device=device), lbl_batch[1].to(device=device), \
                                                                    lbl_batch[2].to(device=device), lbl_batch[3].to(device=device), lbl_batch[-1].detach().to(device=device)
    
        v_stvqrl= primary(frag_vclip).mean((-3,-2,-1)).squeeze()
        a_stvqrl = primary(frag_aclip).mean((-3,-2,-1)).squeeze()
        
        v_feat = auxillary(scale_vclip.view(bs*depth, 3, ps, ps)).view(bs,depth,-1).mean(1)
        a_feat = auxillary(scale_aclip.view(bs*depth//fps, 3, ps, ps)).view(bs,depth//fps,-1).mean(1)
            
        v_score = F.normalize(v_feat) @ F.normalize(text_features).t()
        a_score = F.normalize(a_feat) @ F.normalize(text_features).t()

        v_clip, a_clip = 0, 0
        for k in range(2):
            tmp1 = v_score[:,2*k+1] - v_score[:,2*k]
            tmp2 = a_score[:,2*k+1] - a_score[:,2*k]
            
            v_clip += 1/(1+torch.exp(tmp1/tau))
            a_clip += 1/(1+torch.exp(tmp2/tau))

        n_loss = plcc_loss(v_clip, y) + wt_cons*plcc_loss(v_clip, a_clip) 
        p_loss = plcc_loss(v_stvqrl, y)  + wt_cons*plcc_loss(v_stvqrl, a_stvqrl) 
        
        gt_loss += (plcc_loss(v_stvqrl, y) + plcc_loss(v_clip, y)).item() 
        cons_loss += (plcc_loss(v_clip, a_clip) + plcc_loss(v_stvqrl, a_stvqrl)).item()
         
        if flag:
            unlbl_batch = next(unlbl_dataloader) 
            
            frag_vclip, frag_aclip, scale_vclip, scale_aclip= unlbl_batch[0].to(device=device), unlbl_batch[1].to(device=device), \
                                                                    unlbl_batch[2].to(device=device), unlbl_batch[3].to(device=device)


            v_stvqrl= primary(frag_vclip).mean((-3,-2,-1)).squeeze()
            a_stvqrl = primary(frag_aclip).mean((-3,-2,-1)).squeeze()
            
            v_feat = auxillary(scale_vclip.view(bs*depth, 3, ps, ps)).view(bs,depth,-1).mean(1)
            a_feat = auxillary(scale_aclip.view(bs*depth//fps, 3, ps, ps)).view(bs,depth//fps,-1).mean(1)
                
            v_score = F.normalize(v_feat) @ F.normalize(text_features).t()
            a_score = F.normalize(a_feat) @ F.normalize(text_features).t()
            v_clip, a_clip = 0, 0
            for k in range(2):
                tmp1 = v_score[:,2*k+1] - v_score[:,2*k]
                tmp2 = a_score[:,2*k+1] - a_score[:,2*k]
                
                v_clip += 1/(1+torch.exp(tmp1/tau))
                a_clip += 1/(1+torch.exp(tmp2/tau))
            
            eps_stvqrl = plcc_loss(v_stvqrl,a_stvqrl).item()
            eps_clip = plcc_loss(v_clip, a_clip).item()
            
            mask = int(eps_stvqrl > eps_clip)
            p_loss += mask*wt_stab*plcc_loss(v_stvqrl, v_clip.detach()) + wt_cons*plcc_loss(v_stvqrl, a_stvqrl)                                        
            n_loss += (1-mask)*wt_stab*plcc_loss(v_stvqrl.detach(), v_clip) + wt_cons*plcc_loss(v_clip, a_clip)                                         
  
            cons_loss += (plcc_loss(v_clip, a_clip) + plcc_loss(v_stvqrl, a_stvqrl)).item()
            stab_loss += (mask*(plcc_loss(v_stvqrl, v_clip.detach())) + (1-mask)*(plcc_loss(v_stvqrl.detach(), v_clip))).item()       

        p_opt.zero_grad()
        p_loss.backward()
        p_opt.step()
        p_scheduler.step()
        
        n_opt.zero_grad()
        n_loss.backward()
        n_opt.step()
        n_scheduler.step()
        
        gc.collect()
    torch.cuda.empty_cache()

    n_count+=1
    elapsed_time = (time.time() - start_time)/3600
    print('epoch = %4d , gt_loss = %4.4f , cons_loss = %4.4f , stab_loss = %4.4f , time = %4.2f hr' 
                    % (epoch + 1,  gt_loss/n_count, cons_loss/n_count, stab_loss/n_count, elapsed_time))

    if epoch and epoch%5 ==0: 
        save_model(primary, auxillary, epoch)
out_file.close()
