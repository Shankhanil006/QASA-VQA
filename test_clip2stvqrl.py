import torch
# import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
# from demo_extract_features_resnet_RGB_diff import *
import torch.optim as optim
import glob
import cv2
from scipy import io as sio
import numpy as np
import os
from operator import itemgetter
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from scipy.stats import spearmanr as srocc
from scipy.stats import pearsonr as plcc

from scipy.io import loadmat
import time
import gc
import decord

from fastvqa.datasets.fusion_datasets import *
from swin_backbone import SwinTransformer3D as VideoBackbone
from fastvqa.models.swin_backbone import SwinTransformer3D as VideoBackbone

import clip
torch.cuda.empty_cache()
device = 'cpu'#torch.device('cuda:1')

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

class BaseEvaluator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.backbone = VideoBackbone()
        self.vqa_head = VQAHead()

    def forward(self, vclip, inference=False, **kwargs):
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            return score

classes = ['good', 'bad', 'high quality', 'low quality']
text_list = torch.cat([clip.tokenize(f"a {c} photo.") for c in classes]).to(device)

clip_model, preprocess = clip.load('ViT-B/32', device)
clip_model.float()

clip_image = clip_model.visual
clip_text = TextCLIP(clip_model)

# clip_image.eval()
clip_text.eval()

with torch.no_grad():
    text_features = clip_text(text_list)#.cpu().numpy()
    
stvqrl_model = BaseEvaluator().to(device)

path = 'path_to_model'
state_dict = torch.load(path + 'clip2stvqrl500_25.pth', map_location=device)#['primary']

stvqrl_model.load_state_dict(state_dict['primary'], strict=True)
stvqrl_model.eval()

clip_image.load_state_dict(state_dict['auxillary'], strict=True)
clip_image.eval()

################################# authentic niqe feat #######################################################
clip_len = 64
frame_interval = 2
t_frag = 8
num_clips = 1
tau = 0.1
mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
# sampler = UnifiedFrameSampler(clip_len, num_clips, frame_interval)
sampler = UnifiedFrameSampler(clip_len//t_frag, t_frag, frame_interval, num_clips)     

    
video = decord.VideoReader('path_to_test_video') 
frames = sampler(len(video))
frame_dict = {idx: video[idx] for idx in np.unique(frames)}
imgs = [frame_dict[idx] for idx in frames]
org_video = torch.stack(imgs, 0)

org_video = ((org_video - mean)/ std)
scale_video = get_resized_video(org_video.permute(3,0,1,2))

sampled_video = get_spatial_fragments(org_video.permute(3,0,1,2))
#            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
  

with torch.no_grad():
    vision_embeddings = clip_image(scale_video.to(device))
    score = F.normalize(vision_embeddings.mean(0, keepdim=True)) @ F.normalize(text_features).t()
    score = score.cpu().numpy().squeeze()

    pred = 0
    for k in range(2):
        tmp = score[2*k+1] - score[2*k]
        pred += 1/(1+np.exp(tmp/tau))
    clip_score = pred
    
    pred = stvqrl_model(sampled_video.to(device)).cpu().numpy().mean()
 
    pred = 1/(1+np.exp(-pred*.1))
    stvqrl_score = pred

gc.collect()

overall_score = stvqrl_score* clip_score

