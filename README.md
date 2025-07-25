# QASA-VQA
This is the official repository for the paper titled " Vision-Language Model Guided Semi-supervised Learning for No-Reference Video Quality Assessment" published in **IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2025** by **Shankhanil Mitra** and Rajiv Soundararajan. 

In this work, we address the challenge of requirement of large scale human annotated videos for training by first designing a self- supervised Spatio-Temporal Visual Quality Representation Learning (ST-VQRL) framework to generate robust quality aware features for videos. Then, we propose a dual-model based Semi Supervised Learning (SSL) method specifically designed for the Video Quality Assessment (SSL-VQA) task, through a novel knowledge transfer of quality predictions between the two models.

![SSL-VQA](https://github.com/Shankhanil006/QASA-VQA/blob/main/icassp_mainfig.png)

## Installation 
>conda env create -f environment.yml

## Self-supervised Video Quality Representation Learning (ST-VQRL) Model
We have used pre-trained ST-VQRL models from SSL-VQA which is trained on 200x12 synthetically distorted LSVQ videos.

Google Drive: [pretrained-stvqrl](https://drive.google.com/file/d/1uE0QgCZAsjXrvRHP_bdC8xVu5xb4eZUa/view?usp=drive_link)

## Training Semi-supervised Video Quality Assessment (QASA-VQA) Model

To train QASA-VQA with 500 labelled and 1500 unlabelled samples from LSVQ official train set run the following script:

> python3 KD_clip_stvqrl.py

We have provided 2000 video names in semisupervised.json files and randomly chosen 500 labelled samples from this 2000. User can choose any other set of labelled and unlabelled videos from entire LSVQ train set of 28053 in LSVQ_train.json file. 

Pre-trained weights of QASA-VQA trained on 1 random split of 500 labelled and 1500 unlabelled video in semisupervised.json:

[pretrained QASA-VQA](https://drive.google.com/file/d/1gv-bP6xZnywv1jzzbU_c8oAGs1wGgxx4/view?usp=drive_link)

## Infer with Semi-supervised Video Quality Assessment (QASA-VQA) Model
To estimate quality of any video using QASA-VQA, use the above pre-trained weights and provide the test video path to run the following scipt:
> python3 test_clip2stvqrl.py

## Acknowledgement 
Video fragment generation code is taken from FAST-VQA [link](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/tree/dev?tab=readme-ov-file)
Clip baseline model with pre-trained weights for ViT-B/32 variant is taken from [linl](https://github.com/openai/CLIP)
## Citation
If you find this work useful for your research, please cite our paper:
> @inproceedings{mitra2025vision,
  title={Vision-language model guided semi-supervised learning for no-reference video quality assessment},
  author={Mitra, Shankhanil and Soundararajan, Rajiv},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
