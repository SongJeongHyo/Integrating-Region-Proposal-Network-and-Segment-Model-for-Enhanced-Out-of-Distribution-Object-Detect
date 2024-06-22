# Integrating Region Proposal Network and Segment Model for Enhanced Out-of-Distribution Object Detection

This repository contains the implementation of an Out-of-Distribution (OOD) detection model that combines the strengths of [Segment Anything Model (SAM)](https://ai.meta.com/research/publications/segment-anything/) and [Faster R-CNN](https://arxiv.org/abs/1506.01497).

![image](https://github.com/SongJeongHyo/SAM_OOD/assets/79832986/1cbf3952-71f7-4ec9-97db-ffc7fef49723)

## Paper
Jeong-hyo Song, Jae-ho Cho, Seung-Ik Lee, "Integrating Region Proposal Network and Segment Model for Enhanced Out-of-Distribution Object Detection", KSC2023, 2023

## Main functionality
- input: image
- output: bounding box, label

## How to use
```bash
$ conda create -n SAM_OOD python=3.8
$ conda activate SAM_OOD
$ conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
$ git clone https://github.com/SongJeongHyo/SAM_OOD.git
```

The weights that I learned are [HERE](https://drive.google.com/file/d/1TXgAI5KW82CS22t1DZl74-K3b5Y_B6Rk/view?usp=sharing). if you want to use, it is on ./scaled_cosine/model_final.pth. This file allows you to evaluate and visualize what model detect. 

But if you want to train this model, specify the dataset path in train.py and type this command in terminal : python train.py

## Running on The Dataset Folder:
1. Prepare the dataset you want to test and put it under SAM-OOD-Detection directory. For example, if you use [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/): ./SAM_OOD/voc_data
2. And you can train with your dataset if you specify the path in train.py file.

To evaluate and visualize, check the train.ipynb file and follow the section of evaluation.

## Required program
- Please follow the instruction to install the faster -rcnn dependencies (https://github.com/jwyang/faster-rcnn.pytorch) and detectron2 (https://detectron2.readthedocs.io/en/latest/tutorials/install.html) 

