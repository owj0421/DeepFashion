![header](https://capsule-render.vercel.app/api?type=rounded&height=200&color=gradient&customColorList=20&text=🧷%20Deep%20Fashion&fontSize=36&fontColor=FFFFFF&fontAlignY=45&desc=PyTorch%20implementation%20of%20deep-learning%20based%20fashion%20recommendation%20models&descSize=12&descAlignY=65)

<br>
<div align="center">

![Python Versions](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10-blue)
[![GitHub Issues](https://img.shields.io/github/issues/owj0421/DeepFashion.svg
)](https://github.com/owj0421/DeepFashion/issues)
<br>
[![License](https://img.shields.io/github/license/owj0421/DeepFashion.svg)](https://github.com/owj0421/DeepFashion/blob/master/LICENSE)

</div>


## 🤗 Introduction

Deep Fashion is a **Easy-to-use**, **Modular** and **Extendible** package of deep-learning based fashion recommendation models with PyTorch. <br><br>
Behind the fact that none of the numerous papers released since 2018 have been implemented, we implement and distribute the model ourselves. We aimed to implement the paper as much as possible, but since it is a personal project, there may be some other aspects. Therefore, if there is a better way, please contribute.<br><br>
Only the models studied for the purpose of retrieval were developed, so the prediction-based models(SCE-Net, Outfit-Transformer etc) were not implemented.
<br><br>
In addition, the repository includes a DataLoader that enables easy and fast implementation of Polyvore Dataset to enable more diverse and modular work. It is configured based on type-aware-net code.<br><br>

## 📚 Supported Models
<div align="center">

|Model|Paper|FITB<br>Acc.<br>(Ours)|FITB<br>Acc.<br>(Original)|
|:-:|:-|:-:|:-:|
|type-aware-net|[ECCV 2018] [Learning Type-Aware Embeddings for Fashion Compatibility](https://arxiv.org/abs/1803.09196)|?<br>32, ResNet18 <br>Image|**55.65**<br>64, ResNet18 <br>Image + Text|
|csa-net|[CVPR 2020] [Category-based Subspace Attention Network (CSA-Net)](https://arxiv.org/abs/1912.08967?ref=dl-staging-website.ghost.io)|**56.7**<br>32, ResNet18 <br>Image|**59.3**<br>64, ResNet18 <br>Image|
|sat|[IEEE2022] [SAT: Self-adaptive training for fashion compatibility prediction](https://arxiv.org/abs/2206.12622)|**?**<br>32, ResNet18 <br>Image|**62.2**<br>128, VGG13 <br>Image|
|fashion-swin|[IEEE 2023] [Fashion Compatibility Learning Via Triplet-Swin Transformer](https://ieeexplore.ieee.org/abstract/document/10105392)|?<br>32, Swin-t <br>Image|**60.7**<br>64, Swin-t <br>Image + Text|

</div>

**Notes**
 - The model implementation is based on the above papers, but there may be other parts of it.
 - All models used **ResNet18** as Backbone. <br>(However, some other models are different depending on the purpose of papers.)
 - All Embedding dimension was fixed at **32**. 
 - All learning was carried out using the **online mining method**.

## ⚙ Requirements
This project recommends Python 3.7 or higher.
```
python -m pip install -r requirements.txt
```

## 📦 Datasets
Download the Polyvore dataset from [here](https://github.com/xthan/polyvore-dataset?tab=readme-ov-file).


## 🧱 Train
`$MODEL` is same as above mentioned sheet.

```
!python train.py --model $MODEL --embedding_dim $NUM --img_backbone resnet-18 --dataset_type outfit --train_batch 64 --valid_batch 64 --fitb_batch 32 --n_epochs 5 --work_dir $DIR --data_dir $DIR --num_workers 4 --scheduler_step_size 500 --learning_rate 5e-5
```

## 🧶 Demos
Preparing for demos...

## 🔔 Note
- A paper review of implementation can be found at [here](). (Only Available in Korean)
- This is **NON-OFFICIAL** implementation.
- The part that uses the HGLMM Fisher vector is replaced by **SBERT Embedding**. If you want to use Fisher vector, you can change txt_type to 'hglmm'. but it requires to revise model codes.
