# <div align="center"> Deep Fashion </div>

<div align="center"> Implementation of deep-learning based fashion recommendation models with PyTorch </div>

## 🤗 Introduction

Deep Fashion is a package of deep-learning based fashion recommendation models with PyTorch. <br><br>
Behind the fact that none of the numerous papers released since 2018 have been implemented, we implement and distribute the model ourselves. We aimed to implement the paper as much as possible, but since it is a private project, there may be some other aspects. Therefore, if there is a better way, please contribute.<br><br>
In addition, the repository includes a DataLoader that enables easy and fast implementation of Polyvore Dataset. The Loader is configured to enable more diverse and modular work by referring to Type-aware-net, so I hope it will help researchers.<br><br>

## 📦 Datasets
Download the Polyvore Outfits dataset including the splits and questions for the compatibility and fill-in-the-blank tasks from [here](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view) which is first released in type-aware-net.<br>

## 📚 Models List
<div align="center">

|Name|Paper|FITB|
|:-:|:-|:-:|
|type-aware-net|[ECCV 2018] [Learning Type-Aware Embeddings for Fashion Compatibility](https://arxiv.org/abs/1803.09196)|0|
|csa-net|[CVPR 2020] [Category-based Subspace Attention Network (CSA-Net)](https://arxiv.org/abs/1912.08967?ref=dl-staging-website.ghost.io)|0|
|fashion-swin|[IEEE 2023] [Fashion Compatibility Learning Via Triplet-Swin Transformer](https://ieeexplore.ieee.org/abstract/document/10105392)|54|

</div>


## 🔔 Note
- A paper review of implementation can be found at [here](). (Only Available in Korean)
- This is **NON-OFFICIAL** implementation.
- The part that uses the HGLMM Fisher vector is replaced by **SBERT Embedding**. If you want to use Fisher vector, you can change txt_type to 'hglmm'. but it requires to revise model codes.
