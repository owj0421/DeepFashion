![header](https://capsule-render.vercel.app/api?type=rounded&height=200&color=gradient&customColorList=20&text=🧷%20Deep%20Fashion&fontSize=36&fontColor=FFFFFF&fontAlignY=45&desc=PyTorch%20implementation%20of%20deep-learning%20based%20fashion%20recommendation%20models&descSize=12&descAlignY=65)

<br>
<div align="center">

![Python Versions](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9%20|%203.10-blue)
[![License](https://img.shields.io/github/license/owj0421/DeepFashion.svg)](https://github.com/owj0421/DeepFashion/blob/master/LICENSE)

</div>


## 🤗 Introduction

Deep Fashion is a **Easy-to-use**, **Modular** and **Extendible** package of deep-learning based fashion recommendation models with PyTorch. <br><br>
Behind the fact that none of the numerous papers released since 2018 have been implemented, we implement and distribute the model ourselves. We aimed to implement the paper as much as possible, but since it is a personal project, there may be some other aspects. Therefore, if there is a better way, please contribute.<br><br>
**What is included**
- Data proprocessor that can easily configure Outfit as Dataset and Batch unit
- Fashion compatibility Models
- Metric learning loss that can be applied immediately to Batch configured with Outfit-wise


## 📚 Supported Models

<div align="center">

|Model|Paper|FITB<br>Acc.<br>(Ours)|FITB<br>Acc.<br>(Original)|
|:-:|:-|:-:|:-:|
|siamese-net|Baseline|**50.7**<br>32, ResNet18 <br>Image|**54.0**<br>64, ResNet18 <br>Image|
|type-aware-net|[ECCV 2018] [Learning Type-Aware Embeddings for Fashion Compatibility](https://arxiv.org/abs/1803.09196)|**52.6**<br>32, ResNet18 <br>Image|**54.5**<br>64, ResNet18 <br>Image + Text|
|csa-net|[CVPR 2020] [Category-based Subspace Attention Network (CSA-Net)](https://arxiv.org/abs/1912.08967?ref=dl-staging-website.ghost.io)|**55.8**<br>32, ResNet18 <br>Image|**59.3**<br>64, ResNet18 <br>Image|
|fashion-swin|[IEEE 2023] [Fashion Compatibility Learning Via Triplet-Swin Transformer](https://ieeexplore.ieee.org/abstract/document/10105392)|?<br>32, Swin-t <br>Image|**60.7**<br>64, Swin-t <br>Image + Text|

</div>

**Notes**
 - The model implementation is based on the above papers, but there may be other parts of it.
 - In the test environment, for fairness, the embedding size was fixed at **32**, and **only images** were used.
 - Learning was conducted with **online mining batch-all method**.
 - Only the models studied for the purpose of **retrieval** were developed, so the prediction-based models(SCE-Net, Outfit-Transformer etc) were not implemented.


## ⚙ Requirements
This project recommends Python 3.7 or higher.
```
python -m pip install -r requirements.txt
```

## 🧱 To Train with Polyvore Dataset and Exsisting Models
1. Download the Polyvore dataset from [here](https://github.com/xthan/polyvore-dataset?tab=readme-ov-file).

2. `$MODEL` is same as above mentioned sheet.

    ```
    !python train.py --model $MODEL --embedding_dim $NUM --dataset_type outfit --train_batch 64 --valid_batch 64 --fitb_batch 32 --n_epochs 5 --work_dir $DIR --data_dir $DIR --num_workers 4 --scheduler_step_size 500 --learning_rate 5e-5
    ```

## 🧱 To Build Your Own Dataset and Model
Please refer to `datasets/polyvore.py` and `models/csa_net.py` for detailed implementation. The overall usage is as follows.


### 1. Build Custom dataset using `DeepFashionProcessor` in `datasets/processor.py`.
    
```
class CustomDataset(Dataset):
    def __init__(...):
        ...
        image_processor = DeepFashionImageProcessor(...)
        self.input_processor = DeepFashionInputProcessor(...)
        ...

    def __getitem__(...):
        ...
        return self.input_processor(...)
```

**Note**
- After using Pytorch Dataset, Dataloader, and input processor, Batch is **[B, O, ...]** dimensions, where B is size of batch and O is the maximum length of items in the outfit.


### 2. Build Custom model by inheriting `DeepFashionModel` in `models/baseline.py`.<br>

Since the DeepFashion Model basically includes methods such as logging and iteration required for training, only the following **three methods need to be added**.<br>

In order to efficiently calculate the [B, O, ...] dimensional input containing Pad, it is necessary to **modify the input to the [B * V, ...] dimensions** through `stack_inputs` at the beginning of forward. (In this case, V is the actual number of items in the Batch.)

Please write a code separately from the case of **returning embeds for all categories** because the target category to be used for training is not given, and the case where **the target category to be used for Inference is given**.

In addition, in order to change the resulting embedding back to the form of input, it must be transformed using the `unstack_tensor` and returned.

Use the **DeepFashionOutput** class to return the output according to the form. It is convenient to use the loss functions that has already been written in the library.

```
class CustomModel(DeepFashionModel):
    def __init__(...):
            ...

    def forward(self, inputs) -> DeepFashionOutput:
        inputs = stack_dict(inputs)
        ...
        if target_category is not None:
            ...
        else:
            embed_by_category = []
            for i in range(self.num_category):
                ...
            outputs.embed_by_category = embed_by_category
        return DeepFashionOutput(...)

    def iteration_step(self, batch, device):
        ...

    def evalutaion_step(self, batch, device):
        ...

```
### 3. Train!
Refer to `train.py` and use given `trainer`.

## 🧶 Demos & Inference
Preparing for demos...


## 🔔 Note
- A paper review of implementation can be found at [here](). (Only Available in Korean)
- This is **NON-OFFICIAL** implementation.
- The part that uses the HGLMM Fisher vector is replaced by **SBERT Embedding**. If you want to use Fisher vector, you can change txt_type to 'hglmm'. but it requires to revise model codes.
