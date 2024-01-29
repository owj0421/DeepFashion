# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
Reference:
    [1] Mariya I. Vasileva, Bryan A. Plummer, et al. Learning Type-Aware Embeddings for Fashion Compatibility. ECCV, 2018.
    (https://arxiv.org/abs/1803.09196)
"""
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfashion.utils.utils import *
from deepfashion.models.encoder.builder import *
from deepfashion.models.baseline import *
from deepfashion.loss.triplet_margin_loss import *
import math

class SiameseNet(DeepFashionModel):

    def __init__(
            self,
            embedding_dim: int = 64,
            categories: Optional[List[str]] = None,
            img_backbone: str = 'resnet-18',
            ):
        super().__init__(embedding_dim, categories, img_backbone)


    def forward(self, inputs, target_category=None):
        inputs = stack_dict(inputs)
        outputs = DeepFashionOutput(
            mask=inputs['mask'],
            category=inputs['category'],
            )

        general_img_embed = self.img_encoder(inputs['image_features'])

        if target_category is not None:
            target_category = stack_tensors(inputs['mask'], target_category)
            outputs.embed = general_img_embed
        else:
            embed_by_category = []
            for i in range(self.num_category):
                embed_by_category.append(general_img_embed)
            outputs.embed_by_category = embed_by_category
        
        return unstack_output(outputs)
    

    def iteration_step(self, batch, device) -> np.ndarray:
        outfits = {key: value.to(device) for key, value in batch['outfits'].items()}
        outfit_outs = self(outfits)
        loss = triplet_loss(outfit_outs, self.margin)
        
        return loss
    