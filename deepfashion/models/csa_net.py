# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
Reference:
    [1] Yen-liang L, Son Tran, et al. Category-based Subspace Attention Network (CSA-Net). CVPR, 2020.
    (https://arxiv.org/abs/1912.08967?ref=dl-staging-website.ghost.io)
"""
import os
import math
import wandb
from tqdm import tqdm
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy import ndarray

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfashion.utils.utils import *
from deepfashion.models.encoder.builder import *
from deepfashion.models.baseline import *
from deepfashion.loss.outfit_ranking_loss import *


class CSANet(DeepFashionModel):
    
    def __init__(
            self,
            embedding_dim: int = 64,
            categories: Optional[List[str]] = None,
            num_subspace: int = 5,
            img_backbone: str = 'resnet-18'
            ):
        super().__init__(embedding_dim, categories, img_backbone)
        self.num_subspace = num_subspace
        self.mask = nn.Parameter(torch.empty((num_subspace, embedding_dim), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.mask.data, a=math.sqrt(5))
        self.attention = nn.Sequential(
            nn.Linear(self.num_category * 2, num_subspace),
            nn.ReLU(),
            nn.Linear(num_subspace, num_subspace)
            )


    def _get_mask(self, input_category, target_category):
        input_category = one_hot(input_category, self.num_category)
        target_category = one_hot(target_category, self.num_category)
        attention_query = torch.concat([input_category, target_category], dim=-1)
        attention = F.softmax(self.attention(attention_query), dim=-1)

        return torch.matmul(self.mask.T.unsqueeze(0), attention.unsqueeze(2)).squeeze(2)


    def forward(self, inputs, target_category=None):
        inputs = stack_dict(inputs)
        outputs = DeepFashionOutput(
            mask=inputs['mask'],
            category=inputs['category'],
            )

        embed = self.img_encoder(inputs['image_features']) # Get genreral embedding from inputs
        if target_category is not None:
            target_category = stack_tensors(inputs['mask'], target_category)
            outputs.embed = embed * self._get_mask(inputs['category'], target_category)
        else: # returns embedding for all categories
            embed_by_category = []
            for i in range(self.num_category):
                target_category = torch.ones((inputs['category'].shape[0]), dtype=torch.long, device=inputs['category'].get_device()) * i
                embed_by_category.append(embed * self._get_mask(inputs['category'], target_category))
            outputs.embed_by_category = embed_by_category

        return unstack_output(outputs)


    def iteration_step(self, batch, device):
        outfits = {key: value.to(device) for key, value in batch['outfits'].items()}
        outfit_outs = self(outfits)
        loss = outfit_ranking_loss(outfit_outs, self.margin)
        
        return loss
