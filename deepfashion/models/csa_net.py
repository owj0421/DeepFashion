"""
Author:
    Wonjun Oh, owj0421@naver.com
Reference:
    [1] Yen-liang L, Son Tran, et al. Category-based Subspace Attention Network (CSA-Net). CVPR, 2020.
    (https://arxiv.org/abs/1912.08967?ref=dl-staging-website.ghost.io)
"""
from numpy import ndarray
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfashion.utils.utils import *
from deepfashion.models.encoder.builder import *
from deepfashion.models.baseline import *
from deepfashion.models.loss import *


class CSANet(DeepFashionModel):
    
    def __init__(
            self,
            embedding_dim: int = 64,
            num_category: int = 12,
            num_subspace: int = 5,
            img_backbone: str = 'resnet-18'
            ):
        super().__init__(embedding_dim, num_category, img_backbone)
        self.num_subspace = num_subspace

        self.mask = nn.Parameter(torch.ones((num_subspace, embedding_dim)))
        initrange = 0.1
        self.mask.data.uniform_(-initrange, initrange)
        self.attention = nn.Sequential(
            nn.Linear(num_category * 2, num_category * 2),
            nn.ReLU(),
            nn.Linear(num_category * 2, num_subspace)
            )
    

    def _get_embedding(self, inputs, target_category):
        # Get genreral embedding from inputs
        embed = self.img_encoder(inputs['image_features'])
        # Compute Attention Score
        input_category = self._one_hot(inputs['category'])
        target_category = self._one_hot(target_category)
        attention_query = torch.concat([input_category, target_category], dim=-1)
        attention = F.softmax(self.attention(attention_query), dim=-1)
        # Compute Subspace Mask via Attetion
        mask = torch.matmul(self.mask.T.unsqueeze(0), attention.unsqueeze(2)).squeeze(2)
        masked_embed = embed * mask
        masked_embed = unstack_tensors(inputs['mask'], masked_embed)

        return masked_embed


    def forward(self, inputs, target_category=None):
        outputs = DeepFashionOutput(mask=inputs['mask'], category=inputs['category'])
        inputs = stack_dict(inputs)

        if target_category is not None:
            target_category = stack_tensors(inputs['mask'], target_category)
            embed = self._get_embedding(inputs, target_category)
            outputs.embed = embed
        else: # returns embedding for all categories
            embed_by_category = []
            for i in range(self.num_category):
                target_category = torch.ones((inputs['category'].shape[0]), dtype=torch.long, device=inputs['category'].get_device()) * i
                embed_by_category.append(self._get_embedding(inputs, target_category))
            outputs.embed_by_category = embed_by_category

        return outputs
    

    def iteration_step(self, batch, device) -> ndarray:
        ancs = {key: value.to(device) for key, value in batch['anchors'].items()}
        poss = {key: value.to(device) for key, value in batch['positives'].items()}
        negs = {key: value.to(device) for key, value in batch['negatives'].items()}

        anc_outs = self(ancs)
        pos_outs = self(poss)
        neg_outs = self(negs)

        loss = outfit_ranking_loss(anc_outs, pos_outs, neg_outs, self.margin)

        return loss
