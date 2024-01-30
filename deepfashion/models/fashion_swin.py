# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
Reference:
    [1] Hosna Darvishi, Reza Azmi, et al. Fashion Compatibility Learning Via Triplet-Swin Transformer. IEEE, 2023.
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


class FashionSwin(DeepFashionModel):

    def __init__(
            self,
            embedding_dim: int = 64,
            categories: Optional[List[str]] = None,
            img_backbone: str = 'swin-transformer',
            txt_backbone: str = 'none'
            ):
        super().__init__(embedding_dim, categories, img_backbone, txt_backbone)

        self.category_pairs = [(i, j) for i in range(self.num_category) for j in range(self.num_category) if i >= j]
        self.category_pair2id = dict()
        for i, category_pair in enumerate(self.category_pairs):
            self.category_pair2id[(category_pair[0], category_pair[1])] = i
            self.category_pair2id[(category_pair[1], category_pair[0])] = i

        self.category_embedding = nn.Embedding(num_embeddings=len(self.category_pairs), embedding_dim=embedding_dim)
        nn.init.kaiming_uniform_(self.category_embedding.weight.data, a=math.sqrt(5))


    def _get_mask(self, input_category, target_category):
        category_ids = torch.LongTensor(list(map(lambda x: self.category_pair2id[x], zip(input_category.tolist(), target_category.tolist()))))
        category_mask = self.category_embedding(category_ids.to(input_category.get_device()))
        return category_mask


    def forward(self, inputs, target_category=None):
        inputs = stack_dict(inputs)
        outputs = DeepFashionOutput(
            mask=inputs['mask'],
            category=inputs['category'],
            )
        general_img_embed = self.img_encoder(inputs['image_features'])
        if target_category is not None:
            target_category = stack_tensors(inputs['mask'], target_category)
            outputs.embed = general_img_embed * self._get_mask(inputs['category'], target_category)
        else: # returns embedding for all categories
            embed_by_category = []
            for i in range(self.num_category):
                target_category = torch.ones((inputs['category'].shape[0]), dtype=torch.long, device=inputs['category'].get_device()) * i
                embed_by_category.append(general_img_embed * self._get_mask(inputs['category'], target_category))
            outputs.embed_by_category = embed_by_category
        outputs.general_img_embed = general_img_embed
        return unstack_output(outputs)
    

    def iteration_step(self, batch, device) -> np.ndarray:
        outfits = {key: value.to(device) for key, value in batch['outfits'].items()}
        outfit_outs = self(outfits)
        loss = triplet_loss(outfit_outs, self.margin)
        
        return loss