import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfashion.utils.dataset_utils import *
from deepfashion.models.embedder.builder import *


'''
제작중
'''

class CSANet(nn.Module):
    def __init__(
            self,
            img_backbone: str = 'resnet-18',
            embedding_dim: int = 64,
            num_category: int = 12
            ):
        super().__init__()
        self.img_embedder = build_img_embedder(img_backbone, embedding_dim)
        self.category_embedder = nn.Embedding(num_embeddings=num_category, embedding_dim=embedding_dim)
        self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        initrange = 0.1
        self.category_embedder.weight.data.uniform_(-initrange, initrange)

    def forward(self, anchors, positives, negatives):
        anchor_embeds = self.encode(anchors)
        positive_embeds = self.encode(positives)
        negative_embeds = self.encode(negatives)
        return anchor_embeds, positive_embeds, negative_embeds
    
    def encode(self, inputs, target_categories):
        inputs = stack_dict(inputs)
        
        category_embed = self.category_embedder(inputs['category']).squeeze(-2)
        img_embed = self.img_embedder(inputs['img'])
        masked_embed = img_embed * category_embed

        outputs = unstack_dict({
            'input_mask': inputs['input_mask'],
            'embed': masked_embed
            })
        return outputs
