import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfashion.utils.dataset_utils import *
from deepfashion.models.embedder.builder import *


class TypeAwareNet(nn.Module):
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
        anchors = self.encode(anchors)
        positives = self.encode(positives)
        negatives = self.encode(negatives)

        input_mask = anchors['input_mask']
        mask = input_mask.view(-1)
        B, S, E = anchors['embed'].shape
        B, N, E = negatives['embed'].shape
        # [B, S, E] -> [B, S, 1, E] -> [B, S, N, E] -> [B * S, N, E] -> [B * V, N, E] -> [B * V * N, E]
        anchors = anchors['embed'].unsqueeze(2).expand(B, S, N, E).contiguous().view(B * S, N, E)[~mask].view(-1, E)
        # [B, 1, E] -> [B, V * N, E]
        positives = positives['embed'].unsqueeze(2).expand(B, S, N, E).contiguous().view(B * S, N, E)[~mask].view(-1, E)
        # [B, N, E] -> [B, 1, N, E] -> [B, S, N, E] -> [B * S, N, E] -> [B * V, N, E] -> [B * V * N, E]
        negatives = negatives['embed'].unsqueeze(1).expand(B, S, N, E).contiguous().view(B * S, N, E)[~mask].view(-1, E)
        
        return anchors, positives, negatives
    
    def encode(self, inputs):
        inputs = stack_dict(inputs)
        
        category_embed = self.category_embedder(inputs['category']).squeeze(-2)
        img_embed = self.img_embedder(inputs['img'])
        masked_embed = img_embed * category_embed

        outputs = unstack_dict({
            'input_mask': inputs['input_mask'],
            'embed': masked_embed
            })
        return outputs
