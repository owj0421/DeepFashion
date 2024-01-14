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
from deepfashion.utils.dataset_utils import *
from deepfashion.models.encoder.builder import *
from deepfashion.models.baseline import *


class CSANet(DeepFashionModel):
    def __init__(
            self,
            embedding_dim: int = 64,
            num_category: int = 12,
            img_backbone: str = 'resnet-18'
            ):
        super().__init__(embedding_dim, num_category, img_backbone)
        self.mask = nn.Parameter(torch.ones((num_category, embedding_dim)))
        initrange = 0.1
        self.mask.data.uniform_(-initrange, initrange)
        self.attention = nn.Sequential(
            nn.Linear(num_category * 2, num_category * 2),
            nn.ReLU(),
            nn.Linear(num_category * 2, num_category)
            )
    
    def _get_embedding(self, inputs, target_category):
        # Get genreral embedding from inputs
        embed = self.img_encoder(inputs['img'])
        # Compute Attention Score
        input_category = self._one_hot(inputs['category'])
        target_category = self._one_hot(target_category)
        attention_query = torch.concat([input_category, target_category], dim=-1)
        attention = F.softmax(self.attention(attention_query), dim=-1)
        # Compute Subspace Mask via Attetion
        mask = torch.matmul(self.mask.T.unsqueeze(0), attention.unsqueeze(2)).squeeze(2)
        masked_embed = embed * mask
        masked_embed = unstack_tensors(inputs['input_mask'], masked_embed)

        return masked_embed

    def forward(self, inputs, target_category=None):
        inputs = stack_dict(inputs)
        
        if target_category is not None:
            target_category = stack_tensors(inputs['input_mask'], target_category)
            embed = self._get_embedding(inputs, target_category)
        else: # returns embedding for all categories
            embed_list = []
            for i in range(self.num_category):
                target_category = torch.ones((inputs['img'].shape[0]), dtype=torch.long, device=inputs['category'].get_device()) * i
                embed_list.append(self._get_embedding(inputs, target_category))
            embed = torch.stack(embed_list)
        
        outputs = DeepFashionOutput(mask=inputs['input_mask'], embed=embed)
        return outputs
    
    def evalutaion_step(self, batch, device) -> ndarray:
        questions = {key: value.to(device) for key, value in batch['questions'].items()}
        candidates = {key: value.to(device) for key, value in batch['candidates'].items()}

        question_outputs = self(questions)
        candidate_outputs = self(candidates)

        ans = []
        for batch_i in range(candidate_outputs.mask.shape[0]):
            dists = []
            for c_i in range(torch.sum(~(candidate_outputs.mask[batch_i]))):
                score = 0.
                for q_i in range(torch.sum(~(question_outputs.mask[batch_i]))):
                    q_category = questions['category'][batch_i][q_i]
                    c_category = candidates['category'][batch_i][c_i]

                    q = question_outputs.embed[c_category][batch_i][q_i]
                    c = candidate_outputs.embed[q_category][batch_i][c_i]
                    score += float(nn.PairwiseDistance(p=2)(q, c))
                dists.append(score)
            ans.append(np.argmin(np.array(dists)))
        ans = np.array(ans)
        return ans
    
    def iteration_step(self, batch, device) -> ndarray:
        anchors = {key: value.to(device) for key, value in batch['anchors'].items()}
        positives = {key: value.to(device) for key, value in batch['positives'].items()}
        negatives = {key: value.to(device) for key, value in batch['negatives'].items()}

        anc_outputs = self(anchors)
        pos_outputs = self(positives)
        neg_outputs = self(negatives)

        running_loss = []
        for b_i in range(anc_outputs.mask.shape[0]):
            for a_i in range(torch.sum(~anc_outputs.mask[b_i])):
                for n_i in range(torch.sum(~neg_outputs.mask[b_i])):
                    anc_category = anchors['category'][b_i][a_i]
                    pos_category = positives['category'][b_i][0]
                    anc_embed = anc_outputs.embed[pos_category][b_i][a_i]
                    pos_embed = pos_outputs.embed[anc_category][b_i][0]
                    neg_embed = neg_outputs.embed[anc_category][b_i][n_i]
                    running_loss.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, pos_embed, neg_embed))
        running_loss = torch.mean(torch.stack(running_loss))
        return running_loss
