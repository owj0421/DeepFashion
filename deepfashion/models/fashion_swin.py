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
from deepfashion.utils.dataset_utils import *
from deepfashion.models.encoder.builder import *
from deepfashion.models.baseline import *

from itertools import combinations


class TypeAwareNet(DeepFashionModel):
    def __init__(
            self,
            embedding_dim: int = 64,
            num_category: int = 12,
            img_backbone: str = 'swin-transformer',
            txt_backbone: str = 'bert'
            ):
        super().__init__(embedding_dim, num_category, img_backbone, txt_backbone)

        self.category_embedding = nn.Embedding(num_embeddings=num_category, embedding_dim=embedding_dim)
        initrange = 0.1
        self.category_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        inputs = stack_dict(inputs)
        outputs = DeepFashionOutput()
        outputs.mask = inputs['input_mask']
        category_embed = self.category_embedding(inputs['category'])
        outputs.img_general_embed = self.img_encoder(inputs['img'])
        outputs.embed = outputs.img_general_embed * category_embed
        outputs.txt_general_embed = self.txt_encoder(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])
        outputs.txt_embed = outputs.txt_general_embed * category_embed
        outputs = unstack_output(outputs)

        return outputs
    
    def compute_loss_triplet(self, anc_outputs, pos_outputs, neg_outputs):
        loss_triplet = []
        for batch_i in range(anc_outputs.mask.shape[0]):
            for anc_i in range(torch.sum(~anc_outputs.mask[batch_i])):
                for neg_i in range(torch.sum(~neg_outputs.mask[batch_i])):
                    anc_embed = anc_outputs.embed[batch_i][anc_i]
                    pos_embed = pos_outputs.embed[batch_i][0]
                    neg_embed = neg_outputs.embed[batch_i][neg_i]
                    loss_triplet.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, pos_embed, neg_embed))
        loss_triplet = torch.mean(torch.stack(loss_triplet))
        return loss_triplet
    
    def compute_loss_vse(self, output_i, output_j):
        loss_vse = []
        for batch_i in range(output_i.mask.shape[0]):
            for i_idx in range(torch.sum(~output_i.mask[batch_i])):
                for j_idx in range(torch.sum(~output_j.mask[batch_i])):
                    i_img_embed = output_i.img_general_embed[batch_i][i_idx]
                    i_txt_embed = output_i.txt_general_embed[batch_i][i_idx]
                    j_txt_embed = output_j.txt_general_embed[batch_i][j_idx]
                    loss_vse.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(i_img_embed, i_txt_embed, j_txt_embed))
        loss_vse = torch.mean(torch.stack(loss_vse))
        return loss_vse
        
    def evalutaion_step(self, batch, device) -> np.ndarray:
        questions = {key: value.to(device) for key, value in batch['questions'].items()}
        candidates = {key: value.to(device) for key, value in batch['candidates'].items()}

        question_outputs = self(questions)
        candidate_outputs = self(candidates)

        ans = []
        for batch_i in range(candidate_outputs.mask.shape[0]):
            dists = []
            for c_i in range(torch.sum(~candidate_outputs.mask[batch_i])):
                score = 0.
                for q_i in range(torch.sum(~question_outputs.mask[batch_i])):
                    q = question_outputs.embed[batch_i][q_i]
                    c = candidate_outputs.embed[batch_i][c_i]
                    score += float(nn.PairwiseDistance(p=2)(q, c))
                dists.append(score)
            ans.append(np.argmin(np.array(dists)))
        ans = np.array(ans)
        return ans
    
    def iteration_step(self, batch, device) -> np.ndarray:
        anchors = {key: value.to(device) for key, value in batch['anchors'].items()}
        positives = {key: value.to(device) for key, value in batch['positives'].items()}
        negatives = {key: value.to(device) for key, value in batch['negatives'].items()}

        anc_outputs = self(anchors)
        pos_outputs = self(positives)
        neg_outputs = self(negatives)

        # Loss Triplet
        loss_comp = self.compute_loss_triplet(anc_outputs, pos_outputs, neg_outputs)
        # Loss VSE
        loss_vse = []
        for i, j in combinations([anc_outputs, pos_outputs, neg_outputs], 2):
            loss_vse.append(self.compute_loss_vse(i, j))
        loss_vse = torch.mean(torch.stack(loss_vse))
        running_loss = (loss_comp + loss_vse) / 2

        return running_loss