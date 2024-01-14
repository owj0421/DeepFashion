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


class TypeAwareNet(DeepFashionModel):
    def __init__(
            self,
            embedding_dim: int = 64,
            num_category: int = 12,
            img_backbone: str = 'resnet-18',
            txt_backbone: str = 'bert'
            ):
        super().__init__(embedding_dim, num_category, img_backbone, txt_backbone)

        self.category_pairs = [(i, j) for i in range(num_category) for j in range(num_category) if i >= j]
        self.category_pair2id = dict()
        for i, category_pair in enumerate(self.category_pairs):
            self.category_pair2id[(category_pair[0], category_pair[1])] = i
            self.category_pair2id[(category_pair[1], category_pair[0])] = i

        self.category_embedding = nn.Embedding(num_embeddings=len(self.category_pairs), embedding_dim=embedding_dim)
        initrange = 0.1
        self.category_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, target_category=None):
        inputs = stack_dict(inputs)
        general_img_embed = self.img_encoder(inputs['image_features'])
        general_txt_embed = self.txt_encoder(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])

        outputs = DeepFashionOutput(mask=inputs['mask'])
        
        if target_category is not None:
            target_category = stack_tensors(inputs['mask'], target_category)
            category_ids = torch.LongTensor(list(map(lambda x: self.category_pair2id[x], zip(inputs['category'].tolist(), target_category.tolist()))))
            category_mask = self.category_embedding(category_ids.to(inputs['category'].get_device()))
            embed = unstack_tensors(inputs['mask'], general_img_embed * category_mask)
            txt_embed = unstack_tensors(inputs['mask'], general_txt_embed * category_mask)

            outputs.embed = embed

            return outputs

        else: # returns embedding for all categories
            embed_by_category = []
            txt_embed_by_category = []

            for i in range(self.num_category):
                target_category = torch.ones((inputs['category'].shape[0]), dtype=torch.long, device=inputs['category'].get_device()) * i
                category_ids = torch.LongTensor(list(map(lambda x: self.category_pair2id[x], zip(inputs['category'].tolist(), target_category.tolist()))))
                category_mask = self.category_embedding(category_ids.to(inputs['category'].get_device()))
                embed_by_category.append(unstack_tensors(inputs['mask'], general_img_embed * category_mask))
                txt_embed_by_category.append(unstack_tensors(inputs['mask'], general_txt_embed * category_mask))

            outputs.embed_by_category = embed_by_category
            outputs.txt_embed_by_category = txt_embed_by_category
            outputs.general_img_embed = unstack_tensors(inputs['mask'], general_img_embed)
            outputs.general_txt_embed = unstack_tensors(inputs['mask'], general_txt_embed)

            return outputs

    def compute_loss_comp(self, anchors, positives, negatives, anc_outputs, pos_outputs, neg_outputs):
        loss_comp = []
        for b_i in range(anc_outputs.mask.shape[0]):
            for a_i in range(torch.sum(~anc_outputs.mask[b_i])):
                for p_i in range(torch.sum(~pos_outputs.mask[b_i])):
                    for n_i in range(torch.sum(~neg_outputs.mask[b_i])):
                        anc_category = anchors['category'][b_i][a_i]
                        pos_category = positives['category'][b_i][p_i]
                        anc_embed = anc_outputs.embed_by_category[pos_category][b_i][a_i]
                        pos_embed = pos_outputs.embed_by_category[anc_category][b_i][p_i]
                        neg_embed = neg_outputs.embed_by_category[anc_category][b_i][n_i]
                        loss_comp.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, pos_embed, neg_embed))
        loss_comp = torch.mean(torch.stack(loss_comp))
        return loss_comp
    
    def compute_loss_sim(self, anchors, positives, negatives, anc_outputs, pos_outputs, neg_outputs, l_1, l_2):
        loss_sim_1 = []
        loss_sim_2 = []
        for b_i in range(anc_outputs.mask.shape[0]):
            for a_i in range(torch.sum(~anc_outputs.mask[b_i])):
                for p_i in range(torch.sum(~pos_outputs.mask[b_i])):
                    for n_i in range(torch.sum(~neg_outputs.mask[b_i])):
                        anc_category = anchors['category'][b_i][a_i]
                        pos_category = positives['category'][b_i][p_i]

                        anc_embed = anc_outputs.embed_by_category[pos_category][b_i][a_i]
                        pos_embed = pos_outputs.embed_by_category[anc_category][b_i][p_i]
                        neg_embed = neg_outputs.embed_by_category[anc_category][b_i][n_i]
                        loss_sim_1.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, neg_embed, pos_embed))

                        anc_embed = anc_outputs.txt_embed_by_category[pos_category][b_i][a_i]
                        pos_embed = pos_outputs.txt_embed_by_category[anc_category][b_i][p_i]
                        neg_embed = neg_outputs.txt_embed_by_category[anc_category][b_i][n_i]
                        loss_sim_2.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, neg_embed, pos_embed))

        loss_sim = l_1 * torch.mean(torch.stack(loss_sim_1)) + l_2 * torch.mean(torch.stack(loss_sim_2))
        return loss_sim

    def compute_loss_vse(self, anchors, positives, negatives, anc_outputs, pos_outputs, neg_outputs):
        loss_vse = []
        for b_i in range(anc_outputs.mask.shape[0]):
            for a_i in range(torch.sum(~anc_outputs.mask[b_i])):
                for p_i in range(torch.sum(~pos_outputs.mask[b_i])):
                    for n_i in range(torch.sum(~neg_outputs.mask[b_i])):
                        anc_category = anchors['category'][b_i][a_i]
                        pos_category = positives['category'][b_i][p_i]

                        anc_embed = anc_outputs.embed_by_category[pos_category][b_i][a_i]
                        pos_embed = pos_outputs.txt_embed_by_category[anc_category][b_i][p_i]
                        neg_embed = neg_outputs.txt_embed_by_category[anc_category][b_i][n_i]
                        loss_vse.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, pos_embed, neg_embed))

                        anc_embed = anc_outputs.txt_embed_by_category[pos_category][b_i][a_i]
                        pos_embed = pos_outputs.embed_by_category[anc_category][b_i][p_i]
                        neg_embed = neg_outputs.txt_embed_by_category[anc_category][b_i][n_i]
                        loss_vse.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, pos_embed, neg_embed))

                        anc_embed = anc_outputs.txt_embed_by_category[pos_category][b_i][a_i]
                        pos_embed = pos_outputs.txt_embed_by_category[anc_category][b_i][p_i]
                        neg_embed = neg_outputs.embed_by_category[anc_category][b_i][n_i]
                        loss_vse.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, pos_embed, neg_embed))
        loss_vse = torch.mean(torch.stack(loss_vse))
        return loss_vse
    
    def compute_loss_l2(self, anchors, positives, negatives, anc_outputs, pos_outputs, neg_outputs):
        loss_l2 = []
        loss_l2.append(torch.norm(torch.cat([stack_tensors(anc_outputs.mask, i) for i in anc_outputs.embed_by_category], dim=1), p='fro'))
        loss_l2.append(torch.norm(torch.cat([stack_tensors(pos_outputs.mask, i) for i in pos_outputs.embed_by_category], dim=1), p='fro'))
        loss_l2.append(torch.norm(torch.cat([stack_tensors(neg_outputs.mask, i) for i in neg_outputs.embed_by_category], dim=1), p='fro'))
        running_loss = torch.mean(torch.stack(loss_l2))
        return running_loss
    
    def compute_loss_l1(self):
        return torch.norm(self.category_embedding.weight.clone(), p='nuc')
        
    def evalutaion_step(self, batch, device):
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

                    q = question_outputs.embed_by_category[c_category][batch_i][q_i]
                    c = candidate_outputs.embed_by_category[q_category][batch_i][c_i]
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

        l_1, l_2, l_3, l_4, l_5 = 5e-4, 5e-4, 5e-1, 5e-4, 5e-4

        loss_comp = self.compute_loss_comp(anchors, positives, negatives, anc_outputs, pos_outputs, neg_outputs)
        loss_sim = self.compute_loss_sim(anchors, positives, negatives, anc_outputs, pos_outputs, neg_outputs, l_1=l_1, l_2=l_2)
        loss_vse = self.compute_loss_vse(anchors, positives, negatives, anc_outputs, pos_outputs, neg_outputs)
        loss_l2 = self.compute_loss_l2(anchors, positives, negatives, anc_outputs, pos_outputs, neg_outputs)
        loss_l1 = self.compute_loss_l1()
        running_loss = loss_comp + loss_sim + l_3 * loss_vse + l_4 * loss_l2 + l_5 * loss_l1

        return running_loss
    