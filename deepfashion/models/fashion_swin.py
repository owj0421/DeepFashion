"""
Author:
    Wonjun Oh, owj0421@naver.com
Reference:
    [1] Hosna Darvishi, Reza Azmi, et al. Fashion Compatibility Learning Via Triplet-Swin Transformer. IEEE, 2023.
    (https://ieeexplore.ieee.org/abstract/document/10105392)
"""
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepfashion.utils.dataset_utils import *
from deepfashion.models.encoder.builder import *

from itertools import combinations


class FashionSwin(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 64,
            num_category: int = 12,
            img_backbone: str = 'swin-transformer',
            txt_backbone: str = 'bert'
            ):
        super().__init__()
        self.img_encoder = build_img_encoder(img_backbone, embedding_dim=embedding_dim)
        self.txt_encoder = build_txt_encoder(txt_backbone, embedding_dim=embedding_dim)

        self.category_embedding = nn.Embedding(num_embeddings=num_category, embedding_dim=embedding_dim)
        initrange = 0.1
        self.category_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        inputs = stack_dict(inputs)
        outputs = {
            'input_mask': inputs['input_mask'],
            'img_general_embed': None,
            'img_embed': None,
            'txt_general_embed' : None,
            'txt_embed' : None
            }

        category_embed = self.category_embedding(inputs['category'])
        outputs['img_general_embed'] = self.img_encoder(inputs['img'])
        outputs['img_embed'] = outputs['img_general_embed'] * category_embed
        outputs['txt_general_embed'] = self.txt_encoder(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])
        outputs['txt_embed'] = outputs['txt_general_embed'] * category_embed
        outputs = unstack_dict(outputs)

        return outputs
    
    def evaluation(self, dataloader, epoch, device, use_wandb=False):
        epoch_iterator = tqdm(dataloader)

        correct = 0.
        total = 0.
        for iter, batch in enumerate(epoch_iterator, start=1):
            questions = {key: value.to(device) for key, value in batch['questions'].items()}
            candidates = {key: value.to(device) for key, value in batch['candidates'].items()}

            question_outputs = self(questions)
            candidate_outputs = self(candidates)

            ans = []
            for batch_i in range(len(candidate_outputs['input_mask'])):
                dists = []
                for c_i in range(torch.sum(~candidate_outputs['input_mask'][batch_i])):
                    score = 0.
                    for q_i in range(torch.sum(~question_outputs['input_mask'][batch_i])):
                        q = question_outputs['img_embed'][batch_i][q_i]
                        c = candidate_outputs['img_embed'][batch_i][c_i]
                        score += float(nn.PairwiseDistance(p=2)(q, c))
                    dists.append(score)
                ans.append(np.argmin(np.array(dists)))
                total += 1.
            ans = np.array(ans)

            running_correct = np.sum(np.array(ans)==0)
            running_acc = running_correct / len(ans)
            correct += running_correct
            epoch_iterator.set_description(f'[FITB] Epoch: {epoch + 1:03} | Acc: {running_acc:.5f}')
            if use_wandb:
                log = {
                    f'FITB_acc': running_acc, 
                    f'FITB_step': epoch * len(epoch_iterator) + iter
                    }
                wandb.log(log)
        # Final Log
        total_acc = correct / total
        print( f'[FITB END] Epoch: {epoch + 1:03} | Acc: {total_acc:.5f} ' + '\n')

    def iteration(self, dataloader, epoch, is_train, device,
                  optimizer=None, scheduler=None, use_wandb=False):
        type_str = 'Train' if is_train else 'Valid'
        epoch_iterator = tqdm(dataloader)

        total_loss = 0.
        for iter, batch in enumerate(epoch_iterator, start=1):
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

            total_loss += running_loss.item()
            if is_train == True:
                optimizer.zero_grad()
                running_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.step()
                if scheduler:
                    scheduler.step()
            # Log
            epoch_iterator.set_description(f'[{type_str}] Epoch: {epoch + 1:03} | Loss: {running_loss:.5f}')
            if use_wandb:
                log = {
                    f'{type_str}_loss': running_loss, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                if is_train == True:
                    log["learning_rate"] = scheduler.get_last_lr()[0]
                wandb.log(log)

        # Final Log
        total_loss = total_loss / iter
        print( f'[{type_str} END] Epoch: {epoch + 1:03} | loss: {total_loss:.5f} ' + '\n')

        return total_loss
    
    def compute_loss_triplet(self, anc_outputs, pos_outputs, neg_outputs):
        loss_triplet = []
        for batch_i in range(len(anc_outputs['input_mask'])):
            for anc_i in range(torch.sum(~anc_outputs['input_mask'][batch_i])):
                for neg_i in range(torch.sum(~neg_outputs['input_mask'][batch_i])):
                    anc_embed = anc_outputs['img_embed'][batch_i][anc_i]
                    pos_embed = pos_outputs['img_embed'][batch_i][0]
                    neg_embed = neg_outputs['img_embed'][batch_i][neg_i]
                    loss_triplet.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, pos_embed, neg_embed))
        loss_triplet = torch.mean(torch.stack(loss_triplet))
        return loss_triplet
    
    def compute_loss_vse(self, output_i, output_j):
        loss_vse = []
        for batch_i in range(len(output_i['input_mask'])):
            for i_idx in range(torch.sum(~output_i['input_mask'][batch_i])):
                for j_idx in range(torch.sum(~output_j['input_mask'][batch_i])):
                    i_img_embed = output_i['img_general_embed'][batch_i][i_idx]
                    i_txt_embed = output_i['txt_general_embed'][batch_i][i_idx]
                    j_txt_embed = output_j['txt_general_embed'][batch_i][j_idx]
                    loss_vse.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(i_img_embed, i_txt_embed, j_txt_embed))
        loss_vse = torch.mean(torch.stack(loss_vse))
        return loss_vse
        