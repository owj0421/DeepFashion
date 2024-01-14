"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from deepfashion.utils.dataset_utils import *
from deepfashion.models.encoder.builder import *

from itertools import combinations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

from deepfashion.utils.dataset_utils import *


@ dataclass
class DeepFashionOutput:
    mask: Optional[Tensor] = None
    embed: Optional[Tensor] = None


class DeepFashionModel(nn.Module):
    def __init__(
            self,
            embedding_dim: Optional[int] = 64,
            num_category: Optional[int] = 12,
            img_backbone: Literal['resnet-18', 'vgg-13', 'swin-transformer', 'vit'] = 'resnet-18',
            txt_backbone: Literal['bert'] = 'bert'
            ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_category = num_category
        self.img_encoder = build_img_encoder(img_backbone, embedding_dim=embedding_dim)
        self.txt_encoder = build_txt_encoder(txt_backbone, embedding_dim=embedding_dim)
        self.model = None
        pass

    def forward(self, inputs) -> DeepFashionOutput:
        inputs = stack_dict(inputs)
        outputs = None
        pass

        return outputs
    
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

        running_loss = []
        for batch_i in range(anc_outputs.mask.shape[0]):
            for anc_i in range(torch.sum(~anc_outputs.mask[batch_i])):
                for neg_i in range(torch.sum(~neg_outputs.mask[batch_i])):
                    anc_embed = anc_outputs.embed[batch_i][anc_i]
                    pos_embed = pos_outputs.embed[batch_i][0]
                    neg_embed = neg_outputs.embed[batch_i][neg_i]
                    running_loss.append(nn.TripletMarginLoss(margin=0.3, reduction='mean')(anc_embed, pos_embed, neg_embed))
        running_loss = torch.mean(torch.stack(running_loss))
        return running_loss
    
    def evaluation(self, dataloader, epoch, device, use_wandb=False):
        type_str = 'FITB'
        epoch_iterator = tqdm(dataloader)
        
        total_correct = 0.
        total_item = 0.
        for iter, batch in enumerate(epoch_iterator, start=1):
            ans = self.evalutaion_step(batch, device)

            run_correct = np.sum(ans==0)
            run_item = len(ans)
            run_acc = run_correct / run_item

            total_correct += run_correct
            total_item += run_item
            epoch_iterator.set_description(f'[{type_str}] Epoch: {epoch + 1:03} | Acc: {run_acc:.5f}')
            if use_wandb:
                log = {
                    f'{type_str}_acc': run_acc, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                wandb.log(log)
        total_acc = total_correct / total_item
        print( f'[{type_str} END] Epoch: {epoch + 1:03} | Acc: {total_acc:.5f} ' + '\n')

        return total_acc

    def iteration(self, dataloader, epoch, is_train, device,
                  optimizer=None, scheduler=None, use_wandb=False):
        type_str = 'TRAIN' if is_train else 'VALID'
        epoch_iterator = tqdm(dataloader)

        total_loss = 0.
        for iter, batch in enumerate(epoch_iterator, start=1):
            running_loss = self.iteration_step(batch, device)

            total_loss += running_loss.item()
            if is_train == True:
                optimizer.zero_grad()
                running_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.step()
                if scheduler:
                    scheduler.step()
            epoch_iterator.set_description(f'[{type_str}] Epoch: {epoch + 1:03} | Loss: {running_loss:.5f}')
            if use_wandb:
                log = {
                    f'{type_str}_loss': running_loss, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                if is_train == True:
                    log["learning_rate"] = scheduler.get_last_lr()[0]
                wandb.log(log)
        total_loss = total_loss / iter
        print( f'[{type_str} END] Epoch: {epoch + 1:03} | loss: {total_loss:.5f} ' + '\n')

        return total_loss
    
    def _one_hot(self, x):
        return F.one_hot(x, num_classes=self.num_category).to(torch.float32)