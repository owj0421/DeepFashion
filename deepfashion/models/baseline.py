# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
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
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepfashion.utils.utils import *
from deepfashion.models.baseline import *
from deepfashion.models.encoder.builder import *



@ dataclass
class DeepFashionOutput:
    mask: Optional[Tensor] = None
    category: Optional[Tensor] = None
    embed: Optional[Tensor] = None
    embed_by_category: Optional[Tensor] = None


class DeepFashionModel(nn.Module):
    def __init__(
            self,
            embedding_dim: Optional[int] = 32,
            categories: Optional[List[str]] = None,
            img_backbone: Literal['resnet-18', 'vgg-13', 'swin-transformer', 'vit', 'none'] = 'resnet-18',
            txt_backbone: Literal['bert', 'none'] = 'none',
            margin: float = 0.3
            ):
        super().__init__()
        self.embedding_dim = embedding_dim
        if 'pad' not in categories:
            categories += ['pad']
        self.num_category = len(categories)
        self.img_encoder = build_img_encoder(img_backbone, embedding_dim=embedding_dim)
        if txt_backbone != 'none':
            self.txt_encoder = build_txt_encoder(txt_backbone, embedding_dim=embedding_dim)
        self.model = None
        self.margin = margin
        pass

    def forward(self, inputs) -> DeepFashionOutput:
        return NotImplementedError("DeepFashionModel must implement its own forward method")
    
    def iteration_step(self, batch, device) -> np.ndarray:
        return NotImplementedError("DeepFashionModel must implement its own iteration_step method")
    
    def evalutaion_step(self, batch, device):
        questions = {key: value.to(device) for key, value in batch['questions'].items()}
        candidates = {key: value.to(device) for key, value in batch['candidates'].items()}

        question_outs = self(questions)
        candidate_outs = self(candidates)

        ans = []
        for b_i in range(candidate_outs.mask.shape[0]):
            dists = []
            for c_i in range(torch.sum(~(candidate_outs.mask[b_i]))):
                score = 0.
                for q_i in range(torch.sum(~(question_outs.mask[b_i]))):
                    q = question_outs.embed_by_category[candidate_outs.category[b_i][c_i]][b_i][q_i]
                    c = candidate_outs.embed_by_category[question_outs.category[b_i][q_i]][b_i][c_i]
                    score += float(nn.PairwiseDistance(p=2)(q, c))
                dists.append(score)
            # print(dists)
            ans.append(np.argmin(np.array(dists)))
        ans = np.array(ans)
        return ans
    
    def evaluation(self, dataloader, epoch, is_test, device, use_wandb=False):
        type_str = 'test' if is_test else 'fitb'
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
        type_str = 'train' if is_train else 'valid'
        epoch_iterator = tqdm(dataloader)

        total_loss = 0.
        for iter, batch in enumerate(epoch_iterator, start=1):
            running_loss = self.iteration_step(batch, device)

            total_loss += running_loss.item()
            if is_train == True:
                optimizer.zero_grad()
                running_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
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