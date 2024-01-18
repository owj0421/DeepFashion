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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepfashion.utils.utils import *
from deepfashion.models.baseline import *

def _safe_divide(a, b, eps=1e-7):
    return a / (b + eps)


def _get_pos_mask(dist_matrix, outfit_outs, device):
    mask_self_eq = torch.ones(dist_matrix.shape, dtype=torch.bool, device=device).fill_diagonal_(False)

    label = torch.Tensor([idx for idx, i in enumerate(torch.sum(~outfit_outs.mask, dim=-1).tolist()) for j in range(i)]).to(device)
    mask_label_eq = torch.ne(label.unsqueeze(0), label.unsqueeze(1))

    pos_mask = torch.logical_and(mask_self_eq, ~mask_label_eq).float()
    return pos_mask


def _get_neg_mask(dist_matrix, outfit_outs, device, pos_indices=None):
    label = torch.Tensor([idx for idx, i in enumerate(torch.sum(~outfit_outs.mask, dim=-1).tolist()) for j in range(i)]).to(device)
    mask_label_eq = torch.ne(label.unsqueeze(0), label.unsqueeze(1))

    if pos_indices is not None:
        mask_pos_eq = torch.ones((dist_matrix.shape[0]), dtype=torch.bool, device=device)\
            .scatter_(0, pos_indices.flatten(), False).unsqueeze(-1).expand_as(dist_matrix)
        neg_mask = torch.logical_and(mask_pos_eq, mask_label_eq).float()
    else:
        neg_mask = mask_label_eq
    return neg_mask
    

def outfit_ranking_loss(
          outfit_outs: DeepFashionOutput, 
          margin: float = 0.3,
          method: Literal['batch_all', 'batch_hard'] = 'batch_all',
          aggregation: Literal['mean', 'sum', 'none'] = 'mean'
          ) -> Tensor:
        """Outfit ranking loss, first appeared in `Category-based Subspace Attention Network(2020, Yen-liang L, Son Tran, et al.)`.
        Implemented in Online mining manner.

        :param outfit_outs: Should instance of DeepFashionOutput
        :param margin: Margin for Triplet loss.
        :param method: How to handle a negative sample.
            - 'batch_all': Use the average of all negatives in the mini-batch.
            - 'batch_hard': Use the most hardest sample in the mini-batch.
        :param aggregation: How to handle a final output. same as torch triplet loss' one.
        :return: Outfit Ranking Loss
        """
        device=outfit_outs.mask.get_device()

        outfit_outs.embed_by_category = torch.stack(list(map(lambda x: stack_tensors(outfit_outs.mask, x), outfit_outs.embed_by_category)))
        category_idx = stack_tensors(outfit_outs.mask,  outfit_outs.category)
        dist_matrix = nn.PairwiseDistance()(outfit_outs.embed_by_category[category_idx], outfit_outs.embed_by_category[category_idx].transpose(0, 1))

        # Compute Positive distances
        pos_matrix = dist_matrix * _get_pos_mask(dist_matrix, outfit_outs, device)
        outfit_wise_pos_matrix = _safe_divide(torch.sum(unstack_tensors(outfit_outs.mask, pos_matrix), dim=1), torch.sum(~outfit_outs.mask, dim=-1, keepdim=True) - 1)
        dist_pos, pos_indices = torch.max(outfit_wise_pos_matrix, dim=-1, keepdim=True)
        
        # Compute Negative distances
        neg_matrix = dist_matrix * _get_neg_mask(dist_matrix, outfit_outs, device, pos_indices=pos_indices)
        outfit_wise_neg_matrix = _safe_divide(torch.sum(unstack_tensors(outfit_outs.mask, neg_matrix), dim=1), torch.sum(~outfit_outs.mask, dim=-1, keepdim=True) - 1)
        if method == 'batch_all':
            dist_neg = _safe_divide(torch.sum(outfit_wise_neg_matrix, dim=-1, keepdim=True), torch.sum(torch.where(outfit_wise_neg_matrix == 0., 0., 1.), dim=-1, keepdim=True))
        elif method == 'batch_hard':
            outfit_wise_neg_matrix = outfit_wise_neg_matrix + torch.where(outfit_wise_neg_matrix == 0., torch.max(outfit_wise_neg_matrix), 0.)
            dist_neg, _ = torch.min(outfit_wise_neg_matrix, dim=-1, keepdim=True)
        else:
            raise ValueError('')
        
        hinge_dist = torch.clamp(margin + dist_pos - dist_neg, min=0.0)
        if aggregation == 'mean':
            loss = torch.mean(hinge_dist)
        elif aggregation == 'sum':
            loss = torch.sum(hinge_dist)
        elif aggregation == 'none':
            loss = hinge_dist
        else:
            raise ValueError('')
        
        return loss

def triplet_loss(
          outfit_outs: DeepFashionOutput, 
          margin: float = 0.3,
          method: Literal['batch_all', 'batch_hard'] = 'batch_all',
          aggregation: Literal['mean', 'sum', 'none'] = 'mean'
          ) -> Tensor:
        """Triplet margin loss.
        Implemented in Online mining manner.

        :param outfit_outs: Should instance of DeepFashionOutput
        :param margin: Margin for Triplet loss.
        :param method: How to handle a negative sample.
            - 'batch_all': Use the average of all negatives in the mini-batch.
            - 'batch_hard': Use the most hardest sample in the mini-batch.
        :param aggregation: How to handle a final output. same as torch triplet loss' one.
        :return: Outfit Ranking Loss
        """
        device=outfit_outs.mask.get_device()

        outfit_outs.embed_by_category = torch.stack(list(map(lambda x: stack_tensors(outfit_outs.mask, x), outfit_outs.embed_by_category)))
        category_idx = stack_tensors(outfit_outs.mask,  outfit_outs.category)
        dist_matrix = nn.PairwiseDistance()(outfit_outs.embed_by_category[category_idx], outfit_outs.embed_by_category[category_idx].transpose(0, 1))

        # Compute Positive distances
        pos_matrix = dist_matrix * _get_pos_mask(dist_matrix, outfit_outs, device)
        dist_pos, pos_indices = torch.max(pos_matrix, dim=-1, keepdim=True)
        
        # Compute Negative distances
        neg_matrix = dist_matrix * _get_neg_mask(dist_matrix, outfit_outs, device)
        if method == 'batch_all':
            dist_neg = _safe_divide(torch.sum(neg_matrix, dim=-1, keepdim=True), torch.sum(torch.where(neg_matrix == 0., 0., 1.), dim=-1, keepdim=True))
        elif method == 'batch_hard':
            neg_matrix = neg_matrix + torch.where(neg_matrix == 0., torch.max(neg_matrix), 0.)
            dist_neg, _ = torch.min(neg_matrix, dim=-1, keepdim=True)
        else:
            raise ValueError('')
        
        hinge_dist = torch.clamp(margin + dist_pos - dist_neg, min=0.0)
        if aggregation == 'mean':
            loss = torch.mean(hinge_dist)
        elif aggregation == 'sum':
            loss = torch.sum(hinge_dist)
        elif aggregation == 'none':
            loss = hinge_dist
        else:
            raise ValueError('')
        
        return loss


# def vse_loss(
#           anc_outs: DeepFashionOutput, 
#           pos_outs: DeepFashionOutput, 
#           neg_outs: DeepFashionOutput, 
#           margin: float = 0.3
#           ):
#         n_outfit = anc_outs.mask.shape[0]
#         ans_per_batch = torch.sum(~anc_outs.mask, dim=-1)
#         neg_per_batch = torch.sum(~neg_outs.mask, dim=-1)

#         loss = []
#         for b_i in range(n_outfit):
#             for a_i in range(ans_per_batch[b_i]):
#                 for n_i in range(neg_per_batch[b_i]):
#                     anc_embed = anc_outs.embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
#                     pos_embed = pos_outs.txt_embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
#                     neg_embed = neg_outs.txt_embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
#                     loss.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(anc_embed, pos_embed, neg_embed))

#                     anc_embed = anc_outs.txt_embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
#                     pos_embed = pos_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
#                     neg_embed = neg_outs.txt_embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
#                     loss.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(anc_embed, pos_embed, neg_embed))

#                     anc_embed = anc_outs.txt_embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
#                     pos_embed = pos_outs.txt_embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
#                     neg_embed = neg_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
#                     loss.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(anc_embed, pos_embed, neg_embed))
#         loss = torch.mean(torch.stack(loss))
#         return loss


# def sim_loss(
#           anc_outs: DeepFashionOutput, 
#           pos_outs: DeepFashionOutput, 
#           neg_outs: DeepFashionOutput, 
#           margin: float = 0.3,
#           l_1: float = 5e-4,
#           l_2: float = 5e-4,
#           ):
#         n_outfit = anc_outs.mask.shape[0]
#         ans_per_batch = torch.sum(~anc_outs.mask, dim=-1)
#         neg_per_batch = torch.sum(~neg_outs.mask, dim=-1)

#         loss_1 = []
#         loss_2 = []
#         for b_i in range(n_outfit):
#             for a_i in range(ans_per_batch[b_i]):
#                 for n_i in range(neg_per_batch[b_i]):
#                     anc_embed = anc_outs.embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
#                     pos_embed = pos_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
#                     neg_embed = neg_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
#                     loss_1.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(pos_embed, neg_embed, anc_embed))

#                     if hasattr(anc_outs, 'txt_embed_by_category'):
#                         anc_embed = anc_outs.embed_by_category[pos_outs.category[b_i][0]][b_i][a_i]
#                         pos_embed = pos_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][0]
#                         neg_embed = neg_outs.embed_by_category[anc_outs.category[b_i][a_i]][b_i][n_i]
#                         loss_2.append(nn.TripletMarginLoss(margin=margin, reduction='mean')(pos_embed, neg_embed, anc_embed))
#         loss = l_1 * torch.mean(torch.stack(loss_1))
#         if loss_2:
#              loss += l_2 * torch.mean(torch.stack(loss_2))
#         return loss