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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepfashion.utils.utils import *
from deepfashion.models.baseline import *
from deepfashion.loss.utils import *

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
        :return: Triplet margin loss
        """
        device=outfit_outs.mask.get_device()

        outfit_outs.embed_by_category = torch.stack(list(map(lambda x: stack_tensors(outfit_outs.mask, x), outfit_outs.embed_by_category)))
        category_idx = stack_tensors(outfit_outs.mask,  outfit_outs.category)
        dist_matrix = nn.PairwiseDistance()(outfit_outs.embed_by_category[category_idx], outfit_outs.embed_by_category[category_idx].transpose(0, 1))

        # Compute Positive distances
        pos_matrix = dist_matrix * get_pos_mask(dist_matrix, outfit_outs)
        dist_pos, pos_indices = torch.max(pos_matrix, dim=-1, keepdim=True)

        # Compute Negative distances
        neg_matrix = dist_matrix * get_neg_mask(dist_matrix, outfit_outs)
        
        if method == 'batch_all':
            dist_neg = safe_divide(torch.sum(neg_matrix, dim=-1), torch.count_nonzero(neg_matrix, dim=-1))
        elif method == 'batch_hard':
            neg_matrix[neg_matrix < dist_pos.expand_as(neg_matrix)] = 0.
            neg_matrix = neg_matrix + torch.where(neg_matrix == 0., torch.max(neg_matrix), 0.)
            dist_neg, _ = torch.min(neg_matrix, dim=-1, keepdim=True)
        else:
            raise ValueError('task_type must be one of `batch_all` and `batch_hard`.')
        
        
        hinge_dist = torch.clamp(margin + dist_pos - dist_neg, min=0.0)
        if aggregation == 'mean':
            loss = torch.mean(hinge_dist)
        elif aggregation == 'sum':
            loss = torch.sum(hinge_dist)
        elif aggregation == 'none':
            loss = hinge_dist
        else:
            raise ValueError('aggregation must be one of `mean`, `sum` and `none`.')
        
        return loss