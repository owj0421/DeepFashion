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


def safe_divide(a, b, eps=1e-7):
    return a / (b + eps)


def get_pos_mask(dist_matrix, outfit_outs):
    device=outfit_outs.mask.get_device()

    mask_self_eq = torch.ones(dist_matrix.shape, dtype=torch.bool, device=device).fill_diagonal_(False)

    label = torch.Tensor([idx for idx, i in enumerate(torch.sum(~outfit_outs.mask, dim=-1).tolist()) for j in range(i)]).to(device)
    mask_label_eq = torch.ne(label.unsqueeze(0), label.unsqueeze(1))

    pos_mask = torch.logical_and(mask_self_eq, ~mask_label_eq).float()
    return pos_mask


def get_neg_mask(dist_matrix, outfit_outs, pos_indices=None):
    device=outfit_outs.mask.get_device()

    label = torch.Tensor([idx for idx, i in enumerate(torch.sum(~outfit_outs.mask, dim=-1).tolist()) for j in range(i)]).to(device)
    mask_label_eq = torch.ne(label.unsqueeze(0), label.unsqueeze(1))

    if pos_indices is not None:
        mask_pos_eq = torch.ones((dist_matrix.shape[0]), dtype=torch.bool, device=device)\
            .scatter_(0, pos_indices.flatten(), False).unsqueeze(-1).expand_as(dist_matrix)
        neg_mask = torch.logical_and(mask_pos_eq, mask_label_eq).float()
    else:
        neg_mask = mask_label_eq
    return neg_mask