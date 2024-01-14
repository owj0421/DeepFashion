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
from deepfashion.utils.utils import *
from deepfashion.models.encoder.builder import *
from deepfashion.models.baseline import *

from itertools import combinations


class FashionSwin(DeepFashionModel):
    pass