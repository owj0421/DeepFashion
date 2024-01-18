"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import numpy as np
import random
import json
import torch
from torch import Tensor
from dataclasses import dataclass
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from tqdm import tqdm
from deepfashion.models.baseline import *


def stack_tensors(mask, tensor):
    B, S = mask.shape
    mask = mask.view(-1)
    s = list(tensor.shape)
    tensor = tensor.contiguous().view([s[0] * s[1]] + s[2:])
    tensor = tensor[~mask]
    return tensor


def unstack_tensors(mask, tensor):
    B, S = mask.shape
    mask = mask.view(-1)
    new_shape = [B * S] + list(tensor.shape)[1:]
    new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.get_device())
    new_tensor[~mask] = tensor
    new_tensor = new_tensor.contiguous().view([B, S] + list(tensor.shape)[1:])
    return new_tensor


def stack_dict(batch):
    batch = batch.copy()
    for i in batch.keys():
        if i == 'mask':
            continue
        batch[i] = stack_tensors(batch['mask'], batch[i])
    return batch


def unstack_dict(batch):
    batch = batch.copy()
    for i in batch.keys():
        if i == 'mask':
            continue
        batch[i] = unstack_tensors(batch['mask'], batch[i])
    return batch


def unstack_output(output):
    for i in output.__dict__.keys():
        if i == 'mask':
            continue
        setattr(output, i, unstack_tensors(output.mask, getattr(output, i)))
    return output

