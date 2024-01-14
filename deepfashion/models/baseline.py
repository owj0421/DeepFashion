"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from DeepFashion.deepfashion.utils.utils import *
from deepfashion.models.encoder.builder import *

from itertools import combinations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

from deepfashion.utils.utils import *


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
        return NotImplementedError("DeepFashionModel must implement its own forward method")
    
    def evalutaion_step(self, batch, device) -> np.ndarray:
        return NotImplementedError("DeepFashionModel must implement its own evalutaion_step method")
    
    def iteration_step(self, batch, device) -> np.ndarray:
        return NotImplementedError("DeepFashionModel must implement its own iteration_step method")
    
    def evaluation(self, dataloader, epoch, device, use_wandb=False):
        type_str = 'fitb'
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
        type_str = 'Train' if is_train else 'Valid'
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