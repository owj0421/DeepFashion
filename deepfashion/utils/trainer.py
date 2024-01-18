"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import math
import wandb
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepfashion.utils.utils import *
from deepfashion.models.baseline import *


@dataclass
class TrainingArguments:
    model: str
    train_batch: int=8
    valid_batch: int=32
    fitb_batch: int=32
    n_epochs: int=100
    learning_rate: float=0.01
    save_every: int=1
    work_dir: str=None
    use_wandb: bool=False
    device: str='cuda'


class Trainer:
    def __init__(
            self,
            args: TrainingArguments,
            model: nn.Module,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            valid_fitb_dataloader: DataLoader,
            test_fitb_dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
            ):
        self.device = torch.device(args.device)
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.valid_fitb_dataloader = valid_fitb_dataloader
        self.test_fitb_dataloader = test_fitb_dataloader
        self.scheduler = scheduler
        self.args = args
        self.best_state = {}

    def fit(self):
        best_criterion = -np.inf
        for epoch in range(self.args.n_epochs):
            train_loss = self._train(epoch)
            valid_loss = self._validate(epoch)
            criterion = self._evaluate(epoch)
            if criterion > best_criterion:
               best_criterion = criterion
               self.best_state['model'] = deepcopy(self.model.state_dict())

            if epoch % self.args.save_every == 0:
                date = datetime.now().strftime('%Y-%m-%d')
                output_dir = os.path.join(self.args.work_dir, 'checkpoints', self.args.model, date)
                model_name = f'{epoch}_{best_criterion:.3f}'
                self._save(output_dir, model_name)
                
            self._test(epoch)


    def _train(self, epoch: int):
        self.model.train()
        loss = self.model.iteration(
            dataloader = self.train_dataloader, 
            epoch = epoch, 
            is_train = True, 
            device = self.device,
            optimizer = self.optimizer, 
            scheduler = self.scheduler, 
            use_wandb = self.args.use_wandb
            )
        return loss


    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()
        loss = self.model.iteration(
            dataloader = self.valid_dataloader, 
            epoch = epoch, 
            is_train = False,
            device = self.device,
            use_wandb = self.args.use_wandb
            )
        return loss


    @torch.no_grad()
    def _evaluate(self, epoch: int):
        self.model.eval()
        criterion = self.model.evaluation(
            dataloader = self.valid_fitb_dataloader,
            epoch = epoch,
            is_test = False,
            device = self.device,
            use_wandb = self.args.use_wandb
            )
        return criterion
    

    @torch.no_grad()
    def _test(self, epoch: int):
        self.model.eval()
        criterion = self.model.evaluation(
            dataloader = self.test_fitb_dataloader,
            epoch = epoch,
            is_test = True,
            device = self.device,
            use_wandb = self.args.use_wandb
            )
        return criterion
    

    def _save(self, dir, model_name, best_model: bool=True):
        def _create_folder(dir):
            try:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            except OSError:
                print('[Error] Creating directory.' + dir)
        _create_folder(dir)

        path = os.path.join(dir, f'{model_name}.pth')
        checkpoint = {
            'model_state_dict': self.best_state['model'] if best_model else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
            }
        torch.save(checkpoint, path)
        print(f'[COMPLETE] Save at {path}')


    def load(self, path, load_optim=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=False)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'], strict=False)
        print(f'[COMPLETE] Load from {path}')