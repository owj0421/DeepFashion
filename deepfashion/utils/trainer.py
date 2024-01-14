import os
import wandb
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from datetime import datetime
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader
from .metric import *
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from deepfashion.utils.utils import *


@dataclass
class TrainingArguments:
    model: str
    train_batch: int=8
    valid_batch: int=32
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
            fitb_dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            metric: MetricCalculator,
            scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
            ):
        self.device = torch.device(args.device)
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.fitb_dataloader = fitb_dataloader
        self.scheduler = scheduler
        self.metric = metric
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


    def _train(self, epoch: int):
        self.model.train()
        is_train=True
        loss = self.model.iteration(
            dataloader = self.train_dataloader, 
            epoch = epoch, 
            is_train = is_train, 
            device = self.device,
            optimizer = self.optimizer, 
            scheduler = self.scheduler, 
            use_wandb = self.args.use_wandb
            )
        return loss


    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()
        is_train=False
        loss = self.model.iteration(
            dataloader = self.valid_dataloader, 
            epoch = epoch, 
            is_train = is_train,
            device = self.device,
            use_wandb = self.args.use_wandb
            )
        return loss

    @torch.no_grad()
    def _evaluate(self, epoch: int):
        self.model.eval()
        criterion = self.model.evaluation(
            dataloader = self.fitb_dataloader,
            epoch = epoch,
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