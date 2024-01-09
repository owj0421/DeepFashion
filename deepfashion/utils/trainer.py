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
from deepfashion.utils.dataset_utils import *

def iteration(
        model, 
        dataloader, 
        epoch, 
        is_train, 
        device, 
        optimizer=None, 
        scheduler=None, 
        use_wandb=False
        ):
    type_str = 'Train' if is_train else 'Valid'
    epoch_iterator = tqdm(dataloader)
    total_loss = 0.
    for iter, batch in enumerate(epoch_iterator, start=1):
        # Forward
        anchors = {key: value.to(device) for key, value in batch['anchors'].items()}
        positives = {key: value.to(device) for key, value in batch['positives'].items()}
        negatives = {key: value.to(device) for key, value in batch['negatives'].items()}

        anchor_embeds, positive_embeds, negative_embeds = model(anchors, positives, negatives)

        # Compute running loss
        running_loss = nn.TripletMarginLoss(margin=2, reduction='mean')(anchor_embeds, positive_embeds, negative_embeds)
        total_loss += running_loss.item()
        if is_train == True:
            optimizer.zero_grad()
            running_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if scheduler:
                scheduler.step()
        # Log
        epoch_iterator.set_description(
            f'[{type_str}] Epoch: {epoch + 1:03} | Loss: {running_loss:.5f}'
            )
        if use_wandb:
            log = {
                f'{type_str}_loss': running_loss, 
                f'{type_str}_step': epoch * len(epoch_iterator) + iter
                }
            if is_train == True:
                log["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log)
    # Final Log
    total_loss = total_loss / iter
    if is_train == False:
        print( f'[E N D] Epoch: {epoch + 1:03} | loss: {total_loss:.5f} ' + '\n')
    return total_loss


def evaluate_cp():
    pass


def evaluate_fitb():
    pass


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
        self.scheduler = scheduler
        self.metric = metric
        self.args = args
        self.best_state = {}

    def fit(self):
        best_loss = np.inf
        for epoch in range(self.args.n_epochs):
            train_loss = self._train(epoch, self.train_dataloader)
            valid_loss = self._validate(epoch, self.valid_dataloader)
            if valid_loss < best_loss:
               best_loss = valid_loss
               self.best_state['model'] = deepcopy(self.model.state_dict())

            if epoch % self.args.save_every == 0:
                date = datetime.now().strftime('%Y-%m-%d')
                output_dir = os.path.join(self.args.work_dir, 'checkpoints', self.args.model, date)
                model_name = f'{epoch}_{best_loss:.3f}'
                self._save(output_dir, model_name)


    def _train(self, epoch: int, dataloader: DataLoader):
        self.model.train()
        is_train=True
        loss = iteration(
            model = self.model, 
            dataloader = dataloader, 
            epoch = epoch, 
            is_train = is_train, 
            device = self.device,
            optimizer = self.optimizer, 
            scheduler = self.scheduler, 
            use_wandb = self.args.use_wandb
            )
        return loss


    @torch.no_grad()
    def _validate(self, epoch: int, dataloader: DataLoader):
        self.model.eval()
        is_train=False
        loss = iteration(
            model = self.model, 
            dataloader = dataloader, 
            epoch = epoch, 
            is_train = is_train,
            use_wandb = self.args.use_wandb
            )
        return loss


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