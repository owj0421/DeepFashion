"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import wandb
import argparse
from itertools import chain

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from deepfashion.models.type_aware_net import TypeAwareNet
from deepfashion.models.csa_net import CSANet
from deepfashion.models.fashion_swin import FashionSwin

from deepfashion.utils.trainer import *
from deepfashion.datasets.polyvore import *

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='DeepFashion')
    parser.add_argument('--model', help='Model', type=str, default='csa-net')
    parser.add_argument('--embedding_dim', help='embedding dim', type=int, default=32)
    parser.add_argument('--img_backbone', default='resnet-18') # 'efficientnet-b0', 'resnet-18', 'swin-transformer', 'vgg-13'

    parser.add_argument('--dataset_type', help='dataset_type', type=str, default='outfit')
    parser.add_argument('--use_text', help='', type=bool, default=False)
    parser.add_argument('--use_text_feature', help='', type=bool, default=False)
    parser.add_argument('--outfit_max_length', help='', type=int, default=16)

    parser.add_argument('--train_batch', help='Size of Batch for Training', type=int, default=32)
    parser.add_argument('--valid_batch', help='Size of Batch for Validation, Test', type=int, default=32)
    parser.add_argument('--fitb_batch', help='Size of Batch for FITB evaluation', type=int, default=12)
    parser.add_argument('--n_epochs', help='Number of epochs', type=int, default=1)
    parser.add_argument('--save_every', help='', type=int, default=1)
    
    parser.add_argument('--work_dir', help='Full working directory', type=str, default='F:\Projects\DeepFashion')
    parser.add_argument('--data_dir', help='Full dataset directory', type=str, default='F:\Projects\datasets\polyvore_outfits')
    parser.add_argument('--num_workers', help='', type=int, default=4)

    parser.add_argument('--scheduler_step_size', help='Step LR', type=int, default=1000)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=1e-5)

    parser.add_argument('--wandb_api_key', default='fa37a3c4d1befcb0a7b9b4d33799c7bdbff1f81f') # 
    parser.add_argument('--checkpoint', default='F:/Projects/DeepFashion/checkpoints/csa-net/2024-01-19/1_0.567.pth') # 

    args = parser.parse_args()

    # Wandb
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        os.environ["WANDB_PROJECT"] = f"deep-fashion-{args.model}"
        os.environ["WANDB_LOG_MODEL"] = "all"
        wandb.login()
        run = wandb.init()

    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = args.dataset_type,
        dataset_type = 'train',
        outfit_max_length=args.outfit_max_length,
        use_text=args.use_text,
        use_text_feature=args.use_text_feature
        )
    valid_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = args.dataset_type,
        dataset_type = 'valid',
        outfit_max_length=args.outfit_max_length,
        use_text=args.use_text,
        use_text_feature=args.use_text_feature
        )
    valid_fitb_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = 'fitb',
        dataset_type = 'valid',
        outfit_max_length=12,
        use_text=args.use_text,
        use_text_feature=args.use_text_feature
        )
    test_fitb_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = 'fitb',
        dataset_type = 'test',
        outfit_max_length=12,
        use_text=args.use_text,
        use_text_feature=args.use_text_feature
        )

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-albert-small-v2') if args.use_text else None
    torch.multiprocessing.freeze_support()
    train_dataloader = DataLoader(PolyvoreDataset(args.data_dir, train_dataset_args, tokenizer),
                                args.train_batch, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(PolyvoreDataset(args.data_dir, valid_dataset_args, tokenizer),
                                args.valid_batch, shuffle=False, num_workers=args.num_workers)
    valid_fitb_dataloader = DataLoader(PolyvoreDataset(args.data_dir, valid_fitb_dataset_args, tokenizer), 
                                       args.fitb_batch, shuffle=False, num_workers=args.num_workers)
    test_fitb_dataloader = DataLoader(PolyvoreDataset(args.data_dir, test_fitb_dataset_args, tokenizer), 
                                       args.fitb_batch, shuffle=False, num_workers=args.num_workers)

    categories = ['accessories', 'all-body', 'bags', 'bottoms', 'hats', 'jewellery', 'outerwear', 'scarves', 'shoes', 'sunglasses', 'tops']
    if args.model == 'type-aware-net':
        model = TypeAwareNet(embedding_dim=args.embedding_dim, categories=categories, img_backbone=args.img_backbone)
    elif args.model == 'csa-net':
        model = CSANet(embedding_dim=args.embedding_dim, categories=categories, img_backbone=args.img_backbone)
    elif args.model == 'fashion-swin':
        model = FashionSwin(embedding_dim=args.embedding_dim, categories=categories, img_backbone=args.img_backbone)
    print('[COMPLETE] Build Model')

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.5)

    training_args = TrainingArguments(
        model=args.model,
        train_batch=args.train_batch,
        valid_batch=args.valid_batch,
        fitb_batch=args.fitb_batch,
        n_epochs=args.n_epochs,
        save_every=args.save_every,
        learning_rate=args.learning_rate,
        work_dir = args.work_dir,
        use_wandb = True if args.wandb_api_key else False
        )
    
    trainer = Trainer(training_args, model, train_dataloader, valid_dataloader, valid_fitb_dataloader, test_fitb_dataloader,
                    optimizer=optimizer, scheduler=scheduler)

    if args.checkpoint != None:
        checkpoint = args.checkpoint
        trainer.load(checkpoint, load_optim=False)
        print(f'[COMPLETE] Load Model from {checkpoint}')

    trainer.fit()
    trainer._test(0)
        