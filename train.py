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
from deepfashion.utils.metric import MetricCalculator

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='DeepFashion')
    parser.add_argument('--model', help='Model', type=str, default='csa-net')
    parser.add_argument('--sampling_type', help='sampling_type', type=str, default='outfit')
    parser.add_argument('--embedding_dim', help='embedding dim', type=int, default=64)
    parser.add_argument('--use_text', help='embedding dim', type=bool, default=False)
    parser.add_argument('--text_type', help='embedding dim', type=str, default='token')
    parser.add_argument('--n_neg', help='embedding dim', type=int, default=4)
    parser.add_argument('--train_batch', help='Size of Batch for Training', type=int, default=2)
    parser.add_argument('--valid_batch', help='Size of Batch for Validation, Test', type=int, default=2)
    parser.add_argument('--fitb_batch', help='Size of Batch for Validation, Test', type=int, default=16)
    parser.add_argument('--n_epochs', help='Number of epochs', type=int, default=2)
    parser.add_argument('--save_every', help='Number of epochs', type=int, default=3)
    parser.add_argument('--scheduler_step_size', help='Step LR', type=int, default=200)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=5e-5)
    parser.add_argument('--work_dir', help='Full working directory', type=str, default='F:\Projects\DeepFashion')
    parser.add_argument('--data_dir', help='Full dataset directory', type=str, default='F:\Projects\datasets\polyvore_outfits')
    parser.add_argument('--wandb_api_key', default=None)
    parser.add_argument('--checkpoint', default=None)
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
        
    train_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = args.sampling_type,
        dataset_type = 'train',
        image_transform = [
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2(),
            ],
        n_neg=args.n_neg,
        use_text=args.use_text,
        text_type=args.text_type
        )

    valid_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = args.sampling_type,
        dataset_type = 'valid',
        n_neg=args.n_neg,
        use_text=args.use_text,
        text_type=args.text_type
        )

    fitb_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = 'fitb',
        dataset_type = 'valid',
        use_text=args.use_text,
        text_type=args.text_type
        )

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-albert-small-v2')

    torch.multiprocessing.freeze_support()

    train_dataloader = DataLoader(PolyvoreDataset(args.data_dir, train_dataset_args, tokenizer),
                                training_args.train_batch, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(PolyvoreDataset(args.data_dir, valid_dataset_args, tokenizer),
                                training_args.valid_batch, shuffle=False, num_workers=4)
    fitb_dataloader = DataLoader(PolyvoreDataset(args.data_dir, fitb_dataset_args, tokenizer), 
                                training_args.fitb_batch, shuffle=False, num_workers=4)

    num_category = 12

    if args.model == 'type-aware-net':
        model = TypeAwareNet(embedding_dim=args.embedding_dim, num_category=num_category).to(device)
    elif args.model == 'csa-net':
        model = CSANet(embedding_dim=args.embedding_dim, num_category=num_category).to(device)
    elif args.model == 'fashion-swin':
        model = FashionSwin(embedding_dim=args.embedding_dim, num_category=num_category).to(device)
    print('[COMPLETE] Build Model')

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate,)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.5)
    metric = MetricCalculator()
    trainer = Trainer(training_args, model, train_dataloader, valid_dataloader, fitb_dataloader,
                    optimizer=optimizer, metric=metric , scheduler=scheduler)

    if args.checkpoint != None:
        checkpoint = args.checkpoint
        trainer.load(checkpoint, load_optim=False)
        print(f'[COMPLETE] Load Model from {checkpoint}')

    # Train
    trainer.fit()
        