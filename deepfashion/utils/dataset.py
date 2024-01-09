import os
from torch.utils.data import Dataset
import numpy as np
import random
import json
import torch
from transformers import AutoTokenizer, BatchEncoding
from dataclasses import dataclass
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from deepfashion.utils.dataset_utils import *

@dataclass
class DatasetArguments:
    polyvore_split: str = 'nondisjoint'
    task_type: str = 'cp'
    dataset_type: str = 'train'
    max_input_len: int = 8
    img_size: Tuple[int] = (224, 224)
    img_transform: Optional[A.Compose] = None
    txt_type: Literal['hglmm', 'token'] = 'hglmm'
    txt_max_token: int = 16
    

class PolyvoreDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            args: DatasetArguments,
            tokenizer: Optional[AutoTokenizer] = None
            ):
        # Dataset configurations
        self.args = args
        self.is_train = (args.dataset_type == 'train')
        # Image Data Configurations
        self.img_dir = os.path.join(data_dir, 'images')
        self.img_transform = args.img_transform
        # Text data configurations
        if args.txt_type == 'token':
            if tokenizer:
                self.tokenizer = tokenizer
            else:
                raise AttributeError(f'txt_type is set to token, but tokenizer has not been passed.')
        else:
            self.desc2hglmm = load_hglmm(data_dir, args)
        # Meta Data preprocessing
        self.max_input_len = args.max_input_len
        self.item_ids, self.item_id2idx, \
        self.item_id2category, self.category2item_ids, self.categories, self.category2category_id, \
            self.outfit_id2item_id, self.item_id2desc = load_data(data_dir, args)
        # Psudo Item for padding
        self.pad_item = [torch.LongTensor([0]), 
                         torch.zeros((3, args.img_size[0], args.img_size[1]), dtype=torch.float32), 
                         '<PAD>']
        # Input
        if args.task_type == 'cp':
            self.data = load_cp_inputs(data_dir, args, self.outfit_id2item_id)
        elif args.task_type == 'fitb':
            self.data = load_fitb_inputs(data_dir, args, self.outfit_id2item_id)
        elif args.task_type == 'triplet':
            self.data = load_triplet_inputs(data_dir, args, self.outfit_id2item_id)
        else:
            raise ValueError('task_type must be one of "cp", "fitb", and "triplet".')
        
    def _load_category(self, item_id):
        id = self.category2category_id[self.item_id2category[item_id]]
        id = torch.LongTensor([id])
        return id

    def _load_img(self, item_id):
        path = os.path.join(self.img_dir, f"{item_id}.jpg")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_transform(image=img)['image']
        return img
    
    def _load_txt(self, item_id):
        desc = self.item_id2desc[item_id] if item_id in self.item_id2desc else self.item_id2category[item_id]
        if self.args.txt_type == 'hglmm':
            if desc in self.desc2hglmm.keys():
                desc = self.desc2hglmm[desc]
            else:
                desc = np.zeros(self.text_feat_dim, np.float32)
        return desc
    
    def _pad_inputs(self, inputs, max_input_len: Optional[int] = None):
        if not max_input_len:
            max_input_len = self.max_input_len
        input_mask = torch.ones((max_input_len), dtype=torch.long)
        input_mask[:len(inputs)] = 0
        input_mask = input_mask.bool()
        inputs = inputs + [self.pad_item for _ in range(max_input_len - len(inputs))]
        return input_mask, inputs
    
    def _get_single_input(self, item_id):
        category = self._load_category(item_id)
        img = self._load_img(item_id)
        txt = self._load_txt(item_id)
        return category, img, txt
    
    def _get_inputs(self, item_ids, pad: bool=False) -> Dict[Literal['input_mask', 'img', 'desc'], Tensor]:
        inputs = [self._get_single_input(item_id) for item_id in item_ids]
        if pad:
            input_mask, inputs = self._pad_inputs(inputs)
        else:
            input_mask = torch.zeros((len(inputs)), dtype=torch.long).bool()
            
        category, img, desc = zip(*inputs)
        category = torch.stack(category)
        img = torch.stack(img)
        if self.args.txt_type == 'token':
            desc = self.tokenizer(desc, max_length=self.args.txt_max_token, padding='max_length', truncation=True, return_tensors='pt')
        elif self.args.txt_type == 'hglmm':
            desc = {'hglmm': torch.stack(desc)}
        inputs = dict({'input_mask': input_mask, 'category': category, 'img': img}, **desc)
        return inputs

    def _get_neg_samples(self, positive_id, n, ignore_ids=None):
        return  random.sample(self.item_ids, n)

    def __getitem__(self, idx):
        if self.args.task_type == 'cp':
            target, outfit_ids = self.data[idx]
            inputs = self._get_inputs(outfit_ids, pad=True)
            return {'target': target, 'inputs': inputs}
        
        elif self.args.task_type =='fitb':
            question_ids, candidate_ids = self.data[idx]
            questions = self._get_inputs(question_ids, pad=True)
            candidates = self._get_inputs(candidate_ids)
            return  {'questions': questions, 'candidates': candidates} # ans is always 0 index
        
        elif self.args.task_type =='triplet':
            outfit_ids = self.data[idx]
            positive_id = [outfit_ids.pop(random.randrange(len(outfit_ids)))]
            anchor_ids = outfit_ids
            negative_ids = self._get_neg_samples(positive_id, n=6, ignore_ids=anchor_ids)

            anchors = self._get_inputs(anchor_ids, pad=True)
            positives = self._get_inputs(positive_id)
            negatives = self._get_inputs(negative_ids)
                
            return {'anchors': anchors, 'positives': positives, 'negatives': negatives}

    def __len__(self):
        return len(self.data)