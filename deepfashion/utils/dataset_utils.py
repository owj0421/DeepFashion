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


def stack_tensors(mask, tensor):
    mask = mask.view(-1)
    s = list(tensor.shape)
    tensor = tensor.view([s[0] * s[1]] + s[2:])
    tensor = tensor[~mask]
    return tensor


def unstack_tensors(mask, tensor):
    B, S = mask.shape
    mask = mask.view(-1)
    new_tensor = torch.zeros([B * S] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.get_device())
    new_tensor[~mask] = tensor
    new_tensor = new_tensor.view([B, S] + list(tensor.shape)[1:])
    return new_tensor


def stack_dict(batch):
    for i in batch.keys():
        if i == 'input_mask':
            continue
        batch[i] = stack_tensors(batch['input_mask'], batch[i])
    return batch


def unstack_dict(batch):
    for i in batch.keys():
        if i == 'input_mask':
            continue
        batch[i] = unstack_tensors(batch['input_mask'], batch[i])
    return batch


def load_fitb_inputs(data_dir, args, outfit_id2item_id):
    fitb_path = os.path.join(data_dir, args.polyvore_split, f'fill_in_blank_{args.dataset_type}.json')
    with open(fitb_path, 'r') as f:
        fitb_data = json.load(f)
        fitb_inputs = []
        for item in fitb_data:
            question_ids = list(map(lambda x: outfit_id2item_id[x], item['question'][:args.max_input_len]))
            candidate_ids = list(map(lambda x: outfit_id2item_id[x], item['answers']))
            fitb_inputs.append((question_ids, candidate_ids))
    return fitb_inputs


def load_cp_inputs(data_dir, args, outfit_id2item_id):
    cp_path = os.path.join(data_dir, args.polyvore_split, f'compatibility_{args.dataset_type}.txt')
    with open(cp_path, 'r') as f:
        cp_data = f.readlines()
        cp_inputs = []
        for d in cp_data:
            target, *item_ids = d.split()
            cp_inputs.append((torch.FloatTensor([int(target)]), list(map(lambda x: outfit_id2item_id[x], item_ids[:args.max_input_len]))))
    return cp_inputs


def load_triplet_inputs(data_dir, args, outfit_id2item_id):
    outfit_data_path = os.path.join(data_dir, args.polyvore_split, f'{args.dataset_type}.json')
    outfit_data = json.load(open(outfit_data_path))
    triplet_inputs = [[outfit['items'][i]['item_id'] 
                       for i in range(min(len(outfit['items']), args.max_input_len))] 
                       for outfit in outfit_data]
    triplet_inputs = list(filter(lambda x: len(x) > 1, triplet_inputs))
    return triplet_inputs


def load_hglmm(data_dir, args):
    txt_dim = 6000
    txt = os.path.join(data_dir, args.polyvore_split, 'train_hglmm_pca6000.txt')
    desc2hglmm = {}
    with open(txt, 'r') as f:
        for line in tqdm(f):
            line = line.strip().split(',')
            if not line:
                continue
            desc = ','.join(line[:-txt_dim])
            vec = np.array([float(x) for x in line[-txt_dim:]], np.float32)
            desc2hglmm[desc] = vec
    return desc2hglmm


def load_data(data_dir, args):
    # Paths
    # data_dir = os.path.join(data_dir, args.polyvore_split)
    outfit_data_path = os.path.join(data_dir, args.polyvore_split, f'{args.dataset_type}.json')
    meta_data_path = os.path.join(data_dir, 'polyvore_item_metadata.json')
    outfit_data = json.load(open(outfit_data_path))
    meta_data = json.load(open(meta_data_path))
    # Load
    item_ids = set()
    categories = set()
    item_id2category = {}
    item_id2desc = {}
    category2item_ids = {}
    outfit_id2item_id = {}
    for outfit in outfit_data:
        outfit_id = outfit['set_id']
        for item in outfit['items'][:args.max_input_len]:
            # Item of cloth
            item_id = item['item_id']
            # Category of cloth
            category = meta_data[item_id]['semantic_category']
            categories.add(category)
            item_id2category[item_id] = category
            if category not in category2item_ids:
                category2item_ids[category] = set()
            category2item_ids[category].add(item_id)
            # Description of cloth
            desc = meta_data[item_id]['title']
            if not desc:
                desc = meta_data[item_id]['url_name']
            item_id2desc[item_id] = desc.replace('\n','').strip().lower()
            # Replace the item code with the outfit number with the image code
            outfit_id2item_id[f"{outfit['set_id']}_{item['index']}"] = item_id
            item_ids.add(item_id)
    item_ids = list(item_ids)
    item_id2idx = {id : idx for idx, id in enumerate(item_ids)}
    categories = ['<PAD>'] + list(categories)
    category2category_id = {category : idx for idx, category in enumerate(categories)}
    return item_ids, item_id2idx, \
        item_id2category, category2item_ids, categories, category2category_id, \
            outfit_id2item_id, item_id2desc