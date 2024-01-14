import os
from torch.utils.data import Dataset
import numpy as np
import random
import json
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from deepfashion.utils.utils import *

from sklearn.preprocessing import LabelEncoder
            

class DeepFashionImageProcessor:
    def __init__(
            self,
            size: int = 224,
            use_normalize: bool = True,
            use_custom_transform: bool = False,
            custom_transform: Optional[List[Any]] = None
            ):
        transform = list()
        if use_custom_transform:
            transform += custom_transform
        else:
            transform.append(A.Resize(size, size))
            if use_normalize:
                transform.append(A.Normalize())
            transform.append(ToTensorV2())

        self.transform = A.Compose(transform)

    def __call__(
            self,
            images: List[np.ndarray]
            ) -> Tensor:
        images = [self.transform(image=image)['image'] for image in images]

        return torch.stack(images)


class DeepFashionInputProcessor:
    def __init__(
            self,
            categories: Optional[List[str]] = None, 
            image_processor: Optional[DeepFashionImageProcessor] = None, 
            text_tokenizer: Optional[AutoTokenizer] = None, 
            text_max_length: int = 32, 
            text_padding: str = 'max_length', 
            text_truncation: bool = True, 
            use_text_feature: bool = False,
            text_feature_dim: Optional[int] = None,
            outfit_max_length: int = 16
            ):
        if categories is None:
            raise ValueError("You need to specify a `categories`.")
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if text_tokenizer is None and use_text_feature is True:
            raise ValueError("If you want to set `use_text_feature` as `False`, You need to specify a `tokenizer`.")
        if use_text_feature is True and text_feature_dim is None:
            raise ValueError("If you want to set `use_text_feature` as `True`, You need to specify a `text_feature_dim`.")
        
        self.categories = ['<PAD>'] + categories
        self.category_encoder = LabelEncoder()
        self.category_encoder.fit(self.categories)

        self.image_processor = image_processor

        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length
        self.text_padding = text_padding
        self.text_truncation = text_truncation

        self.use_text_feature = use_text_feature
        self.text_feature_dim = text_feature_dim

        self.outfit_max_length = outfit_max_length

    def __call__(
            self, 
            category: Optional[List[str]]=None, 
            images: Optional[List[np.ndarray]]=None, 
            texts: Optional[List[str]]=None, 
            text_features: Optional[List[np.ndarray]]=None,
            do_pad: bool=False,
            do_truncation: bool=True,
            **kwargs
            ) -> Dict[str, Tensor]:
        if category is None:
            raise ValueError("You have to specify `category`.")
        if texts is None and images is None:
            raise ValueError("You have to specify either text or images.")
        
        inputs = dict()
        
        if category is not None:
            if do_truncation:
                category = category[:self.outfit_max_length]
            if do_pad:
                category = category + ['<PAD>' for _ in range(self.outfit_max_length - len(category))]
            inputs['category'] = torch.LongTensor(self.category_encoder.transform(category))
        
        if self.use_text_feature and text_features is not None:
            if do_truncation:
                text_features = text_features[:self.outfit_max_length]
            if do_pad:
                text_features = text_features + [np.zeros_like(text_features[0]) for _ in range(self.outfit_max_length - len(text_features))]
            inputs['text_features'] = torch.stack([text_feature for text_feature in text_features])
        
        if ~self.use_text_feature and texts is not None:
            if do_truncation:
                texts = texts[:self.outfit_max_length]
            if do_pad:
                texts = texts + ['<PAD>' for _ in range(self.outfit_max_length - len(texts))]
            encoding = self.text_tokenizer(texts, max_length=self.text_max_length, padding=self.text_padding, truncation=self.text_truncation, return_tensors='pt')
            inputs['input_ids'] = encoding.input_ids
            inputs['attention_mask'] = encoding.attention_mask

        if images is not None:
            if do_truncation:
                images = images[:self.outfit_max_length]
            if do_pad:
                images = images + [np.zeros(list(images[0].shape), dtype='uint8') for _ in range(self.outfit_max_length - len(images))]
            inputs['image_features'] = self.image_processor(images, **kwargs)

        if do_pad:
            mask = torch.ones((self.outfit_max_length), dtype=torch.long)
            mask[:len(category)] = 0
        else:
            mask = torch.zeros((len(category)), dtype=torch.long)
        mask = mask.bool()
        inputs['mask'] = mask

        return inputs