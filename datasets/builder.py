# -------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
# Written by Ze Liu, Zhenda Xie
# Modified by Jiarui Xu
# -------------------------------------------------------------------------
# Modified by Jilan Xu
# -------------------------------------------------------------------------


import os.path as osp
import random
import warnings
from functools import partial

import nltk
import numpy as np
import torch
import torch.distributed as dist

from mmcv.parallel import collate
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
if timm.__version__ == '0.6.12':
    from timm.data.transforms import str_to_pil_interp as _pil_interp
else:
    from timm.data.transforms import _pil_interp
# this works for timm==0.3.2
# from timm.data.transforms import _pil_interp 
from torchvision import transforms
import torch.nn as nn
from PIL import ImageFilter,Image
from torch import Tensor
from typing import Tuple, List, Optional
import numbers
import math
import torchvision.transforms.functional as F
import shutil

from .formatting import ToDataContainer
from .tokenizer import SimpleTokenizer
from .clip_dataset import ClipDataset
from ipdb import set_trace

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_fn(batch):  
    img = torch.stack([b['image'] for b in batch])
    caption = torch.stack([b['caption'] for b in batch])
    raw_caption = [b['raw_caption'] for b in batch] 
    
    raw_question = [b['raw_question'] for b in batch] if 'raw_question' in batch[0].keys() else None
    raw_answer = [b['raw_answer'] for b in batch] if 'raw_answer' in batch[0].keys() else None

    cross_image = torch.stack([b['cross_image'] for b in batch]) if 'cross_image' in batch[0].keys() else None
    cross_entity = [b['cross_entity'] for b in batch] if 'cross_entity' in batch[0].keys() else None
    
    question = torch.stack([b['question'] for b in batch]) if 'question' in batch[0].keys() and batch[0]['question'] is not None else None
    answer = torch.stack([b['answer'] for b in batch]) if 'answer' in batch[0].keys() and batch[0]['answer'] is not None else None
        
    return {    
        'image':img,
        'caption':caption,
        'raw_caption' : raw_caption,
        'raw_question': raw_question,
        'raw_answer': raw_answer,
        
        'cross_image': cross_image,
        'cross_entity': cross_entity, 
        'question': question,
        'answer': answer,
        
    }

def build_loader(config):
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0

    dataset_train = build_dataset(is_train=True, config=config)
    print(f'local rank {local_rank} / global rank {dist.get_rank()} \
        successfully build train dataset')
    dataset_val = build_dataset(is_train=False, config=config)
    print(f'local rank {local_rank} / global rank {dist.get_rank()} \
        successfully build val dataset')

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)        
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    print('train batch size: ', config.train.batch_size)
    print('val batch size: ', config.val.batch_size)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn, ### NOTEL THIS ###
        #shuffle=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.val.batch_size,
        num_workers=config.val.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def build_dataset(is_train, config):
    img_transform = build_img_transform(is_train, config.img_aug, config.with_dc)
    text_transform = build_text_transform(is_train, config.text_aug, config.with_dc)
    split = 'train' if is_train else 'val'

    dataset = ClipDataset(
        root_dir=config[split]['root_dir'],
        meta_file=config[split]['meta_file'],
        img_transform=img_transform,
        text_transform=text_transform,
        read_from=config[split]['read_from'],
        split=split,
        cross_image=config[split].get('cross_image', False),
        mask_type=config[split].get('mask_type', 'class'),
        use_distilbert=config[split].get('use_distilbert', True),
    )
    print('dataset len: ', len(dataset))

    # for i in range(10):
    #     t = dataset.__getitem__(i)
    #     print(t['image'].shape, t['cross_image'].shape)
    #     print(t['caption'].shape, t['target'])
    #     print(t['raw_caption'])
    #     print(t['cross_caption'], '\t',  t['cross_entity'])
    #     print(t['raw_question'], '\t', t['raw_answer'])
    # set_trace()

    return dataset

def build_img_transform(is_train, config, with_dc=True):
    if not config.deit_aug:
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(config.img_size, scale=config.img_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.img_size + 32),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
        return transform

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.img_size,
            is_training=True,
            color_jitter=config.color_jitter if config.color_jitter > 0 else None,
            auto_augment=config.auto_augment if config.auto_augment != 'none' else None,
            re_prob=config.re_prob,
            re_mode=config.re_mode,
            re_count=config.re_count,
            interpolation=config.interpolation,
        )
    else:
        size = int((256 / 224) * config.img_size)
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=_pil_interp(config.interpolation)),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    if with_dc:
        transform = transforms.Compose([*transform.transforms, ToDataContainer()])

    return transform

def build_text_transform(is_train, config, with_dc=True):
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
    if is_train:
        ### only on local rank 0 ###
        if local_rank == 0:
            ### download itself or pre-download and give the nltk dir ###
            # nltk.download('popular')
            nltk.data.path.append('/mnt/petrelfs/xujilan/nltk_data')
            
        transform = WordAugTokenizeWrapper(
            Tokenize(SimpleTokenizer(), max_seq_len=config.max_seq_len),
            max_word=config.multi_label,
            word_type=config.word_type)
    else:
        transform = Tokenize(SimpleTokenizer(), max_seq_len=config.max_seq_len)
            
    if with_dc:
        transform = transforms.Compose([transform, ToDataContainer()])
    return transform
    
class Tokenize:

    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True
        
        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[:self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f'Input {texts[i]} is too long for context length {self.max_seq_len}')
            result[i, :len(tokens)] = torch.tensor(tokens)

        if expanded_dim:
            return result[0]

        return result


class WordAugTokenizeWrapper:

    def __init__(self, tokenize, max_word=3, template_set='full', word_type='noun'):
        self.tokenize = tokenize
        self.max_word = max_word
        from .imagenet_template import (full_imagenet_templates, sub_imagenet_template, simple_imagenet_template,
                                        identity_template)
        assert template_set in ['full', 'subset', 'simple', 'identity']
        if template_set == 'full':
            templates = full_imagenet_templates
        elif template_set == 'subset':
            templates = sub_imagenet_template
        elif template_set == 'simple':
            templates = simple_imagenet_template
        elif template_set == 'identity':
            templates = identity_template
        else:
            raise ValueError
        self.templates = templates
        assert word_type in ['noun', 'noun_phrase']
        self.word_type = word_type

    def get_tag(self, tokenized, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for (word, pos) in nltk.pos_tag(tokenized):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret

    def get_tag_with_loc(self, tokenized, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        loc = []
        for i, (word, pos) in enumerate(nltk.pos_tag(tokenized)):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
                    loc.append(i)
        return ret, loc
    
    def get_noun_phrase(self, tokenized):
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        chunker = nltk.RegexpParser(grammar)

        chunked = chunker.parse(nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if isinstance(subtree, nltk.Tree):
                current_chunk.append(' '.join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = ' '.join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def __call__(self, text):
        """
        Args:
            text: str
        
        """
        assert isinstance(text, str)
        tokenized = nltk.word_tokenize(text)
        
        nouns = []
        if len(tokenized) > 0:
            if self.word_type == 'noun':
                # nouns = self.get_tag(tokenized, ['NN', 'NNS', 'NNP', 'VBG', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ'])
                # nouns = self.get_tag(tokenized, ['NN', 'NNS'])
                # nouns, locs = self.get_tag_with_loc(tokenized, ['NN', 'NNS'])
                nouns, locs = self.get_tag_with_loc(tokenized, ['NN', 'NNS', 'NNP',])
            elif self.word_type == 'noun_phrase':
                nouns = self.get_noun_phrase(tokenized)
            else:
                raise ValueError('word_type must be noun or noun_phrase')
        
        ### By default, we use this ###
        if self.max_word == 0:
            return self.tokenize(text), nouns, locs, text
        
        prompt_texts = []
        if len(nouns) > 0:
            select_nouns = np.random.choice(nouns, min(self.max_word, len(nouns)), replace=False)
            prompt_texts = [np.random.choice(self.templates).format(noun) for noun in select_nouns]
        if len(prompt_texts) < self.max_word:
            prompt_texts += [text] * (self.max_word - len(prompt_texts))

        texts = [text] + prompt_texts
        return self.tokenize(texts), nouns, locs, texts
