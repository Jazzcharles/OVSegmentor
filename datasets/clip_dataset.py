# -------------------------------------------------------------------------
# Written by Jilan Xu
# -------------------------------------------------------------------------

from re import L
import torch
import json
import os.path as osp
import requests
import numpy as np
import time
import ast
from typing import List
from torch.utils.data import Dataset

import random
import os
import pandas as pd
import omegaconf
import clip
from .tokenizer import SimpleTokenizer
from .imagenet_template import full_imagenet_templates
from nltk.stem import WordNetLemmatizer
from PIL import Image
from ipdb import set_trace

import io
lemmatizer = WordNetLemmatizer()

### frequently appeared 100 entities ###
TOP_CLASSES_1=[
    'people', 'man', 'men', 'woman', 'women', 'girl', 'boy', 'lady', 'kid', 'child', 'children', 'baby', 'student', 'bride', 'groom', 'couple', 'prince', 'princess', \
    'car', 'bus', 'truck', 'motorcycle', 'train', 'bicycle', 'boat', 'aeroplane', 'airplane', 'motorbike', 'bike',\
    'cup', 'bottle', 'bowl', 'knife', 'spoon',  'glass', 'fork',\
    'chair', 'table', 'bench', 'clock', 'laptop', 'light', 'vase', 'plant', 'remote', 'microwave', 'toaster', 'oven','mouse', 'keyboard','sofa', 'monitor','desk', 'tv','TV', 'couch', 'flower','refrigerator', \
    'house', 'building', 'hotel',\
    'handbag', 'umbrella','book', 'backpack', 'phone', 'shirt', 'tie', 'suitcase','T-shirt', 'bag',  'box', \
    'sink','bed','toilet',\
    'cat','dog',  'horse', 'bird','cow', 'sheep' ,'elephant', 'bear', 'zebra', 'giraffe', \
    'ball', 'racket', 'skateboard', 'skis', 'snowboard', 'surfboard', 'kite', \
    'pizza', 'cake', 'apple', 'banana', 'sandwich', 'orange', 'carrot', 'donut' ,\
]

### some of the entities are similar, map them to a single one ###
syn_dict = {
    'people':'people', 'man':'people', 'men':'people', 'woman':'people', 'women':'people', 'girl':'people', 'boy':'people', 'lady':'people', 'kid':'people', 'child':'people', 'children':'people', 'baby':'people', 'student':'people', 'bride':'people', 'groom':'people', 'couple':'people', 'prince':'people', 'princess':'people',\
    'airplane': 'aeroplane','motorbike': 'motorcycle','bike': 'bicycle',\
    'TV':'tv', 'desk': 'table', 'couch':'sofa',\
    'building': 'house', 'hotel': 'house', \
    'T-shirt': 'shirt','T-Shirt': 'shirt', 'handbag': 'bag', \
}

### unique entities ###
TOP_UNIQUE_CLASSES = [
    'people', 'car', 'bus', 'truck', 'motorcycle', \
    'train', 'bicycle', 'boat', 'aeroplane', 'cup', \
    'bottle', 'bowl', 'knife', 'spoon',  'glass', \
    'fork', 'chair', 'table', 'bench', 'clock', \
    'laptop', 'light', 'vase', 'plant', 'remote',\
    'microwave', 'toaster', 'oven','mouse', 'keyboard',\
    'sofa', 'monitor', 'tv', 'flower','refrigerator', \
    'house', 'bag', 'umbrella','book', 'backpack', \
    'phone', 'shirt', 'tie', 'suitcase', 'box',\
    'sink','bed','toilet', 'cat','dog', \
    'horse', 'bird','cow', 'sheep' ,'elephant', \
    'bear', 'zebra', 'giraffe',  'ball', 'racket', \
    'skateboard', 'skis', 'snowboard', 'surfboard', 'kite',\
    'pizza', 'cake', 'apple', 'banana', 'sandwich',\
    'orange', 'carrot', 'donut' ,\
]


TOP_UNIQUE_CLASSES_IDX = {}
for i, x in enumerate(TOP_UNIQUE_CLASSES):
    TOP_UNIQUE_CLASSES_IDX[x] = i

class ClipDataset(Dataset):

    def __init__(self, root_dir, meta_file, img_transform=None, text_transform=None,
                 read_from='dir', 
                 label_texts_ensemble='none', split='train',
                 cross_image=False, use_entity=True, mask_type='class', use_distilbert=True
                 ):
        
        self.root_dir = root_dir        
        self.meta_file = meta_file
        self.metas = pd.read_csv(self.meta_file)
        print(f'Total {len(self.metas)} samples')
        
        self.read_from = read_from
        if self.read_from == 'petrel':
            from petrel_client.client import Client
            self.client = Client()
        
        self.img_transform = img_transform
        self.text_transform = text_transform

        self.label_texts_ensemble = label_texts_ensemble
        self.split=split

        self.cross_image = cross_image
        self.use_entity = use_entity
        self.tokenizer = SimpleTokenizer()
        self.mask_type = mask_type
        self.use_distilbert = use_distilbert   

    def __len__(self):
        return len(self.metas)

    def load_image(self, filename):
        filename = os.path.join(self.root_dir, filename)
        if self.read_from == 'dir':
            img = Image.open(filename).convert('RGB')
            return img
        elif self.read_from == 'petrel':
            value = self.client.get(filename)
            img_bytes = np.frombuffer(value, dtype=np.uint8)
            with Image.open(io.BytesIO(img_bytes)) as img:
                img = img.convert('RGB')
            return img
        else:
            raise NotImplementedError

    def _load_meta(self, idx):
        return self.metas.iloc[idx]
                    
    def sample_cross_image(self, curr_meta):
        pair_index = curr_meta['pairindex']
        pair_entity = curr_meta['pairentity']

        pair_index_list = ast.literal_eval(pair_index)
        pair_entity_list = ast.literal_eval(pair_entity)
        
        sample_index = np.random.randint(0, len(pair_index_list))
   
        index = pair_index_list[sample_index]
        entity = pair_entity_list[sample_index]
        
        pair_meta = self._load_meta(index)

        img = self.load_image(pair_meta['image_id'])        
        caption = pair_meta['caption']
        return img, caption, entity

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = curr_meta['image_id']
        raw_caption = caption = curr_meta['caption']
        label = int(curr_meta['label']) if 'label' in curr_meta else -1

        ret_info = {}
        try:
            
            assert self.is_contains_chinese(caption) == False
            img = self.load_image(filename)

            if self.img_transform is not None:
                image = self.img_transform(img)
                    
            if self.text_transform is not None:
                if self.split == 'train':
                    ### for clip TextTransformer, captions are here tokenised ###
                    ### for bert/distilbert, text transform are used to select nouns, captions will be tokensized later ###
                    caption, nouns, locs, prompt_texts = self.text_transform(caption)
                    
                    if self.use_entity:
                        ### A feasible option here is to pre-process question and answers to speed-up data loading ###
                        if self.use_distilbert:
                            ### bert/distilbert-like, questions/answers will be tokenised later ###
                            raw_question, question, raw_answer, answer = self.build_question_and_answer_for_distilbert(raw_caption, nouns)
                        else: 
                            ### clip TextTransformer-like, questions/answers are tokenised ###
                            raw_question, question, raw_answer, answer = self.build_question_and_answer(raw_caption, nouns)
                
                        ret_info['question'] = question
                        ret_info['answer'] = answer
                        ret_info['raw_question'] = raw_question
                        ret_info['raw_answer'] = raw_answer

                    if self.cross_image:
                        crossimg, crosscaption, crossentity = self.sample_cross_image(curr_meta)
                        crossimg = self.img_transform(crossimg)

                        crossentity = 'A photo of ' + crossentity
                        ret_info['cross_image'] = crossimg
                        ret_info['cross_entity'] = crossentity
                        ret_info['cross_caption'] = crosscaption

                else:
                    caption = self.text_transform(caption)
            
            ret_info['image'] = image
            ret_info['caption'] = caption
            ret_info['raw_caption'] = raw_caption
            ret_info['target'] = label
            # ret_info['filename'] = filename
            return ret_info    
                            
        except Exception as e:          
           return self.__getitem__(np.random.randint(0, len(self.metas)))

   
    def judge_noun(self, n):
        n = n.replace('.', '')
        ans = n
        ### conduct Lemmatization ###
        ans = lemmatizer.lemmatize(ans.lower())
        
        if ans in syn_dict:
            ans = syn_dict[ans]
        
        if ans in TOP_UNIQUE_CLASSES:
            return 1, ans
        return 0, n       
    
    def build_question_and_answer(self, caption, nouns):
        words = caption.split(' ')
        question = ''
        ans_list = []

        token_mapper = {}
        word_mapper = {}
        assert self.mask_type == 'class'
        for word in words:
            word = word.strip("'s").strip(' ').strip('\n')
            word_flag, newword = self.judge_noun(word)
            if word_flag == 1:
                question = question + newword + ' '
                ans_list.append(newword)
                token_id = self.tokenizer.encode(newword)[0]
                token_mapper[token_id] = TOP_UNIQUE_CLASSES_IDX[newword]
                word_mapper[token_id] = 332   ### this is 'M'
            else:
                question = question + word + ' '
                    
        question = question.replace("'", '').strip()
        raw_question = question
        
        question, _, _, _ = self.text_transform(raw_question)
        question = torch.tensor([word_mapper[int(word)] if int(word) in word_mapper else word for word in question])
        # raw_answer = 'A photo of ' + ' and '.join(list(set(ans_list))) ## unique words
        raw_answer = random.choice(full_imagenet_templates).split('{}')[0] + ' and '.join(list(set(ans_list)))
        answer, _, _, _ = self.text_transform(raw_answer)
        
        return raw_question, question, raw_answer, answer


    def build_question_and_answer_for_distilbert(self, caption, nouns):
        words = caption.split(' ')
        question = ''
        entity_list = []

        ### default, mask all entites ###
        assert self.mask_type == 'class'
        for word in words:
            word = word.strip("'s").strip(' ').strip('\n')
            word_flag, newword = self.judge_noun(word)
            if word_flag == 1:
                question = question + '[MASK]' + ' '
                entity_list.append(newword)
            else:
                question = question + word + ' '
    
        question = question.replace("'", '').strip()
        raw_question = question
        #### build and transform answers ###
        # raw_answer = 'A photo of ' + ' and '.join(list(set(ans_list))) ## unique words
        raw_answer = random.choice(full_imagenet_templates).split('{}')[0] + ' and '.join(list(set(entity_list)))    
        return raw_question, None, raw_answer, None

    def is_contains_chinese(self, strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False
