# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual
# property and proprietary rights in and to this software, related
# documentation and any modifications thereto.  Any use, reproduction,
# disclosure or distribution of this software and related documentation
# without an express license agreement from NVIDIA CORPORATION is strictly
# prohibited.
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------
# Modified by Jilan Xu
# -------------------------------------------------------------------------

import collections.abc
from collections import OrderedDict
import json
import cv2
import numpy as np
from PIL import Image
from ipdb import set_trace
import scipy

import torch
import torch.distributed as dist
from datasets import template_meta

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def momentum_update(model_on, model_off, coeff):
    for params_on, params_off in zip(model_on.parameters(), model_off.parameters()):
        params_off.data = coeff * params_off.data + (1 - coeff) * params_on.data

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm


def get_batch_size(data):

    if isinstance(data, torch.Tensor):
        return data.size(0)
    elif isinstance(data, collections.abc.Mapping):
        return get_batch_size(data[next(iter(data))])
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
        # check to make sure that the elements in batch have consistent size
        it = iter(data)
        return get_batch_size(next(it))

    raise TypeError


def data2cuda(data):

    if isinstance(data, torch.Tensor):
        batch = data.cuda(non_blocking=True)
        return batch
    elif isinstance(data, collections.abc.Mapping):
        return {key: data2cuda(data[key]) for key in data}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
        return [data2cuda(d) for d in data]
    else:
        raise TypeError

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    return loss, log_vars


def build_dataset_class_tokens(text_transform, template_set, classnames):

    tokens = []
    templates = template_meta[template_set]
    for classname in classnames:
        # format with class
        tokens.append(torch.stack([text_transform(template.format(classname)) for template in templates]))
    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    tokens = torch.stack(tokens)

    return tokens

def build_dataset_class_lists(template_set, classnames):
    tokens = []
    templates = template_meta[template_set]
    for classname in classnames:
        # format with class
        for template in templates:
            tokens.append(template.format(classname))
    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    # tokens = torch.stack(tokens)
    return tokens
            
def cdist_(x, metric='euclidean'):
    assert len(x.shape) == 3
    if metric != 'JS':
        x_ = torch.split(x, 1, dim=0)  # tuple
        return np.mean(tuple(map(lambda a: scipy.spatial.distance.cdist(a.squeeze(), a.squeeze(), metric).mean(), x_)))
    else:
        softmax = torch.nn.Softmax(dim=1)
        x_ = torch.split(softmax(x), 1, dim=0)  # tuple
        return np.mean(tuple(map(lambda a: scipy.spatial.distance.cdist(a.squeeze(), a.squeeze(), metric).mean(), x_)))
        pass
    pass