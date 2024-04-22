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

import mmcv
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from utils import build_dataset_class_tokens, build_dataset_class_lists

from .group_vit_seg import GroupViTSegInference
from ipdb import set_trace


def build_seg_dataset(config):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config.cfg)
    dataset = build_dataset(cfg.data.test)
    return dataset


def build_custom_seg_dataset(config, args):
    """Build a dataset from config."""
    cfg = mmcv.Config.fromfile(config.cfg)
    
    cfg.data.test.data_root = args.image_folder
    cfg.data.test.img_dir = ''
    cfg.data.test.ann_dir = '' ## unsure
    cfg.data.test.split = 'image_list.txt'

    dataset = build_dataset(cfg.data.test)
    return dataset
    
def build_seg_dataloader(dataset):

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=True,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False)
    return data_loader


def build_seg_inference(model, dataset, text_transform, config, tokenizer=None):
    cfg = mmcv.Config.fromfile(config.cfg)
    if len(config.opts):
        cfg.merge_from_dict(OmegaConf.to_container(OmegaConf.from_dotlist(OmegaConf.to_container(config.opts))))
    
    with_bg = dataset.CLASSES[0] == 'background'
    if with_bg:
        classnames = dataset.CLASSES[1:]
    else:
        classnames = dataset.CLASSES
    
    if tokenizer is not None:
        text_tokens = build_dataset_class_lists(config.template, classnames)
        text_embedding = model.build_text_embedding(text_tokens, tokenizer, num_classes=len(classnames))
    else:
        text_tokens = build_dataset_class_tokens(text_transform, config.template, classnames)
        text_embedding = model.build_text_embedding(text_tokens, num_classes=len(classnames))
    kwargs = dict(with_bg=with_bg)

    if hasattr(cfg, 'test_cfg'):
        kwargs['test_cfg'] = cfg.test_cfg
    
    seg_model = GroupViTSegInference(model, text_embedding, **kwargs)
    print('Evaluate during seg inference')

    seg_model.CLASSES = dataset.CLASSES
    seg_model.PALETTE = dataset.PALETTE

    return seg_model

def build_demo_inference(model, text_transform, config, tokenizer=None):
    seg_config = config.evaluate.seg
    cfg = mmcv.Config.fromfile(seg_config.cfg)
    if len(seg_config.opts):
        cfg.merge_from_dict(OmegaConf.to_container(OmegaConf.from_dotlist(OmegaConf.to_container(seg_config.opts))))

    with_bg = True
    from segmentation.datasets.ade20k import ADE20KDataset
    from segmentation.datasets.coco_object import COCOObjectDataset
    from segmentation.datasets.pascal_voc import PascalVOCDataset

    if config.vocab == ['voc']:
        classnames = PascalVOCDataset.CLASSES
        palette = PascalVOCDataset.PALETTE
    elif config.vocab == ['coco']:
        classnames = COCOObjectDataset.CLASSES
        palette = COCOObjectDataset.PALETTE
    elif config.vocab == ['ade']:
        classnames = ADE20KDataset.CLASSES
        palette = ADE20KDataset.PALETTE
    else:
        classnames = config.vocab
        palette = ADE20KDataset.PALETTE[:len(classnames)]
        
    if classnames[0] == 'background':
        classnames = classnames[1:]

    print('candidate CLASSES: ', classnames)
    print('Using palette: ',     palette)
    
    if tokenizer is not None:
        text_tokens = build_dataset_class_lists(seg_config.template, classnames)
        text_embedding = model.build_text_embedding(text_tokens, tokenizer, num_classes=len(classnames))
    else:
        text_tokens = build_dataset_class_tokens(text_transform, seg_config.template, classnames)
        text_embedding = model.build_text_embedding(text_tokens, num_classes=len(classnames))
    kwargs = dict(with_bg=with_bg)

    if hasattr(cfg, 'test_cfg'):
        kwargs['test_cfg'] = cfg.test_cfg
    
    seg_model = GroupViTSegInference(model, text_embedding, **kwargs)
    print('Evaluate during seg inference')

    seg_model.CLASSES = tuple(['background'] + list(classnames))
    seg_model.PALETTE = palette
    return seg_model

class LoadImage:
    """A simple pipeline to load image."""
    cnt = 0
    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def build_seg_demo_pipeline():
    """Build a demo pipeline from config."""
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    test_pipeline = Compose([
        LoadImage(),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 448),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ])
    return test_pipeline
