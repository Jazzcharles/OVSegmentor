# -------------------------------------------------------------------------
# Written by Jilan Xu
# -------------------------------------------------------------------------

import argparse
import os
import os.path as osp
import subprocess

import mmcv
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datasets import build_text_transform
from main_pretrain import validate_seg
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import set_random_seed
from models import build_model
from omegaconf import OmegaConf, read_write

from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_custom_seg_dataset, build_seg_inference, build_demo_inference
from utils import get_config, get_logger, load_checkpoint
from transformers import AutoTokenizer, RobertaTokenizer
from ipdb import set_trace
from main_pretrain import init_distributed_mode

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
    
tokenizer_dict = {
    'Bert': AutoTokenizer.from_pretrained('distilbert-base-uncased', TOKENIZERS_PARALLELISM=False),
    # 'Roberta': RobertaTokenizer.from_pretrained('/mnt/petrelfs/xujilan/roberta-base/'),
    'Roberta': RobertaTokenizer.from_pretrained('roberta-base'),
    'TextTransformer': None,
}

def parse_args():
    parser = argparse.ArgumentParser('OVSegmentor segmentation demo')
    parser.add_argument(
        '--cfg',
        type=str,
        default='./configs/test_voc12.yml',
        help='path to config file',
    )
    parser.add_argument(
        '--resume', 
        type=str,
        required=True,
        help='resume from checkpoint',
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        required=True,
        help='path to the input image folder',
    )
    parser.add_argument(
        '--vocab',
        help='could be a list of candidate vocabularies, use given classes from [voc, coco, ade], or give a custom list of classes',
        default='voc',
        nargs='+',
    )
    
    parser.add_argument(
        '--output_folder', 
        type=str, 
        help='root of output folder',
    )
    parser.add_argument(
        '--vis',
        help='Specify the visualization mode, '
        'could be a list, support input, pred, input_seg, input_pred_seg_label, all_groups, first_group, last_group, mask',
        default='input_pred_seg_label',
        nargs='+',
    )
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )


    # distributed training
    parser.add_argument('--local_rank', type=int, required=False, default=0, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    return args

def generate_imagelist_with_sanity_check(root):
    image_list = []
    for each_file in os.listdir(root):
        ### assume we process all .jpg files, and convert png to jpg files
        if each_file.endswith('.jpg'):
            pass
        elif each_file.endswith('.png'):
            img = mmcv.imread(osp.join(root, each_file))
            mmcv.imwrite(img, osp.join(root, each_file.replace('.png','.jpg')))
        else:
            continue
        
        filename = each_file.split('.')[0]
        image_list.append(filename)
            
    if len(image_list) == 0:
        raise ValueError(f'No image found in {args.image_folder}')
    
    with open(os.path.join(root, 'image_list.txt'), 'w') as f:
        for item in image_list:
            f.write("%s\n" % item)
            
    return image_list
    
def inference(cfg):
    logger = get_logger()
    
    ### check and generate image list ###
    generate_imagelist_with_sanity_check(cfg.image_folder)
    os.makedirs(cfg.output_folder, exist_ok=True)
    
    data_loader = build_seg_dataloader(build_custom_seg_dataset(cfg.evaluate.seg, cfg))
    dataset = data_loader.dataset
    print('whether activating visualization: ', cfg.vis)
    logger.info(f'Evaluating dataset: {dataset}')
    
    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    model.cuda()
    logger.info(str(model))

    if cfg.train.amp_opt_level != 'O0':
        model = amp.initialize(model, None, opt_level=cfg.train.amp_opt_level)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    load_checkpoint(cfg, model, None, None)
    
    global tokenizer
    tokenizer = tokenizer_dict[cfg.model.text_encoder.type]

    if cfg.vis:
        vis_seg(cfg, data_loader, model, cfg.vis)


@torch.no_grad()
def vis_seg(config, data_loader, model, vis_modes):
    dist.barrier()
    model.eval()
    
    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)
    seg_model = build_demo_inference(model_without_ddp, text_transform, config, tokenizer)

    mmddp_model = MMDistributedDataParallel(
        seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    mmddp_model.eval()
    model = mmddp_model.module
    device = next(model.parameters()).device
    dataset = data_loader.dataset
    
    if dist.get_rank() == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = mmddp_model(return_loss=False, **data)
        # set_trace()

        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        for batch_idx, img, img_meta in zip(batch_indices, imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            for vis_mode in vis_modes:
                out_file = osp.join(config.output_folder, vis_mode, f'{batch_idx:04d}.jpg')
                # os.makedirs(osp.join(config.output_folder, 'vis_imgs', vis_mode), exist_ok=True)
                print(osp.join(config.output_folder, vis_mode))
                model.show_result(img_show, img_tensor.to(device), result, out_file, vis_mode)
            if dist.get_rank() == 0:
                batch_size = len(result) * dist.get_world_size()
                for _ in range(batch_size):
                    prog_bar.update()    

def main():
    args = parse_args()
    cfg = get_config(args)

    if cfg.train.amp_opt_level != 'O0':
        assert amp is not None, 'amp not installed!'

    with read_write(cfg):
        cfg.evaluate.eval_only = True
    
    init_distributed_mode(args)
    rank, world_size = args.rank, args.world_size

    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config saved to {path}')

    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    inference(cfg)
    dist.barrier()


if __name__ == '__main__':
    main()
