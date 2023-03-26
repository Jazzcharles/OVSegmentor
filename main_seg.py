# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------
# Modified by Jilan Xu
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

from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from utils import get_config, get_logger, load_checkpoint
from transformers import AutoTokenizer, RobertaTokenizer
from ipdb import set_trace

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
    parser = argparse.ArgumentParser('GroupViT segmentation evaluation and visualization')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='path to config file',
    )
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument(
        '--output', type=str, help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument(
        '--vis',
        help='Specify the visualization mode, '
        'could be a list, support input, pred, input_seg, input_pred_seg_label, all_groups, first_group, last_group',
        default=None,
        nargs='+')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=False, default=0, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    return args


def inference(cfg):
    logger = get_logger()
    data_loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg))
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
    if cfg.model.text_encoder.type == 'Roberta':
        tokenizer = RobertaTokenizer.from_pretrained('/mnt/petrelfs/xujilan/roberta-base/')
        print('Done switching roberta tokenizer')
    
    if 'seg' in cfg.evaluate.task:
        miou = validate_seg(cfg, data_loader, model, tokenizer=tokenizer)
        logger.info(f'mIoU of the network on the {len(data_loader.dataset)} test images: {miou:.2f}%')
    else:
        logger.info('No segmentation evaluation specified')

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
    if config.model.text_encoder['type'] in ['DistilBert', 'Bert','BertMedium','Roberta']:
        seg_model = build_seg_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.seg, tokenizer)
    else:
        seg_model = build_seg_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.seg)

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
                out_file = osp.join(config.output, 'vis_imgs', vis_mode, f'{batch_idx:04d}.jpg')
                model.show_result(img_show, img_tensor.to(device), result, out_file, vis_mode)
            if dist.get_rank() == 0:
                batch_size = len(result) * dist.get_world_size()
                for _ in range(batch_size):
                    prog_bar.update()    


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        #args.rank = int(os.environ['SLURM_PROCID'])
        #args.gpu = args.rank % torch.cuda.device_count()
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = os.environ['SLURM_NTASKS']
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list)
        )
        master_port = os.environ.get('MASTER_PORT', '29499')
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = int(ntasks)
        args.rank = int(proc_id)
        args.gpu = int(proc_id % num_gpus)
        print(f'SLURM MODE: proc_id: {proc_id}, ntasks: {ntasks}, node_list: {node_list}, num_gpus:{num_gpus}, addr:{addr}, master port:{master_port}' )
        
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

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
