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
# Modified by Jilan Xu
# -------------------------------------------------------------------------

import argparse
import datetime
import os
import os.path as osp
import time
from collections import defaultdict
import subprocess
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import build_loader, build_text_transform, imagenet_classes
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import collect_env, get_git_hash
from mmseg.apis import multi_gpu_test
from models import build_model
from omegaconf import OmegaConf, read_write
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from timm.utils import AverageMeter, accuracy
from utils import (auto_resume_helper, build_dataset_class_tokens, build_optimizer, build_scheduler, data2cuda,
                   get_config, get_grad_norm, get_logger, load_checkpoint, parse_losses, reduce_tensor, save_checkpoint, momentum_update,
                   load_checkpoint_stage1, build_dataset_class_lists,cdist_,
                   )

from ipdb import set_trace
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, RobertaTokenizer
from einops import rearrange
tokenizer_dict = {
    'Bert': AutoTokenizer.from_pretrained('distilbert-base-uncased', TOKENIZERS_PARALLELISM=False),
    'TextTransformer': None,
}


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_args():
    parser = argparse.ArgumentParser('GroupViT training and evaluation script')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')

    # easy config modification
    parser.add_argument('--batch-size', type=int, help='batch size for single GPU')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument(
        '--amp-opt-level',
        type=str,
        default='O1',
        choices=['O0', 'O1', 'O2'],
        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument(
        '--output', type=str, help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--wandb', action='store_true', help='Use W&B to log experiments')
    parser.add_argument('--keep', type=int, help='Maximum checkpoint to keep')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=False, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    return args


def train(cfg):
    if cfg.wandb and dist.get_rank() == 0:
        import wandb
        wandb.init(
            project='group_vit',
            name=osp.join(cfg.model_name, cfg.tag),
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=cfg.checkpoint.auto_resume)
    else:
        wandb = None
    # waiting wandb init
    dist.barrier()
    
    dataset_train, dataset_val, \
        data_loader_train, data_loader_val = build_loader(cfg.data)
    print('Done train/val loader')
    data_loader_seg = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg))
    print('Done seg loader')
    
    logger = get_logger()
    if dist.get_rank() == 0:
        writer = SummaryWriter(cfg.output)
    else:
        writer = None

    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(cfg.train, model)
    if cfg.train.amp_opt_level != 'O0':
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.train.amp_opt_level)

    model = MMDistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, find_unused_parameters=True)
    model_without_ddp = model.module
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')
    lr_scheduler = build_scheduler(cfg.train, optimizer, len(data_loader_train))

    ##### load init params from stage 1 here, before auto resuming ######
    if cfg.checkpoint.stage1_checkpoint:
        load_checkpoint_stage1(cfg, model_without_ddp)

    if cfg.checkpoint.auto_resume:
        resume_file = auto_resume_helper(cfg.output)
        if resume_file:
            if cfg.checkpoint.resume:
                logger.warning(f'auto-resume changing resume file from {cfg.checkpoint.resume} to {resume_file}')
            with read_write(cfg):
                cfg.checkpoint.resume = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.output}, ignoring auto resume')

    max_accuracy = max_miou = 0.0
    max_metrics = {'max_accuracy': max_accuracy, 'max_miou': max_miou}

    if cfg.checkpoint.resume:
        max_metrics = load_checkpoint(cfg, model_without_ddp, optimizer, lr_scheduler)
        max_accuracy, max_miou = max_metrics['max_accuracy'], max_metrics['max_miou']
    
    ############# set tokenizer ##############
    global tokenizer
    tokenizer = tokenizer_dict[cfg.model.text_encoder.type]
    tensorbd_logdir = cfg.output + "/logs"

    logger.info('Start training')
    start_time = time.time()
    
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        ### train model ###
        loss_train_dict = train_one_epoch(cfg, model, data_loader_train, optimizer, epoch, lr_scheduler, writer)
        if dist.get_rank() == 0 and (epoch % cfg.checkpoint.save_freq == 0 or epoch == (cfg.train.epochs - 1)):
            save_checkpoint(cfg, epoch, model_without_ddp, {
                'max_accuracy': max_accuracy,
                'max_miou': max_miou
            }, optimizer, lr_scheduler)
        dist.barrier()
        loss_train = loss_train_dict['total_loss']
        logger.info(f'Avg loss of the network on the {len(dataset_train)} train images: {loss_train:.2f}')
                
        # evaluate
        if (epoch % cfg.evaluate.eval_freq == 0 or epoch == (cfg.train.epochs - 1)):
            if 'cls' in cfg.evaluate.task:
                acc1, acc5, loss = validate_cls(cfg, data_loader_val, model)
                logger.info(f'Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%')
                max_metrics['max_accuracy'] = max(max_metrics['max_accuracy'], acc1)
                # if cfg.evaluate.cls.save_best and dist.get_rank() == 0 and acc1 > max_accuracy:
                #     save_checkpoint(
                #         cfg, epoch, model_without_ddp, max_metrics, optimizer, lr_scheduler, suffix='best_acc1')
                dist.barrier()
                max_accuracy = max_metrics['max_accuracy']
                logger.info(f'Max accuracy: {max_accuracy:.2f}%')

            if 'seg' in cfg.evaluate.task:
                miou = validate_seg(cfg, data_loader_seg, model, epoch, writer, tokenizer=tokenizer)
                logger.info(f'mIoU of the network on the {len(data_loader_seg.dataset)} test images: {miou:.2f}%')
                max_metrics['max_miou'] = max(max_metrics['max_miou'], miou)
                if cfg.evaluate.seg.save_best and dist.get_rank() == 0 and miou > max_miou:
                    print('ready saving the best iou model')
                    save_checkpoint(
                        cfg, epoch, model_without_ddp, max_metrics, optimizer, lr_scheduler, suffix='best_miou')
                dist.barrier()
                max_miou = max_metrics['max_miou']
                logger.info(f'Max mIoU: {max_miou:.2f}%')

        if wandb is not None:
            log_stat = {f'epoch/train_{k}': v for k, v in loss_train_dict.items()}
            log_stat.update({
                'epoch/val_acc1': acc1,
                'epoch/val_acc5': acc5,
                'epoch/val_loss': loss,
                'epoch/val_miou': miou,
                'epoch/epoch': epoch,
                'epoch/n_parameters': n_parameters
            })
            wandb.log(log_stat)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    dist.barrier()
    # writer.flush()

def process_text(text_data):
    ### we run all the exps with padding=True, meaning padding to the longest caption ###
    # text_data = tokenizer(text_data, return_tensors='pt', padding=True,
    #                         truncation=True, max_length=77)
    
    ### this is more memory friendly/load balance if we chunk the padding size to max_length ###
    text_data = tokenizer(text_data, return_tensors='pt', padding='max_length',
                            truncation=True, max_length=77)
    text_data = {key: val.cuda() for key, val in text_data.items()}
    return text_data

                    
def generate_entity_masks(text_data):
    text = text_data['input_ids']
    # [b, L]
    entity_masks = text.clone()
    entity_masks[entity_masks != 103] = 0
    entity_masks[entity_masks == 103] = 1
    
    entity_masks  = entity_masks.to(text.device)
    return entity_masks

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, writer):
    logger = get_logger()
    dist.barrier()
    model.train()
    optimizer.zero_grad()
    if config.wandb and dist.get_rank() == 0:
        import wandb
    else:
        wandb = None

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()

    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)
    
    for idx, samples in enumerate(data_loader):        
        batch_size = config.data.train.batch_size
        all_images = samples['image'].cuda()

        all_questions = None
        entity_labels = entity_masks =  None
        all_answers = None
        if config.model.text_encoder['type'] in ['DistilBert','Bert','BertMedium','Roberta']:
            all_texts = process_text(samples['raw_caption'])
            if config.data.train.use_entity is True:
                all_questions = process_text(samples['raw_question'])
                all_answers= process_text(samples['raw_answer'])
                entity_masks = generate_entity_masks(all_questions)

        elif config.model.text_encoder['type'] not in ['TextTransformer'] and config.data.train.use_entity is True:
            all_texts = samples['caption'].cuda()
            all_questions = samples['question'].cuda()
            all_answers = samples['answer'].cuda()
        else:
            all_texts = samples['caption'].cuda()
        
        ### for cross-image mask consistency loss ###
        all_crossimage = samples['cross_image'].cuda() if 'cross_image' in samples and samples['cross_image'] is not None else None
        question_masks = samples['question_mask'].cuda() if 'question_mask' in samples else None
        cross_entity = process_text(samples['cross_entity']) if 'cross_entity' in samples and samples['cross_entity'] is not None else None

        ### forward and compute loss ###
        losses = model(image=all_images, text=all_texts, cross_image=all_crossimage, cross_entity=cross_entity, \
                        question=all_questions, answer=all_answers, entity_masks=entity_masks, question_masks=question_masks)
        loss, log_vars = parse_losses(losses)
        
        if dist.get_rank() == 0:
            writer.add_scalar("Total loss", loss, len(data_loader) * epoch + idx)
            writer.add_scalar("contrastive loss", losses['loss'], len(data_loader) * epoch + idx)
            if 'entity' in losses:
                writer.add_scalar("entity loss", losses['entity'], len(data_loader) * epoch + idx)
            if 'mask' in losses:
                writer.add_scalar("Mask loss", losses['mask'], len(data_loader) * epoch + idx)
            writer.add_scalar("lr",  optimizer.param_groups[0]['lr'], len(data_loader) * epoch + idx)

        if config.train.accumulation_steps > 1:
            loss = loss / config.train.accumulation_steps
            if config.train.amp_opt_level != 'O0':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.train.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.train.clip_grad)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.train.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.train.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
            
            if config.model.use_maskloss:
                maskloss_coeff = 0.99
                momentum_update(model.module.img_encoder, model.module.img_encoder_momentum, maskloss_coeff)
        
        else:
            optimizer.zero_grad()
            if config.train.amp_opt_level != 'O0':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.train.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.train.clip_grad)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.train.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)
            if config.model.use_maskloss:
                maskloss_coeff = 0.99
                momentum_update(model.module.img_encoder, model.module.img_encoder_momentum, maskloss_coeff)


        torch.cuda.synchronize()
        loss_meter.update(loss.item(), batch_size)
        for loss_name in log_vars:
            log_vars_meters[loss_name].update(log_vars[loss_name], batch_size)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            log_vars_str = '\t'.join(f'{n} {m.val:.4f} ({m.avg:.4f})' for n, m in log_vars_meters.items())
            logger.info(f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'{log_vars_str}\t'
                        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')
            if wandb is not None:
                log_stat = {f'iter/train_{n}': m.avg for n, m in log_vars_meters.items()}
                log_stat['iter/train_total_loss'] = loss_meter.avg
                log_stat['iter/learning_rate'] = lr
                wandb.log(log_stat)

    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')
    result_dict = dict(total_loss=loss_meter.avg)
    for n, m in log_vars_meters.items():
        result_dict[n] = m.avg
    dist.barrier()
    return result_dict

@torch.no_grad()
def validate_cls(config, data_loader, model):
    logger = get_logger()
    dist.barrier()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)

    end = time.time()
    logger.info('Building zero shot classifier')

    if config.model.text_encoder['type'] in ['DistilBert', 'Bert','BertMedium','Roberta']:
        text_embedding = model.module.build_text_embedding(
                build_dataset_class_lists(config.evaluate.cls.template, imagenet_classes), tokenizer, len(imagenet_classes))
    else:    
        text_embedding = data2cuda(
            model.module.build_text_embedding(
                build_dataset_class_tokens(text_transform, config.evaluate.cls.template, imagenet_classes)))
        
    logger.info('Zero shot classifier built')
    
    for idx, samples in enumerate(data_loader):
        all_images = samples['image'].cuda()
        target = samples['target'].cuda()
        output = model(image=all_images, text=text_embedding)
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                        f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')
    logger.info('Clearing zero shot classifier')
    torch.cuda.empty_cache()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    dist.barrier()
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

@torch.no_grad()
def validate_seg(config, data_loader, model, epoch=0, writer=None, tokenizer=None):
    logger = get_logger()
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
    results = multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=True,
        efficient_test=False,
        pre_eval=True,
        format_only=False)

    if dist.get_rank() == 0:
        metric = [data_loader.dataset.evaluate(results, metric='mIoU')]
    else:
        metric = [None]
    dist.broadcast_object_list(metric)
    miou_result = metric[0]['mIoU'] * 100

    torch.cuda.empty_cache()
    logger.info(f'Eval Seg mIoU {miou_result:.2f}')
    if writer is not None and dist.get_rank() == 0:
        writer.add_scalar("mIoU", miou_result, epoch)
    dist.barrier()
    return miou_result

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
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = os.environ['SLURM_NTASKS']
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list)
        )
        master_port = os.environ.get('MASTER_PORT', '29484')
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
    '''
    # start faster ref: https://github.com/open-mmlab/mmdetection/pull/7036
    mp.set_start_method('fork', force=True)
    init_dist('pytorch')
    rank, world_size = get_dist_info()
    print(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')
    dist.barrier()
    '''
    init_distributed_mode(args)
    rank, world_size = args.rank, args.world_size

    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = cfg.train.base_lr * cfg.data.train.batch_size * world_size / 4096.0
    linear_scaled_warmup_lr = cfg.train.warmup_lr * cfg.data.train.batch_size * world_size / 4096.0
    linear_scaled_min_lr = cfg.train.min_lr * cfg.data.train.batch_size * world_size / 4096.0


    # gradient accumulation also need to scale the learning rate
    if cfg.train.accumulation_steps > 1:
        linear_scaled_lr = linear_scaled_lr * cfg.train.accumulation_steps
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * cfg.train.accumulation_steps
        linear_scaled_min_lr = linear_scaled_min_lr * cfg.train.accumulation_steps

    with read_write(cfg):
        logger.info(f'Scale base_lr from {cfg.train.base_lr} to {linear_scaled_lr}')
        logger.info(f'Scale warmup_lr from {cfg.train.warmup_lr} to {linear_scaled_warmup_lr}')
        logger.info(f'Scale min_lr from {cfg.train.min_lr} to {linear_scaled_min_lr}')
        cfg.train.base_lr = linear_scaled_lr
        cfg.train.warmup_lr = linear_scaled_warmup_lr
        cfg.train.min_lr = linear_scaled_min_lr

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config saved to {path}')

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    logger.info(f'Git hash: {get_git_hash(digits=7)}')

    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    train(cfg)
    dist.barrier()

if __name__ == '__main__':
    main()
