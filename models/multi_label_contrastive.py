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

import diffdist.functional as diff_dist
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.loss import SoftTargetCrossEntropy
from random import choice

from .builder import MODELS
from .misc import Result
from .losses import HungarianMatcher, dice_loss

from ipdb import set_trace
import torchvision.ops.roi_pool as roi_pool
import cv2
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .group_vit import CrossAttnBlock, AssignAttention, AttnBlock

def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()

class ProjectMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(ProjectMLP, self).__init__()
        # hidden layers
        linear_hidden = []
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Conv1d(in_dim if i == 0 else inner_dim, inner_dim, kernel_size=1))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Conv1d(
            in_dim if num_layers == 1 else inner_dim, out_dim, kernel_size=1) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): output of transformers, shape [B, L, C]

        Returns:

        """
        assert x.ndim in [2, 3], x.ndim
        add_dim = False
        if x.ndim == 2:
            # [B, C] -> [B, L, C]
            x = x.unsqueeze(1)
            add_dim = True

        x = rearrange(x, 'b l c -> b c l')
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        x = rearrange(x, 'b c l -> b l c')

        if add_dim:
            x = x.squeeze(1)

        return x

class MultimodalGroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.,
                 attn_drop=0.,
                 ):
        super(MultimodalGroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        # norm on x
        self.norm_x = norm_layer(dim)
        # self.visual_attn = AttnBlock(
        #     dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer )
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.post_attn = AttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer )
        
        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps,
            attn_drop=attn_drop,
            )
        self.norm_new_x = norm_layer(dim)

    def forward(self, ans_tokens, visual_tokens, text_tokens, entity_masks=None, question_masks=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): group_tokens, [B, k, C]
            group_tokens (torch.Tensor): word tokens, [B, L, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        # [B, K, C], self-attention
        # visual_tokens = self.visual_attn(visual_tokens)
    
        text_tokens = self.norm_tokens(text_tokens)
        visual_tokens = self.norm_x(visual_tokens)
    
        # [B, L, C], cross attention
        projected_text_tokens = self.pre_assign_attn(text_tokens, visual_tokens)
        ### mask needs to be [b, 1, 77, 1] to match [b, nh, 77, k]
        # projected_text_tokens = text_tokens
        # new_x, attn_dict = self.assign(projected_text_tokens, visual_tokens, return_attn=return_attn, mask=question_masks)
        
        if ans_tokens is None:
            ans_temp = projected_text_tokens
        else:
            ans_temp = ans_tokens + projected_text_tokens    
    
        ############## self-attn only ###################
        if question_masks is not None:
            new_x = self.post_attn(ans_temp, mask=question_masks)
        else:
            new_x = self.post_attn(ans_temp)
        new_x += projected_text_tokens
        
        new_x = self.norm_new_x(new_x)
        return new_x


class MultimodalGroupingNetwork(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.,
                 attn_drop=0.,
                 num_layers=1,
                 ):
        super(MultimodalGroupingNetwork, self).__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
                MultimodalGroupingBlock(
                    dim=dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                    hard=hard,
                    gumbel=gumbel,
                    sum_assign=sum_assign,
                    assign_eps=assign_eps,
                    gumbel_tau=gumbel_tau,
                    attn_drop=attn_drop,
                ) for i in range(num_layers)
            ])
        
        
    def forward(self, visual_tokens, text_tokens, entity_masks=None, question_masks=None, return_attn=False, return_feat=False):
        """
        Args:
            x (torch.Tensor): group_tokens, [B, k, C]
            group_tokens (torch.Tensor): word tokens, [B, L, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens

            1. norm
            2. cross-attn
            3. self-attn

        """
        ans_text = None
        for i, blk in enumerate(self.blocks):
            ans_text = blk(ans_text, visual_tokens, text_tokens, entity_masks, question_masks, return_attn)
        
        if return_feat is True:  #[B, L, d_t]
            return ans_text

        answer = ans_text[:, 0]
        return answer
        

@MODELS.register_module()
class MultiLabelContrastive(nn.Module):

    def __init__(self,
                 img_encoder,
                 text_encoder,
                 output_dim=256,
                 contrast_temperature=0.07,
                 proj_num_layers=2,
                 multi_label=0,
                 share_temperature=False,
                 multi_label_loss_weight=1.0,
                 
                 use_entityloss=False,
                 entity_weight=1.0,
                 cross_layers=1,

                 use_maskloss=False,
                 maskloss_weight=0.1,
                 num_deep_stages=1,
                 cost_type='L2',
                 cross_threshold=0.6,
                 topmask_ratio=1.0,
                 dual_dice=False,
                 group_ratio=0.5,
                 ):
        super().__init__()

        self.img_encoder = MODELS.build(img_encoder)
        self.text_encoder = MODELS.build(text_encoder)
        self.img_encoder_type = img_encoder['type']
        self.text_encoder_type = text_encoder['type']
        # add 
        print('self image encoder: ', img_encoder)
        print('self text encoder:', text_encoder)

        self.contrast_temperature = contrast_temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
        
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        self.binary_cross_entropy = nn.BCELoss()
        self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss()
        self.soft_cross_entropy = SoftTargetCrossEntropy()
        self.mse_loss = nn.MSELoss()

        
        self.proj_num_layers = proj_num_layers
        self.multi_label = multi_label
        
        if proj_num_layers > 0:
        # if proj_num_layers > 0 and self.use_clip_visual is False:
            self.img_projector = ProjectMLP(
                in_dim=self.img_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
            self.text_projector = ProjectMLP(
                in_dim=self.text_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
            self.img_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.img_projector)
            self.text_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.text_projector)
        elif proj_num_layers == -1:
            self.img_projector = nn.Linear(self.img_encoder.width, self.text_encoder.width)
            self.text_projector = nn.Identity()
        else:
            self.img_projector = nn.Identity()
            self.text_projector = nn.Identity()
        

        self.share_temperature = share_temperature
        if self.with_multi_label and not self.share_temperature:
            self.multi_label_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
        self.multi_label_loss_weight = multi_label_loss_weight
        
        ### for masked entity loss ###
        self.use_entityloss = use_entityloss
        self.entity_weight = entity_weight
        self.cross_layers = cross_layers
        if self.use_entityloss:
            min_width = min(self.img_encoder.width, self.text_encoder.width)
            max_width = max(self.img_encoder.width, self.text_encoder.width)
            self.align_proj_img = nn.Linear(max_width, min_width) if self.img_encoder.width > self.text_encoder.width else nn.Identity()
            self.align_proj_text = nn.Linear(max_width, min_width) if self.text_encoder.width > self.img_encoder.width else nn.Identity()
            
            ### similar to transformer decoder ###
            self.multimodal_groupingblock = MultimodalGroupingNetwork(
                dim=min_width,
                out_dim=min_width,
                num_heads=8,
                norm_layer=nn.LayerNorm,
                hard=False,
                gumbel=False,
                num_layers=cross_layers,
            )
            self.bridge_projector = ProjectMLP(
                in_dim=min_width, num_layers=proj_num_layers, out_dim=output_dim)
        
        
        ### for mask loss ###
        self.use_maskloss = use_maskloss
        self.maskloss_weight = maskloss_weight

        self.cross_threshold = cross_threshold
        self.topmask_ratio = topmask_ratio
        self.dual_dice = dual_dice
        self.group_ratio = group_ratio
        
        if self.use_maskloss:
            self.num_deep_stages = num_deep_stages
            self.logit_scale_mask = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
            self.img_encoder_momentum = MODELS.build(img_encoder)
            
            self.q_projector = nn.Identity()
            self.k_projector = nn.Identity()
            self.q_projector_momentum = nn.Identity()
            self.k_projector_momentum = nn.Identity()

            ## set momentum branch offline
            for p in self.img_encoder_momentum.parameters():
                p.requires_grad = False
            self.matcher = HungarianMatcher(cost_type=cost_type)
            
                    
    def mask_loss(self, mask1, mask2, threshold, imgtokens=None, text=None, indicator='none'):
        # set_trace()
        bs = mask1.size(0)
        num_masks = mask1.size(1)
        
        ################# hungarian matching #######################################
        #[b, k, hw], make the masks exclusive with softmax???
        ############# Note, we keep the original mask, while using the normed mask to compute matching ########
        mask1 = torch.flatten(mask1, 2).float()
        mask2 = torch.flatten(mask2, 2).float()
        mask1_norm = F.normalize(mask1, dim=-1)
        mask2_norm = F.normalize(mask2, dim=-1)

        idx1, idx2 = self.matcher(mask1_norm, mask2_norm)
        mask1 = mask1[torch.arange(bs).unsqueeze(1), idx1]
        mask2 = mask2[torch.arange(bs).unsqueeze(1), idx2]
        
        ################## norm and contrastive loss ################################
        #[b, k, hw]
        
        ################# BCE loss ##################################################
        ### hard-thresholding ###
        def min_max_norm(x):
            x_max = torch.max(x, dim=-1, keepdim=True)[0]
            x_min = torch.min(x, dim=-1, keepdim=True)[0]
            return (x - x_min) / (x_max - x_min)
        
        ################ THIS IS PERHAPS IMPORTANT HERE ##############
        mask2 = mask2.sigmoid()
        # mask2 = F.softmax(mask2, dim=1)
        # mask2 = min_max_norm(mask2)
        # mask2 = F.normalize(mask2)

        mask2_pseudo = mask2
        mask2_pseudo = rearrange(mask2_pseudo, 'b k d -> (b k) d')

        thres_onehot = torch.max(mask2_pseudo, dim=-1, keepdim=True)[0] * threshold
        mask2_onehot = mask2_pseudo - thres_onehot
        mask2_onehot[mask2_onehot >= 0] = 1.0
        mask2_onehot[mask2_onehot < 0] = 0.0
        mask2_onehot = rearrange(mask2_onehot, '(b k) d -> b k d', k=num_masks)

        # self.draw_attn(rearrange(mask1, 'b k (h w) -> b k h w', k=num_masks, h=224), 'before_sigmoid')
        # set_trace()
        # mask1 = F.softmax(mask1, dim=1)
        # mask1 = torch.sigmoid(mask1)
        mask1 = min_max_norm(mask1)
        
        ####### select topk mask for contrast w.r.t ratio #######
        topk_mask = None
        # if self.topmask_ratio < 1.0:
        #     alltoken_logits = (imgtokens @ text.unsqueeze(-1)).squeeze(-1) #[bs, k]
        #     topk_logits = torch.topk(alltoken_logits, k=int(num_masks * self.topmask_ratio))[1]
        #     topk_mask = torch.zeros_like(alltoken_logits)
        #     topk_mask[torch.arange(bs).unsqueeze(1), topk_logits] = 1.0
            # set_trace()
        #########################################################

        loss = dice_loss(mask1, mask2_onehot, topk_mask=topk_mask) 

        return loss

    @property
    def with_multi_label(self):
        return self.multi_label > 0

    def loss(self, image_x, text_x):
        batch_size = image_x.shape[0]
        # get label globally
        labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()

        image_x = F.normalize(image_x, dim=-1) #[B, C]
        text_x = F.normalize(text_x, dim=-1) #[B, C]

        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)
        return loss

    def multi_label_loss(self, image_feat, text_feat):
        """

        Args:
            image_feat (torch.Tensor): shape [B, L1, C]
            text_feat (torch.Tensor): shape [B, L2, C]

        Returns:

        """
        # [B, L1, C], L1 = 1
        image_feat = F.normalize(image_feat, dim=-1)
        # [B, L2, C]
        text_feat = F.normalize(text_feat, dim=-1)

        # [B, L1, L2]
        dist_per_img = image_feat @ rearrange(text_feat, 'b l c -> b c l')
        # [B, L2, L1]
        dist_per_text = text_feat @ rearrange(image_feat, 'b l c -> b c l')

        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)

        batch = image_feat.shape[0]
        img_len = image_feat.shape[1]
        text_len = text_feat.shape[1]
        # [B, L1, L2]
        pos_labels_batch_img = rearrange(torch.ones_like(dist_per_text) / dist_per_text.size(1), 'b l2 l1 -> b l1 l2')
        # [B, L2, L1]
        pos_labels_batch_text = rearrange(torch.ones_like(dist_per_img) / dist_per_img.size(1), 'b l1 l2 -> b l2 l1')

        image_x = rearrange(image_feat, 'b l c -> (b l) c')
        text_x = rearrange(text_feat, 'b l c -> (b l) c')

        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()

        # get label globally
        # [B, L1, B, L2, W]
        labels_per_img = F.one_hot(
            torch.ones(batch, img_len, batch, text_len, dtype=torch.long, device=image_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(image_x.dtype)
        labels_per_img *= rearrange(pos_labels_batch_img, 'b l1 l2 -> b l1 1 l2 1') * repeat(
            torch.eye(batch, dtype=image_x.dtype, device=image_x.device), 'b1 b2 -> b1 1 b2 1 1')
        # [BxL1, WxBxL2]
        labels_per_img = rearrange(labels_per_img, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
        # [B, L2, B, L1, W]
        labels_per_text = F.one_hot(
            torch.ones(batch, text_len, batch, img_len, dtype=torch.long, device=text_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(text_x.dtype)
        labels_per_text *= rearrange(pos_labels_batch_text, 'b l2 l1 -> b l2 1 l1 1') * repeat(
            torch.eye(batch, dtype=text_x.dtype, device=image_x.device), 'b2 b1 -> b2 1 b1 1 1')
        # [BxL2, WxBxL1]
        labels_per_text = rearrange(labels_per_text, 'b2 l2 b1 l1 w -> (b2 l2) (w b1 l1)')

        loss_img = self.soft_cross_entropy(logits_per_img * logit_scale, labels_per_img)
        loss_text = self.soft_cross_entropy(logits_per_text * logit_scale, labels_per_text)

        loss = 0.5 * (loss_img + loss_text)

        return loss

    def encode_image(self, image, *, return_feat=False, as_dict=False, return_attn=False, momentum=False):
        outs = Result(as_dict)
        ### momentum branch, no gradient update ###
        if momentum:
            with torch.no_grad():
                img_outs = self.img_encoder_momentum(image, return_feat=return_feat, as_dict=True, return_attn=return_attn)
                outs.append(self.img_projector(img_outs['x']), 'image_x')
                if return_feat and 'feat' in img_outs:
                    outs.append(img_outs['x'], 'image_x_before_proj')
                    outs.append(img_outs['feat'], 'image_feat_before_proj')
                
                if return_feat:
                    outs.append(self.img_projector(img_outs['feat']), 'image_feat')
                if return_attn:
                    outs.append(img_outs['attn_dicts'], 'attn_dicts')
                return outs.as_return()            
        else:
        ### online branch ###
            img_outs = self.img_encoder(image, return_feat=return_feat, as_dict=True, return_attn=return_attn)
            # change here
            outs.append(self.img_projector(img_outs['x']), 'image_x')
            if return_feat and 'feat' in img_outs:
                outs.append(img_outs['x'], 'image_x_before_proj')
                outs.append(img_outs['feat'], 'image_feat_before_proj')

            if return_feat:
                outs.append(self.img_projector(img_outs['feat']), 'image_feat')
            if return_attn:
                outs.append(img_outs['attn_dicts'], 'attn_dicts')
            return outs.as_return()
    
    def encode_text(self, text, *, as_dict=False, forward_template=False):
        # assert text.ndim in [2, 3], text.ndim
        squeeze_dim = False
        num_text = 1
        if type(text) is not dict and text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, 'b n l -> (b n) l', n=num_text)
            squeeze_dim = True

        outs = Result(as_dict=as_dict)
        # [B, C]
        text_outs = self.text_encoder(text)
        if 'all_tokens' in text_outs:
            all_tokens = text_outs['all_tokens'].contiguous()

        x = text_outs['x']
        text_x = self.text_projector(x)
        
        outs.append(text_x, 'text_x')
        outs.append(x, 'text_x_before_proj') # add transformer out
        outs.append(all_tokens, 'text_feat_before_proj')
        outs.append(self.text_projector(all_tokens), 'text_feat_after_proj')

        # if squeeze_dim:
        if (squeeze_dim or self.with_multi_label) and self.training:
            # text_x = rearrange(text_x, '(b n) c -> b n c', n=num_text)
            text_x = rearrange(text_x, '(b n) c -> b n c', n=self.multi_label + 1) ### 2 prompts and 1 caption
            text_multi_label_x = text_x[:, 1:]
            text_x = text_x[:, 0]
            ####### here projection !!!! #######
            outs.update(text_x=text_x, text_multi_label_x=text_multi_label_x)

        return outs.as_return()
 

    def project_and_mask(self, q, k, branch='online'):
        scale = self.img_encoder.width ** -0.5

        if branch == 'online':
            q = self.q_projector(q)
            k = self.k_projector(k)
            attn = q @ k.transpose(-2, -1) * scale  ### no softmax for now
        else:
            with torch.no_grad():
                q = self.q_projector_momentum(q)
                k = self.k_projector_momentum(k)
                attn = q @ k.transpose(-2, -1) * scale  ### no softmax for now

        return attn
    
    def forward_train(self, image, text, cross_image=None, cross_entity=None, \
                        question=None, answer=None, entity_masks=None, question_masks=None):
        bs = image.size(0)
        losses_dict = dict()
        
        ############################################################
        ### Encode image and caption, calculate image-caption matching loss ###
        text_outs = self.encode_text(text, as_dict=True)
        text_x = text_outs['text_x']  # [B, C]
        image_outs = self.encode_image(image, as_dict=True, return_feat=True, return_attn=True)
        image_x = image_outs['image_x'] # [B, C]
        
        matchingloss = self.loss(image_x, text_x) 
        losses_dict['matching'] = matchingloss
        
        
        ############################################################
        ### Encode question/answer and calculate masked entity modeling loss (if necessary) ###
        entityloss = .0
        if self.use_entityloss:
            visual_feat = image_outs['image_feat_before_proj'] # unprojected group token features [b, k, d_v]
            ### Encode questions ###
            question_feat = self.encode_text(question, as_dict=True)['text_feat_before_proj']  ## unprojected word tokens, [B, L, d_t]
            current_question_masks = question['attention_mask'] if isinstance(question, dict) else None
            ### Encode answer ###
            answer_feat = self.encode_text(answer, as_dict=True)['text_x']  # projected answer embedding, #[B, d]
            ### project the group feature/question feature to the common multimodal space ###
            visual_feat = self.align_proj_img(visual_feat)
            question_feat = self.align_proj_text(question_feat)
            ### calculate entity loss ### 
            question_out = self.multimodal_groupingblock(visual_feat, question_feat, entity_masks=entity_masks, question_masks=current_question_masks) #[b, d_t]
            question_out = self.bridge_projector(question_out) #[b, d]
            entityloss = self.loss(question_out, answer_feat)            
            
            losses_dict['entity'] = entityloss
        ############################################################
        ### Encode cross-image and calculate mask loss ###
        maskloss = .0
        if self.use_maskloss:                
            assert cross_image is not None and cross_entity is not None
            
            image_outs3 = self.encode_image(cross_image, as_dict=True, return_feat=True, return_attn=True, momentum=True)
            # total_stages = len(image_outs3['attn_dicts'])
            attn_q = image_outs['attn_dicts'][0]['q'].squeeze(1)
            attn_k = image_outs['attn_dicts'][0]['k'].squeeze(1)
            attn_q_cross = image_outs3['attn_dicts'][0]['q'].squeeze(1)
            attn_k_cross = image_outs3['attn_dicts'][0]['k'].squeeze(1)

            attn_map3 = self.project_and_mask(attn_q_cross, attn_k_cross)
            attn_map_cross1 = self.project_and_mask(attn_q, attn_k_cross)   # the mask to match image
            
            
            def compute_cross_loss(mask1, mask2, cross_entity, groups, indicator='none'):
                mask1 = rearrange(mask1, 'b k (h w) -> b k h w', h = 14, w = 14) # hard coded this for now, [b, h, w] 
                mask2 = rearrange(mask2, 'b k (h w) -> b k h w', h = 14, w = 14) # hard coded this for now, [b, h, w] 
                mask1 = F.interpolate(mask1, size=(224, 224), mode='bilinear', align_corners=True)
                mask2 = F.interpolate(mask2, size=(224, 224), mode='bilinear', align_corners=True)
                
                ###### get the representation of the sampled_noun and measure the similarity ###############
                if cross_entity is not None:
                    with torch.no_grad():
                        noun_feat = self.encode_text(cross_entity, as_dict=True)['text_x']  # [bs, d_c]
                        group_logits = (groups @ noun_feat.unsqueeze(-1)).squeeze(-1) #[bs, k]
                        num_groups = group_logits.size(1)
                        topk_logits = torch.topk(group_logits, k=int(num_groups*self.group_ratio), largest=False)[1]
                        
                    mask1[torch.arange(bs).unsqueeze(1), topk_logits] = mask1[torch.arange(bs).unsqueeze(1), topk_logits].detach()
                ############################################################################################
                return self.mask_loss(mask1, mask2.detach(), self.cross_threshold, indicator=indicator)
            
            maskloss_cross = compute_cross_loss(attn_map_cross1, attn_map3, cross_entity, image_outs['image_feat'], indicator='none')
        
            if self.dual_dice:
                dual_image_outs = self.encode_image(image, as_dict=True, return_feat=True, return_attn=True, momentum=True)
                dual_image_outs3 = self.encode_image(cross_image, as_dict=True, return_feat=True, return_attn=True)

                dual_attn_q = dual_image_outs['attn_dicts'][0]['q'].squeeze(1)
                dual_attn_k = dual_image_outs['attn_dicts'][0]['k'].squeeze(1)
                dual_attn_q_cross = dual_image_outs3['attn_dicts'][0]['q'].squeeze(1)
                dual_attn_k_cross = dual_image_outs3['attn_dicts'][0]['k'].squeeze(1)
                
                dual_attn_map = self.project_and_mask(dual_attn_q, dual_attn_k)
                dual_attn_map_cross = self.project_and_mask(dual_attn_q_cross, dual_attn_k)
                
                dual_maskloss = compute_cross_loss(dual_attn_map_cross, dual_attn_map, cross_entity, dual_image_outs3['image_feat'], indicator='cross')
                maskloss_cross = (maskloss_cross + dual_maskloss) * 0.5
                
            maskloss = maskloss_cross
            losses_dict['mask'] = maskloss
            
        ############################################################
        ### total loss ###
        if self.use_entityloss and self.use_maskloss: ### for 2nd stage ###
            losses = matchingloss + self.entity_weight * entityloss + self.maskloss_weight * maskloss
        elif self.use_entityloss: ### for 1st stage ###
            losses = matchingloss + self.entity_weight * entityloss
        else: ### baseline ###
            losses = matchingloss
        
        if self.with_multi_label:
            image_multi_label_x = image_x.unsqueeze(1)
            text_multi_label_x = text_outs['text_multi_label_x']
            loss_multi_label = self.multi_label_loss(image_multi_label_x, text_multi_label_x) * self.multi_label_loss_weight
            losses_dict['multi_label'] = loss_multi_label
            losses += loss_multi_label
            
        losses_dict['loss'] = losses
        return losses_dict

    def forward_test(self, image, text):
        return self.zero_shot_pred(image, text)

    def forward(self, image, text, cross_image=None, cross_entity=None, \
                 question=None, answer=None, entity_masks=None, question_masks=None):
        """
        
        Args:
            image: [b, 3, 224, 224] raw input image
            text: [b, L] caption embedding after tokenisation with length L
            cross_image: [b, 3, 224, 224] the image that shares the same entity with the input image
            cross_entity: [b, L] text embedding of the shared entity after tokenisation 
            question: [b, L] question embedding after tokenisation
            answer: [b, L]  prompted answer embedding after tokenisation
            entity_masks: [b, L] 
            question_masks: [b, L]
            
        """
        if self.training:
            return self.forward_train(image=image, text=text, cross_image=cross_image, cross_entity=cross_entity, \
                                    question=question, answer=answer, entity_masks=entity_masks, question_masks=question_masks)
        else:
            return self.forward_test(image, text)

    @torch.no_grad()
    def build_text_embedding(self, text, tokenizer=None, num_classes=20):
        """

        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH]
            
            distilbert:
                text (list) [classes * numtemplates] for distilbert, num_classes: 20 for voc by default, 1000 for IN1K
                num_classes 暂时没用
        Returns:

        """
        if self.text_encoder_type in ['DistilBert','Bert', 'BertMedium', 'Roberta']:
            assert tokenizer is not None
            text_data = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            text_data = {key: val.cuda() for key, val in text_data.items()}
            text_tokens = self.encode_text(text_data, as_dict=True, forward_template=True)['text_x']
        else:
            text = text.to(next(self.parameters()).device)
            num_classes, num_templates = text.shape[:2]
            text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
            text_tokens = self.encode_text(text, as_dict=True, forward_template=True)['text_x']
        # [N, T, C]
        # text_tokens = rearrange(text_tokens, '(n t) c -> n t c', n=num_classes, t=num_templates)
        text_tokens = rearrange(text_tokens, '(n t) c -> n t c', n=num_classes)
        # [N, C]
        text_tokens = text_tokens.mean(dim=1)
        text_tokens = F.normalize(text_tokens, dim=-1)

        return text_tokens


    @torch.no_grad()
    def build_text_embedding_without_projection(self, text):
        """

        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH]

        Returns:

        """
        text = text.to(next(self.parameters()).device)
        num_classes, num_templates = text.shape[:2]
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        text_tokens = self.encode_text(text, as_dict=True, forward_template=True)['text_x_before_proj']
        
        # [N, T, C]
        text_tokens = rearrange(text_tokens, '(n t) c -> n t c', n=num_classes, t=num_templates)
        # [N, C]
        text_tokens = text_tokens.mean(dim=1)
        text_tokens = F.normalize(text_tokens, dim=-1)

        return text_tokens

    @torch.no_grad()
    def zero_shot_pred(self, image, text):
        # [B, C]
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        # cosine similarity as logits
        logits_per_image = image_features @ text.t()
        return logits_per_image
        

