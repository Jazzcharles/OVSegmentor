# -------------------------------------------------------------------------
# Written by Jilan Xu
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from torch import linalg as LA

from scipy.optimize import linear_sum_assignment
# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ipdb import set_trace
import torch.distributed as dist
import diffdist.functional as diff_dist
from ipdb import set_trace


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

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_type='L2'):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_type = cost_type
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        NewParams:
            outputs: [b, k, h * w], k normalized masks 
            targets: [b, k, h * w]  k normalized masks
            
        Params:s
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs.shape[:2]
        # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        if self.cost_type == 'L2':
            cost_mask = torch.cdist(outputs, targets, p=2) #[b, k, k]
        elif self.cost_type == 'cosine':
            ##### <a, b> / (||a|| * ||b||) ######
            cos_sim = outputs @ targets.transpose(-2, -1)   #[b, k, k]
            dist_a = LA.norm(outputs, dim=-1).unsqueeze(-1) #[b, k, 1]
            dist_b = LA.norm(targets, dim=-1).unsqueeze(-2) #[b, 1, k]
            eps = 1e-6
            ### negative cosine similarity as cost matrix
            cost_mask = -1 * (cos_sim / (dist_a + eps) / (dist_b + eps)) 
        else:
            return ValueError
        # set_trace()
        inds = []
        inds2 = []
        for i in range(bs):
            xx, yy = linear_sum_assignment(cost_mask[i].cpu())
            inds.append(xx)
            inds2.append(yy)
        # indices = [linear_sum_assignment(cost_mask[i]) for i in range(bs)]
        # indices = [linear_sum_assignment(c[i].cpu()) for i, c in enumerate(cost_mask.split(bs, -1))]
        # indices = [linear_sum_assignment(c[i].cpu()) for i, c in zip(range(bs), cost_mask)]
        inds = torch.tensor(inds).long().cuda()
        inds2 = torch.tensor(inds2).long().cuda()
        return inds, inds2
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
def dice_loss(inputs, targets, num_masks=None, threshold=0.0, topk_mask=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        
        1. norm the input and the target to [0, 1] with sigmoid
        2. binarize the target
        3. compute dice loss
    """
    if num_masks is None:
        num_masks = inputs.size(1)

    if topk_mask is not None:
        ### [bs, k, nm] * [bs, k, 1], filter the masked clusters
        inputs = inputs * topk_mask.unsqueeze(-1)
        targets = targets * topk_mask.unsqueeze(-1) 

    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def get_logits(dense_feat_1, selected_feat_2, logit_scale):
    # logit_scale_dense = self.logit_scale.exp()
    logit_scale_dense = torch.clamp(logit_scale.exp(), max=100)
    
    i, j, k = dense_feat_1.shape
    l, m, k = selected_feat_2.shape
    dense_feat_1 = dense_feat_1.reshape(-1, k)
    selected_feat_2 = selected_feat_2.reshape(-1, k)
    final_logits_1 = logit_scale_dense * dense_feat_1 @ selected_feat_2.t()
    final_logits_1 = final_logits_1.reshape(i, j, l, m).permute(0,2,1,3)
    return final_logits_1


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """

    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class NormSoftmaxLoss(nn.Module):

    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j
