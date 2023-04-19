#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner


class ComputeLoss:
    '''Loss computation func.'''
    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=4,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                     'class': 0.0,
                     'iou': 0.0,
                     'dfl': 0.0,
                     'kps': 1.0}
                 ):

        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size

        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.kps_loss = KpsLoss()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight


    def __call__(
        self,
        outputs,
        targets,
        epoch_num,
        step_num,
        images,
        debug=False
    ):
        # pred_scores / pred_distri / kps_dist --> bs * num_total_anchors * 1/4/10
        feats, pred_scores, pred_distri, kps_dist = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)

        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.full((1,4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # targets --> bs x n_max_boxes x 20
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:5] #xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        gt_kps = targets[:, :, 5:]  # [x y z] * 5
        mask_gt_kps = (gt_kps[:, :, 2::3].sum(-1, keepdim=True) > -1 * 5).float()  # kpss w is not all(-1)
        mask_gt = mask_gt * mask_gt_kps  # for new landmark modules gt_kpss must be valid

        # pboxes
        # anchor_points --> num_total_anchors * 2
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri) #xyxy

        pred_kps = kps_dist + anchor_points_s.unsqueeze(0).repeat([batch_size, 1, 5])

        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_kps, target_scores, fg_mask = \
                    self.warmup_assigner(
                        anchors,
                        n_anchors_list,
                        gt_labels,
                        gt_bboxes,
                        gt_kps,
                        mask_gt,
                        pred_bboxes.detach() * stride_tensor)
            else:
                target_labels, target_bboxes, target_kps, target_scores, fg_mask = \
                    self.formal_assigner(
                        pred_scores.detach(),
                        pred_bboxes.detach() * stride_tensor,
                        anchor_points,
                        gt_labels,
                        gt_bboxes,
                        gt_kps,
                        mask_gt)

        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            if epoch_num < self.warmup_epoch:
                _anchors = anchors.cpu().float()
                _n_anchors_list = n_anchors_list
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _gt_kps = gt_kps.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_kps, target_scores, fg_mask = \
                    self.warmup_assigner(
                        _anchors,
                        _n_anchors_list,
                        _gt_labels,
                        _gt_bboxes,
                        _gt_kps,
                        _mask_gt,
                        _pred_bboxes * _stride_tensor)

            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _gt_kps = gt_kps.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_kps, target_scores, fg_mask = \
                    self.formal_assigner(
                        _pred_scores,
                        _pred_bboxes * _stride_tensor,
                        _anchor_points,
                        _gt_labels,
                        _gt_bboxes,
                        _gt_kps,
                        _mask_gt)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()
        #Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss
        # target_labels --> bs * num_total_anchors
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson 
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum

        # kpss loss
        # pos_decode_kps_pred or pos_decode_kps_targets --> num_total_anchors * 10
        # pos_kps_weights --> num_total_anchors * 1
        # kps_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 15])
        # target_kps = torch.where(kps_mask > 0, target_kps, torch.ones_like(target_kps) * -1.0)
        # kps_mask_wt = (target_kps[:, :, 2::3].sum(-1, keepdim=True) > -1 * 5).float()
        # pos_kps_weights = torch.where(kps_mask_wt > 0, torch.ones_like(kps_mask_wt), torch.zeros_like(kps_mask_wt))
        # pos_kps_weights = torch.reshape(pos_kps_weights, (-1, 1))
        # pos_kps_weights = target_scores.max(axis=-1)[0].reshape((-1, 1)) * pos_kps_weights
        # pos_kps_targets = torch.reshape(target_kps, (-1, 5, 3))
        # pos_kps_targets = pos_kps_targets[..., :2]
        # anchor_points_rpt = anchor_points.unsqueeze(0).repeat([fg_mask.size(0), 1, 5])
        # anchor_points_rpt = anchor_points_rpt.reshape((-1, 10))
        # pos_kps_targets = pos_kps_targets.reshape((-1, 10))
        # stride_tensor_rpt = stride_tensor.unsqueeze(0).repeat([fg_mask.size(0), 1, 10]).reshape((-1, 10))
        # pos_decode_kps_targets = pos_kps_targets / stride_tensor_rpt
        # pos_decode_kps_pred = pred_kps.reshape((-1, 10))
        loss_kps = self.kps_loss(pred_kps, target_kps, stride_tensor, target_scores, target_scores_sum, fg_mask)

        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)

        # debug
        if debug:
            self._plot_and_save_targets(images, pred_kps, target_kps, target_bboxes,
                                        anchor_points_s, anchors, stride_tensor, fg_mask, epoch_num)


        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['kps'] * loss_kps

        return loss, \
            torch.cat(((self.loss_weight['kps'] * loss_kps).unsqueeze(0),
                         (self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                         (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                         (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()

    @staticmethod
    def _plot_and_save_targets(
            images, pred_kps, target_kps, target_bboxes, anchor_points_s, anchors, stride_tensor, fg_mask, epoch_num,
            root_dir='./debug'):
        """
        plot infos

        Args:
              images: bs x c x h x w.
              pred_kps: bs x na x 10.
              target_kps: bs x na x 15.
              target_bboxes: bs x na x 4.
              anchor_points_s: na x 2.
              anchors: na x 4.
              stride_tensor: na x 1.
              fg_mask: bs x na.
        """
        bs = images.size(0)
        if isinstance(images, torch.Tensor):
            images = images.permute([0, 2, 3, 1]).cpu().float().numpy()
        if isinstance(pred_kps, torch.Tensor):
            pred_kps = pred_kps.detach().cpu().numpy()
        if isinstance(target_kps, torch.Tensor):
            target_kps = target_kps.cpu().numpy()
        if isinstance(target_bboxes, torch.Tensor):
            target_bboxes = target_bboxes.cpu().numpy()
        if isinstance(anchor_points_s, torch.Tensor):
            anchor_points_s = anchor_points_s.unsqueeze(0).repeat([bs, 1, 1]).cpu().numpy()
        if isinstance(anchors, torch.Tensor):
            anchors = anchors.unsqueeze(0).repeat([bs, 1, 1]).cpu().numpy()
        if isinstance(stride_tensor, torch.Tensor):
            stride_tensor = stride_tensor.unsqueeze(0).repeat([bs, 1, 1]).cpu().numpy()
        if isinstance(fg_mask, torch.Tensor):
            fg_mask = fg_mask.cpu().numpy()
        if np.max(images[0]) <= 1:
            images *= 255

        anchor_points = anchor_points_s * stride_tensor
        target_bboxes *= stride_tensor
        pred_kps *= stride_tensor

        for i, image in enumerate(images):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            anchor_points_si = anchor_points[i]
            stride_tensor_si = stride_tensor[i]
            anchors_si = anchors[i]
            fg_mask_si = fg_mask[i]
            target_bboxes_si = target_bboxes[i]
            target_kps_si = target_kps[i]
            pred_kps_si = pred_kps[i]

            for s_pick in [8, 16, 32]:
                image_draw = image.copy()
                img_name = f'yolov6_train_e{epoch_num}_{i}_{s_pick}.bmp'
                for kps_p, kps, (tx0, ty0, tx1, ty1), (x0, y0, x1, y1), (x, y), s, m in zip(
                        pred_kps_si.reshape(-1, 10),
                        target_kps_si.reshape(-1, 15),
                        target_bboxes_si.reshape(-1, 4),
                        anchors_si.reshape(-1, 4),
                        anchor_points_si.reshape(-1, 2),
                        stride_tensor_si.reshape(-1),
                        fg_mask_si.reshape(-1)):
                    if m:
                        cv2.rectangle(image_draw, (int(tx0), int(ty0)), (int(tx1), int(ty1)), (255, 0, 0), thickness=1)
                        color = [(255, 0, 0), (0, 0, 255), (255, 255, 255), (255, 0, 128), (128, 0, 255)]
                        for k, (kpx, kpy, _) in enumerate(kps.reshape(-1, 3)):
                            color_pick = color[k]
                            cv2.circle(image_draw, (int(kpx), int(kpy)), 1, color_pick, thickness=1)
                        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                        for kpx, kpy in kps_p.reshape(-1, 2):
                            cv2.circle(image_draw, (int(kpx), int(kpy)), 1, color, thickness=1)
                    if s == s_pick:
                        if m:
                            cv2.rectangle(image_draw, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), thickness=1)
                            cv2.circle(image_draw, (int(x), int(y)), 1, (0, 255, 0), thickness=1)
                        # else:
                        #     cv2.rectangle(image_draw, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), thickness=1)
                        # cv2.circle(image_draw, (int(x), int(y)), 1, (0, 0, 255), thickness=1)

                cv2.imwrite(os.path.join(root_dir, img_name), image_draw)

        return target_kps


    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5 + 3 * 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:5] = xywh2xyxy(batch_target)
        scale_tensor_kps = torch.full((1, 5), scale_tensor[0][0]).type_as(scale_tensor)
        targets[..., 5::3] = targets[..., 5::3].mul_(scale_tensor_kps)
        targets[..., 6::3] = targets[..., 6::3].mul_(scale_tensor_kps)
        return targets  # bs x nmt x 20


    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)


def smooth_l1_loss(pred, target, weight, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        weight (torch.Tensor): loss weights.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    loss *= weight.repeat([1, loss.size(-1)])
    return loss.mean()


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
        """
        loss = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta)
        return loss


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score,gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


class KpsLoss(nn.Module):
    def __init__(self):
        super(KpsLoss, self).__init__()
        self.smooth_l1_loss = SmoothL1Loss().cuda()

    def forward(self, pred_kps, target_kps, stride_tensor, target_scores, target_scores_sum, fg_mask):
        """
        kps loss computation

        Args:
              pred_kps: bs x na x 10.
              target_kps: bs x na x 15.
              stride_tensor: na x 1.
              target_scores: bs x na x 1.
              fg_mask: bs x na.
        """

        stride_tensor_rp = stride_tensor.unsqueeze(0).repeat([fg_mask.size(0), 1, 10])

        target_kps = target_kps.reshape((-1, 3))
        target_kps = target_kps[:, :2]
        target_kps = target_kps.reshape((fg_mask.size(0), fg_mask.size(1), -1))
        target_kps /= stride_tensor_rp

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            kps_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 10])
            pred_kps_pos = torch.masked_select(pred_kps, kps_mask).reshape([-1, 10])
            target_kps_pos = torch.masked_select(target_kps, kps_mask).reshape([-1, 10])
            kps_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_sll = self.smooth_l1_loss(pred_kps_pos, target_kps_pos, weight=kps_weight.reshape((-1, 1)))
            if target_scores_sum == 0:
                loss_kps = loss_sll.sum()
            else:
                loss_kps = loss_sll.sum() / target_scores_sum
        else:
            loss_kps = pred_kps.sum() * 0.

        return loss_kps


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
               
            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                        target_ltrb_pos) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.

        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
