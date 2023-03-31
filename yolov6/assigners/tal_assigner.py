import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.assigners.assigner_utils import select_candidates_in_gts, select_highest_overlaps, iou_calculator, dist_calculator

class TaskAlignedAssigner(nn.Module):
    def __init__(self,
                 topk=13,
                 num_classes=80,
                 alpha=1.0,
                 beta=6.0, 
                 eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes,
                gt_ldmks,
                mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            gt_ldmks (Tensor): shape(bs, n_max_boxes, 10)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_ldmks (Tensor): shape(bs, num_total_anchors, 10)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), \
                   torch.zeros_like(pd_bboxes).to(device), \
                   torch.zeros(pd_bboxes.shape[0], pd_bboxes.shape[1], 10).to(device), \
                   torch.zeros_like(pd_scores).to(device), \
                   torch.zeros_like(pd_scores[..., 0]).bool().to(device)

        cycle, step, self.bs = (1, self.bs, self.bs) if self.n_max_boxes <= 100 else (self.bs, 1, 1)
        target_labels_lst, target_bboxes_lst, target_ldmks_lst, target_scores_lst, fg_mask_lst = [], [], [], [], []
        # loop batch dim in case of numerous object box
        for i in range(cycle):
            start, end = i*step, (i+1)*step
            pd_scores_ = pd_scores[start:end, ...]
            pd_bboxes_ = pd_bboxes[start:end, ...]
            gt_labels_ = gt_labels[start:end, ...]
            gt_bboxes_ = gt_bboxes[start:end, ...]
            gt_ldmks_  = gt_ldmks[start:end, ...]
            mask_gt_   = mask_gt[start:end, ...]

            # (bs, n_max_boxes, num_total_anchors)
            mask_pos, align_metric, overlaps = self.get_pos_mask(
                pd_scores_, pd_bboxes_, gt_labels_, gt_bboxes_, anc_points, mask_gt_)

            # target_gt_idx, fg_mask: (bs, num_total_anchors)
            # mask_pos: (bs, n_max_boxes, num_total_anchors)
            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
                mask_pos, overlaps, self.n_max_boxes)

            # assigned target (bs, num_total_anchors, num_classes)
            target_labels, target_bboxes, target_ldmks, target_scores = self.get_targets(
                gt_labels_, gt_bboxes_, gt_ldmks_, target_gt_idx, fg_mask)

            # normalize
            align_metric *= mask_pos  # (bs, n_max_boxes, num_total_anchors)
            pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]  # (bs, n_max_boxes, 1)
            pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]  # (bs, n_max_boxes, 1)
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)  # (bs, num_total_anchors, 1)
            target_scores = target_scores * norm_align_metric  # (bs, num_total_anchors, num_classes)

            # append
            target_labels_lst.append(target_labels)
            target_bboxes_lst.append(target_bboxes)
            target_ldmks_lst.append(target_ldmks)
            target_scores_lst.append(target_scores)
            fg_mask_lst.append(fg_mask)

        # concat
        target_labels = torch.cat(target_labels_lst, 0)
        target_bboxes = torch.cat(target_bboxes_lst, 0)
        target_ldmks = torch.cat(target_ldmks_lst, 0)
        target_scores = torch.cat(target_scores_lst, 0)
        fg_mask = torch.cat(fg_mask_lst, 0)

        return target_labels, target_bboxes, target_ldmks, target_scores, fg_mask.bool()

    def get_pos_mask(self,
                     pd_scores,
                     pd_bboxes,
                     gt_labels,
                     gt_bboxes,
                     anc_points,
                     mask_gt):
        """

        :param pd_scores: shape(bs, num_total_anchors, num_classes)
        :param pd_bboxes: shape(bs, num_total_anchors, 4)
        :param gt_labels: shape(bs, n_max_boxes, 1)
        :param gt_bboxes: shape(bs, n_max_boxes, 4)
        :param anc_points: shape(num_total_anchors, 2)
        :param mask_gt: shape(bs, n_max_boxes, 1)
        :return: shape(bs, n_max_boxes, num_total_anchors)
        """

        # get anchor_align metric (bs, n_max_boxes, num_total_anchors)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask (bs, n_max_boxes, num_total_anchors)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask (bs, n_max_boxes, num_total_anchors)
        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self,
                        pd_scores,
                        pd_bboxes,
                        gt_labels,
                        gt_bboxes):
        """

        :param pd_scores: shape(bs, num_total_anchors, num_classes)
        :param pd_bboxes: shape(bs, num_total_anchors, 4)
        :param gt_labels: shape(bs, n_max_boxes, 1)
        :param gt_bboxes: shape(bs, n_max_boxes, 4)
        :return:
        """

        pd_scores = pd_scores.permute(0, 2, 1)  # bs x num_classes x num_total_anchors
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # bs x n_max_boxes
        ind[1] = gt_labels.squeeze(-1)  # bs x n_max_boxes
        bbox_scores = pd_scores[ind[0], ind[1]]  # bs x n_max_boxes x num_total_anchors

        overlaps = iou_calculator(gt_bboxes, pd_bboxes)  # shape(bs, n_max_boxes, num_total_anchors)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)  # (bs, n_max_boxes, num_total_anchors)

        return align_metric, overlaps

    def select_topk_candidates(self,
                               metrics,
                               largest=True,
                               topk_mask=None):
        """

        :param metrics: shape(bs, n_max_boxes, num_total_anchors)
        :param largest:
        :param topk_mask: shape(bs, n_max_boxes, topk)
        :return:
        """

        num_anchors = metrics.shape[-1]
        # shape(bs, n_max_boxes, topk)
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, axis=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile(
                [1, 1, self.topk])
        # shape(bs, n_max_boxes, topk)
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        # (bs, n_max_boxes, topk, num_total_anchors).sum(axis=-2) --> (bs, n_max_boxes, num_total_anchors)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1,
            torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self,
                    gt_labels,
                    gt_bboxes,
                    gt_ldmks,
                    target_gt_idx,
                    fg_mask):
        """

        :param gt_labels: shape(bs, n_max_boxes, 1)
        :param gt_bboxes: shape(bs, n_max_boxes, 4)
        :param gt_ldmks: shape(bs, n_max_boxes, 10)
        :param target_gt_idx: shape(bs, num_total_anchors)
        :param fg_mask: shape(bs, num_total_anchors)
        :return:
        """

        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]  # (bs, 1)
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (bs, num_total_anchors)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (bs, num_total_anchors)

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx]  # (bs, num_total_anchors, 4)
        target_ldmks = gt_ldmks.reshape([-1, 10])[target_gt_idx]  # (bs, num_total_anchors, 10)

        # assigned target scores
        target_labels[target_labels<0] = 0  # (bs, num_total_anchors)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (bs, num_total_anchors, num_classes)
        fg_scores_mask  = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (bs, num_total_anchors, num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores,
                                        torch.full_like(target_scores, 0))

        return target_labels, target_bboxes, target_ldmks, target_scores
