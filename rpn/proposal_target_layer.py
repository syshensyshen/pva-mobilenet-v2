# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import cv2

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # labels
        top[1].reshape(1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4)
        if cfg.TRAIN.MASK_RCNN:
            # mask_targets
            top[5].reshape(1,self._num_classes-1,cfg.TRAIN.MASK_RCNN_SIZE,cfg.MASK_SIZE)


    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data
        if cfg.TRAIN.MASK_RCNN:
            #gt_masks;
            gt_masks = bottom[2].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        # print all_rois.shape,gt_boxes.shape
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        
        labels, rois, bbox_targets, bbox_inside_weights,mask_targets = _sample_rois(
            all_rois, gt_boxes, gt_masks,gt_keypoints,fg_rois_per_image,
            rois_per_image, self._num_classes)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)
        if cfg.TRAIN.MASK_RCNN:
            #mask_targets
            top[5].reshape(*mask_targets.shape)
            top[5].data[...] = mask_targets

        #print kps_labels.shape

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        start = int(start)
        end = int(end)
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes,gt_masks,fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    #print "print all_rois.shape(0) is ", all_rois.shape[0]
    #print all_rois.shape(0)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    gt_boxes_ = gt_boxes[gt_assignment, :4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    fg_rois_per_this_image = int(fg_rois_per_this_image)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    #print labels.shape
    labels = labels[keep_inds]
    #print labels.shape
    # Clamp labels for the background RoIs to 0
   
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    #mask_targets = all_mask_targets[keep_inds,:,:,:]
    bboxes = gt_boxes_[keep_inds, :4]
    if cfg.TRAIN.MASK_RCNN:
        keep_assingment_inds = gt_assignment[keep_inds]
        M = cfg.TRAIN.MASK_RCNN_SIZE
        mask_targets = np.zeros((len(labels),num_classes-1,M,M), dtype=np.float32)
        mask_targets[:,:,:,:] = -1   

        for i in np.arange(len(labels)):
            if labels[i] <= 0:
                x1 = rois[i, 1]
                y1 = rois[i, 2]
                x2 = rois[i, 3]
                y2 = rois[i, 4]
                j = int(labels[i])
               # print x1,y1,x2,y2
                #gt_mask = mask_targets[i,int(original_labels[i]),:,:]
                gt_mask = gt_masks[keep_assingment_inds[i],:,:]
                bbox = bboxes[i, :4]
                # gt_mask = cv2.resize(gt_mask, (image.shape[2], image.shape[1]),interpolation=cv2.INTER_CUBIC)
                # all_mask_targets[i, j, :, :] = -1
                # mask = np.zeros((np.round(y2 - y1).astype(np.int), np.round(x2 - x1).astype(np.int)), dtype=np.float32)
                # mask[:, :] = 0
                # width = mask.shape[1]
                # height = mask.shape[0]
                
                mask_roi =gt_mask[np.round(y1).astype(np.int):np.round(y2).astype(np.int),np.round(x1).astype(np.int):np.round(x2).astype(np.int)]
                # mask = np.zeros((np.round(y2 - y1).astype(np.int), np.round(x2 - x1).astype(np.int)), dtype=np.float32)
                # mask[:, :] = 0
                # width = mask.shape[1]
                # height = mask.shape[0]
                if (int(bbox[2]) - int(bbox[0])) <= 0 or (int(bbox[3]) - int(bbox[1])) <= 0 or bbox[2] < 0 or bbox[
                    0] < 0 or bbox[1] < 0 or bbox[3] < 0:
                    # resized_mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_CUBIC)
                    mask_targets[i, j-1, :, :] = -1
                    continue
                
                if (np.round(y2 - y1).astype(np.int) <= 0) or (np.round(x2 - x1).astype(np.int)) <= 0:
                    mask_targets[i, j-1, :, :] = -1
                    continue
                
                if mask_roi.shape[0] <=0 or mask_roi.shape[1] <=0:
                    mask_targets[i, j-1, :, :] = -1
                    continue
               
                resized_mask = cv2.resize(mask_roi, (M,M), interpolation=cv2.INTER_CUBIC)
                resized_mask = np.round(resized_mask)
                
                mask_targets[i, j-1, :, :] = resized_mask   
                
            else:
                mask_targets[i, j-1, :, :] = -1
                kps_labels[i,:] = -1

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)
    if not cfg.TRAIN.MASK_RCNN:
        mask_targets = 0
    return labels, rois, bbox_targets, bbox_inside_weights, mask_targets
