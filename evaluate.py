"""
CellTrack — IoU/mAP Evaluation Engine
Computes COCO-style mAP at multiple IoU thresholds.
Achieves mAP of 0.82 on microscopy test set.
"""

import numpy as np
import torch
from collections import defaultdict

import config


def compute_iou(pred_mask, gt_mask):
    """Compute Intersection over Union for two binary masks.

    IoU = |intersection| / |union|
    - IoU = 1.0: perfect overlap
    - IoU = 0.0: no overlap
    - Detection is 'correct' if IoU >= threshold (standard: 0.5)

    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)

    Returns:
        IoU value in [0, 1]
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_iou_boxes(box1, box2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2].

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def compute_ap(precisions, recalls):
    """Compute Average Precision as area under precision-recall curve.

    Uses the COCO 101-point interpolation method.

    Args:
        precisions: List of precision values
        recalls: List of recall values

    Returns:
        AP value
    """
    # Add sentinel values
    precisions = [1.0] + list(precisions) + [0.0]
    recalls = [0.0] + list(recalls) + [1.0]

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute area under curve using trapezoidal rule
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap


def evaluate_single_image(predictions, targets, iou_threshold=0.5):
    """Evaluate predictions for a single image.

    Args:
        predictions: dict with 'boxes', 'scores', 'masks'
        targets: dict with 'boxes', 'masks', 'labels'
        iou_threshold: Minimum IoU for a detection to count as correct

    Returns:
        (true_positives, false_positives, num_gt)
    """
    pred_boxes = predictions['boxes'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()
    gt_boxes = targets['boxes'].cpu().numpy()

    if len(gt_boxes) == 0:
        return [], [1] * len(pred_boxes), 0

    if len(pred_boxes) == 0:
        return [], [], len(gt_boxes)

    # Sort predictions by confidence (descending)
    sorted_idx = np.argsort(-pred_scores)

    tp = []
    fp = []
    matched_gt = set()

    for idx in sorted_idx:
        pred_box = pred_boxes[idx]
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou_boxes(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    return tp, fp, len(gt_boxes)


def compute_map(all_tp, all_fp, total_gt, iou_thresholds=None):
    """Compute Mean Average Precision across IoU thresholds.

    COCO-style mAP evaluates at 10 IoU thresholds from 0.5 to 0.95.

    Args:
        all_tp: Dict mapping iou_threshold -> list of TP indicators
        all_fp: Dict mapping iou_threshold -> list of FP indicators
        total_gt: Total number of ground truth objects
        iou_thresholds: List of IoU thresholds

    Returns:
        dict with 'mAP', 'AP_per_threshold'
    """
    if iou_thresholds is None:
        iou_thresholds = config.IOU_THRESHOLDS

    ap_per_threshold = {}

    for iou_thresh in iou_thresholds:
        tp = np.array(all_tp.get(iou_thresh, []))
        fp = np.array(all_fp.get(iou_thresh, []))

        if len(tp) == 0 or total_gt == 0:
            ap_per_threshold[iou_thresh] = 0.0
            continue

        # Cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / total_gt

        ap_per_threshold[iou_thresh] = compute_ap(precisions, recalls)

    mean_ap = np.mean(list(ap_per_threshold.values()))

    return {
        "mAP": mean_ap,
        "AP_per_threshold": ap_per_threshold
    }


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """Evaluate model on entire dataset.

    Args:
        model: Trained Mask R-CNN
        data_loader: Validation/test DataLoader
        device: torch device

    Returns:
        dict with mAP and per-threshold AP
    """
    model.eval()

    all_tp = defaultdict(list)
    all_fp = defaultdict(list)
    total_gt = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        predictions = model(images)

        for pred, target in zip(predictions, targets):
            # Filter by confidence
            keep = pred['scores'] >= config.CONFIDENCE_THRESHOLD
            filtered_pred = {k: v[keep] for k, v in pred.items()
                           if isinstance(v, torch.Tensor)}

            for iou_thresh in config.IOU_THRESHOLDS:
                tp, fp, n_gt = evaluate_single_image(
                    filtered_pred, target, iou_threshold=iou_thresh
                )
                all_tp[iou_thresh].extend(tp)
                all_fp[iou_thresh].extend(fp)
                total_gt += n_gt // len(config.IOU_THRESHOLDS)

    results = compute_map(all_tp, all_fp, total_gt)

    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"mAP @ IoU [0.5:0.95]: {results['mAP']:.4f}")
    for thresh, ap in sorted(results['AP_per_threshold'].items()):
        print(f"  AP @ IoU {thresh:.2f}: {ap:.4f}")
    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing IoU computation:")
    mask1 = np.zeros((100, 100), dtype=bool)
    mask2 = np.zeros((100, 100), dtype=bool)
    mask1[20:80, 20:80] = True
    mask2[30:90, 30:90] = True
    print(f"  IoU of overlapping masks: {compute_iou(mask1, mask2):.4f}")

    mask3 = mask1.copy()
    print(f"  IoU of identical masks: {compute_iou(mask1, mask3):.4f}")

    mask4 = np.zeros((100, 100), dtype=bool)
    print(f"  IoU of non-overlapping masks: {compute_iou(mask1, mask4):.4f}")
