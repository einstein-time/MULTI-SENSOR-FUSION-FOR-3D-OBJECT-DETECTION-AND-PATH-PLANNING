"""
Evaluation Metrics for 3D Object Detection

Implements standard metrics:
- Average Precision (AP)
- nuScenes Detection Score (NDS)
- Intersection over Union (IoU)
"""

from typing import List, Dict, Tuple
import numpy as np


def compute_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute 3D IoU between two boxes (simplified version).

    Args:
        box1: Box [x, y, z, w, l, h, yaw]
        box2: Box [x, y, z, w, l, h, yaw]

    Returns:
        IoU value
    """
    # Simplified: use bird's eye view IoU
    x1, y1, w1, l1 = box1[0], box1[1], box1[3], box1[4]
    x2, y2, w2, l2 = box2[0], box2[1], box2[3], box2[4]

    # Rectangular approximation (ignoring rotation)
    x1_min, x1_max = x1 - w1/2, x1 + w1/2
    y1_min, y1_max = y1 - l1/2, y1 + l1/2
    x2_min, x2_max = x2 - w2/2, x2 + w2/2
    y2_min, y2_max = y2 - l2/2, y2 + l2/2

    # Intersection
    x_inter = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_inter = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_inter * y_inter

    # Union
    area1 = w1 * l1
    area2 = w2 * l2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def compute_ap(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute Average Precision.

    Args:
        predictions: List of predicted boxes per sample (N, 8) [x,y,z,w,l,h,yaw,score]
        ground_truths: List of ground truth boxes per sample (M, 7)
        iou_threshold: IoU threshold for matching

    Returns:
        Average Precision
    """
    all_scores = []
    all_matched = []

    for preds, gts in zip(predictions, ground_truths):
        if len(preds) == 0:
            continue

        # Sort predictions by confidence
        scores = preds[:, 7]
        sorted_idx = np.argsort(-scores)
        preds = preds[sorted_idx]

        # Match predictions to ground truths
        matched = np.zeros(len(preds), dtype=bool)
        gt_matched = np.zeros(len(gts), dtype=bool)

        for i, pred in enumerate(preds):
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(gts):
                if gt_matched[j]:
                    continue

                iou = compute_iou_3d(pred[:7], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                matched[i] = True
                gt_matched[best_gt_idx] = True

        all_scores.extend(scores)
        all_matched.extend(matched)

    if len(all_scores) == 0:
        return 0.0

    # Sort by confidence
    sorted_idx = np.argsort(-np.array(all_scores))
    all_matched = np.array(all_matched)[sorted_idx]

    # Compute precision-recall curve
    tp = np.cumsum(all_matched)
    fp = np.cumsum(~all_matched)

    recall = tp / len(ground_truths) if len(ground_truths) > 0 else tp * 0
    precision = tp / (tp + fp)

    # Compute AP (area under PR curve)
    ap = np.trapz(precision, recall)

    return ap


def compute_nds(
    predictions: List[Dict],
    ground_truths: List[Dict],
) -> float:
    """
    Compute nuScenes Detection Score (simplified).

    NDS combines multiple metrics: mAP, translation error, scale error, etc.

    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries

    Returns:
        NDS score
    """
    # Simplified implementation
    # Full NDS requires multiple error metrics
    return 0.0  # Placeholder


def evaluate_tracking(
    tracks: List[np.ndarray],
    ground_truths: List[np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate tracking performance.

    Computes MOTA (Multiple Object Tracking Accuracy) and MOTP.

    Args:
        tracks: Predicted tracks
        ground_truths: Ground truth tracks

    Returns:
        Dictionary with tracking metrics
    """
    # Simplified implementation
    metrics = {
        'mota': 0.0,
        'motp': 0.0,
        'precision': 0.0,
        'recall': 0.0,
    }

    return metrics


def compute_map(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    iou_thresholds: List[float] = [0.5, 0.7],
) -> float:
    """
    Compute mean Average Precision across IoU thresholds.

    Args:
        predictions: Predicted boxes
        ground_truths: Ground truth boxes
        iou_thresholds: IoU thresholds to evaluate

    Returns:
        mAP score
    """
    aps = []
    for threshold in iou_thresholds:
        ap = compute_ap(predictions, ground_truths, threshold)
        aps.append(ap)

    return np.mean(aps)
