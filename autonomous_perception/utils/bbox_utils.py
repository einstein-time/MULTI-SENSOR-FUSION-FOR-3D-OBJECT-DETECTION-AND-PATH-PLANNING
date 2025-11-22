"""
Bounding Box Utilities

Functions for manipulating 3D bounding boxes.
"""

from typing import List
import numpy as np


def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes to 8 corner points.

    Args:
        boxes: Boxes (N, 7+) [x, y, z, w, l, h, yaw, ...]

    Returns:
        Corners (N, 8, 3)
    """
    if len(boxes) == 0:
        return np.zeros((0, 8, 3))

    centers = boxes[:, :3]
    dims = boxes[:, 3:6]
    yaws = boxes[:, 6]

    # Create local corners
    w, l, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]

    corners_local = np.stack([
        np.concatenate([-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]),
        np.concatenate([-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2]),
        np.concatenate([-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]),
    ], axis=0)  # (3, 8)

    # Rotate
    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)

    rotation_matrices = np.stack([
        np.stack([cos_yaw, -sin_yaw, np.zeros_like(yaws)], axis=1),
        np.stack([sin_yaw, cos_yaw, np.zeros_like(yaws)], axis=1),
        np.stack([np.zeros_like(yaws), np.zeros_like(yaws), np.ones_like(yaws)], axis=1),
    ], axis=1)  # (N, 3, 3)

    corners = np.einsum('nij,jk->nik', rotation_matrices, corners_local)  # (N, 3, 8)
    corners = corners.transpose(0, 2, 1)  # (N, 8, 3)

    # Translate
    corners += centers[:, np.newaxis, :]

    return corners


def nms_3d(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    Non-maximum suppression for 3D boxes.

    Args:
        boxes: Boxes (N, 7) [x, y, z, w, l, h, yaw]
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold

    Returns:
        Indices of kept boxes
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # Sort by score
    sorted_idx = np.argsort(-scores)

    keep = []

    while len(sorted_idx) > 0:
        # Keep highest scoring box
        idx = sorted_idx[0]
        keep.append(idx)

        if len(sorted_idx) == 1:
            break

        # Compute IoU with remaining boxes
        ious = np.array([
            compute_iou_bev(boxes[idx], boxes[i])
            for i in sorted_idx[1:]
        ])

        # Keep boxes with IoU below threshold
        mask = ious < iou_threshold
        sorted_idx = sorted_idx[1:][mask]

    return np.array(keep)


def compute_iou_bev(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute bird's eye view IoU.

    Args:
        box1: Box [x, y, z, w, l, h, yaw]
        box2: Box [x, y, z, w, l, h, yaw]

    Returns:
        IoU value
    """
    x1, y1, w1, l1 = box1[0], box1[1], box1[3], box1[4]
    x2, y2, w2, l2 = box2[0], box2[1], box2[3], box2[4]

    # Rectangular approximation
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


def filter_boxes_by_range(
    boxes: np.ndarray,
    point_cloud_range: List[float],
) -> np.ndarray:
    """
    Filter boxes within point cloud range.

    Args:
        boxes: Boxes (N, 7+)
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        Filtered boxes
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

    mask = (
        (boxes[:, 0] >= x_min) & (boxes[:, 0] < x_max) &
        (boxes[:, 1] >= y_min) & (boxes[:, 1] < y_max) &
        (boxes[:, 2] >= z_min) & (boxes[:, 2] < z_max)
    )

    return boxes[mask]
