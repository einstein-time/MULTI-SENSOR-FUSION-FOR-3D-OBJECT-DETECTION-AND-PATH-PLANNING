"""
Hungarian Algorithm for Data Association

Implements the Hungarian algorithm (Kuhn-Munkres algorithm) for solving
the assignment problem in multi-object tracking. This finds the optimal
matching between predicted tracks and new detections.
"""

from typing import Tuple, List
import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_algorithm(
    cost_matrix: np.ndarray,
    max_cost: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve assignment problem using Hungarian algorithm.

    Finds optimal matching between tracks and detections that minimizes
    total cost. Typically used with IoU or distance as cost metric.

    Args:
        cost_matrix: (N, M) cost matrix where N is number of tracks and
                    M is number of detections. Lower cost = better match.
        max_cost: Maximum cost threshold for valid matches

    Returns:
        matched_indices: (K, 2) array of (track_idx, detection_idx) pairs
        unmatched_tracks: Indices of tracks without matches
        unmatched_detections: Indices of detections without matches
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )

    # Solve assignment using scipy
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Filter out matches with high cost
    matches = []
    matched_tracks = set()
    matched_detections = set()

    for track_idx, det_idx in zip(row_indices, col_indices):
        if cost_matrix[track_idx, det_idx] < max_cost:
            matches.append([track_idx, det_idx])
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)

    # Find unmatched tracks and detections
    all_tracks = set(range(cost_matrix.shape[0]))
    all_detections = set(range(cost_matrix.shape[1]))

    unmatched_tracks = np.array(list(all_tracks - matched_tracks))
    unmatched_detections = np.array(list(all_detections - matched_detections))

    if matches:
        matched_indices = np.array(matches)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    return matched_indices, unmatched_tracks, unmatched_detections


def associate_detections_to_tracks(
    detections: np.ndarray,
    tracks: np.ndarray,
    iou_threshold: float = 0.3,
    use_iou: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate detections to existing tracks.

    Args:
        detections: (M, 7) array of detections [x, y, z, w, l, h, yaw]
        tracks: (N, 7) array of track predictions
        iou_threshold: IoU threshold for matching
        use_iou: If True, use IoU; otherwise use Euclidean distance

    Returns:
        matched_indices: (K, 2) array of (track_idx, detection_idx) pairs
        unmatched_tracks: Indices of tracks without matches
        unmatched_detections: Indices of detections without matches
    """
    if len(tracks) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty(0, dtype=int),
            np.arange(len(detections)),
        )

    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(tracks)),
            np.empty(0, dtype=int),
        )

    if use_iou:
        # Compute IoU-based cost (1 - IoU)
        iou_matrix = compute_iou_matrix(tracks, detections)
        cost_matrix = 1 - iou_matrix
        max_cost = 1 - iou_threshold
    else:
        # Compute distance-based cost
        cost_matrix = compute_distance_matrix(tracks, detections)
        max_cost = 5.0  # Maximum distance threshold

    return hungarian_algorithm(cost_matrix, max_cost)


def compute_iou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.

    Uses bird's eye view IoU for efficiency.

    Args:
        boxes1: (N, 7) boxes
        boxes2: (M, 7) boxes

    Returns:
        IoU matrix (N, M)
    """
    N = len(boxes1)
    M = len(boxes2)
    iou_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = compute_iou_bev(boxes1[i], boxes2[j])

    return iou_matrix


def compute_iou_bev(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute bird's eye view IoU between two boxes.

    Args:
        box1: Box [x, y, z, w, l, h, yaw]
        box2: Box [x, y, z, w, l, h, yaw]

    Returns:
        IoU value
    """
    # Extract parameters
    x1, y1, w1, l1 = box1[0], box1[1], box1[3], box1[4]
    x2, y2, w2, l2 = box2[0], box2[1], box2[3], box2[4]

    # Compute rectangular bounds (ignoring rotation for simplicity)
    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - l1 / 2, y1 + l1 / 2
    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - l2 / 2, y2 + l2 / 2

    # Intersection
    x_inter_min = max(x1_min, x2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_min = max(y1_min, y2_min)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max > x_inter_min and y_inter_max > y_inter_min:
        intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    else:
        intersection = 0

    # Union
    area1 = w1 * l1
    area2 = w2 * l2
    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def compute_distance_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distance matrix between box centers.

    Args:
        boxes1: (N, 7) boxes
        boxes2: (M, 7) boxes

    Returns:
        Distance matrix (N, M)
    """
    centers1 = boxes1[:, :3]  # (N, 3)
    centers2 = boxes2[:, :3]  # (M, 3)

    # Compute pairwise distances
    distances = np.linalg.norm(
        centers1[:, np.newaxis, :] - centers2[np.newaxis, :, :],
        axis=2
    )

    return distances


def greedy_assignment(
    cost_matrix: np.ndarray,
    max_cost: float = 0.7,
) -> Tuple[List, List, List]:
    """
    Greedy assignment algorithm (faster but suboptimal).

    Useful as a faster alternative to Hungarian algorithm for large
    numbers of tracks/detections.

    Args:
        cost_matrix: (N, M) cost matrix
        max_cost: Maximum cost threshold

    Returns:
        matches: List of (track_idx, detection_idx) tuples
        unmatched_tracks: List of unmatched track indices
        unmatched_detections: List of unmatched detection indices
    """
    N, M = cost_matrix.shape

    if N == 0:
        return [], [], list(range(M))
    if M == 0:
        return [], list(range(N)), []

    matches = []
    matched_tracks = set()
    matched_detections = set()

    # Flatten cost matrix and get sorted indices
    flat_costs = cost_matrix.flatten()
    sorted_indices = np.argsort(flat_costs)

    # Greedily match lowest costs
    for idx in sorted_indices:
        track_idx = idx // M
        det_idx = idx % M
        cost = cost_matrix[track_idx, det_idx]

        if cost >= max_cost:
            break

        if track_idx not in matched_tracks and det_idx not in matched_detections:
            matches.append((track_idx, det_idx))
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)

    unmatched_tracks = [i for i in range(N) if i not in matched_tracks]
    unmatched_detections = [j for j in range(M) if j not in matched_detections]

    return matches, unmatched_tracks, unmatched_detections
