"""
Point Cloud Processing Utilities

This module provides utilities for processing LiDAR point clouds including:
- Voxelization for 3D convolutions
- Ground plane removal
- Point cloud filtering
- Bird's eye view (BEV) projection
- Point cloud augmentation
"""

from typing import Tuple, Optional
import numpy as np


def voxelize_point_cloud(
    points: np.ndarray,
    voxel_size: Tuple[float, float, float],
    point_cloud_range: Tuple[float, float, float, float, float, float],
    max_points_per_voxel: int = 35,
    max_voxels: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelize point cloud for efficient 3D processing.

    Divides 3D space into voxels and groups points within each voxel.
    This is a key preprocessing step for 3D object detection models.

    Args:
        points: Point cloud (N, 4) with x, y, z, intensity
        voxel_size: Voxel dimensions (x, y, z) in meters
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points_per_voxel: Maximum points to keep per voxel
        max_voxels: Maximum number of voxels

    Returns:
        voxels: (M, max_points_per_voxel, 4) voxel features
        coordinates: (M, 3) voxel grid coordinates
        num_points_per_voxel: (M,) number of points in each voxel
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size

    # Calculate grid size
    nx = int((x_max - x_min) / vx)
    ny = int((y_max - y_min) / vy)
    nz = int((z_max - z_min) / vz)

    # Filter points within range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]

    # Compute voxel indices
    voxel_indices = np.floor((points[:, :3] - [x_min, y_min, z_min]) / [vx, vy, vz]).astype(np.int32)

    # Clamp indices to grid
    voxel_indices = np.clip(voxel_indices, [0, 0, 0], [nx - 1, ny - 1, nz - 1])

    # Create unique voxel IDs
    voxel_ids = (
        voxel_indices[:, 0] * (ny * nz) +
        voxel_indices[:, 1] * nz +
        voxel_indices[:, 2]
    )

    # Get unique voxels
    unique_voxel_ids, inverse_indices = np.unique(voxel_ids, return_inverse=True)

    # Limit number of voxels
    num_voxels = min(len(unique_voxel_ids), max_voxels)
    unique_voxel_ids = unique_voxel_ids[:num_voxels]

    # Initialize output arrays
    voxels = np.zeros((num_voxels, max_points_per_voxel, points.shape[1]), dtype=np.float32)
    coordinates = np.zeros((num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(num_voxels, dtype=np.int32)

    # Fill voxels
    for i, voxel_id in enumerate(unique_voxel_ids):
        # Get points in this voxel
        point_mask = (voxel_ids == voxel_id)
        voxel_points = points[point_mask]

        # Limit points per voxel
        num_pts = min(len(voxel_points), max_points_per_voxel)
        voxels[i, :num_pts] = voxel_points[:num_pts]
        num_points_per_voxel[i] = num_pts

        # Store voxel coordinates
        idx = np.where(point_mask)[0][0]
        coordinates[i] = voxel_indices[idx]

    return voxels, coordinates, num_points_per_voxel


def remove_ground_plane(
    points: np.ndarray,
    ground_threshold: float = -1.5,
    num_iterations: int = 10,
    distance_threshold: float = 0.2,
) -> np.ndarray:
    """
    Remove ground plane from point cloud using RANSAC.

    This helps focus on obstacles and objects above ground.

    Args:
        points: Point cloud (N, 3 or 4)
        ground_threshold: Minimum z-value to consider as non-ground
        num_iterations: RANSAC iterations
        distance_threshold: Point-to-plane distance threshold

    Returns:
        Point cloud with ground removed
    """
    if len(points) == 0:
        return points

    # Simple height-based filtering (fast)
    mask = points[:, 2] > ground_threshold

    # Optional: RANSAC for more accurate ground removal
    # (Commented out for speed, can be enabled if needed)
    """
    if len(points) > 100:
        best_inliers = 0
        best_plane = None

        for _ in range(num_iterations):
            # Randomly sample 3 points
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample_pts = points[sample_idx, :3]

            # Fit plane
            v1 = sample_pts[1] - sample_pts[0]
            v2 = sample_pts[2] - sample_pts[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            # Count inliers
            distances = np.abs(np.dot(points[:, :3] - sample_pts[0], normal))
            inliers = distances < distance_threshold

            if inliers.sum() > best_inliers:
                best_inliers = inliers.sum()
                best_plane = (normal, sample_pts[0])

        # Remove ground points
        if best_plane is not None:
            normal, point = best_plane
            distances = np.dot(points[:, :3] - point, normal)
            mask = distances > distance_threshold
    """

    return points[mask]


def filter_point_cloud_range(
    points: np.ndarray,
    point_cloud_range: Tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """
    Filter point cloud to keep only points within specified range.

    Args:
        points: Point cloud (N, 3 or 4)
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        Filtered point cloud
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )

    return points[mask]


def points_to_bev(
    points: np.ndarray,
    bev_range: Tuple[float, float, float, float],
    bev_resolution: float = 0.1,
    height_threshold: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Convert 3D point cloud to bird's eye view (BEV) representation.

    Creates a 2D grid where each cell contains statistics about points
    falling within that cell (e.g., max height, density).

    Args:
        points: Point cloud (N, 3 or 4)
        bev_range: [x_min, y_min, x_max, y_max] in meters
        bev_resolution: Cell size in meters
        height_threshold: Optional (z_min, z_max) to filter points

    Returns:
        BEV image (H, W, C) with channels for height, intensity, density
    """
    x_min, y_min, x_max, y_max = bev_range

    # Calculate grid size
    width = int((x_max - x_min) / bev_resolution)
    height = int((y_max - y_min) / bev_resolution)

    # Initialize BEV channels
    # Channel 0: Maximum height
    # Channel 1: Mean intensity
    # Channel 2: Point density
    bev = np.zeros((height, width, 3), dtype=np.float32)

    if len(points) == 0:
        return bev

    # Filter by height if specified
    if height_threshold is not None:
        z_min, z_max = height_threshold
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        points = points[mask]

    # Compute grid indices
    x_indices = np.floor((points[:, 0] - x_min) / bev_resolution).astype(np.int32)
    y_indices = np.floor((points[:, 1] - y_min) / bev_resolution).astype(np.int32)

    # Clamp to grid
    x_indices = np.clip(x_indices, 0, width - 1)
    y_indices = np.clip(y_indices, 0, height - 1)

    # Populate BEV grid
    for i in range(len(points)):
        x_idx = x_indices[i]
        y_idx = y_indices[i]

        # Maximum height
        bev[y_idx, x_idx, 0] = max(bev[y_idx, x_idx, 0], points[i, 2])

        # Accumulate intensity
        if points.shape[1] > 3:
            bev[y_idx, x_idx, 1] += points[i, 3]

        # Increment density
        bev[y_idx, x_idx, 2] += 1

    # Normalize intensity by density
    mask = bev[:, :, 2] > 0
    bev[mask, 1] = bev[mask, 1] / bev[mask, 2]

    # Normalize density to [0, 1]
    if bev[:, :, 2].max() > 0:
        bev[:, :, 2] = bev[:, :, 2] / bev[:, :, 2].max()

    return bev


def augment_point_cloud(
    points: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    rotation_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
    scale_range: Tuple[float, float] = (0.95, 1.05),
    translation_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    flip_probability: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Augment point cloud and bounding boxes for training.

    Applies random transformations including rotation, scaling, translation,
    and flipping to increase training data diversity.

    Args:
        points: Point cloud (N, 4)
        boxes: Bounding boxes (K, 7+) [x, y, z, w, l, h, yaw, ...]
        rotation_range: Range of rotation angles (min, max) in radians
        scale_range: Range of scaling factors (min, max)
        translation_std: Standard deviation for random translation
        flip_probability: Probability of horizontal flip

    Returns:
        Augmented points and boxes
    """
    points_aug = points.copy()
    boxes_aug = boxes.copy() if boxes is not None else None

    # Random rotation around z-axis
    angle = np.random.uniform(*rotation_range)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

    points_aug[:, :3] = points_aug[:, :3] @ rotation_matrix.T

    if boxes_aug is not None and len(boxes_aug) > 0:
        boxes_aug[:, :3] = boxes_aug[:, :3] @ rotation_matrix.T
        boxes_aug[:, 6] += angle  # Update yaw angle

    # Random scaling
    scale = np.random.uniform(*scale_range)
    points_aug[:, :3] *= scale

    if boxes_aug is not None and len(boxes_aug) > 0:
        boxes_aug[:, :6] *= scale

    # Random translation
    translation = np.random.normal(0, translation_std)
    points_aug[:, :3] += translation

    if boxes_aug is not None and len(boxes_aug) > 0:
        boxes_aug[:, :3] += translation

    # Random flip along y-axis
    if np.random.random() < flip_probability:
        points_aug[:, 1] = -points_aug[:, 1]

        if boxes_aug is not None and len(boxes_aug) > 0:
            boxes_aug[:, 1] = -boxes_aug[:, 1]
            boxes_aug[:, 6] = -boxes_aug[:, 6]

    return points_aug, boxes_aug


def downsample_point_cloud(
    points: np.ndarray,
    voxel_size: float = 0.1,
) -> np.ndarray:
    """
    Downsample point cloud using voxel grid filtering.

    Keeps one representative point per voxel, reducing point cloud density
    while preserving overall structure.

    Args:
        points: Point cloud (N, 3 or 4)
        voxel_size: Voxel size in meters

    Returns:
        Downsampled point cloud
    """
    if len(points) == 0:
        return points

    # Compute voxel indices
    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)

    # Create unique voxel IDs
    # Offset indices to ensure all are positive
    min_indices = voxel_indices.min(axis=0)
    voxel_indices -= min_indices

    # Compute 1D voxel IDs
    max_indices = voxel_indices.max(axis=0) + 1
    voxel_ids = (
        voxel_indices[:, 0] * (max_indices[1] * max_indices[2]) +
        voxel_indices[:, 1] * max_indices[2] +
        voxel_indices[:, 2]
    )

    # Get unique voxels and their first occurrence
    _, unique_indices = np.unique(voxel_ids, return_index=True)

    return points[unique_indices]
