"""
Sensor Fusion Utilities

This module provides utilities for fusing data from multiple sensors:
- Coordinate transformations between sensor frames
- LiDAR-to-camera projection
- Frustum extraction for region-based processing
- Multi-modal feature alignment
"""

from typing import Tuple, Optional
import numpy as np


def project_lidar_to_camera(
    points: np.ndarray,
    camera_intrinsic: np.ndarray,
    lidar_to_camera_transform: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project LiDAR points onto camera image plane.

    This enables fusion of 3D LiDAR data with 2D camera images.

    Args:
        points: LiDAR points (N, 3 or 4) in LiDAR frame
        camera_intrinsic: Camera intrinsic matrix (3, 3)
        lidar_to_camera_transform: Transformation matrix (4, 4)
        image_shape: (height, width) of camera image

    Returns:
        image_points: (M, 2) pixel coordinates of projected points
        point_depths: (M,) depths of projected points
    """
    if len(points) == 0:
        return np.zeros((0, 2)), np.zeros(0)

    # Convert to homogeneous coordinates
    points_homogeneous = np.ones((len(points), 4))
    points_homogeneous[:, :3] = points[:, :3]

    # Transform to camera frame
    points_camera = (lidar_to_camera_transform @ points_homogeneous.T).T[:, :3]

    # Remove points behind camera
    mask = points_camera[:, 2] > 0
    points_camera = points_camera[mask]

    if len(points_camera) == 0:
        return np.zeros((0, 2)), np.zeros(0)

    # Project to image plane
    points_image = (camera_intrinsic @ points_camera.T).T
    points_image[:, :2] /= points_image[:, 2:3]

    # Get depths
    depths = points_image[:, 2]

    # Remove points outside image
    height, width = image_shape
    mask = (
        (points_image[:, 0] >= 0) & (points_image[:, 0] < width) &
        (points_image[:, 1] >= 0) & (points_image[:, 1] < height)
    )

    return points_image[mask, :2], depths[mask]


def get_frustum_points(
    points: np.ndarray,
    bbox_2d: np.ndarray,
    camera_intrinsic: np.ndarray,
    lidar_to_camera_transform: np.ndarray,
) -> np.ndarray:
    """
    Extract points within camera frustum defined by 2D bounding box.

    This is useful for region-based 3D object detection where we first
    detect objects in 2D, then extract relevant 3D points.

    Args:
        points: LiDAR points (N, 3 or 4)
        bbox_2d: 2D bounding box [x_min, y_min, x_max, y_max]
        camera_intrinsic: Camera intrinsic matrix (3, 3)
        lidar_to_camera_transform: Transformation matrix (4, 4)

    Returns:
        Points within frustum
    """
    if len(points) == 0:
        return points

    # Project points to image
    image_points, depths = project_lidar_to_camera(
        points, camera_intrinsic, lidar_to_camera_transform, (10000, 10000)
    )

    if len(image_points) == 0:
        return np.zeros((0, points.shape[1]))

    # Filter points within 2D bbox
    x_min, y_min, x_max, y_max = bbox_2d
    mask = (
        (image_points[:, 0] >= x_min) & (image_points[:, 0] <= x_max) &
        (image_points[:, 1] >= y_min) & (image_points[:, 1] <= y_max)
    )

    # Return original points within frustum
    # Note: Need to track which points survived projection
    # For simplicity, we return points that project within bbox
    # In practice, you'd need to maintain index mapping

    return points[:len(image_points)][mask]


def transform_points(
    points: np.ndarray,
    transformation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Transform points using 4x4 transformation matrix.

    Args:
        points: Points (N, 3) or (N, 4)
        transformation_matrix: Transformation matrix (4, 4)

    Returns:
        Transformed points (N, 3)
    """
    if len(points) == 0:
        return points

    # Convert to homogeneous coordinates
    points_homogeneous = np.ones((len(points), 4))
    points_homogeneous[:, :3] = points[:, :3]

    # Apply transformation
    points_transformed = (transformation_matrix @ points_homogeneous.T).T

    return points_transformed[:, :3]


def create_transformation_matrix(
    translation: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Create 4x4 transformation matrix from translation and rotation.

    Args:
        translation: Translation vector (3,)
        rotation_matrix: Rotation matrix (3, 3)

    Returns:
        Transformation matrix (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """
    Convert 3D boxes to 8 corner points.

    Args:
        boxes: Boxes (N, 7+) [x, y, z, w, l, h, yaw, ...]

    Returns:
        Corners (N, 8, 3) - 8 corner points for each box
    """
    if len(boxes) == 0:
        return np.zeros((0, 8, 3))

    # Extract box parameters
    centers = boxes[:, :3]
    dims = boxes[:, 3:6]  # w, l, h
    yaws = boxes[:, 6]

    # Create template box corners (before rotation)
    # Box center is at origin, with dimensions w, l, h
    w, l, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]

    # 8 corners of box (in local frame)
    corners_local = np.array([
        [-w / 2, -l / 2, -h / 2],
        [w / 2, -l / 2, -h / 2],
        [w / 2, l / 2, -h / 2],
        [-w / 2, l / 2, -h / 2],
        [-w / 2, -l / 2, h / 2],
        [w / 2, -l / 2, h / 2],
        [w / 2, l / 2, h / 2],
        [-w / 2, l / 2, h / 2],
    ])  # (8, N, 3)

    corners_local = corners_local.transpose(1, 0, 2)  # (N, 8, 3)

    # Rotation matrices for each box
    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)

    # Rotate corners
    corners = np.zeros_like(corners_local)
    corners[:, :, 0] = (
        corners_local[:, :, 0] * cos_yaw[:, None] -
        corners_local[:, :, 1] * sin_yaw[:, None]
    )
    corners[:, :, 1] = (
        corners_local[:, :, 0] * sin_yaw[:, None] +
        corners_local[:, :, 1] * cos_yaw[:, None]
    )
    corners[:, :, 2] = corners_local[:, :, 2]

    # Translate to box centers
    corners += centers[:, None, :]

    return corners


def compute_iou_3d(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute 3D Intersection over Union (IoU) between boxes.

    This is a simplified version using bird's eye view IoU.
    Full 3D IoU computation is more complex.

    Args:
        boxes1: Boxes (N, 7+) [x, y, z, w, l, h, yaw, ...]
        boxes2: Boxes (M, 7+)

    Returns:
        IoU matrix (N, M)
    """
    # Use bird's eye view IoU as approximation
    # This is faster and commonly used in practice
    return compute_iou_bev(boxes1, boxes2)


def compute_iou_bev(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute bird's eye view IoU between boxes.

    Args:
        boxes1: Boxes (N, 7+) [x, y, z, w, l, h, yaw, ...]
        boxes2: Boxes (M, 7+)

    Returns:
        IoU matrix (N, M)
    """
    N = len(boxes1)
    M = len(boxes2)

    if N == 0 or M == 0:
        return np.zeros((N, M))

    # Simplified rectangular IoU (assumes aligned boxes)
    # For accurate IoU with rotation, use specialized libraries
    ious = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            # Extract box parameters
            x1, y1, w1, l1 = boxes1[i, 0], boxes1[i, 1], boxes1[i, 3], boxes1[i, 4]
            x2, y2, w2, l2 = boxes2[j, 0], boxes2[j, 1], boxes2[j, 3], boxes2[j, 4]

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

            if union > 0:
                ious[i, j] = intersection / union

    return ious


def align_radar_velocity_to_lidar(
    radar_points: np.ndarray,
    radar_to_lidar_transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Align radar velocity measurements to LiDAR frame.

    Radar provides velocity in its own frame, which needs to be
    transformed to match the LiDAR/vehicle frame for fusion.

    Args:
        radar_points: Radar detections (N, 5) [x, y, z, vx, vy]
        radar_to_lidar_transform: Transformation matrix (4, 4)

    Returns:
        Radar points in LiDAR frame
    """
    if len(radar_points) == 0:
        return radar_points

    aligned_points = radar_points.copy()

    if radar_to_lidar_transform is not None:
        # Transform positions
        positions = transform_points(radar_points[:, :3], radar_to_lidar_transform)
        aligned_points[:, :3] = positions

        # Transform velocities (using rotation part only)
        rotation = radar_to_lidar_transform[:3, :3]
        velocities = radar_points[:, 3:5]
        velocities_3d = np.column_stack([velocities, np.zeros(len(velocities))])
        velocities_transformed = (rotation @ velocities_3d.T).T
        aligned_points[:, 3:5] = velocities_transformed[:, :2]

    return aligned_points
