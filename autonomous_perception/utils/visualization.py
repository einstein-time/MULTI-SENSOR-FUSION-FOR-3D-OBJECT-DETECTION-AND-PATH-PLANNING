"""
Visualization Utilities

Functions for visualizing point clouds, detections, and trajectories.
"""

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def visualize_bev(
    points: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    tracks: Optional[List] = None,
    trajectory: Optional[np.ndarray] = None,
    point_cloud_range: List[float] = [0, -40, -3, 70, 40, 1],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize bird's eye view of scene.

    Args:
        points: Point cloud (N, 3 or 4)
        boxes: Bounding boxes (M, 7) [x, y, z, w, l, h, yaw]
        tracks: List of tracks
        trajectory: Planned trajectory (K, 2)
        point_cloud_range: Point cloud range
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot point cloud
    if points is not None and len(points) > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=0.5,
            c='gray',
            alpha=0.3,
            label='Point Cloud'
        )

    # Plot bounding boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            corners = get_box_corners_2d(box)
            rect = plt.Polygon(
                corners,
                fill=False,
                edgecolor='red',
                linewidth=2,
                label='Detection'
            )
            ax.add_patch(rect)

            # Draw heading direction
            x, y, yaw = box[0], box[1], box[6]
            dx = np.cos(yaw) * 2
            dy = np.sin(yaw) * 2
            ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.5, fc='red', ec='red')

    # Plot tracks
    if tracks is not None:
        for track in tracks:
            if hasattr(track, 'history') and len(track.history) > 1:
                history = np.array(track.history)
                ax.plot(
                    history[:, 0],
                    history[:, 1],
                    'b-',
                    alpha=0.5,
                    linewidth=2,
                )

    # Plot trajectory
    if trajectory is not None and len(trajectory) > 0:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            'g-',
            linewidth=3,
            label='Planned Path'
        )
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            s=100,
            c='green',
            marker='o',
            label='Start'
        )
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            s=100,
            c='blue',
            marker='*',
            label='Goal'
        )

    # Set limits
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    x_max, y_max = point_cloud_range[3], point_cloud_range[4]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Bird\'s Eye View', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def get_box_corners_2d(box: np.ndarray) -> np.ndarray:
    """
    Get 2D corners of bounding box.

    Args:
        box: Box [x, y, z, w, l, h, yaw]

    Returns:
        Corners (4, 2)
    """
    x, y, w, l, yaw = box[0], box[1], box[3], box[4], box[6]

    # Local corners
    corners_local = np.array([
        [-w/2, -l/2],
        [w/2, -l/2],
        [w/2, l/2],
        [-w/2, l/2],
    ])

    # Rotation matrix
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
    ])

    # Transform to world frame
    corners = (rot_mat @ corners_local.T).T + np.array([x, y])

    return corners


def visualize_3d_boxes(
    points: np.ndarray,
    boxes: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize 3D bounding boxes with point cloud.

    Args:
        points: Point cloud (N, 3)
        boxes: Bounding boxes (M, 7)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(15, 5))

    # Bird's eye view
    ax1 = fig.add_subplot(131)
    ax1.scatter(points[:, 0], points[:, 1], s=0.1, c='gray', alpha=0.3)
    for box in boxes:
        corners = get_box_corners_2d(box)
        rect = plt.Polygon(corners, fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Bird\'s Eye View')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Front view
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 2], s=0.1, c='gray', alpha=0.3)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Front View')
    ax2.grid(True, alpha=0.3)

    # Side view
    ax3 = fig.add_subplot(133)
    ax3.scatter(points[:, 1], points[:, 2], s=0.1, c='gray', alpha=0.3)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_trajectory(
    trajectory: np.ndarray,
    velocities: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot trajectory with velocity profile.

    Args:
        trajectory: Trajectory positions (N, 2)
        velocities: Velocity profile (N,)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if velocities is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], s=100, c='green', marker='o', label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], s=100, c='red', marker='*', label='Goal')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Planned Trajectory', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')

    # Plot velocity profile
    if velocities is not None:
        distances = np.cumsum([0] + [
            np.linalg.norm(trajectory[i+1] - trajectory[i])
            for i in range(len(trajectory) - 1)
        ])
        ax2.plot(distances, velocities, 'r-', linewidth=2)
        ax2.set_xlabel('Distance (m)', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.set_title('Velocity Profile', fontsize=14)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_occupancy_grid(
    grid: np.ndarray,
    path: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize occupancy grid with path.

    Args:
        grid: Occupancy grid (H, W)
        path: Path as list of grid coordinates
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Show grid
    ax.imshow(grid, cmap='gray_r', origin='lower')

    # Plot path
    if path is not None:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=3, label='Path')
        ax.scatter(path_array[0, 0], path_array[0, 1], s=100, c='green', marker='o', label='Start')
        ax.scatter(path_array[-1, 0], path_array[-1, 1], s=100, c='blue', marker='*', label='Goal')

    ax.set_xlabel('Grid X', fontsize=12)
    ax.set_ylabel('Grid Y', fontsize=12)
    ax.set_title('Occupancy Grid', fontsize=14)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
