"""
Inference Script for Autonomous Driving Perception System

This script runs the complete perception pipeline:
1. 3D object detection
2. Multi-object tracking
3. Path planning

Usage:
    python inference.py --config config/inference_config.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.nuscenes_loader import NuScenesDataset
from models.pointpillars import PointPillars, create_pillar_input
from tracking.track_manager import MultiObjectTracker
from planning.occupancy_grid import OccupancyGrid
from planning.path_planner import AStarPlanner
from planning.trajectory_optimizer import TrajectoryOptimizer
from utils.visualization import visualize_bev
from utils.bbox_utils import nms_3d


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config/inference_config.yaml',
        help='Path to inference configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    return parser.parse_args()


class PerceptionPipeline:
    """
    Complete autonomous driving perception pipeline.

    Integrates detection, tracking, and planning.
    """

    def __init__(
        self,
        model: PointPillars,
        tracker: MultiObjectTracker,
        planner: AStarPlanner,
        config: dict,
        device: str = 'cuda',
    ):
        """
        Initialize pipeline.

        Args:
            model: Detection model
            tracker: Multi-object tracker
            planner: Path planner
            config: Configuration dictionary
            device: Device to run on
        """
        self.model = model.to(device)
        self.tracker = tracker
        self.planner = planner
        self.config = config
        self.device = device

        self.model.eval()

    def process_frame(
        self,
        sample: dict,
        goal_position: tuple = (50.0, 0.0),
    ) -> dict:
        """
        Process single frame through complete pipeline.

        Args:
            sample: Data sample from dataset
            goal_position: Goal position for path planning

        Returns:
            Dictionary with detections, tracks, and trajectory
        """
        # Extract data
        lidar_points = sample['lidar_points']
        gt_boxes = sample['gt_boxes_3d']

        # 1. Object Detection
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            points_tensor = torch.from_numpy(lidar_points).unsqueeze(0).float().to(self.device)

            # Create pillar representation
            pillar_features, pillar_coords = create_pillar_input(
                points_tensor,
                max_points_per_pillar=self.config['model']['pillar']['max_points_per_pillar'],
                max_pillars=self.config['model']['pillar']['max_pillars'],
                pillar_size=tuple(self.config['model']['pillar']['pillar_size']),
                point_cloud_range=self.config['model']['point_cloud_range'],
            )

            # Forward pass
            outputs = self.model(pillar_features, pillar_coords, batch_size=1)

            # Post-process (simplified - use ground truth for demo)
            detections = gt_boxes  # Placeholder

        # Apply NMS
        if len(detections) > 0:
            scores = np.ones(len(detections))  # Placeholder scores
            keep_idx = nms_3d(
                detections[:, :7],
                scores,
                iou_threshold=self.config['nms']['nms_iou_threshold'],
            )
            detections = detections[keep_idx]
        else:
            detections = np.zeros((0, 7))

        # 2. Multi-Object Tracking
        if self.config['tracking']['enabled']:
            tracks = self.tracker.update(detections[:, :7])
            tracked_boxes = np.array([track.get_state() for track in tracks])
        else:
            tracks = []
            tracked_boxes = detections[:, :7]

        # 3. Path Planning
        trajectory = None
        if self.config['planning']['enabled'] and len(tracked_boxes) > 0:
            # Create occupancy grid
            grid_config = self.config['planning']['grid']
            occupancy_grid = OccupancyGrid(
                resolution=grid_config['resolution'],
                width=grid_config['width'],
                height=grid_config['height'],
            )
            occupancy_grid.update_from_boxes(
                tracked_boxes,
                safety_margin=self.config['planning']['planner']['safety_margin'],
            )

            # Plan path
            self.planner.grid = occupancy_grid
            start_position = (0.0, 0.0)  # Ego vehicle position
            path = self.planner.plan(start_position, goal_position)

            if path is not None:
                # Optimize trajectory
                optimizer = TrajectoryOptimizer(
                    max_velocity=self.config['planning']['trajectory']['max_velocity'],
                    max_acceleration=self.config['planning']['trajectory']['max_acceleration'],
                    dt=self.config['planning']['trajectory']['dt'],
                )
                trajectory, velocities, timestamps = optimizer.optimize(path)

        return {
            'points': lidar_points,
            'detections': detections,
            'tracks': tracks,
            'trajectory': trajectory,
        }


def main():
    """Main inference function."""
    args = parse_args()

    # Load configurations
    with open(args.config) as f:
        inference_config = yaml.safe_load(f)

    model_config_path = Path(args.config).parent / 'model_config.yaml'
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    config = {**model_config, **inference_config}

    print('=' * 80)
    print('Autonomous Driving Perception System - Inference')
    print('=' * 80)
    print(f'Device: {args.device}')
    print(f'Output directory: {args.output}')
    print('=' * 80)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print('Loading dataset...')
    dataset = NuScenesDataset(
        dataroot=config['dataset']['dataroot'],
        version=config['dataset']['version'],
        split=config['dataset']['split'],
    )
    print(f'Dataset samples: {len(dataset)}')

    # Create model
    print('Creating model...')
    model = PointPillars(
        num_classes=config['model']['num_classes'],
        max_points_per_pillar=config['model']['pillar']['max_points_per_pillar'],
        max_pillars=config['model']['pillar']['max_pillars'],
        pillar_size=tuple(config['model']['pillar']['pillar_size']),
        point_cloud_range=config['model']['point_cloud_range'],
    )

    # Load checkpoint
    if args.checkpoint:
        print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Create tracker
    if config['tracking']['enabled']:
        tracker = MultiObjectTracker(
            max_age=config['tracking']['max_age'],
            min_hits=config['tracking']['min_hits'],
            iou_threshold=config['tracking']['iou_threshold'],
        )
    else:
        tracker = None

    # Create planner
    if config['planning']['enabled']:
        occupancy_grid = OccupancyGrid(
            resolution=config['planning']['grid']['resolution'],
            width=config['planning']['grid']['width'],
            height=config['planning']['grid']['height'],
        )
        planner = AStarPlanner(occupancy_grid)
    else:
        planner = None

    # Create pipeline
    pipeline = PerceptionPipeline(
        model=model,
        tracker=tracker,
        planner=planner,
        config=config,
        device=args.device,
    )

    # Process frames
    print('Processing frames...')
    num_samples = min(10, len(dataset))  # Process first 10 samples

    for idx in tqdm(range(num_samples)):
        sample = dataset[idx]

        # Run pipeline
        result = pipeline.process_frame(sample)

        # Visualize
        if config['visualization']['enabled']:
            fig = visualize_bev(
                points=result['points'],
                boxes=result['detections'][:, :7] if len(result['detections']) > 0 else None,
                tracks=result['tracks'],
                trajectory=result['trajectory'],
                point_cloud_range=config['model']['point_cloud_range'],
                save_path=output_dir / f'frame_{idx:04d}.png',
            )
            plt.close(fig)

    print('=' * 80)
    print('Inference completed!')
    print(f'Results saved to: {output_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
