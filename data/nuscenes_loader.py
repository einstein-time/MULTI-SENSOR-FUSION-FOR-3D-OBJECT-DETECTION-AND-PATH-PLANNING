"""
nuScenes Dataset Loader for Multi-Sensor Fusion

This module provides a PyTorch Dataset class for loading nuScenes data including:
- Camera images (6 cameras)
- LiDAR point clouds
- Radar detections
- 3D bounding box annotations
- Calibration matrices

The dataset handles synchronization across sensors and provides data in a
format suitable for 3D object detection and multi-sensor fusion.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("Warning: nuscenes-devkit not installed. Using mock data.")


class NuScenesDataset:
    """
    nuScenes dataset loader for multi-sensor autonomous driving data.

    This dataset provides synchronized multi-sensor data including camera images,
    LiDAR point clouds, radar detections, and 3D bounding box annotations.

    Attributes:
        nusc: NuScenes API instance
        scenes: List of scene tokens for the specified split
        samples: List of sample tokens
        sensors: List of sensor names to load
    """

    def __init__(
        self,
        dataroot: str,
        version: str = 'v1.0-mini',
        split: str = 'train',
        sensors: Optional[List[str]] = None,
        point_cloud_range: Optional[List[float]] = None,
    ):
        """
        Initialize nuScenes dataset loader.

        Args:
            dataroot: Path to nuScenes data directory
            version: Dataset version ('v1.0-mini' or 'v1.0-trainval')
            split: Data split ('train', 'val', or 'test')
            sensors: List of sensors to load (default: front camera and LiDAR)
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
                              for filtering point clouds
        """
        self.dataroot = Path(dataroot)
        self.version = version
        self.split = split

        if sensors is None:
            self.sensors = ['CAM_FRONT', 'LIDAR_TOP']
        else:
            self.sensors = sensors

        if point_cloud_range is None:
            self.point_cloud_range = [0, -40, -3, 70, 40, 1]
        else:
            self.point_cloud_range = point_cloud_range

        # Initialize nuScenes API
        if NUSCENES_AVAILABLE and self.dataroot.exists():
            self.nusc = NuScenes(
                version=version,
                dataroot=str(dataroot),
                verbose=False
            )
            self._build_sample_index()
        else:
            # Mock data for testing without dataset
            self.nusc = None
            self.samples = []
            print(f"Warning: Running in mock mode. Data will be simulated.")

    def _build_sample_index(self) -> None:
        """Build index of samples for the specified split."""
        if self.nusc is None:
            return

        # Get scenes for split
        scenes = self._get_scenes_for_split()

        # Collect all samples from these scenes
        self.samples = []
        for scene in scenes:
            scene_rec = self.nusc.get('scene', scene)
            sample_token = scene_rec['first_sample_token']

            while sample_token:
                self.samples.append(sample_token)
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']

    def _get_scenes_for_split(self) -> List[str]:
        """Get scene tokens for the specified split."""
        if self.nusc is None:
            return []

        # For mini version, simple split
        all_scenes = [s['token'] for s in self.nusc.scene]

        if self.split == 'train':
            return all_scenes[:int(0.8 * len(all_scenes))]
        elif self.split == 'val':
            return all_scenes[int(0.8 * len(all_scenes)):]
        else:
            return all_scenes

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        if self.nusc is None:
            return 100  # Mock data
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get multi-sensor sample at index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - camera_images: Dict[str, np.ndarray] - sensor_name -> image
            - lidar_points: np.ndarray (N, 4) - x, y, z, intensity
            - radar_points: np.ndarray (M, 5) - x, y, z, vx, vy
            - gt_boxes_3d: np.ndarray (K, 9) - x, y, z, w, l, h, yaw, class, id
            - calibration: Dict with transformation matrices
            - sample_token: str
        """
        if self.nusc is None:
            return self._get_mock_data(idx)

        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)

        # Load multi-sensor data
        camera_data = self._load_cameras(sample)
        lidar_data = self._load_lidar(sample)
        radar_data = self._load_radar(sample)
        gt_boxes = self._load_annotations(sample)
        calibration = self._load_calibration(sample)

        return {
            'camera_images': camera_data,
            'lidar_points': lidar_data,
            'radar_points': radar_data,
            'gt_boxes_3d': gt_boxes,
            'calibration': calibration,
            'sample_token': sample_token
        }

    def _load_cameras(self, sample: dict) -> Dict[str, np.ndarray]:
        """
        Load camera images for all requested camera sensors.

        Args:
            sample: nuScenes sample record

        Returns:
            Dictionary mapping sensor names to images
        """
        camera_data = {}
        camera_sensors = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        for sensor_name in camera_sensors:
            if sensor_name not in self.sensors:
                continue

            cam_token = sample['data'][sensor_name]
            cam_data = self.nusc.get('sample_data', cam_token)

            # Load image
            img_path = self.dataroot / cam_data['filename']
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._preprocess_image(img)
                camera_data[sensor_name] = img

        return camera_data

    def _load_lidar(self, sample: dict) -> np.ndarray:
        """
        Load LiDAR point cloud.

        Args:
            sample: nuScenes sample record

        Returns:
            Point cloud array (N, 4) with x, y, z, intensity
        """
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)

        # Load point cloud file
        pcl_path = self.dataroot / lidar_data['filename']
        pc = LidarPointCloud.from_file(str(pcl_path))

        # Points are in shape (4, N) - x, y, z, intensity
        points = pc.points.T  # Now (N, 4)

        # Filter points within range
        points = self._filter_lidar_points(points)

        return points

    def _filter_lidar_points(self, points: np.ndarray) -> np.ndarray:
        """
        Filter LiDAR points to remove noise and out-of-range points.

        Removes:
        - Points too close (< 1m) - likely from vehicle itself
        - Points outside defined range
        - Points with invalid intensities

        Args:
            points: Point cloud (N, 4)

        Returns:
            Filtered point cloud
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range

        # Distance filter
        distances = np.linalg.norm(points[:, :3], axis=1)
        mask = distances > 1.0

        # Range filter
        mask &= (points[:, 0] >= x_min) & (points[:, 0] < x_max)
        mask &= (points[:, 1] >= y_min) & (points[:, 1] < y_max)
        mask &= (points[:, 2] >= z_min) & (points[:, 2] < z_max)

        return points[mask]

    def _load_radar(self, sample: dict) -> np.ndarray:
        """
        Load radar detections from all radar sensors.

        Args:
            sample: nuScenes sample record

        Returns:
            Radar points array (M, 5) with x, y, z, vx, vy
        """
        radar_sensors = [
            'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
        ]

        all_radar_points = []

        for sensor_name in radar_sensors:
            if sensor_name not in sample['data']:
                continue

            radar_token = sample['data'][sensor_name]
            radar_data = self.nusc.get('sample_data', radar_token)

            # Load radar point cloud
            radar_path = self.dataroot / radar_data['filename']
            radar_pc = RadarPointCloud.from_file(str(radar_path))

            # Extract x, y, z, vx, vy (velocity components)
            points = radar_pc.points[:5].T  # (M, 5)
            all_radar_points.append(points)

        if all_radar_points:
            radar_points = np.vstack(all_radar_points)
        else:
            radar_points = np.zeros((0, 5), dtype=np.float32)

        return radar_points

    def _load_annotations(self, sample: dict) -> np.ndarray:
        """
        Load 3D bounding box annotations.

        Args:
            sample: nuScenes sample record

        Returns:
            Array of boxes (K, 9):
            [x, y, z, width, length, height, yaw, class_id, track_id]
        """
        boxes = []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Get box parameters
            center = ann['translation']  # [x, y, z]
            size = ann['size']  # [width, length, height]
            orientation = Quaternion(ann['rotation'])

            # Convert quaternion to yaw angle
            yaw = self._quaternion_to_yaw(orientation)

            # Get class ID
            class_name = ann['category_name']
            class_id = self._get_class_id(class_name)

            # Get track ID
            instance_token = ann['instance_token']
            track_id = hash(instance_token) % 100000

            # Combine into box array
            box = [
                center[0], center[1], center[2],
                size[0], size[1], size[2],
                yaw, class_id, track_id
            ]
            boxes.append(box)

        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
        else:
            boxes = np.zeros((0, 9), dtype=np.float32)

        return boxes

    def _load_calibration(self, sample: dict) -> Dict:
        """
        Load calibration matrices for sensor fusion.

        Args:
            sample: nuScenes sample record

        Returns:
            Dictionary with transformation matrices
        """
        calibration = {}

        # Get LiDAR calibration
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd = self.nusc.get('sample_data', lidar_token)
        lidar_cs = self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])

        # Camera calibrations
        if 'CAM_FRONT' in sample['data']:
            cam_token = sample['data']['CAM_FRONT']
            cam_sd = self.nusc.get('sample_data', cam_token)
            cam_cs = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])

            # Camera intrinsic matrix
            calibration['CAM_FRONT_intrinsic'] = np.array(cam_cs['camera_intrinsic'])

            # Extrinsic transformations (for projecting LiDAR to camera)
            cam_translation = np.array(cam_cs['translation'])
            cam_rotation = Quaternion(cam_cs['rotation']).rotation_matrix

            lidar_translation = np.array(lidar_cs['translation'])
            lidar_rotation = Quaternion(lidar_cs['rotation']).rotation_matrix

            calibration['lidar_to_cam'] = {
                'cam_translation': cam_translation,
                'cam_rotation': cam_rotation,
                'lidar_translation': lidar_translation,
                'lidar_rotation': lidar_rotation,
            }

        return calibration

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess camera image.

        Args:
            img: Input image

        Returns:
            Preprocessed image (normalized, resized)
        """
        # Resize to standard size
        img = cv2.resize(img, (800, 450))

        # Convert to float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def _quaternion_to_yaw(self, q: Quaternion) -> float:
        """
        Convert quaternion to yaw angle (rotation around z-axis).

        Args:
            q: Quaternion

        Returns:
            Yaw angle in radians
        """
        # Extract yaw from quaternion
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )
        return yaw

    def _get_class_id(self, class_name: str) -> int:
        """
        Map class name to integer ID.

        Args:
            class_name: nuScenes category name

        Returns:
            Class ID
        """
        class_mapping = {
            'car': 0,
            'truck': 1,
            'bus': 2,
            'trailer': 3,
            'construction_vehicle': 4,
            'pedestrian': 5,
            'motorcycle': 6,
            'bicycle': 7,
            'traffic_cone': 8,
            'barrier': 9,
        }

        # Check if any keyword matches
        for key, value in class_mapping.items():
            if key in class_name.lower():
                return value

        return 10  # Other/unknown

    def _get_mock_data(self, idx: int) -> Dict:
        """
        Generate mock data for testing without nuScenes dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with simulated multi-sensor data
        """
        np.random.seed(idx)

        # Mock camera image
        camera_images = {
            'CAM_FRONT': np.random.rand(450, 800, 3).astype(np.float32)
        }

        # Mock LiDAR points
        num_points = np.random.randint(10000, 30000)
        lidar_points = np.random.randn(num_points, 4).astype(np.float32)
        lidar_points[:, 0] = lidar_points[:, 0] * 20 + 30  # x: 10-50m
        lidar_points[:, 1] = lidar_points[:, 1] * 30  # y: -30 to 30m
        lidar_points[:, 2] = lidar_points[:, 2] * 2 - 1  # z: -3 to 1m
        lidar_points[:, 3] = np.abs(lidar_points[:, 3])  # intensity: positive

        # Mock radar points
        num_radar = np.random.randint(50, 200)
        radar_points = np.random.randn(num_radar, 5).astype(np.float32)

        # Mock ground truth boxes
        num_boxes = np.random.randint(5, 15)
        gt_boxes = np.zeros((num_boxes, 9), dtype=np.float32)
        gt_boxes[:, 0] = np.random.uniform(10, 50, num_boxes)  # x
        gt_boxes[:, 1] = np.random.uniform(-20, 20, num_boxes)  # y
        gt_boxes[:, 2] = np.random.uniform(-1, 0, num_boxes)  # z
        gt_boxes[:, 3:6] = np.random.uniform(1, 4, (num_boxes, 3))  # w, l, h
        gt_boxes[:, 6] = np.random.uniform(-np.pi, np.pi, num_boxes)  # yaw
        gt_boxes[:, 7] = np.random.randint(0, 10, num_boxes)  # class
        gt_boxes[:, 8] = np.arange(num_boxes)  # track_id

        return {
            'camera_images': camera_images,
            'lidar_points': lidar_points,
            'radar_points': radar_points,
            'gt_boxes_3d': gt_boxes,
            'calibration': {},
            'sample_token': f'mock_{idx}'
        }
