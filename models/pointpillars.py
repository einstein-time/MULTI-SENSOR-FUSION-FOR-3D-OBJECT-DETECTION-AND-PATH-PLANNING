"""
PointPillars: Fast Encoders for Object Detection from Point Clouds

This module implements the PointPillars architecture for 3D object detection
from LiDAR point clouds. PointPillars converts 3D point clouds to a 2D
pseudo-image representation (bird's eye view) for efficient processing.

Reference:
    Lang et al. "PointPillars: Fast Encoders for Object Detection from Point Clouds"
    CVPR 2019

Architecture:
    1. Pillar Feature Network: Extract features from point pillars
    2. Scatter: Create BEV pseudo-image
    3. Backbone: 2D CNN for feature extraction
    4. Detection Head: Predict 3D bounding boxes
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointPillars(nn.Module):
    """
    PointPillars 3D object detection model.

    This model provides real-time 3D object detection from LiDAR point clouds
    by converting them to a bird's eye view representation.

    Attributes:
        pillar_encoder: Encodes point pillars into features
        backbone: 2D CNN backbone for BEV feature extraction
        detection_head: Predicts 3D bounding boxes
    """

    def __init__(
        self,
        num_classes: int = 10,
        max_points_per_pillar: int = 100,
        max_pillars: int = 12000,
        pillar_size: Tuple[float, float, float] = (0.16, 0.16, 4.0),
        point_cloud_range: List[float] = [0, -40, -3, 70, 40, 1],
        num_input_features: int = 4,
    ):
        """
        Initialize PointPillars model.

        Args:
            num_classes: Number of object classes
            max_points_per_pillar: Maximum points per pillar
            max_pillars: Maximum number of pillars
            pillar_size: (x, y, z) pillar dimensions in meters
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            num_input_features: Number of input features per point (default: 4 for x,y,z,intensity)
        """
        super().__init__()

        self.num_classes = num_classes
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars
        self.pillar_size = pillar_size
        self.point_cloud_range = point_cloud_range

        # Calculate grid size
        self.grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / pillar_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / pillar_size[1]),
            1  # Single layer in z-direction (BEV)
        ]

        # Pillar feature encoder
        self.pillar_encoder = PointPillarEncoder(
            in_channels=num_input_features,
            feat_channels=[64],
            point_cloud_range=point_cloud_range,
            voxel_size=pillar_size,
        )

        # Backbone (2D CNN)
        self.backbone = Backbone2D(
            in_channels=64,
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
            num_filters=[64, 128, 256],
        )

        # Detection head
        self.detection_head = DetectionHead(
            in_channels=sum([64, 128, 256]),  # Multi-scale features
            num_classes=num_classes,
            num_anchors=2,  # 0° and 90° anchors
        )

    def forward(
        self,
        pillar_features: torch.Tensor,
        pillar_coords: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pillar_features: (num_pillars, num_points, num_features)
            pillar_coords: (num_pillars, 4) - batch_idx, z, y, x
            batch_size: Batch size

        Returns:
            Dictionary with:
            - cls_preds: (B, H, W, num_anchors, num_classes)
            - box_preds: (B, H, W, num_anchors, 7)
            - dir_preds: (B, H, W, num_anchors, 2)
        """
        # Encode pillars
        encoded_features = self.pillar_encoder(pillar_features)

        # Scatter to BEV
        bev_features = self.scatter_features(
            encoded_features,
            pillar_coords,
            batch_size,
        )

        # Backbone
        spatial_features = self.backbone(bev_features)

        # Detection head
        predictions = self.detection_head(spatial_features)

        return predictions

    def scatter_features(
        self,
        pillar_features: torch.Tensor,
        pillar_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Scatter pillar features to create BEV pseudo-image.

        Args:
            pillar_features: (num_pillars, num_channels)
            pillar_coords: (num_pillars, 4) - batch_idx, z, y, x
            batch_size: Batch size

        Returns:
            BEV feature map (B, C, H, W)
        """
        num_channels = pillar_features.shape[1]
        nx, ny = self.grid_size[0], self.grid_size[1]

        # Initialize BEV canvas
        canvas = torch.zeros(
            (batch_size, num_channels, ny, nx),
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        # Scatter features
        batch_idx = pillar_coords[:, 0].long()
        x_idx = pillar_coords[:, 3].long()
        y_idx = pillar_coords[:, 2].long()

        # Clamp indices
        x_idx = torch.clamp(x_idx, 0, nx - 1)
        y_idx = torch.clamp(y_idx, 0, ny - 1)

        canvas[batch_idx, :, y_idx, x_idx] = pillar_features

        return canvas


class PointPillarEncoder(nn.Module):
    """
    Pillar feature encoder using simplified PointNet.

    Extracts features from each pillar independently using 1D convolutions
    and max pooling.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: List[int] = [64],
        point_cloud_range: Optional[List[float]] = None,
        voxel_size: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Initialize pillar encoder.

        Args:
            in_channels: Number of input channels per point
            feat_channels: List of feature dimensions
            point_cloud_range: Point cloud range for normalization
            voxel_size: Voxel size for computing pillar centers
        """
        super().__init__()

        self.in_channels = in_channels
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # Feature extraction uses 9 channels:
        # x, y, z, intensity (4) + xc, yc, zc (3) + xp, yp (2)
        # xc, yc, zc: offset from pillar center
        # xp, yp: offset from point mean
        feat_in_channels = in_channels + 5

        # Build feature layers
        layers = []
        prev_channels = feat_in_channels
        for out_channels in feat_channels:
            layers.append(nn.Linear(prev_channels, out_channels, bias=False))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            prev_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        self.out_channels = feat_channels[-1]

    def forward(self, pillar_features: torch.Tensor) -> torch.Tensor:
        """
        Encode pillar features.

        Args:
            pillar_features: (num_pillars, num_points, in_channels)

        Returns:
            Encoded features (num_pillars, out_channels)
        """
        num_pillars, num_points, num_features = pillar_features.shape

        # Augment features with geometric encodings
        features_aug = self.augment_features(pillar_features)

        # Reshape for processing: (num_pillars * num_points, feat_channels)
        features_flat = features_aug.view(-1, features_aug.shape[-1])

        # Extract features
        features_encoded = self.encoder(features_flat)

        # Reshape back: (num_pillars, num_points, out_channels)
        features_encoded = features_encoded.view(num_pillars, num_points, -1)

        # Max pooling over points
        features_pooled = torch.max(features_encoded, dim=1)[0]

        return features_pooled

    def augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Augment point features with geometric information.

        Adds:
        - Offset from pillar center (xc, yc, zc)
        - Offset from point mean (xp, yp)

        Args:
            features: (num_pillars, num_points, in_channels)

        Returns:
            Augmented features (num_pillars, num_points, in_channels + 5)
        """
        # For simplicity, compute offsets from local statistics
        # In full implementation, these would use actual pillar positions

        # Compute point mean within each pillar
        point_mean = features[:, :, :3].mean(dim=1, keepdim=True)

        # Offset from mean
        offset_mean = features[:, :, :2] - point_mean[:, :, :2]

        # Offset from center (simplified: use mean as center)
        offset_center = features[:, :, :3] - point_mean

        # Concatenate features
        features_aug = torch.cat([
            features,  # Original features
            offset_center,  # xc, yc, zc
            offset_mean,  # xp, yp
        ], dim=-1)

        return features_aug


class Backbone2D(nn.Module):
    """
    2D CNN backbone for processing BEV features.

    Uses multi-scale feature extraction with upsampling and concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        layer_nums: List[int],
        layer_strides: List[int],
        num_filters: List[int],
    ):
        """
        Initialize 2D backbone.

        Args:
            in_channels: Number of input channels
            layer_nums: Number of layers per block
            layer_strides: Stride for each block
            num_filters: Number of filters per block
        """
        super().__init__()

        assert len(layer_nums) == len(layer_strides) == len(num_filters)

        # Build encoder blocks
        self.blocks = nn.ModuleList()
        prev_channels = in_channels

        for i, (num_layers, stride, filters) in enumerate(
            zip(layer_nums, layer_strides, num_filters)
        ):
            block = self._make_block(prev_channels, filters, num_layers, stride)
            self.blocks.append(block)
            prev_channels = filters

        # Build decoder (upsampling) blocks
        self.deblocks = nn.ModuleList()
        for i, (stride, filters) in enumerate(zip(layer_strides, num_filters)):
            deblock = nn.Sequential(
                nn.ConvTranspose2d(
                    filters,
                    filters,
                    stride,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(filters),
                nn.ReLU(inplace=True),
            )
            self.deblocks.append(deblock)

        self.out_channels = sum(num_filters)

    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        stride: int,
    ) -> nn.Module:
        """Create a residual block."""
        layers = []

        # First layer with stride
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Remaining layers
        for _ in range(num_layers - 1):
            layers.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features.

        Args:
            x: (B, C, H, W) BEV features

        Returns:
            Multi-scale features (B, sum(num_filters), H', W')
        """
        features = []

        # Encode
        for block in self.blocks:
            x = block(x)
            features.append(x)

        # Decode and upsample
        upsampled_features = []
        for feat, deblock in zip(features, self.deblocks):
            upsampled = deblock(feat)
            upsampled_features.append(upsampled)

        # Concatenate multi-scale features
        output = torch.cat(upsampled_features, dim=1)

        return output


class DetectionHead(nn.Module):
    """
    Detection head for predicting 3D bounding boxes.

    Predicts:
    - Class scores
    - Box parameters (x, y, z, w, l, h, yaw)
    - Direction classification (for resolving yaw ambiguity)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 2,
    ):
        """
        Initialize detection head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of object classes
            num_anchors: Number of anchors per location
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.cls_head = nn.Conv2d(
            256,
            num_anchors * num_classes,
            1,
        )

        # Box regression head
        self.box_head = nn.Conv2d(
            256,
            num_anchors * 7,  # x, y, z, w, l, h, yaw
            1,
        )

        # Direction classification head
        self.dir_head = nn.Conv2d(
            256,
            num_anchors * 2,  # Binary classification
            1,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict 3D bounding boxes.

        Args:
            x: (B, C, H, W) spatial features

        Returns:
            Dictionary with predictions
        """
        # Shared features
        features = self.shared_conv(x)

        # Predictions
        cls_preds = self.cls_head(features)
        box_preds = self.box_head(features)
        dir_preds = self.dir_head(features)

        # Reshape predictions
        B, _, H, W = x.shape

        cls_preds = cls_preds.view(B, self.num_anchors, self.num_classes, H, W)
        cls_preds = cls_preds.permute(0, 3, 4, 1, 2).contiguous()

        box_preds = box_preds.view(B, self.num_anchors, 7, H, W)
        box_preds = box_preds.permute(0, 3, 4, 1, 2).contiguous()

        dir_preds = dir_preds.view(B, self.num_anchors, 2, H, W)
        dir_preds = dir_preds.permute(0, 3, 4, 1, 2).contiguous()

        return {
            'cls_preds': cls_preds,
            'box_preds': box_preds,
            'dir_preds': dir_preds,
        }


def create_pillar_input(
    points: torch.Tensor,
    max_points_per_pillar: int,
    max_pillars: int,
    pillar_size: Tuple[float, float, float],
    point_cloud_range: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert point cloud to pillar representation.

    Args:
        points: (B, N, 4) point clouds
        max_points_per_pillar: Maximum points per pillar
        max_pillars: Maximum number of pillars
        pillar_size: (x, y, z) pillar dimensions
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        pillar_features: (total_pillars, max_points_per_pillar, 4)
        pillar_coords: (total_pillars, 4) - batch_idx, z, y, x
    """
    batch_size = points.shape[0]
    device = points.device

    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = pillar_size

    # Calculate grid size
    nx = int((x_max - x_min) / vx)
    ny = int((y_max - y_min) / vy)

    all_pillar_features = []
    all_pillar_coords = []

    for b in range(batch_size):
        pc = points[b]  # (N, 4)

        # Filter points in range
        mask = (
            (pc[:, 0] >= x_min) & (pc[:, 0] < x_max) &
            (pc[:, 1] >= y_min) & (pc[:, 1] < y_max) &
            (pc[:, 2] >= z_min) & (pc[:, 2] < z_max)
        )
        pc = pc[mask]

        if len(pc) == 0:
            continue

        # Compute pillar coordinates
        x_coords = ((pc[:, 0] - x_min) / vx).long()
        y_coords = ((pc[:, 1] - y_min) / vy).long()

        # Clamp to grid
        x_coords = torch.clamp(x_coords, 0, nx - 1)
        y_coords = torch.clamp(y_coords, 0, ny - 1)

        # Unique pillar indices
        pillar_indices = y_coords * nx + x_coords
        unique_pillars = torch.unique(pillar_indices)

        # Limit number of pillars
        num_unique = min(len(unique_pillars), max_pillars)
        unique_pillars = unique_pillars[:num_unique]

        # Create pillar features
        for pillar_id in unique_pillars:
            # Get points in this pillar
            point_mask = (pillar_indices == pillar_id)
            pillar_points = pc[point_mask]

            # Limit points per pillar
            num_pts = min(len(pillar_points), max_points_per_pillar)

            # Create padded pillar
            pillar_feat = torch.zeros(
                (max_points_per_pillar, 4),
                dtype=pc.dtype,
                device=device,
            )
            pillar_feat[:num_pts] = pillar_points[:num_pts]

            all_pillar_features.append(pillar_feat)

            # Store coordinates
            y = pillar_id // nx
            x = pillar_id % nx
            coords = torch.tensor([b, 0, y, x], dtype=torch.long, device=device)
            all_pillar_coords.append(coords)

    if len(all_pillar_features) == 0:
        # Return empty tensors
        return (
            torch.zeros((1, max_points_per_pillar, 4), device=device),
            torch.zeros((1, 4), dtype=torch.long, device=device),
        )

    pillar_features = torch.stack(all_pillar_features)
    pillar_coords = torch.stack(all_pillar_coords)

    return pillar_features, pillar_coords
