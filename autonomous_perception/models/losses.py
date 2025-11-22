"""
Loss Functions for 3D Object Detection

Implements multi-task loss for PointPillars including:
- Focal Loss for classification (handles class imbalance)
- Smooth L1 Loss for box regression
- Cross-entropy for direction classification
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses on hard negatives.
    This is especially useful in object detection where most locations
    are background.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive examples
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: (N, num_classes) logits
            targets: (N,) target class indices
            weights: (N,) optional sample weights

        Returns:
            Loss value
        """
        # Convert to probabilities
        probs = torch.sigmoid(predictions)

        # Get target probabilities
        num_classes = predictions.shape[-1]
        targets_one_hot = F.one_hot(targets.long(), num_classes).float()

        # Compute focal weight
        pt = torch.where(targets_one_hot == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets_one_hot,
            reduction='none',
        )

        # Apply focal weight and alpha
        loss = focal_weight * bce_loss
        loss = torch.where(
            targets_one_hot == 1,
            self.alpha * loss,
            (1 - self.alpha) * loss,
        )

        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss for box regression.

    Combines L1 and L2 loss for robust regression:
    - L2 loss for small errors (smooth gradients)
    - L1 loss for large errors (robust to outliers)
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = 'mean',
    ):
        """
        Initialize Smooth L1 Loss.

        Args:
            beta: Threshold for switching between L1 and L2
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute smooth L1 loss.

        Args:
            predictions: (N, D) predicted values
            targets: (N, D) target values
            weights: (N,) optional sample weights

        Returns:
            Loss value
        """
        diff = torch.abs(predictions - targets)

        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta,
        )

        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PointPillarsLoss(nn.Module):
    """
    Multi-task loss for PointPillars.

    Combines:
    1. Classification loss (Focal Loss)
    2. Box regression loss (Smooth L1)
    3. Direction classification loss (Cross-entropy)
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        box_weight: float = 2.0,
        dir_weight: float = 0.2,
        num_classes: int = 10,
    ):
        """
        Initialize PointPillars loss.

        Args:
            cls_weight: Weight for classification loss
            box_weight: Weight for box regression loss
            dir_weight: Weight for direction classification loss
            num_classes: Number of object classes
        """
        super().__init__()

        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.dir_weight = dir_weight
        self.num_classes = num_classes

        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.smooth_l1_loss = SmoothL1Loss(beta=1.0)
        self.dir_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            predictions: Dictionary with:
                - cls_preds: (B, H, W, num_anchors, num_classes)
                - box_preds: (B, H, W, num_anchors, 7)
                - dir_preds: (B, H, W, num_anchors, 2)

            targets: Dictionary with:
                - cls_targets: (B, H, W, num_anchors)
                - box_targets: (B, H, W, num_anchors, 7)
                - dir_targets: (B, H, W, num_anchors)
                - positive_mask: (B, H, W, num_anchors)

        Returns:
            Dictionary with loss components and total loss
        """
        # Extract predictions
        cls_preds = predictions['cls_preds']
        box_preds = predictions['box_preds']
        dir_preds = predictions['dir_preds']

        # Extract targets
        cls_targets = targets['cls_targets']
        box_targets = targets['box_targets']
        dir_targets = targets['dir_targets']
        positive_mask = targets['positive_mask']

        # Flatten tensors for loss computation
        batch_size = cls_preds.shape[0]

        cls_preds_flat = cls_preds.view(-1, self.num_classes)
        box_preds_flat = box_preds.view(-1, 7)
        dir_preds_flat = dir_preds.view(-1, 2)

        cls_targets_flat = cls_targets.view(-1)
        box_targets_flat = box_targets.view(-1, 7)
        dir_targets_flat = dir_targets.view(-1)
        positive_mask_flat = positive_mask.view(-1)

        # Classification loss (all anchors)
        cls_loss = self.focal_loss(cls_preds_flat, cls_targets_flat)

        # Box regression loss (only positive anchors)
        num_positives = positive_mask_flat.sum().clamp(min=1.0)

        if positive_mask_flat.sum() > 0:
            box_loss = self.smooth_l1_loss(
                box_preds_flat[positive_mask_flat],
                box_targets_flat[positive_mask_flat],
            )
        else:
            box_loss = torch.tensor(0.0, device=cls_preds.device)

        # Direction loss (only positive anchors)
        if positive_mask_flat.sum() > 0:
            dir_loss = self.dir_loss(
                dir_preds_flat[positive_mask_flat],
                dir_targets_flat[positive_mask_flat].long(),
            )
        else:
            dir_loss = torch.tensor(0.0, device=cls_preds.device)

        # Total loss
        total_loss = (
            self.cls_weight * cls_loss +
            self.box_weight * box_loss +
            self.dir_weight * dir_loss
        )

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'dir_loss': dir_loss,
            'num_positives': num_positives,
        }


def compute_iou_3d_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 3D IoU loss for boxes.

    This is an alternative to regression loss that directly optimizes IoU.

    Args:
        pred_boxes: (N, 7) predicted boxes
        target_boxes: (N, 7) target boxes

    Returns:
        IoU loss
    """
    # Simplified: use GIoU or DIoU in production
    # For now, return placeholder
    return torch.tensor(0.0, device=pred_boxes.device)


def encode_box_targets(
    gt_boxes: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """
    Encode ground truth boxes relative to anchors.

    This converts absolute box parameters to residuals relative to anchors,
    which are easier to learn.

    Args:
        gt_boxes: (N, 7) ground truth boxes [x, y, z, w, l, h, yaw]
        anchors: (N, 7) anchor boxes

    Returns:
        Encoded box targets (N, 7)
    """
    # Extract box parameters
    gt_x, gt_y, gt_z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
    gt_w, gt_l, gt_h = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
    gt_yaw = gt_boxes[:, 6]

    anc_x, anc_y, anc_z = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    anc_w, anc_l, anc_h = anchors[:, 3], anchors[:, 4], anchors[:, 5]
    anc_yaw = anchors[:, 6]

    # Encode as residuals
    # Position: normalize by anchor dimensions
    dx = (gt_x - anc_x) / anc_w
    dy = (gt_y - anc_y) / anc_l
    dz = (gt_z - anc_z) / anc_h

    # Size: log-space encoding
    dw = torch.log(gt_w / anc_w)
    dl = torch.log(gt_l / anc_l)
    dh = torch.log(gt_h / anc_h)

    # Rotation: direct difference
    dyaw = gt_yaw - anc_yaw

    encoded = torch.stack([dx, dy, dz, dw, dl, dh, dyaw], dim=1)

    return encoded


def decode_box_predictions(
    box_preds: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """
    Decode box predictions from anchor residuals to absolute coordinates.

    Args:
        box_preds: (N, 7) predicted box residuals
        anchors: (N, 7) anchor boxes

    Returns:
        Decoded boxes (N, 7) in absolute coordinates
    """
    # Extract predictions
    dx, dy, dz = box_preds[:, 0], box_preds[:, 1], box_preds[:, 2]
    dw, dl, dh = box_preds[:, 3], box_preds[:, 4], box_preds[:, 5]
    dyaw = box_preds[:, 6]

    # Extract anchor parameters
    anc_x, anc_y, anc_z = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    anc_w, anc_l, anc_h = anchors[:, 3], anchors[:, 4], anchors[:, 5]
    anc_yaw = anchors[:, 6]

    # Decode position
    x = dx * anc_w + anc_x
    y = dy * anc_l + anc_y
    z = dz * anc_h + anc_z

    # Decode size
    w = torch.exp(dw) * anc_w
    l = torch.exp(dl) * anc_l
    h = torch.exp(dh) * anc_h

    # Decode rotation
    yaw = dyaw + anc_yaw

    decoded = torch.stack([x, y, z, w, l, h, yaw], dim=1)

    return decoded
