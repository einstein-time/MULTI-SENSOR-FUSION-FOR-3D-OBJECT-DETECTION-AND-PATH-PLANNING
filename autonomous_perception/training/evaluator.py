"""
Model Evaluator

Evaluates trained model on validation/test set.
"""

from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_ap, compute_map


class Evaluator:
    """
    Evaluator for 3D object detection model.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = 'cuda',
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            data_loader: Data loader
            device: Device to run on
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on dataset.

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for batch in tqdm(self.data_loader, desc='Evaluating'):
                points = batch['lidar_points'].to(self.device)
                gt_boxes = batch['gt_boxes_3d']

                # Forward pass
                outputs = self.model(points)

                # Post-process predictions
                predictions = self._postprocess(outputs)

                all_predictions.extend(predictions)
                all_ground_truths.extend([gt.numpy() for gt in gt_boxes])

        # Compute metrics
        metrics = {}
        metrics['mAP'] = compute_map(all_predictions, all_ground_truths)
        metrics['AP@0.5'] = compute_ap(all_predictions, all_ground_truths, 0.5)
        metrics['AP@0.7'] = compute_ap(all_predictions, all_ground_truths, 0.7)

        return metrics

    def _postprocess(self, outputs: Dict) -> List[np.ndarray]:
        """
        Post-process model outputs to get final predictions.

        Args:
            outputs: Model outputs

        Returns:
            List of predictions
        """
        # Simplified post-processing
        # In production, apply NMS and convert to absolute coordinates
        predictions = []

        # Extract predictions (placeholder)
        # Real implementation needs to decode boxes and apply NMS

        return predictions
