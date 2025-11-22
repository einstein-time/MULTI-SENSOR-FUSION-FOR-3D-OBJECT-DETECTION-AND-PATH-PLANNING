"""
Training Pipeline for PointPillars

Handles model training with data loading, optimization, and checkpointing.
"""

from typing import Dict, Optional
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import yaml


class Trainer:
    """
    Trainer for 3D object detection model.

    Manages training loop, logging, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        log_interval: int = 10,
        scheduler: Optional[_LRScheduler] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Logging interval in iterations
            scheduler: Learning rate scheduler
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.scheduler = scheduler

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            points = batch['lidar_points'].to(self.device)
            gt_boxes = batch['gt_boxes_3d'].to(self.device)

            # Forward pass (simplified - needs pillar conversion)
            # In production, convert points to pillars here
            outputs = self.model(points)

            # Compute loss (simplified - needs targets preparation)
            loss_dict = self.criterion(outputs, {'gt_boxes': gt_boxes})
            loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        return total_loss / num_batches

    def validate(self) -> float:
        """
        Validate model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                points = batch['lidar_points'].to(self.device)
                gt_boxes = batch['gt_boxes_3d'].to(self.device)

                outputs = self.model(points)
                loss_dict = self.criterion(outputs, {'gt_boxes': gt_boxes})
                loss = loss_dict['total_loss']

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, num_epochs: int) -> None:
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

            # Save checkpoint
            self.save_checkpoint(val_loss)

    def save_checkpoint(self, val_loss: float) -> None:
        """
        Save model checkpoint.

        Args:
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'Saved best model with val_loss={val_loss:.4f}')

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f'Loaded checkpoint from epoch {self.epoch}')


def create_trainer_from_config(
    config_path: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Trainer:
    """
    Create trainer from configuration file.

    Args:
        config_path: Path to config YAML
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader

    Returns:
        Configured trainer
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_config = config['training']

    # Create optimizer
    if train_config['optimizer']['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config['optimizer']['lr'],
            weight_decay=train_config['optimizer']['weight_decay'],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config['optimizer']['lr'],
        )

    # Create scheduler
    scheduler = None
    if train_config['lr_scheduler']['type'] == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_config['lr_scheduler']['max_lr'],
            epochs=train_config['num_epochs'],
            steps_per_epoch=len(train_loader),
        )

    # Create loss function
    from ..models.losses import PointPillarsLoss
    criterion = PointPillarsLoss(
        cls_weight=config['loss']['cls_weight'],
        box_weight=config['loss']['box_weight'],
        dir_weight=config['loss']['dir_weight'],
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=config['checkpoint']['save_dir'],
        log_interval=config['logging']['log_interval'],
        scheduler=scheduler,
    )

    return trainer
