"""
Training Script for PointPillars 3D Object Detection

This script trains the PointPillars model on nuScenes dataset.

Usage:
    python train.py --config config/train_config.yaml
"""

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from data.nuscenes_loader import NuScenesDataset
from models.pointpillars import PointPillars
from training.trainer import Trainer, create_trainer_from_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PointPillars model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )
    return parser.parse_args()


def create_data_loaders(config: dict) -> tuple:
    """
    Create training and validation data loaders.

    Args:
        config: Configuration dictionary

    Returns:
        train_loader, val_loader
    """
    dataset_config = config['dataset']

    # Training dataset
    train_dataset = NuScenesDataset(
        dataroot=dataset_config['dataroot'],
        version=dataset_config['version'],
        split='train',
        sensors=dataset_config['sensors'],
    )

    # Validation dataset
    val_dataset = NuScenesDataset(
        dataroot=dataset_config['dataroot'],
        version=dataset_config['version'],
        split='val',
        sensors=dataset_config['sensors'],
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
    )

    return train_loader, val_loader


def create_model(config: dict) -> PointPillars:
    """
    Create PointPillars model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        PointPillars model
    """
    model_config = config['model']

    model = PointPillars(
        num_classes=model_config['num_classes'],
        max_points_per_pillar=model_config['pillar']['max_points_per_pillar'],
        max_pillars=model_config['pillar']['max_pillars'],
        pillar_size=tuple(model_config['pillar']['pillar_size']),
        point_cloud_range=model_config['point_cloud_range'],
    )

    return model


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        train_config = yaml.safe_load(f)

    # Also load model config
    model_config_path = Path(args.config).parent / 'model_config.yaml'
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Merge configs
    config = {**model_config, **train_config}

    print('=' * 80)
    print('Training PointPillars for 3D Object Detection')
    print('=' * 80)
    print(f'Device: {args.device}')
    print(f'Dataset: {config["dataset"]["dataroot"]}')
    print(f'Batch size: {config["training"]["batch_size"]}')
    print(f'Epochs: {config["training"]["num_epochs"]}')
    print('=' * 80)

    # Create data loaders
    print('Creating data loaders...')
    train_loader, val_loader = create_data_loaders(config)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')

    # Create model
    print('Creating model...')
    model = create_model(config)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Create trainer
    print('Creating trainer...')
    trainer = create_trainer_from_config(
        args.config,
        model,
        train_loader,
        val_loader,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        trainer.load_checkpoint(args.resume)

    # Train
    print('Starting training...')
    trainer.train(num_epochs=config['training']['num_epochs'])

    print('=' * 80)
    print('Training completed!')
    print(f'Best validation loss: {trainer.best_val_loss:.4f}')
    print(f'Checkpoints saved to: {trainer.checkpoint_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
