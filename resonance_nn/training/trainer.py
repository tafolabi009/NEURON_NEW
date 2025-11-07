"""
Training utilities for Resonance Neural Networks
Includes stable gradient handling for oscillatory parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any
from tqdm import tqdm
import numpy as np


class ResonanceTrainer:
    """
    Trainer for Resonance Neural Networks with stable gradient handling
    
    Features:
    - Gradient clipping for oscillatory parameters
    - Gradient statistics monitoring
    - Adaptive learning rate for magnitude/phase parameters
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        phase_lr_multiplier: float = 0.1,  # Lower LR for phase parameters
    ):
        """
        Args:
            model: Resonance neural network model
            learning_rate: Base learning rate
            weight_decay: L2 regularization
            gradient_clip: Maximum gradient norm
            device: Device to train on
            phase_lr_multiplier: Learning rate multiplier for phase parameters
        """
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        
        # Separate parameter groups for magnitude and phase
        magnitude_params = []
        phase_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'magnitude' in name:
                magnitude_params.append(param)
            elif 'phase' in name:
                phase_params.append(param)
            else:
                other_params.append(param)
                
        # Optimizer with different learning rates
        self.optimizer = optim.AdamW([
            {'params': magnitude_params, 'lr': learning_rate, 'name': 'magnitude'},
            {'params': phase_params, 'lr': learning_rate * phase_lr_multiplier, 'name': 'phase'},
            {'params': other_params, 'lr': learning_rate, 'name': 'other'},
        ], weight_decay=weight_decay)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Statistics tracking
        self.train_losses = []
        self.gradient_stats = []
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: Callable,
    ) -> float:
        """
        Single training step
        
        Args:
            batch: Dictionary containing 'input' and 'target' tensors
            criterion: Loss function
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        inputs = batch['input'].to(self.device)
        targets = batch['target'].to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )
        
        # Collect gradient statistics
        if hasattr(self.model, 'get_gradient_stats'):
            grad_stats = self.model.get_gradient_stats()
            grad_stats['total_norm'] = total_norm.item()
            self.gradient_stats.append(grad_stats)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: Callable,
        epoch: int,
    ) -> float:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Average loss for epoch
        """
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch in pbar:
            loss = self.train_step(batch, criterion)
            epoch_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        
        # Step scheduler
        self.scheduler.step()
        
        return avg_loss
    
    def validate(
        self,
        dataloader: DataLoader,
        criterion: Callable,
    ) -> Dict[str, float]:
        """
        Validation loop
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())
                
        return {
            'val_loss': np.mean(val_losses),
            'val_loss_std': np.std(val_losses),
        }
    
    def get_gradient_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive gradient statistics
        
        Returns:
            Dictionary of gradient statistics
        """
        if not self.gradient_stats:
            return {}
            
        stats = {}
        
        # Aggregate statistics
        for key in self.gradient_stats[0].keys():
            values = [s[key] for s in self.gradient_stats if key in s]
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_min'] = np.min(values)
                
        return stats
    
    def check_gradient_stability(self) -> Dict[str, bool]:
        """
        Check if gradients are stable (no explosion)
        
        Returns:
            Dictionary of stability checks
        """
        stats = self.get_gradient_statistics()
        
        checks = {
            'magnitude_stable': stats.get('magnitude_grad_max_max', 0) < 10.0,
            'phase_stable': stats.get('phase_grad_max_max', 0) < 10.0,
            'total_stable': stats.get('total_norm_max', 0) < 100.0,
        }
        
        checks['all_stable'] = all(checks.values())
        
        return checks
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """
        Save training checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            **kwargs: Additional data to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'gradient_stats': self.gradient_stats,
            **kwargs,
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> int:
        """
        Load training checkpoint
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.gradient_stats = checkpoint.get('gradient_stats', [])
        
        return checkpoint['epoch']


class ResonanceAutoEncoderTrainer(ResonanceTrainer):
    """
    Specialized trainer for autoencoder models
    """
    
    def __init__(self, *args, reconstruction_weight: float = 1.0, 
                 latent_regularization: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_weight = reconstruction_weight
        self.latent_regularization = latent_regularization
        
    def train_step(self, batch: Dict[str, torch.Tensor], criterion: Callable) -> float:
        """
        Training step for autoencoder with latent regularization
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        inputs = batch['input'].to(self.device)
        
        # Forward pass (autoencoder returns reconstruction and latent)
        reconstruction, latent = self.model(inputs)
        
        # Reconstruction loss
        recon_loss = criterion(reconstruction, inputs)
        
        # Latent regularization (encourage small latent codes)
        latent_loss = torch.mean(latent ** 2)
        
        # Total loss
        loss = self.reconstruction_weight * recon_loss + self.latent_regularization * latent_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()


class ResonanceClassifierTrainer(ResonanceTrainer):
    """
    Specialized trainer for classification models
    """
    
    def validate(self, dataloader: DataLoader, criterion: Callable) -> Dict[str, float]:
        """
        Validation with accuracy metrics
        """
        self.model.eval()
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                
                # Accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return {
            'val_loss': np.mean(val_losses),
            'val_accuracy': 100.0 * correct / total,
        }


def create_criterion(task: str = 'regression', **kwargs) -> Callable:
    """
    Create loss function based on task
    
    Args:
        task: Task type ('regression', 'classification', 'autoencoder')
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function
    """
    if task == 'regression':
        return nn.MSELoss(**kwargs)
    elif task == 'classification':
        return nn.CrossEntropyLoss(**kwargs)
    elif task == 'autoencoder':
        return nn.MSELoss(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


def create_trainer(
    model: nn.Module,
    task: str = 'regression',
    **kwargs
) -> ResonanceTrainer:
    """
    Create appropriate trainer for task
    
    Args:
        model: Model to train
        task: Task type
        **kwargs: Additional trainer arguments
        
    Returns:
        Trainer instance
    """
    if task == 'autoencoder':
        return ResonanceAutoEncoderTrainer(model, **kwargs)
    elif task == 'classification':
        return ResonanceClassifierTrainer(model, **kwargs)
    else:
        return ResonanceTrainer(model, **kwargs)
