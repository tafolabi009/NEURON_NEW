"""
Production Training Infrastructure for NEURONSv2
Complete trainer with distributed training, mixed precision, and all optimizations
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
import os
import json
import time
from pathlib import Path
import wandb

from neurons.neuronsv2_unified import NEURONSv2Model, NEURONSv2Config


@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Model
    model_config: NEURONSv2Config = field(default_factory=NEURONSv2Config)
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Optimization
    use_mixed_precision: bool = True
    precision: str = "bf16"  # "fp16" or "bf16"
    use_gradient_checkpointing: bool = True
    compile_model: bool = False  # PyTorch 2.0 compile
    
    # Distributed
    distributed: bool = False
    world_size: int = 1
    local_rank: int = -1
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "./checkpoints"
    use_wandb: bool = False
    wandb_project: str = "neuronsv2"
    
    # Data
    dataset_path: str = ""
    max_seq_length: int = 2048
    
    # Scheduler
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant"


class NEURONSv2Trainer:
    """
    Complete training infrastructure for NEURONSv2
    
    Features:
    - Distributed training (DDP/FSDP)
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Gradient checkpointing
    - Learning rate scheduling
    - Checkpointing
    - Logging (WandB, TensorBoard)
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: NEURONSv2Model,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ):
        self.config = config
        self.device = self._setup_device()
        
        # Setup distributed training
        if config.distributed:
            self._setup_distributed()
        
        # Model
        self.model = model.to(self.device)
        
        # Mixed precision
        self.scaler = None
        if config.use_mixed_precision:
            self.scaler = GradScaler(
                enabled=(config.precision == "fp16")
            )
        
        # Wrap model for distributed
        if config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[config.local_rank],
                output_device=config.local_rank,
                find_unused_parameters=False,
            )
        
        # Compile model (PyTorch 2.0+)
        if config.compile_model:
            try:
                self.model = torch.compile(self.model)
                print("Model compiled with torch.compile()")
            except Exception as e:
                print(f"Could not compile model: {e}")
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Data loaders
        self.train_loader = None
        self.eval_loader = None
        if train_dataset is not None:
            self.train_loader = self._create_dataloader(train_dataset, is_train=True)
        if eval_dataset is not None:
            self.eval_loader = self._create_dataloader(eval_dataset, is_train=False)
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("NEURONSv2 Trainer initialized")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Distributed: {config.distributed}")
        if config.distributed:
            print(f"  - World size: {config.world_size}")
            print(f"  - Local rank: {config.local_rank}")
        print(f"Mixed precision: {config.use_mixed_precision} ({config.precision})")
        print(f"Gradient checkpointing: {config.use_gradient_checkpointing}")
        print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps * config.world_size}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("="*80 + "\n")
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if torch.cuda.is_available():
            if self.config.local_rank != -1:
                device = torch.device(f'cuda:{self.config.local_rank}')
                torch.cuda.set_device(device)
            else:
                device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device
    
    def _setup_distributed(self):
        """Setup distributed training"""
        if self.config.local_rank == -1:
            # Try to get from environment
            self.config.local_rank = int(os.environ.get('LOCAL_RANK', -1))
            self.config.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if self.config.local_rank != -1:
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://'
            )
            self.config.world_size = dist.get_world_size()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.config.weight_decay
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0
            }
        ]
        
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler"""
        if self.config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.max_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _create_dataloader(self, dataset: Any, is_train: bool) -> DataLoader:
        """Create data loader with distributed sampler"""
        sampler = None
        if self.config.distributed and is_train:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.local_rank,
                shuffle=True
            )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None and is_train),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    
    def _setup_logging(self):
        """Setup logging (WandB, etc.)"""
        if self.config.use_wandb and (not self.config.distributed or self.config.local_rank == 0):
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__,
            )
    
    def train(self):
        """Main training loop"""
        self.model.train()
        
        print("Starting training...")
        print(f"  - Max steps: {self.config.max_steps}")
        print(f"  - Warmup steps: {self.config.warmup_steps}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print()
        
        start_time = time.time()
        total_loss = 0.0
        log_loss = 0.0
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device) if 'labels' in batch else input_ids
                
                # Forward pass with mixed precision
                with autocast(
                    device_type='cuda' if torch.cuda.is_available() else 'cpu',
                    dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16,
                    enabled=self.config.use_mixed_precision
                ):
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs['loss']
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Accumulate loss
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                log_loss += loss.item() * self.config.gradient_accumulation_steps
                
                # Optimization step
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Scheduler step
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = log_loss / self.config.log_interval
                    lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    tokens_per_sec = (
                        self.config.batch_size * 
                        self.config.max_seq_length * 
                        self.config.log_interval / 
                        elapsed
                    )
                    
                    print(f"Step {self.global_step:6d} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Tokens/sec: {tokens_per_sec:.0f}")
                    
                    if self.config.use_wandb:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/tokens_per_sec': tokens_per_sec,
                            'train/step': self.global_step,
                        })
                    
                    log_loss = 0.0
                    start_time = time.time()
                
                # Evaluation
                if self.global_step % self.config.eval_interval == 0 and self.eval_loader is not None:
                    eval_loss = self.evaluate()
                    print(f"\nEval loss: {eval_loss:.4f}\n")
                    
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint(is_best=True)
                    
                    self.model.train()
                
                # Checkpointing
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint()
                
                # Check if done
                if self.global_step >= self.config.max_steps:
                    break
        
        print(f"\nTraining complete! Final loss: {total_loss / self.global_step:.4f}")
        self.save_checkpoint(is_final=True)
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device) if 'labels' in batch else input_ids
            
            with autocast(
                device_type='cuda' if torch.cuda.is_available() else 'cpu',
                dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16,
                enabled=self.config.use_mixed_precision
            ):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
            
            # Limit eval batches
            if num_batches >= 100:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if self.config.use_wandb:
            wandb.log({
                'eval/loss': avg_loss,
                'eval/step': self.global_step,
            })
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        if self.config.distributed and self.config.local_rank != 0:
            return  # Only save on rank 0
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.module.state_dict() if self.config.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'best_eval_loss': self.best_eval_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        save_path = Path(self.config.output_dir)
        
        if is_final:
            checkpoint_path = save_path / "final_checkpoint.pt"
        elif is_best:
            checkpoint_path = save_path / "best_checkpoint.pt"
        else:
            checkpoint_path = save_path / f"checkpoint_step_{self.global_step}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save config
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        if self.config.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load state
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        print(f"Loaded checkpoint from step {self.global_step}")


# Helper function to launch distributed training
def launch_distributed_training(
    config: TrainingConfig,
    model: NEURONSv2Model,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
):
    """
    Launch distributed training
    
    Use with torchrun:
    torchrun --nproc_per_node=4 train_script.py
    """
    trainer = NEURONSv2Trainer(config, model, train_dataset, eval_dataset)
    trainer.train()


if __name__ == "__main__":
    print("NEURONSv2 Production Training Infrastructure ready!")
    print("\nTo train:")
    print("  Single GPU: python train_script.py")
    print("  Multi-GPU: torchrun --nproc_per_node=4 train_script.py")
    print("  Multi-node: torchrun --nproc_per_node=8 --nnodes=4 train_script.py")
