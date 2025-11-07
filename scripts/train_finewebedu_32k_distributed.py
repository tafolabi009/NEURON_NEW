"""
Distributed Training Script for Pretokenized FineWebEdu 32k Data
Genovo Technologies Research Team
Lead: Oluwatosin Afolabi (afolabi@genovotech.com)

Features:
- Multi-GPU training with DDP
- Gradient accumulation
- Mixed precision
- Checkpointing
- WandB integration (optional)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_finewebedu_32k import (
    PreTokenizedFineWebEduDataset,
    ResonanceLanguageModel,
)


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_distributed(
    data_path: str,
    output_dir: str = 'checkpoints',
    # Model config
    vocab_size: int = 50257,
    model_dim: int = 2048,
    num_frequencies: int = 256,
    num_layers: int = 16,
    # Training config
    batch_size: int = 2,  # Per GPU
    gradient_accumulation_steps: int = 16,  # Effective batch = 2 * num_gpus * 16
    max_seq_len: int = 32768,
    num_epochs: int = 3,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 2000,
    # Optimization
    use_amp: bool = True,
    # Logging
    log_every: int = 10,
    save_every: int = 1000,
    use_wandb: bool = False,
):
    """
    Distributed training for Resonance LM
    """
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    if is_main:
        print("="*80)
        print("RESONANCE NEURAL NETWORK - DISTRIBUTED TRAINING")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Effective batch size: {batch_size * world_size * gradient_accumulation_steps}")
        print(f"Model: {model_dim}D, {num_layers}L, {num_frequencies}F")
    
    # Create output directory
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    
    # WandB initialization
    if use_wandb and is_main:
        import wandb
        wandb.init(
            project="resonance-finewebedu",
            config={
                'vocab_size': vocab_size,
                'model_dim': model_dim,
                'num_frequencies': num_frequencies,
                'num_layers': num_layers,
                'batch_size': batch_size * world_size * gradient_accumulation_steps,
                'max_seq_len': max_seq_len,
                'learning_rate': learning_rate,
            }
        )
    
    # Load dataset
    if is_main:
        print(f"\nLoading dataset from {data_path}...")
    
    dataset = PreTokenizedFineWebEduDataset(
        data_path=data_path,
        seq_length=max_seq_len,
        embedding_dim=model_dim,
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    if is_main:
        print("\nCreating model...")
    
    model = ResonanceLanguageModel(
        vocab_size=vocab_size,
        input_dim=model_dim,
        num_frequencies=num_frequencies,
        hidden_dim=model_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Scheduler
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps // 3,
        T_mult=2,
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Training loop
    if is_main:
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
    
    global_step = 0
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        
        if is_main:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}") if is_main else dataloader
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)
            
            # Forward pass
            with torch.cuda.amp.autocast() if use_amp else torch.cuda.amp.autocast(enabled=False):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / gradient_accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if is_main and global_step % log_every == 0:
                    if isinstance(pbar, tqdm):
                        pbar.set_postfix({
                            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                        })
                    
                    if use_wandb:
                        wandb.log({
                            'loss': loss.item() * gradient_accumulation_steps,
                            'learning_rate': scheduler.get_last_lr()[0],
                            'step': global_step,
                        })
                
                # Save checkpoint
                if is_main and global_step % save_every == 0:
                    checkpoint_path = os.path.join(output_dir, f'checkpoint-{global_step}.pt')
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item() * gradient_accumulation_steps,
                    }, checkpoint_path)
                    print(f"\n✓ Saved checkpoint: {checkpoint_path}")
        
        if is_main:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
    # Save final model
    if is_main:
        final_path = os.path.join(output_dir, 'final_model.pt')
        torch.save(model.module.state_dict(), final_path)
        print(f"\n✓ Saved final model: {final_path}")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
    
    cleanup_distributed()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed training for Resonance LM')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--vocab-size', type=int, default=50257)
    parser.add_argument('--model-dim', type=int, default=2048)
    parser.add_argument('--num-frequencies', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--max-seq-len', type=int, default=32768)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    
    args = parser.parse_args()
    
    train_distributed(
        data_path=args.data_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_frequencies=args.num_frequencies,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_seq_len=args.max_seq_len,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        use_amp=not args.no_amp,
        use_wandb=args.wandb,
    )
