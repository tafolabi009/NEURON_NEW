#!/usr/bin/env python3
"""
Training Script for L40 GPU
Optimized for NVIDIA L40 with 48GB memory

Usage:
    python scripts/train_l40.py --model language --dataset wikitext --epochs 10
    python scripts/train_l40.py --model vision --dataset imagenet --batch-size 64
    python scripts/train_l40.py --model multimodal --mixed-precision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
import time
from typing import Optional, Dict
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train Resonance Models on L40 GPU')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       choices=['language', 'code', 'vision', 'audio', 'multimodal'],
                       help='Model type to train')
    parser.add_argument('--vocab-size', type=int, default=50000,
                       help='Vocabulary size for language/code models')
    parser.add_argument('--max-seq-len', type=int, default=4096,
                       help='Maximum sequence length')
    parser.add_argument('--embed-dim', type=int, default=768,
                       help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=12,
                       help='Number of layers')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup steps')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping')
    
    # Optimization
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for optimization')
    
    # Data
    parser.add_argument('--dataset', type=str, default='dummy',
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory')
    
    # Checkpointing
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--log-every', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    return parser.parse_args()


def create_model(args):
    """Create model based on type"""
    print(f"Creating {args.model} model...")
    
    if args.model == 'language':
        from resonance_nn.models.specialized.language_model import ResonanceLanguageModel
        model = ResonanceLanguageModel(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            max_seq_length=args.max_seq_len,
            use_long_context=(args.max_seq_len > 4096),
        )
    
    elif args.model == 'code':
        from resonance_nn.models.specialized.code_model import ResonanceCodeModel
        model = ResonanceCodeModel(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            max_seq_length=args.max_seq_len,
        )
    
    elif args.model == 'vision':
        from resonance_nn.models.specialized.vision_model import ResonanceVisionModel
        model = ResonanceVisionModel(
            num_classes=1000,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
        )
    
    elif args.model == 'audio':
        from resonance_nn.models.specialized.audio_model import ResonanceAudioModel
        model = ResonanceAudioModel(
            num_classes=50,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
        )
    
    elif args.model == 'multimodal':
        from resonance_nn.multimodal.fusion import MultiModalResonanceFusion
        modality_dims = {
            'text': args.embed_dim,
            'vision': args.embed_dim,
            'audio': args.embed_dim // 2,
        }
        model = MultiModalResonanceFusion(
            modality_dims=modality_dims,
            hidden_dim=args.hidden_dim,
            num_classes=1000,
        )
    
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    return model


def create_dummy_dataloader(args, model_type):
    """Create dummy dataloader for testing"""
    print("Creating dummy dataloader...")
    
    if model_type in ['language', 'code']:
        def collate_fn(batch):
            input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.max_seq_len))
            labels = torch.randint(0, args.vocab_size, (args.batch_size, args.max_seq_len))
            return {'input_ids': input_ids, 'labels': labels}
    
    elif model_type == 'vision':
        def collate_fn(batch):
            images = torch.randn(args.batch_size, 3, 224, 224)
            labels = torch.randint(0, 1000, (args.batch_size,))
            return {'images': images, 'labels': labels}
    
    elif model_type == 'audio':
        def collate_fn(batch):
            audio = torch.randn(args.batch_size, 22050 * 5)  # 5 seconds
            labels = torch.randint(0, 50, (args.batch_size,))
            return {'audio': audio, 'labels': labels}
    
    elif model_type == 'multimodal':
        def collate_fn(batch):
            return {
                'text': torch.randn(args.batch_size, 128, args.embed_dim),
                'vision': torch.randn(args.batch_size, 196, args.embed_dim),
                'audio': torch.randn(args.batch_size, 200, args.embed_dim // 2),
                'labels': torch.randint(0, 1000, (args.batch_size,)),
            }
    
    # Create dummy dataset
    dataset = [None] * 1000  # 1000 dummy samples
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Batch created in collate_fn
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    return dataloader


def train_epoch(model, dataloader, optimizer, scaler, device, args, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            if args.model in ['language', 'code']:
                logits = model(batch['input_ids'])
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    batch['labels'].view(-1)
                )
            
            elif args.model == 'vision':
                logits = model(batch['images'])
                loss = nn.CrossEntropyLoss()(logits, batch['labels'])
            
            elif args.model == 'audio':
                logits = model(batch['audio'])
                loss = nn.CrossEntropyLoss()(logits, batch['labels'])
            
            elif args.model == 'multimodal':
                modality_inputs = {k: v for k, v in batch.items() if k != 'labels'}
                logits = model(modality_inputs)
                loss = nn.CrossEntropyLoss()(logits, batch['labels'])
            
            loss = loss / args.gradient_accumulation
        
        # Backward pass
        if args.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (step + 1) % args.gradient_accumulation == 0:
            if args.mixed_precision:
                scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            if args.mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation
        num_batches += 1
        
        # Logging
        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - start_time
            throughput = (step + 1) * args.batch_size / elapsed
            
            print(f"Epoch {epoch} | Step {step+1} | "
                  f"Loss: {total_loss/num_batches:.4f} | "
                  f"Throughput: {throughput:.1f} samples/sec")
        
        # Checkpointing
        if (step + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, step, args)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, step, args):
    """Save training checkpoint"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }
    
    path = output_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint to {path}")


def main():
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create model
    model = create_model(args)
    model = model.to(device)
    
    # Compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    
    # Create dataloader
    dataloader = create_dummy_dataloader(args, args.model)
    
    # Training loop
    print("\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print("="*80)
    
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, scaler, device, args, epoch)
        print(f"\nEpoch {epoch} completed | Average loss: {avg_loss:.4f}\n")
        
        # Save epoch checkpoint
        save_checkpoint(model, optimizer, epoch, 'final', args)
    
    print("="*80)
    print("✓ Training completed!")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
