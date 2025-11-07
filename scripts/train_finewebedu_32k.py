"""
Training Script for Pretokenized FineWebEdu 32k Data
Genovo Technologies Research Team
Lead: Oluwatosin Afolabi (afolabi@genovotech.com)

Optimized for:
- Pretokenized 32k chunk sequences
- L40S GPU (44 GB VRAM)
- Resonance Neural Networks
- Distributed training support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from resonance_nn import ResonanceNet
from resonance_nn.layers.resonance import optimize_resonance_model


class PreTokenizedFineWebEduDataset(Dataset):
    """
    Dataset for pretokenized FineWebEdu data at 32k chunks
    
    Expected data format:
    - Directory containing .npy or .pt files
    - Each file contains tokenized sequences of shape (n_sequences, 32768)
    """
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 32768,
        embedding_dim: int = 768,
        cache_size: int = 1000,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.cache_size = cache_size
        
        # Find all data files
        self.data_files = sorted(list(self.data_path.glob('*.npy')) + 
                                list(self.data_path.glob('*.pt')))
        
        if len(self.data_files) == 0:
            raise ValueError(f"No data files found in {data_path}")
        
        print(f"Found {len(self.data_files)} data files")
        
        # Load first file to get sequence count
        first_file = self._load_file(self.data_files[0])
        self.sequences_per_file = len(first_file)
        self.total_sequences = len(self.data_files) * self.sequences_per_file
        
        print(f"Sequences per file: {self.sequences_per_file}")
        print(f"Total sequences: {self.total_sequences:,}")
        
        # Simple LRU cache
        self._cache = {}
        self._cache_order = []
    
    def _load_file(self, filepath):
        """Load a data file"""
        if filepath.suffix == '.npy':
            return np.load(filepath)
        elif filepath.suffix == '.pt':
            return torch.load(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _get_file_cached(self, file_idx):
        """Get file with caching"""
        if file_idx in self._cache:
            return self._cache[file_idx]
        
        # Load file
        data = self._load_file(self.data_files[file_idx])
        
        # Add to cache
        self._cache[file_idx] = data
        self._cache_order.append(file_idx)
        
        # Evict old entries
        if len(self._cache) > self.cache_size:
            old_idx = self._cache_order.pop(0)
            del self._cache[old_idx]
        
        return data
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Calculate file and sequence index
        file_idx = idx // self.sequences_per_file
        seq_idx = idx % self.sequences_per_file
        
        # Load file (cached)
        data = self._get_file_cached(file_idx)
        
        # Get sequence
        sequence = data[seq_idx]
        
        # Convert to tensor if needed
        if isinstance(sequence, np.ndarray):
            sequence = torch.from_numpy(sequence).long()
        
        # For language modeling, input is seq[:-1], target is seq[1:]
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        
        return {
            'input_ids': input_seq,
            'labels': target_seq,
        }


class ResonanceLanguageModel(nn.Module):
    """
    Language model wrapper for Resonance Neural Networks
    """
    
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 768,
        num_frequencies: int = 128,
        hidden_dim: int = 768,
        num_layers: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 32768,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, input_dim)
        
        # Positional encoding (learnable for long context)
        self.pos_embedding = nn.Embedding(max_seq_len, input_dim)
        
        # Resonance backbone
        self.backbone = ResonanceNet(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Output head
        self.lm_head = nn.Linear(input_dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Layer norm
        self.norm = nn.LayerNorm(input_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)
        
        # Positional encoding
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        # Backbone
        x = self.backbone(x)
        
        # Norm
        x = self.norm(x)
        
        # LM head
        logits = self.lm_head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            'logits': logits,
            'loss': loss,
        }


def train(
    data_path: str,
    output_dir: str = 'checkpoints',
    # Model config
    vocab_size: int = 50257,  # GPT-2 vocab size
    model_dim: int = 2048,
    num_frequencies: int = 256,
    num_layers: int = 16,
    # Training config
    batch_size: int = 4,  # Per GPU
    gradient_accumulation_steps: int = 8,  # Effective batch size = 32
    max_seq_len: int = 32768,
    num_epochs: int = 3,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 2000,
    # Optimization
    use_amp: bool = True,
    gradient_checkpointing: bool = True,
    compile_model: bool = False,  # Disabled for complex tensors
    # Logging
    log_every: int = 10,
    save_every: int = 1000,
    eval_every: int = 500,
):
    """
    Train Resonance LM on pretokenized FineWebEdu data
    """
    print("="*80)
    print("RESONANCE NEURAL NETWORK TRAINING")
    print("="*80)
    print(f"Data path: {data_path}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {model_dim}D, {num_layers}L, {num_frequencies}F")
    print(f"Batch size: {batch_size} x {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}")
    print(f"Max sequence length: {max_seq_len:,}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load dataset
    print(f"\nLoading dataset from {data_path}...")
    dataset = PreTokenizedFineWebEduDataset(
        data_path=data_path,
        seq_length=max_seq_len,
        embedding_dim=model_dim,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    print("\nCreating model...")
    model = ResonanceLanguageModel(
        vocab_size=vocab_size,
        input_dim=model_dim,
        num_frequencies=num_frequencies,
        hidden_dim=model_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    
    # Optimize model
    if compile_model:
        print("Compiling model... (disabled for complex tensors)")
        # model = optimize_resonance_model(model, use_compile=True)
    
    model = model.to(device)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Scheduler
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps // 3,
        T_mult=2,
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    global_step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
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
                if global_step % log_every == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    })
                
                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint_path = os.path.join(output_dir, f'checkpoint-{global_step}.pt')
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item() * gradient_accumulation_steps,
                    }, checkpoint_path)
                    print(f"\n✓ Saved checkpoint: {checkpoint_path}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
    # Save final model
    final_path = os.path.join(output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Saved final model: {final_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Resonance LM on FineWebEdu')
    parser.add_argument('--data-path', type=str, required=True, help='Path to pretokenized data')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--vocab-size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--model-dim', type=int, default=2048, help='Model dimension')
    parser.add_argument('--num-frequencies', type=int, default=256, help='Number of frequencies')
    parser.add_argument('--num-layers', type=int, default=16, help='Number of layers')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--gradient-accumulation', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--max-seq-len', type=int, default=32768, help='Max sequence length')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    
    args = parser.parse_args()
    
    train(
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
    )
