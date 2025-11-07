"""
Data Preparation Utilities for FineWebEdu 32k Training
Genovo Technologies Research Team

Utilities for:
- Validating pretokenized data format
- Creating train/val splits
- Computing dataset statistics
- Preparing data for distributed training
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def validate_data_directory(data_path: str) -> Dict:
    """
    Validate pretokenized data directory
    
    Returns:
        Dict with validation results and statistics
    """
    print("="*80)
    print("VALIDATING DATA DIRECTORY")
    print("="*80)
    print(f"Path: {data_path}\n")
    
    data_path = Path(data_path)
    
    # Check directory exists
    if not data_path.exists():
        return {'valid': False, 'error': 'Directory does not exist'}
    
    # Find data files
    npy_files = sorted(list(data_path.glob('*.npy')))
    pt_files = sorted(list(data_path.glob('*.pt')))
    all_files = npy_files + pt_files
    
    if len(all_files) == 0:
        return {'valid': False, 'error': 'No .npy or .pt files found'}
    
    print(f"Found {len(all_files)} files ({len(npy_files)} .npy, {len(pt_files)} .pt)")
    
    # Load first file to check format
    first_file = all_files[0]
    print(f"\nInspecting: {first_file.name}")
    
    if first_file.suffix == '.npy':
        data = np.load(first_file)
    else:
        data = torch.load(first_file)
    
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    
    # Validate shape
    if len(data.shape) != 2:
        return {'valid': False, 'error': f'Expected 2D array, got {len(data.shape)}D'}
    
    num_sequences, seq_length = data.shape
    print(f"  Sequences per file: {num_sequences}")
    print(f"  Sequence length: {seq_length:,}")
    
    # Check all files have same shape
    print("\nValidating all files...")
    total_sequences = 0
    for i, filepath in enumerate(tqdm(all_files)):
        try:
            if filepath.suffix == '.npy':
                file_data = np.load(filepath)
            else:
                file_data = torch.load(filepath)
            
            if file_data.shape != data.shape:
                return {
                    'valid': False,
                    'error': f'Shape mismatch in {filepath.name}: expected {data.shape}, got {file_data.shape}'
                }
            
            total_sequences += file_data.shape[0]
        except Exception as e:
            return {'valid': False, 'error': f'Error loading {filepath.name}: {str(e)}'}
    
    # Calculate statistics
    print("\nCalculating statistics...")
    
    # Sample first file for token stats
    if isinstance(data, np.ndarray):
        sample_data = data
    else:
        sample_data = data.numpy()
    
    vocab_size = int(sample_data.max()) + 1
    unique_tokens = len(np.unique(sample_data))
    avg_token_value = float(sample_data.mean())
    
    results = {
        'valid': True,
        'num_files': len(all_files),
        'sequences_per_file': num_sequences,
        'total_sequences': total_sequences,
        'sequence_length': seq_length,
        'vocab_size': vocab_size,
        'unique_tokens': unique_tokens,
        'avg_token_value': avg_token_value,
        'file_format': first_file.suffix,
        'total_tokens': total_sequences * seq_length,
    }
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"✓ Valid: {results['valid']}")
    print(f"✓ Files: {results['num_files']}")
    print(f"✓ Total sequences: {results['total_sequences']:,}")
    print(f"✓ Sequence length: {results['sequence_length']:,}")
    print(f"✓ Total tokens: {results['total_tokens']:,} ({results['total_tokens']/1e9:.2f}B)")
    print(f"✓ Vocab size: {results['vocab_size']:,}")
    print(f"✓ Unique tokens: {results['unique_tokens']:,}")
    print(f"✓ Avg token value: {results['avg_token_value']:.2f}")
    print("="*80)
    
    return results


def create_train_val_split(
    data_path: str,
    output_path: str,
    val_ratio: float = 0.01,
    seed: int = 42,
):
    """
    Create train/val split by organizing files into subdirectories
    """
    print("="*80)
    print("CREATING TRAIN/VAL SPLIT")
    print("="*80)
    
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Find all files
    all_files = sorted(list(data_path.glob('*.npy')) + list(data_path.glob('*.pt')))
    print(f"Total files: {len(all_files)}")
    
    # Shuffle with seed
    np.random.seed(seed)
    indices = np.random.permutation(len(all_files))
    
    # Split
    num_val = max(1, int(len(all_files) * val_ratio))
    val_indices = set(indices[:num_val])
    
    print(f"Train files: {len(all_files) - num_val}")
    print(f"Val files: {num_val}")
    
    # Create directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy/symlink files
    print("\nOrganizing files...")
    for i, filepath in enumerate(tqdm(all_files)):
        target_dir = val_dir if i in val_indices else train_dir
        target_path = target_dir / filepath.name
        
        # Create symlink (more efficient than copying)
        if not target_path.exists():
            target_path.symlink_to(filepath.absolute())
    
    print("\n✓ Split complete!")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    
    return {
        'train_dir': str(train_dir),
        'val_dir': str(val_dir),
        'train_files': len(all_files) - num_val,
        'val_files': num_val,
    }


def estimate_training_time(
    total_tokens: int,
    batch_size: int,
    seq_length: int,
    throughput: float,  # tokens/sec
    num_epochs: int = 1,
) -> Dict:
    """
    Estimate training time based on throughput
    """
    tokens_per_batch = batch_size * seq_length
    batches_per_epoch = total_tokens // tokens_per_batch
    total_batches = batches_per_epoch * num_epochs
    
    seconds = (total_batches * tokens_per_batch) / throughput
    hours = seconds / 3600
    days = hours / 24
    
    return {
        'total_tokens': total_tokens,
        'batches_per_epoch': batches_per_epoch,
        'total_batches': total_batches,
        'estimated_seconds': seconds,
        'estimated_hours': hours,
        'estimated_days': days,
    }


def print_training_recommendations(validation_results: Dict):
    """
    Print recommended training configurations
    """
    print("\n" + "="*80)
    print("TRAINING RECOMMENDATIONS")
    print("="*80)
    
    total_tokens = validation_results['total_tokens']
    seq_length = validation_results['sequence_length']
    
    # Small model (good for testing)
    print("\n1. SMALL MODEL (Testing)")
    print("   " + "-"*40)
    print("   python scripts/train_finewebedu_32k.py \\")
    print(f"     --data-path /path/to/data \\")
    print("     --model-dim 768 \\")
    print("     --num-frequencies 64 \\")
    print("     --num-layers 6 \\")
    print("     --batch-size 4 \\")
    print("     --gradient-accumulation 8")
    
    # Medium model
    print("\n2. MEDIUM MODEL (50M-200M params)")
    print("   " + "-"*40)
    print("   python scripts/train_finewebedu_32k.py \\")
    print(f"     --data-path /path/to/data \\")
    print("     --model-dim 1024 \\")
    print("     --num-frequencies 128 \\")
    print("     --num-layers 12 \\")
    print("     --batch-size 4 \\")
    print("     --gradient-accumulation 8")
    
    # Large model
    print("\n3. LARGE MODEL (500M-1B params)")
    print("   " + "-"*40)
    print("   python scripts/train_finewebedu_32k.py \\")
    print(f"     --data-path /path/to/data \\")
    print("     --model-dim 2048 \\")
    print("     --num-frequencies 256 \\")
    print("     --num-layers 16 \\")
    print("     --batch-size 2 \\")
    print("     --gradient-accumulation 16")
    
    # Distributed training
    print("\n4. DISTRIBUTED TRAINING (Multi-GPU)")
    print("   " + "-"*40)
    print("   torchrun --nproc_per_node=4 \\")
    print("     scripts/train_finewebedu_32k_distributed.py \\")
    print(f"     --data-path /path/to/data \\")
    print("     --model-dim 2048 \\")
    print("     --num-frequencies 256 \\")
    print("     --num-layers 24 \\")
    print("     --batch-size 1 \\")
    print("     --gradient-accumulation 16 \\")
    print("     --wandb")
    
    # Time estimates
    print("\n" + "="*80)
    print("TIME ESTIMATES")
    print("="*80)
    
    # Assuming L40S throughput ~50k tokens/sec for 32k context
    throughput = 50000  # tokens/sec
    
    configs = [
        ("Small (batch=32, 1 GPU)", 32),
        ("Medium (batch=32, 1 GPU)", 32),
        ("Large (batch=32, 1 GPU)", 32),
        ("Distributed (batch=64, 4 GPUs)", 64),
    ]
    
    for name, batch_size in configs:
        est = estimate_training_time(
            total_tokens=total_tokens,
            batch_size=batch_size,
            seq_length=seq_length,
            throughput=throughput * (4 if '4 GPUs' in name else 1),
            num_epochs=1,
        )
        print(f"\n{name}:")
        print(f"  Batches: {est['batches_per_epoch']:,}")
        print(f"  Time: {est['estimated_hours']:.1f} hours ({est['estimated_days']:.2f} days)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data preparation utilities')
    parser.add_argument('command', choices=['validate', 'split'])
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--val-ratio', type=float, default=0.01)
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        results = validate_data_directory(args.data_path)
        if results['valid']:
            print_training_recommendations(results)
            
            # Save results
            output_file = Path(args.data_path) / 'validation_results.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
    
    elif args.command == 'split':
        if not args.output_path:
            print("Error: --output-path required for split command")
            sys.exit(1)
        
        results = create_train_val_split(
            data_path=args.data_path,
            output_path=args.output_path,
            val_ratio=args.val_ratio,
        )
        
        # Save results
        output_file = Path(args.output_path) / 'split_info.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
