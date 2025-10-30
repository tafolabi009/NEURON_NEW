"""
Complete training script for NEURONSv2
Run with: python train_complete.py or torchrun --nproc_per_node=4 train_complete.py
"""

import torch
from torch.utils.data import Dataset
import argparse
from pathlib import Path

from neurons.neuronsv2_unified import (
    NEURONSv2Model,
    NEURONSv2Config,
    create_neuronsv2_small,
    create_neuronsv2_medium,
    create_neuronsv2_large,
)
from neurons.training.trainer import NEURONSv2Trainer, TrainingConfig


class SimpleTextDataset(Dataset):
    """Simple dataset for text generation"""
    
    def __init__(self, data_path: str, max_seq_length: int = 2048):
        self.max_seq_length = max_seq_length
        
        # Load data (placeholder - implement your data loading)
        # For now, create dummy data
        self.data = torch.randint(0, 50257, (1000, max_seq_length))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.data[idx],  # For language modeling
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train NEURONSv2')
    
    # Model
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['small', 'medium', 'large', 'custom'],
                        help='Model size')
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='Hidden size (for custom model)')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of layers (for custom model)')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum training steps')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Warmup steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    
    # Optimization
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--precision', type=str, default='bf16',
                        choices=['fp16', 'bf16'],
                        help='Precision for mixed precision')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='Use gradient checkpointing')
    parser.add_argument('--compile', action='store_true',
                        help='Compile model with torch.compile()')
    
    # Distributed
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save interval')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='neuronsv2',
                        help='WandB project name')
    
    # Data
    parser.add_argument('--data_path', type=str, default='',
                        help='Path to training data')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("NEURONSv2 Training")
    print("="*80)
    
    # Create model
    if args.model_size == 'small':
        model = create_neuronsv2_small()
        print("Created small model (124M parameters)")
    elif args.model_size == 'medium':
        model = create_neuronsv2_medium()
        print("Created medium model (350M parameters)")
    elif args.model_size == 'large':
        model = create_neuronsv2_large()
        print("Created large model (774M parameters)")
    else:
        config = NEURONSv2Config(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            max_seq_length=args.max_seq_length,
        )
        model = NEURONSv2Model(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Created custom model ({num_params/1e6:.0f}M parameters)")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SimpleTextDataset(args.data_path, args.max_seq_length)
    eval_dataset = SimpleTextDataset(args.data_path, args.max_seq_length)
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval: {len(eval_dataset)} examples")
    
    # Create training config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_mixed_precision=args.mixed_precision,
        precision=args.precision,
        use_gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile,
        distributed=(args.local_rank != -1),
        local_rank=args.local_rank,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        max_seq_length=args.max_seq_length,
    )
    
    # Create trainer
    trainer = NEURONSv2Trainer(
        config=training_config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
