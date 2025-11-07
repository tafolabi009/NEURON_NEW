"""
Sequence Modeling Example
Train Resonance Neural Network on sequence prediction task
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from resonance_nn import ResonanceNet
from resonance_nn.training import ResonanceTrainer, create_criterion


class SyntheticSequenceDataset(Dataset):
    """Synthetic sequence dataset for testing"""
    
    def __init__(self, num_samples: int, seq_len: int, input_dim: int, task: str = 'next_step'):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.task = task
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate sequence with temporal patterns
        t = torch.linspace(0, 4 * np.pi, self.seq_len)
        
        # Multiple frequency components
        seq = torch.zeros(self.seq_len, self.input_dim)
        for i in range(self.input_dim):
            freq = (i + 1) * 0.5
            seq[:, i] = torch.sin(freq * t) + 0.3 * torch.cos(freq * 2 * t)
            
        # Add noise
        seq += torch.randn_like(seq) * 0.1
        
        if self.task == 'next_step':
            # Predict next timestep
            input_seq = seq[:-1]
            target_seq = seq[1:]
        elif self.task == 'autoencoder':
            # Reconstruct input
            input_seq = seq
            target_seq = seq
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        return {'input': input_seq, 'target': target_seq}


def train_sequence_model(
    task: str = 'next_step',
    seq_len: int = 128,
    input_dim: int = 64,
    num_epochs: int = 20,
    batch_size: int = 32,
):
    """Train resonance network on sequence modeling"""
    
    print("=" * 80)
    print(f"SEQUENCE MODELING: {task.upper()}")
    print("=" * 80)
    print()
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Task: {task}")
    print(f"Sequence length: {seq_len}")
    print(f"Input dimension: {input_dim}")
    print()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SyntheticSequenceDataset(
        num_samples=1000,
        seq_len=seq_len,
        input_dim=input_dim,
        task=task,
    )
    
    val_dataset = SyntheticSequenceDataset(
        num_samples=200,
        seq_len=seq_len,
        input_dim=input_dim,
        task=task,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("Creating Resonance Neural Network...")
    model = ResonanceNet(
        input_dim=input_dim,
        num_frequencies=32,
        hidden_dim=input_dim,
        num_layers=4,
        holographic_capacity=500,
        dropout=0.1,
        use_multi_scale=False,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    complexity = model.get_complexity_estimate(seq_len)
    print(f"Complexity: {complexity['complexity_class']}")
    print()
    
    # Create trainer
    trainer = ResonanceTrainer(
        model=model,
        learning_rate=1e-3,
        gradient_clip=1.0,
        device=device,
    )
    
    criterion = create_criterion('regression')
    
    # Training loop
    print("Starting training...")
    print("-" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, criterion, epoch + 1)
        
        # Validate
        val_metrics = trainer.validate(val_loader, criterion)
        val_loss = val_metrics['val_loss']
        
        # Check gradient stability
        stability = trainer.check_gradient_stability()
        
        # Print metrics
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Gradients:  {'Stable ✓' if stability['all_stable'] else 'Unstable ✗'}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint('best_model.pt', epoch)
            print(f"  → Best model saved")
        
        print()
    
    print("-" * 80)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
    print()
    
    # Test gradient statistics
    print("Gradient Statistics:")
    print("-" * 80)
    grad_stats = trainer.get_gradient_statistics()
    for key, value in grad_stats.items():
        print(f"  {key}: {value:.6f}")
    print()
    
    # Test with holographic memory
    print("Testing with holographic memory...")
    print("-" * 80)
    
    model.eval()
    with torch.no_grad():
        # Get a test batch
        test_batch = next(iter(val_loader))
        test_input = test_batch['input'].to(device)
        test_target = test_batch['target'].to(device)
        
        # Without memory
        output_no_mem = model(test_input, use_memory=False)
        loss_no_mem = criterion(output_no_mem, test_target).item()
        
        # Store some patterns to memory
        for batch in train_loader:
            model.encode_to_memory(batch['input'].to(device))
            break
        
        # With memory
        output_with_mem = model(test_input, use_memory=True)
        loss_with_mem = criterion(output_with_mem, test_target).item()
        
        print(f"Loss without memory: {loss_no_mem:.6f}")
        print(f"Loss with memory:    {loss_with_mem:.6f}")
        
        improvement = (loss_no_mem - loss_with_mem) / loss_no_mem * 100
        print(f"Improvement: {improvement:+.2f}%")
    
    print()
    print("=" * 80)
    print("Sequence modeling complete!")
    print("=" * 80)
    
    return model, trainer


def main():
    # Train on next-step prediction
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Next-Step Prediction")
    print("=" * 80 + "\n")
    
    model1, trainer1 = train_sequence_model(
        task='next_step',
        seq_len=128,
        input_dim=64,
        num_epochs=15,
        batch_size=32,
    )
    
    # Train as autoencoder
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Sequence Autoencoding")
    print("=" * 80 + "\n")
    
    model2, trainer2 = train_sequence_model(
        task='autoencoder',
        seq_len=128,
        input_dim=64,
        num_epochs=15,
        batch_size=32,
    )
    
    print("\n" + "=" * 80)
    print("All experiments complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
