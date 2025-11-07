"""
Quick Start Example for Resonance Neural Networks
Demonstrates basic usage of the library
"""

import torch
from resonance_nn import ResonanceNet, ResonanceTrainer
from resonance_nn.training import create_criterion

def main():
    print("=" * 70)
    print("Resonance Neural Networks - Quick Start Example")
    print("=" * 70)
    print()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Configuration
    input_dim = 128
    seq_len = 64
    batch_size = 16
    num_epochs = 5
    
    # Create model
    print("Creating Resonance Neural Network...")
    model = ResonanceNet(
        input_dim=input_dim,
        num_frequencies=32,
        hidden_dim=128,
        num_layers=3,
        holographic_capacity=100,
        dropout=0.1,
    )
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Input dimension: {input_dim}")
    print(f"Number of layers: 3")
    print(f"Frequencies per layer: 32")
    print()
    
    # Get complexity estimate
    complexity = model.get_complexity_estimate(seq_len)
    print(f"Computational complexity: {complexity['complexity_class']}")
    print(f"Estimated operations: {complexity['total']:.0f}")
    print()
    
    # Create synthetic training data
    print("Generating synthetic training data...")
    train_data = []
    for _ in range(100):
        x = torch.randn(seq_len, input_dim)
        # Target is input with some transformation
        target = torch.sin(x * 0.5) + torch.cos(x * 0.3)
        train_data.append({'input': x, 'target': target})
    
    # Create data loader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    
    # Create trainer
    print("Initializing trainer...")
    trainer = ResonanceTrainer(
        model=model,
        learning_rate=1e-3,
        gradient_clip=1.0,
        device=device,
    )
    
    # Loss function
    criterion = create_criterion('regression')
    
    print("Starting training...")
    print("-" * 70)
    
    # Training loop
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(train_loader, criterion, epoch + 1)
        
        # Check gradient stability
        stability = trainer.check_gradient_stability()
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  Gradient stable: {stability['all_stable']}")
        print()
    
    print("-" * 70)
    print("Training complete!")
    print()
    
    # Test holographic memory
    print("Testing Holographic Memory...")
    print("-" * 70)
    
    # Store some patterns
    test_patterns = [torch.randn(input_dim) for _ in range(5)]
    for i, pattern in enumerate(test_patterns):
        model.holographic_memory.encode(pattern.unsqueeze(0))
        print(f"Stored pattern {i + 1}")
    
    # Reconstruct
    reconstructed = model.holographic_memory.reconstruct()
    print(f"\nReconstructed pattern shape: {reconstructed.shape}")
    
    # Test reconstruction fidelity
    fidelity = model.holographic_memory.get_reconstruction_fidelity(test_patterns[0].unsqueeze(0))
    print(f"Reconstruction fidelity: {fidelity:.4f}")
    
    capacity_util = model.holographic_memory.get_capacity_utilization()
    print(f"Memory capacity utilization: {capacity_util:.2%}")
    print()
    
    # Test inference
    print("Testing inference...")
    print("-" * 70)
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, seq_len, input_dim).to(device)
        output = model(test_input, use_memory=True)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
    print()
    
    print("=" * 70)
    print("Quick start example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
