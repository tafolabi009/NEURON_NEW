"""
Example demonstrating the NEURONSv2 spiking neural network architecture.

This example shows the novel features:
- Spiking neurons (LIF dynamics)
- Dendritic computation
- Oscillatory communication (replaces attention)
- Hebbian plasticity (local learning)
"""

import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neurons import (
    NEURONSv2,
    NEURONSv2Config,
    setup_logging,
    create_data_loader,
)
import logging

def main():
    """Train NEURONSv2 on digit classification."""
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading digit dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Split and normalize
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)
    
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Configure NEURONSv2
    config = NEURONSv2Config(
        # Neuron dynamics
        tau_mem=20.0,           # Membrane time constant (ms)
        tau_syn=5.0,            # Synaptic time constant (ms)
        threshold=1.0,          # Spike threshold
        
        # Dendritic computation
        n_basal_dendrites=8,    # Number of basal branches
        n_apical_dendrites=4,   # Number of apical branches
        tau_dendrite=10.0,      # Dendritic time constant
        
        # Oscillatory dynamics
        natural_frequency=40.0, # Hz (gamma band)
        coupling_strength=0.1,  # Oscillator coupling
        
        # Hebbian plasticity
        learning_rate_fast=0.01,  # Fast Hebbian learning
        learning_rate_slow=0.001, # Slow Hebbian learning
        use_stdp=True,          # Spike-timing dependent plasticity
        use_bcm=True,           # BCM rule
        
        # Simulation
        dt=1.0,                 # Time step (ms)
    )
    
    # Create model
    logger.info("Creating NEURONSv2 model...")
    model = NEURONSv2(
        layer_sizes=[64, 128, 64, 10],  # Input: 64, Hidden: 128, 64, Output: 10
        config=config,
        use_dendrites=True,      # Enable dendritic computation
        use_oscillators=True,    # Enable oscillatory dynamics
        use_plasticity=True,     # Enable Hebbian plasticity
        output_mode='rate'       # Output firing rates
    ).to(device)
    
    logger.info(f"\n{model}\n")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    n_epochs = 50
    batch_size = 32
    time_steps = 50  # Simulate 50ms of spiking activity
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Forward pass with temporal dynamics
            optimizer.zero_grad()
            output = model(batch_X, time_steps=time_steps)
            
            # Compute loss
            loss = criterion(output, batch_y)
            
            # Backward pass (combines backprop with Hebbian updates)
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        # Evaluate
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(X_test, time_steps=time_steps)
                _, test_pred = torch.max(test_output, 1)
                test_acc = (test_pred == y_test).sum().item() / len(y_test)
            
            train_acc = correct / total
            avg_loss = total_loss / (len(X_train) // batch_size)
            
            logger.info(
                f"Epoch {epoch+1}/{n_epochs} - "
                f"Loss: {avg_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Test Acc: {test_acc:.4f}"
            )
    
    # Final evaluation
    logger.info("\nFinal evaluation...")
    model.eval()
    with torch.no_grad():
        # Get firing rates
        test_output = model.get_firing_rates(X_test, time_steps=100)
        _, predictions = torch.max(test_output, 1)
        
        accuracy = (predictions == y_test).sum().item() / len(y_test)
        logger.info(f"Final Test Accuracy: {accuracy:.4f}")
    
    # Save model
    model.save('neuronsv2_digits.pth')
    logger.info("Model saved to neuronsv2_digits.pth")
    
    # Demonstrate spiking activity
    logger.info("\nDemonstrating spiking activity...")
    model.eval()
    with torch.no_grad():
        sample = X_test[0:1]
        result = model.forward(sample, time_steps=100, return_all_layers=True)
        
        logger.info(f"Input shape: {sample.shape}")
        logger.info(f"Output shape: {result['output'].shape}")
        logger.info(f"Number of layers: {len(result['layers'])}")
        
        for i, layer_output in enumerate(result['layers']):
            spikes = layer_output['spikes']
            phases = layer_output['phases']
            logger.info(
                f"Layer {i+1}: "
                f"Spikes shape: {spikes.shape}, "
                f"Mean firing rate: {spikes.mean().item():.4f}, "
                f"Phase coherence: {torch.cos(phases).mean().item():.4f}"
            )


if __name__ == '__main__':
    main()
