"""
Complete example of training NEURONSv2 spiking neural network.

This example demonstrates the unique features of the architecture:
- Spiking dynamics with temporal coding
- Dendritic computation for spatial processing
- Oscillatory communication between layers
- Hebbian plasticity for local learning
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
    calculate_metrics,
)

import logging


def poisson_encode(x: torch.Tensor, time_steps: int, max_rate: float = 100.0) -> torch.Tensor:
    """
    Encode input as Poisson spike trains.
    
    Args:
        x: Input tensor (batch_size, features)
        time_steps: Number of time steps
        max_rate: Maximum firing rate in Hz
        
    Returns:
        Spike trains (batch_size, time_steps, features)
    """
    # Normalize to [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    # Generate Poisson spikes
    rates = x_norm * max_rate / 1000.0  # Convert to probability per ms
    spikes = torch.rand(x.size(0), time_steps, x.size(1)) < rates.unsqueeze(1)
    
    return spikes.float()


def main():
    """Main training function."""
    # Setup
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("NEURONSv2 Spiking Neural Network Training")
    logger.info("="*60)
    
    # Load dataset
    logger.info("\nLoading digit classification dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Input features: {X_train.shape[1]}")
    logger.info(f"Output classes: {len(torch.unique(y_train))}")
    
    # Configure spiking network
    logger.info("\nConfiguring NEURONSv2 architecture...")
    config = NEURONSv2Config(
        # Neuron dynamics
        tau_mem=20.0,          # Membrane time constant (ms)
        tau_syn=5.0,           # Synaptic time constant (ms)
        threshold=1.0,         # Firing threshold
        reset_mechanism='subtract',
        
        # Dendritic parameters
        n_basal_dendrites=5,
        n_apical_dendrites=3,
        tau_dendrite=10.0,
        dendritic_coupling=0.5,
        
        # Oscillatory parameters
        natural_frequency=40.0,  # 40 Hz (gamma oscillations)
        coupling_strength=0.1,
        
        # Plasticity parameters
        learning_rate_fast=0.01,
        learning_rate_slow=0.001,
        stdp_tau_plus=20.0,
        stdp_tau_minus=20.0,
        bcm_tau=1000.0,
        use_stdp=True,
        use_bcm=True,
        
        # Simulation
        dt=1.0,  # Time step (ms)
    )
    
    # Create model
    logger.info("\nCreating NEURONSv2 model...")
    model = NEURONSv2(
        layer_sizes=[64, 128, 64, 10],
        config=config,
        use_dendrites=True,
        use_oscillators=True,
        use_plasticity=True,
        output_mode='rate'  # Output spike rates for classification
    )
    
    logger.info(f"\n{model}")
    
    # Training parameters
    n_epochs = 50
    time_steps = 100  # Simulation time steps per input
    batch_size = 32
    
    # Simple training loop (Hebbian + supervised fine-tuning)
    logger.info(f"\nTraining for {n_epochs} epochs...")
    logger.info(f"Time steps per input: {time_steps}")
    logger.info(f"Batch size: {batch_size}")
    
    # Optimizer for supervised learning (fine-tuning on top of Hebbian)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Mini-batch training
        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            # Encode as spike trains
            spike_input = poisson_encode(batch_x, time_steps)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(spike_input)
            
            # Loss (on spike rates)
            loss = criterion(output, batch_y)
            
            # Backward pass (fine-tuning)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Evaluate on test set every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Encode test data
                test_spikes = poisson_encode(X_test, time_steps)
                
                # Get predictions
                test_output = model(test_spikes)
                predictions = torch.argmax(test_output, dim=1)
                
                # Calculate metrics
                metrics = calculate_metrics(predictions, y_test)
                
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} - "
                    f"Loss: {avg_loss:.4f} - "
                    f"Test Acc: {metrics['accuracy']:.4f} - "
                    f"F1: {metrics['f1_score']:.4f}"
                )
                
                if metrics['accuracy'] > best_acc:
                    best_acc = metrics['accuracy']
                    model.save('best_neuronsv2_model.pth')
                    logger.info(f"  → Saved best model (acc: {best_acc:.4f})")
    
    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("Final Evaluation")
    logger.info("="*60)
    
    model.eval()
    with torch.no_grad():
        test_spikes = poisson_encode(X_test, time_steps)
        test_output = model(test_spikes)
        predictions = torch.argmax(test_output, dim=1)
        
        metrics = calculate_metrics(predictions, y_test)
        
        logger.info(f"\nTest Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    # Demonstrate unique features
    logger.info("\n" + "="*60)
    logger.info("Analyzing Spiking Dynamics")
    logger.info("="*60)
    
    with torch.no_grad():
        # Get detailed outputs for a single sample
        sample_input = poisson_encode(X_test[:1], time_steps)
        detailed_output = model(sample_input, return_all_layers=True)
        
        logger.info(f"\nLayer-by-layer analysis:")
        for i, layer_output in enumerate(detailed_output['layers']):
            spikes = layer_output['spikes']
            phases = layer_output['phases']
            
            # Spike statistics
            spike_rate = spikes.mean().item() * 1000  # Convert to Hz
            
            logger.info(f"\nLayer {i+1}:")
            logger.info(f"  Average firing rate: {spike_rate:.2f} Hz")
            logger.info(f"  Spike shape: {spikes.shape}")
            
            if phases is not None:
                logger.info(f"  Phase shape: {phases.shape}")
                logger.info(f"  Phase range: [{phases.min():.2f}, {phases.max():.2f}]")
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
