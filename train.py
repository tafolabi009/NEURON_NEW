"""
Simple training script for NEURONSv2.

Demonstrates basic usage of the spiking neural network architecture.
"""

import torch
import numpy as np
from sklearn.datasets import make_classification
from neurons import NEURONSv2, NEURONSv2Config, setup_logging
import logging


def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    
    # Create spiking network
    model = NEURONSv2(
        layer_sizes=[20, 50, 3],
        config=NEURONSv2Config(),
        output_mode='rate'
    )
    
    logger.info(f"Model: {model.get_num_parameters():,} parameters")
    
    # Simple training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(20):
        # Encode input as spike train
        spike_input = (torch.rand(X.size(0), 50, X.size(1)) < torch.sigmoid(X).unsqueeze(1)).float()
        
        optimizer.zero_grad()
        output = model(spike_input)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
