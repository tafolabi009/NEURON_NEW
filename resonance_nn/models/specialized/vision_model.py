"""
Resonance Vision Model
Specialized for image classification and understanding
Uses frequency-domain vision processing
"""

import torch
import torch.nn as nn
from typing import Optional, List
from resonance_nn.multimodal.vision import ResonanceVisionEncoder


class ResonanceVisionModel(nn.Module):
    """
    Complete vision model for image tasks
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = ResonanceVisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        
        print(f"ResonanceVisionModel initialized:")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Classes: {num_classes}")
        print(f"  Frequency-domain processing (NO CNN)")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classify images
        
        Args:
            images: (batch, channels, height, width)
            
        Returns:
            Class logits (batch, num_classes)
        """
        return self.encoder(images)
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features"""
        return self.encoder(images, return_features=False)
