"""
Resonance Audio Model
Specialized for audio processing and recognition
"""

import torch
import torch.nn as nn
from typing import Optional
from resonance_nn.multimodal.audio import ResonanceAudioEncoder


class ResonanceAudioModel(nn.Module):
    """
    Complete audio model for audio tasks
    """
    
    def __init__(
        self,
        num_classes: int = 50,  # e.g., for audio event classification
        sample_rate: int = 22050,
        n_mels: int = 128,
        embed_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = ResonanceAudioEncoder(
            sample_rate=sample_rate,
            n_mels=n_mels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
        )
        
        print(f"ResonanceAudioModel initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Classes: {num_classes}")
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Classify audio
        
        Args:
            audio: (batch, samples) or (batch, n_mels, time)
            
        Returns:
            Class logits (batch, num_classes)
        """
        return self.encoder(audio)
    
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract audio features"""
        return self.encoder(audio, return_features=False)
