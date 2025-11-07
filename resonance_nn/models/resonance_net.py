"""
Complete Resonance Neural Network Architecture
Integrates resonance layers and holographic memory
"""

import torch
import torch.nn as nn
from typing import Optional, List
from resonance_nn.layers.resonance import ResonanceLayer, MultiScaleResonanceLayer
from resonance_nn.layers.holographic import HolographicMemory


class ResonanceNet(nn.Module):
    """
    Complete Resonance Neural Network
    
    Architecture:
    1. Input embedding
    2. Stack of resonance layers with O(n log n) complexity
    3. Holographic memory for long-term storage
    4. Output projection
    """
    
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 64,
        hidden_dim: Optional[int] = None,
        num_layers: int = 4,
        holographic_capacity: int = 1000,
        dropout: float = 0.1,
        use_multi_scale: bool = False,
        num_scales: int = 3,
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_frequencies: Number of frequencies per layer
            hidden_dim: Hidden dimension (default: input_dim)
            num_layers: Number of resonance layers
            holographic_capacity: Capacity of holographic memory
            dropout: Dropout rate
            use_multi_scale: Whether to use multi-scale processing
            num_scales: Number of scales for multi-scale processing
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.num_layers = num_layers
        self.use_multi_scale = use_multi_scale
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        
        # Resonance layers
        self.resonance_layers = nn.ModuleList()
        for i in range(num_layers):
            if use_multi_scale:
                layer = MultiScaleResonanceLayer(
                    input_dim=self.hidden_dim,
                    num_scales=num_scales,
                    frequencies_per_scale=num_frequencies // num_scales,
                    dropout=dropout,
                )
            else:
                layer = ResonanceLayer(
                    input_dim=self.hidden_dim,
                    num_frequencies=num_frequencies,
                    dropout=dropout,
                    init_magnitude=0.1 / (i + 1),  # Decay with depth
                )
            self.resonance_layers.append(layer)
            
        # Holographic memory
        self.holographic_memory = HolographicMemory(
            pattern_dim=self.hidden_dim,
            hologram_dim=self.hidden_dim * 2,
            capacity=holographic_capacity,
        )
        
        # Memory gating
        self.memory_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, input_dim)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        use_memory: bool = True,
        store_to_memory: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through Resonance Network
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            use_memory: Whether to retrieve from holographic memory
            store_to_memory: Whether to store patterns to memory
            
        Returns:
            Output tensor (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to hidden dimension
        h = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # Pass through resonance layers
        for layer in self.resonance_layers:
            h = layer(h)
            
        # Holographic memory integration
        if use_memory:
            # Retrieve from memory
            memory_content = self.holographic_memory.reconstruct()
            
            # Gate memory contribution
            gate = self.memory_gate(h)
            memory_contribution = memory_content.unsqueeze(0).unsqueeze(0)
            memory_contribution = memory_contribution.expand(batch_size, seq_len, -1)
            h = h + gate * memory_contribution
            
        if store_to_memory:
            # Store current patterns to memory
            # Store mean pattern across sequence
            pattern_to_store = h.mean(dim=1)  # (batch, hidden_dim)
            self.holographic_memory.encode(pattern_to_store)
            
        # Final normalization
        h = self.final_norm(h)
        
        # Project to output dimension
        output = self.output_projection(h)
        
        return output
    
    def encode_to_memory(self, x: torch.Tensor):
        """
        Encode input patterns to holographic memory
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
        """
        with torch.no_grad():
            h = self.input_projection(x)
            for layer in self.resonance_layers:
                h = layer(h)
            pattern = h.mean(dim=1)
            self.holographic_memory.encode(pattern)
            
    def retrieve_from_memory(self, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Retrieve pattern from holographic memory
        
        Args:
            query: Optional query pattern
            
        Returns:
            Retrieved pattern
        """
        if query is not None:
            query = self.input_projection(query)
            
        return self.holographic_memory.reconstruct(query)
    
    def get_complexity_estimate(self, seq_len: int) -> dict:
        """
        Estimate computational complexity
        
        Returns:
            Dictionary with complexity estimates
        """
        import math
        
        # O(n log n) for each resonance layer
        resonance_complexity = self.num_layers * seq_len * math.log2(seq_len)
        
        # O(k^2) for cross-frequency interference
        num_freq = self.resonance_layers[0].num_frequencies if hasattr(
            self.resonance_layers[0], 'num_frequencies'
        ) else 64
        interference_complexity = self.num_layers * (num_freq ** 2)
        
        # Total
        total_complexity = resonance_complexity + interference_complexity
        
        return {
            'resonance': resonance_complexity,
            'interference': interference_complexity,
            'total': total_complexity,
            'complexity_class': 'O(n log n + k²)',
        }
    
    def get_gradient_stats(self) -> dict:
        """
        Get gradient statistics from all layers
        """
        stats = {}
        for i, layer in enumerate(self.resonance_layers):
            if hasattr(layer, 'get_gradient_stats'):
                layer_stats = layer.get_gradient_stats()
                for key, value in layer_stats.items():
                    stats[f'layer_{i}_{key}'] = value
        return stats


class ResonanceEncoder(nn.Module):
    """
    Encoder-only Resonance Network for representation learning
    """
    
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 64,
        hidden_dim: Optional[int] = None,
        num_layers: int = 4,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim or input_dim
        self.output_dim = output_dim or self.hidden_dim
        
        # Resonance encoder
        self.encoder = ResonanceNet(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
            hidden_dim=self.hidden_dim,
            num_layers=num_layers,
            holographic_capacity=0,  # No memory for encoder-only
            dropout=dropout,
        )
        
        # Pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, self.output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to fixed-size representation
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Encoded representation (batch, output_dim)
        """
        # Pass through resonance network
        h = self.encoder(x, use_memory=False)
        
        # Pool across sequence
        h = h.transpose(1, 2)  # (batch, input_dim, seq_len)
        h = self.pooling(h).squeeze(-1)  # (batch, input_dim)
        
        # Project to output
        output = self.output_proj(h)
        
        return output


class ResonanceAutoencoder(nn.Module):
    """
    Autoencoder using Resonance Networks
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_frequencies: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Encoder
        self.encoder = ResonanceEncoder(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
            hidden_dim=input_dim,
            num_layers=num_layers,
            output_dim=latent_dim,
            dropout=dropout,
        )
        
        # Decoder
        self.decoder = ResonanceNet(
            input_dim=latent_dim,
            num_frequencies=num_frequencies,
            hidden_dim=input_dim,
            num_layers=num_layers,
            holographic_capacity=0,
            dropout=dropout,
        )
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Encode and decode input
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (reconstruction, latent)
        """
        # Encode
        latent = self.encoder(x)  # (batch, latent_dim)
        
        # Expand for decoder
        batch_size, seq_len = x.shape[0], x.shape[1]
        latent_expanded = latent.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Decode
        reconstruction = self.decoder(latent_expanded, use_memory=False)
        
        return reconstruction, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode only"""
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode only"""
        batch_size = latent.shape[0]
        latent_expanded = latent.unsqueeze(1).expand(-1, seq_len, -1)
        return self.decoder(latent_expanded, use_memory=False)


class ResonanceClassifier(nn.Module):
    """
    Classification model using Resonance Network
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_frequencies: int = 64,
        hidden_dim: Optional[int] = None,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Resonance encoder
        self.encoder = ResonanceEncoder(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
            hidden_dim=hidden_dim or input_dim,
            num_layers=num_layers,
            output_dim=hidden_dim or input_dim,
            dropout=dropout,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim or input_dim, (hidden_dim or input_dim) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((hidden_dim or input_dim) // 2, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input sequence
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Class logits (batch, num_classes)
        """
        # Encode
        h = self.encoder(x)
        
        # Classify
        logits = self.classifier(h)
        
        return logits
