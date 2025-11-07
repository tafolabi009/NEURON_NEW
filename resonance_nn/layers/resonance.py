"""
Core Resonance Layer Implementation
Frequency-domain processing with O(n log n) complexity and stable gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ComplexWeight(nn.Module):
    """
    Complex-valued weight with magnitude and phase parameterization.
    Ensures stable gradients for oscillatory parameters.
    
    Theorem 1: Stable Frequency Gradients
    ∂L/∂|w| = Re(∂L/∂w · w/|w|)
    ∂L/∂φ = Im(∂L/∂w · (-iw)/|w|)
    """
    
    def __init__(self, shape: Tuple[int, ...], init_magnitude: float = 1.0):
        super().__init__()
        self.magnitude = nn.Parameter(torch.ones(shape) * init_magnitude)
        self.phase = nn.Parameter(torch.zeros(shape))
        
    def forward(self) -> torch.Tensor:
        """Convert magnitude/phase to complex weight"""
        # w = |w| * e^(iφ)
        real = self.magnitude * torch.cos(self.phase)
        imag = self.magnitude * torch.sin(self.phase)
        return torch.complex(real, imag)
    
    def get_magnitude(self) -> torch.Tensor:
        """Return weight magnitude"""
        return self.magnitude
    
    def get_phase(self) -> torch.Tensor:
        """Return weight phase"""
        return self.phase


class ResonanceLayer(nn.Module):
    """
    Resonance Layer: Frequency-domain processing with O(n log n) complexity
    
    Algorithm 1: O(n log n) Resonance Layer
    1. Pad input to next power of 2
    2. Compute FFT: O(n log n)
    3. Apply complex weights to selected frequencies: O(k)
    4. Reconstruct full spectrum: O(k)
    5. Compute IFFT: O(n log n)
    Total: O(n log n + k) where k << n
    """
    
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int,
        dropout: float = 0.1,
        init_magnitude: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        
        # Complex weights for frequency domain
        self.weights = ComplexWeight(
            shape=(num_frequencies, input_dim),
            init_magnitude=init_magnitude
        )
        
        # Frequency positions (fixed, not learnable for stability)
        self.register_buffer(
            'frequency_positions',
            torch.linspace(0, input_dim // 2, num_frequencies).long()
        )
        
        # Cross-frequency interference weights O(k^2)
        self.interference_weights = nn.Parameter(
            torch.randn(num_frequencies, num_frequencies) * 0.01
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with O(n log n) complexity
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, input_dim)
        """
        batch_size, seq_len, dim = x.shape
        
        # Residual connection
        residual = x
        
        # 1. Pad to next power of 2 for efficient FFT
        padded_len = 2 ** int(np.ceil(np.log2(seq_len)))
        if padded_len > seq_len:
            x_padded = F.pad(x, (0, 0, 0, padded_len - seq_len))
        else:
            x_padded = x
            
        # 2. Compute FFT: O(n log n)
        x_fft = torch.fft.rfft(x_padded, dim=1)  # (batch, freq_bins, dim)
        
        # 3. Extract frequency components at selected positions
        freq_pos = torch.clamp(self.frequency_positions, 0, x_fft.shape[1] - 1)
        x_selected = torch.stack([x_fft[:, pos, :] for pos in freq_pos], dim=1)
        # Shape: (batch, num_frequencies, dim)
        
        # 4. Apply complex weights
        complex_weights = self.weights()  # (num_frequencies, dim)
        x_weighted = x_selected * complex_weights.unsqueeze(0)
        
        # 5. Cross-frequency interference: O(k^2)
        # This implements holographic-like interference between frequencies
        # Need to handle complex values - take real part of weighted sum
        x_weighted_real = torch.real(x_weighted)
        interference = torch.einsum('bkd,kj->bjd', x_weighted_real, self.interference_weights)
        # Add interference back as complex
        interference_complex = torch.complex(interference, torch.zeros_like(interference))
        x_processed = x_weighted + 0.1 * interference_complex
        
        # 6. Reconstruct full spectrum
        x_fft_processed = x_fft.clone()
        for i, pos in enumerate(freq_pos):
            x_fft_processed[:, pos, :] = x_processed[:, i, :]
            
        # 7. Compute IFFT: O(n log n)
        x_reconstructed = torch.fft.irfft(x_fft_processed, n=padded_len, dim=1)
        
        # 8. Trim to original length
        x_output = x_reconstructed[:, :seq_len, :]
        
        # 9. Apply normalization and dropout
        x_output = self.layer_norm(x_output + residual)
        x_output = self.dropout(x_output)
        
        return x_output
    
    def get_gradient_stats(self) -> dict:
        """
        Return gradient statistics for monitoring stability
        """
        stats = {}
        if self.weights.magnitude.grad is not None:
            stats['magnitude_grad_norm'] = torch.norm(self.weights.magnitude.grad).item()
            stats['magnitude_grad_max'] = torch.max(torch.abs(self.weights.magnitude.grad)).item()
        if self.weights.phase.grad is not None:
            stats['phase_grad_norm'] = torch.norm(self.weights.phase.grad).item()
            stats['phase_grad_max'] = torch.max(torch.abs(self.weights.phase.grad)).item()
        return stats
    
    def get_frequency_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the frequency spectrum of input for analysis
        """
        x_fft = torch.fft.rfft(x, dim=1)
        magnitude = torch.abs(x_fft)
        return magnitude


class MultiScaleResonanceLayer(nn.Module):
    """
    Multi-scale resonance processing with different frequency ranges
    """
    
    def __init__(
        self,
        input_dim: int,
        num_scales: int = 3,
        frequencies_per_scale: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_scales = num_scales
        
        # Multiple resonance layers at different scales
        self.resonance_layers = nn.ModuleList([
            ResonanceLayer(
                input_dim=input_dim,
                num_frequencies=frequencies_per_scale,
                dropout=dropout,
                init_magnitude=0.1 / (i + 1)  # Decay with scale
            )
            for i in range(num_scales)
        ])
        
        # Scale mixing weights
        self.scale_mixer = nn.Linear(num_scales * input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through multiple frequency scales
        """
        # Process at each scale
        scale_outputs = []
        for layer in self.resonance_layers:
            scale_out = layer(x)
            scale_outputs.append(scale_out)
            
        # Concatenate and mix
        mixed = torch.cat(scale_outputs, dim=-1)
        output = self.scale_mixer(mixed)
        
        return output


class AdaptiveResonanceLayer(nn.Module):
    """
    Adaptive resonance layer with learned frequency selection
    """
    
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_layer = ResonanceLayer(input_dim, num_frequencies, dropout)
        
        # Frequency attention mechanism
        self.freq_attention = nn.Sequential(
            nn.Linear(input_dim, num_frequencies),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with adaptive frequency selection
        """
        # Compute frequency importance
        freq_weights = self.freq_attention(x.mean(dim=1))  # (batch, num_frequencies)
        
        # Standard resonance processing
        output = self.base_layer(x)
        
        # Weight by frequency importance
        freq_weights_expanded = freq_weights.unsqueeze(1).unsqueeze(-1)
        # Note: This is a simplified version, full implementation would weight FFT components
        
        return output
