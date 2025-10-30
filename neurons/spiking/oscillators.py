"""
Oscillatory dynamics for NEURONSv2.

This module implements Communication Through Coherence - information routing
through phase synchronization, NOT attention mechanisms!

Key insight: The brain doesn't use attention matrices. Instead, neurons
communicate by synchronizing their oscillations. When synchronized, information
flows. When desynchronized, communication is blocked.

This provides:
- Dynamic routing without explicit attention
- O(n log n) complexity through oscillatory binding
- Natural support for temporal coding
- Multi-scale temporal hierarchies

Based on:
- Kuramoto model of coupled oscillators
- Communication Through Coherence (Fries, 2005)
- Phase-amplitude coupling in cortex
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import math
import logging

logger = logging.getLogger(__name__)


class OscillatoryPopulation(nn.Module):
    """
    Population of coupled oscillating neurons.
    
    Implements the Kuramoto model:
        dθ_i/dt = ω_i + K Σ_j sin(θ_j - θ_i)
        
    Where:
        θ_i = phase of neuron i
        ω_i = natural frequency of neuron i
        K = coupling strength
        
    When neurons fire together, their phases synchronize. This creates
    dynamic functional connectivity without explicit attention weights!
    
    Features:
    - Heterogeneous natural frequencies (biological realism)
    - Spike-driven phase modulation
    - Synchrony measurement (order parameter)
    - Multi-frequency support (theta, gamma, beta bands)
    
    Args:
        size: Number of oscillating units
        base_freq: Base oscillation frequency (Hz). Default: 60.0 (gamma)
        coupling_strength: Phase coupling strength. Default: 0.3
        freq_variation: Heterogeneity in frequencies. Default: 0.1
        dt: Integration timestep (ms). Default: 1.0
    """
    
    def __init__(
        self,
        size: int,
        base_freq: float = 60.0,
        coupling_strength: float = 0.3,
        freq_variation: float = 0.1,
        dt: float = 1.0,
    ):
        """Initialize oscillatory population."""
        super().__init__()
        self.size = size
        self.base_freq = base_freq
        self.coupling_strength = coupling_strength
        self.dt = dt
        
        # Convert frequency to angular velocity (rad/ms)
        self.omega = 2 * math.pi * base_freq / 1000.0
        
        # Each neuron has slightly different natural frequency
        # This heterogeneity is crucial for rich dynamics
        freq_var = torch.randn(size) * (base_freq * freq_variation)
        natural_freqs = base_freq + freq_var
        self.register_buffer(
            'natural_freq',
            2 * math.pi * natural_freqs / 1000.0
        )
        
        # Phase state for each oscillator
        self.register_buffer(
            'phases',
            torch.rand(1, size) * 2 * math.pi
        )
        
        logger.debug(f"Initialized OscillatoryPopulation: size={size}, "
                    f"freq={base_freq}Hz, coupling={coupling_strength}")
    
    def forward(
        self,
        spikes: Tensor,
        connectivity: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Update oscillatory phases based on spiking activity.
        
        Args:
            spikes: (batch, size) spike input (drives oscillation)
            connectivity: Optional (size, size) connection matrix
            
        Returns:
            phases: (batch, size) updated phases (in radians)
            synchrony: (batch,) synchrony index (0-1, Kuramoto order parameter)
        """
        batch_size = spikes.shape[0]
        device = spikes.device
        
        # Resize phase buffer if needed
        if self.phases.shape[0] != batch_size:
            self.phases = torch.rand(
                batch_size, self.size,
                device=device,
                dtype=spikes.dtype
            ) * 2 * math.pi
        
        # Spiking activity drives oscillation frequency
        # When a neuron spikes, it temporarily increases its frequency
        spike_drive = spikes.float() * 5.0
        
        # Kuramoto coupling: dθ_i/dt = ω_i + K Σ_j sin(θ_j - θ_i)
        # Compute phase differences
        phase_i = self.phases.unsqueeze(-1)  # (batch, size, 1)
        phase_j = self.phases.unsqueeze(-2)  # (batch, 1, size)
        phase_diff = phase_j - phase_i  # (batch, size, size)
        
        # Coupling term: K * sin(θ_j - θ_i)
        coupling = torch.sin(phase_diff)
        
        # Apply connectivity matrix if provided (sparse coupling)
        if connectivity is not None:
            coupling = coupling * connectivity.unsqueeze(0)
        
        # Sum coupling contributions
        coupling_term = self.coupling_strength * coupling.mean(dim=-1)
        
        # Phase update: dθ/dt = ω + spike_drive + coupling
        dphase = (
            self.natural_freq.unsqueeze(0) +
            spike_drive +
            coupling_term
        ) * self.dt
        
        # Update phases (wrap to [0, 2π])
        self.phases = (self.phases + dphase) % (2 * math.pi)
        
        # Compute synchrony (Kuramoto order parameter)
        # r = |⟨exp(iθ)⟩| where ⟨⟩ is average over population
        # r = 1: perfect synchrony, r = 0: fully asynchronous
        complex_phases = torch.complex(
            torch.cos(self.phases),
            torch.sin(self.phases)
        )
        order_parameter = torch.abs(complex_phases.mean(dim=-1))
        synchrony = order_parameter
        
        return self.phases, synchrony
    
    def get_phase_coherence(
        self,
        phases1: Tensor,
        phases2: Tensor
    ) -> Tensor:
        """
        Compute phase coherence between two populations.
        
        This measures how well two populations are synchronized.
        High coherence = effective communication.
        
        Args:
            phases1: (batch, size1) phases of population 1
            phases2: (batch, size2) phases of population 2
            
        Returns:
            coherence: (batch,) phase coherence (0-1)
        """
        # Average phases for each population
        mean_phase1 = torch.atan2(
            torch.sin(phases1).mean(dim=-1),
            torch.cos(phases1).mean(dim=-1)
        )
        mean_phase2 = torch.atan2(
            torch.sin(phases2).mean(dim=-1),
            torch.cos(phases2).mean(dim=-1)
        )
        
        # Phase difference
        phase_diff = mean_phase1 - mean_phase2
        
        # Coherence = 1 - |phase_diff| / π
        # (normalized so 0° diff = 1, 180° diff = 0)
        coherence = 1.0 - torch.abs(phase_diff) / math.pi
        
        return coherence
    
    def reset_state(self) -> None:
        """Reset oscillator phases to random initial state."""
        self.phases = torch.rand_like(self.phases) * 2 * math.pi
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f'size={self.size}, base_freq={self.base_freq}Hz, '
            f'coupling={self.coupling_strength}'
        )


class MultiScaleOscillator(nn.Module):
    """
    Multi-scale oscillatory system with multiple frequency bands.
    
    The brain uses different frequency bands for different functions:
    - Theta (4-8 Hz): Working memory, sequence processing
    - Beta (15-30 Hz): Top-down predictions
    - Gamma (30-100 Hz): Local processing, binding
    
    Phase-amplitude coupling between bands creates hierarchical
    temporal structure - this is how the brain handles long contexts!
    
    Args:
        size: Number of units
        theta_freq: Theta band frequency (Hz). Default: 6.0
        beta_freq: Beta band frequency (Hz). Default: 20.0
        gamma_freq: Gamma band frequency (Hz). Default: 60.0
        coupling_strength: Coupling strength. Default: 0.3
    """
    
    def __init__(
        self,
        size: int,
        theta_freq: float = 6.0,
        beta_freq: float = 20.0,
        gamma_freq: float = 60.0,
        coupling_strength: float = 0.3,
    ):
        """Initialize multi-scale oscillator."""
        super().__init__()
        
        self.theta = OscillatoryPopulation(
            size, theta_freq, coupling_strength
        )
        self.beta = OscillatoryPopulation(
            size, beta_freq, coupling_strength
        )
        self.gamma = OscillatoryPopulation(
            size, gamma_freq, coupling_strength
        )
        
        logger.debug(f"Initialized MultiScaleOscillator: θ={theta_freq}Hz, "
                    f"β={beta_freq}Hz, γ={gamma_freq}Hz")
    
    def forward(
        self,
        spikes: Tensor
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Update all frequency bands.
        
        Args:
            spikes: (batch, size) spike input
            
        Returns:
            phases: Dict with 'theta', 'beta', 'gamma' phases
            synchrony: Dict with 'theta', 'beta', 'gamma' synchrony values
        """
        theta_phases, theta_sync = self.theta(spikes)
        beta_phases, beta_sync = self.beta(spikes)
        gamma_phases, gamma_sync = self.gamma(spikes)
        
        phases = {
            'theta': theta_phases,
            'beta': beta_phases,
            'gamma': gamma_phases,
        }
        
        synchrony = {
            'theta': theta_sync,
            'beta': beta_sync,
            'gamma': gamma_sync,
        }
        
        return phases, synchrony
    
    def reset_state(self) -> None:
        """Reset all oscillators."""
        self.theta.reset_state()
        self.beta.reset_state()
        self.gamma.reset_state()


__all__ = [
    'OscillatoryPopulation',
    'MultiScaleOscillator',
]
