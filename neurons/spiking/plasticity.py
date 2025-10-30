"""
Hebbian plasticity for NEURONSv2.

This module implements local learning rules - NO BACKPROPAGATION!

Key principles:
1. STDP (Spike-Timing Dependent Plasticity): "Neurons that fire together, wire together"
2. BCM (Bienenstock-Cooper-Munro): Homeostatic threshold adaptation
3. Fast-slow dynamics: Meta-learning through synaptic consolidation

These are biologically plausible learning rules that work with:
- Only local information (pre/post synaptic activity)
- No error signals propagated backwards
- Natural regularization through homeostasis
- Online learning (no batch updates needed)

References:
- Hebb (1949): The Organization of Behavior
- Bienenstock, Cooper, Munro (1982): BCM theory
- Abbott & Nelson (2000): Synaptic plasticity review
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class HebbianSynapse(nn.Module):
    """
    Synapse with Hebbian plasticity rules.
    
    Implements multiple plasticity mechanisms:
    1. **STDP**: Synaptic weight changes based on spike timing
       - Potentiation: post-spike shortly after pre-spike (causality)
       - Depression: pre-spike shortly after post-spike (anti-causality)
       
    2. **BCM Rule**: Homeostatic learning
       - Weights strengthen when post-synaptic activity exceeds threshold
       - Threshold adapts based on average activity (prevents runaway)
       
    3. **Fast-slow dynamics**: Meta-learning
       - Fast weights: Rapid adaptation (working memory)
       - Slow weights: Long-term consolidation (learned patterns)
       
    NO BACKPROPAGATION - all learning is purely local!
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        tau_fast: Fast weight timescale (seconds). Default: 1.0
        tau_slow: Slow consolidation timescale (seconds). Default: 10000.0
        sparsity: Connection sparsity (0-1). Default: 0.1
        tau_trace: Spike trace timescale (ms). Default: 20.0
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        tau_fast: float = 1.0,
        tau_slow: float = 10000.0,
        sparsity: float = 0.1,
        tau_trace: float = 20.0,
    ):
        """Initialize Hebbian synapse."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.tau_trace = tau_trace
        
        # Weight components
        # w_slow: Long-term structural weights (learned patterns)
        # w_fast: Short-term plastic weights (working memory)
        self.w_slow = nn.Parameter(torch.randn(input_dim, output_dim) * 0.02)
        self.register_buffer('w_fast', torch.zeros(input_dim, output_dim))
        
        # Enforce biological sparsity
        with torch.no_grad():
            mask = (torch.rand_like(self.w_slow) < sparsity).float()
            self.w_slow.data *= mask
            self.register_buffer('connection_mask', mask)
        
        # BCM sliding threshold for homeostasis
        self.register_buffer('threshold', torch.ones(output_dim))
        
        # Spike traces for STDP
        # Exponentially decaying traces of recent spikes
        self.register_buffer('pre_trace', torch.zeros(1, input_dim))
        self.register_buffer('post_trace', torch.zeros(1, output_dim))
        
        logger.debug(f"Initialized HebbianSynapse: {input_dim}→{output_dim}, "
                    f"sparsity={sparsity*100:.0f}%")
    
    def forward(self, input_spikes: Tensor) -> Tensor:
        """
        Forward pass through synapse.
        
        Args:
            input_spikes: (batch, input_dim) presynaptic spikes (binary or rate)
            
        Returns:
            output_current: (batch, output_dim) postsynaptic current
        """
        # Effective weights = slow (structural) + fast (plastic)
        w_eff = (self.w_slow + self.w_fast) * self.connection_mask
        
        # Synaptic transmission
        output_current = torch.matmul(input_spikes.float(), w_eff)
        
        return output_current
    
    def update_plasticity(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        learning_rate: float = 0.01,
        dt: float = 1.0,
    ) -> None:
        """
        Update synaptic weights using Hebbian rules.
        
        This is where the magic happens - LOCAL learning without backprop!
        
        Args:
            pre_spikes: (batch, input_dim) presynaptic spikes
            post_spikes: (batch, output_dim) postsynaptic spikes
            learning_rate: Learning rate for plasticity
            dt: Timestep (ms)
        """
        batch_size = pre_spikes.shape[0]
        device = pre_spikes.device
        
        # Expand traces to batch size
        if self.pre_trace.shape[0] != batch_size:
            self.pre_trace = torch.zeros(
                batch_size, self.input_dim,
                device=device,
                dtype=pre_spikes.dtype
            )
            self.post_trace = torch.zeros(
                batch_size, self.output_dim,
                device=device,
                dtype=post_spikes.dtype
            )
        
        # Update spike traces (exponential decay)
        # Trace tracks recent spiking history
        decay_factor = dt / self.tau_trace
        self.pre_trace = (
            self.pre_trace * (1 - decay_factor) +
            pre_spikes.float() * decay_factor
        )
        self.post_trace = (
            self.post_trace * (1 - decay_factor) +
            post_spikes.float() * decay_factor
        )
        
        # ===================================================================
        # STDP (Spike-Timing Dependent Plasticity)
        # ===================================================================
        # Core principle: "Neurons that fire together, wire together"
        # But timing matters!
        #
        # Potentiation: Δw ∝ pre_trace * post_spike
        #   - Post spikes shortly after pre spikes → strengthen
        #   - This captures causality (pre causes post)
        #
        # Depression: Δw ∝ -pre_spike * post_trace  
        #   - Pre spikes shortly after post spikes → weaken
        #   - This eliminates spurious correlations
        # ===================================================================
        
        # Compute weight changes (outer product over batch)
        potentiation = torch.matmul(
            self.pre_trace.transpose(-2, -1),
            post_spikes.float()
        )
        depression = torch.matmul(
            pre_spikes.float().transpose(-2, -1),
            self.post_trace
        )
        
        # STDP rule with asymmetry (potentiation > depression)
        dw_stdp = learning_rate * (potentiation - 0.5 * depression)
        dw_stdp = dw_stdp.mean(dim=0)  # Average over batch
        
        # ===================================================================
        # BCM Rule (Bienenstock-Cooper-Munro)
        # ===================================================================
        # Homeostatic learning rule that prevents runaway activity
        #
        # Δw ∝ pre * post * (post - θ)
        # where θ is a sliding threshold that adapts to activity
        #
        # - If post > θ: potentiation (neuron should respond more)
        # - If post < θ: depression (neuron should respond less)
        # - θ adapts based on post²: moves to stabilize activity
        #
        # This provides automatic regularization without explicit terms!
        # ===================================================================
        
        # Compute average firing rates
        post_rate = post_spikes.float().mean(dim=0)
        pre_rate = pre_spikes.float().mean(dim=0)
        
        # BCM modification term
        bcm_modulation = post_rate * (post_rate - self.threshold)
        
        # Weight update
        dw_bcm = learning_rate * 0.1 * torch.matmul(
            pre_rate.unsqueeze(-1),
            bcm_modulation.unsqueeze(0)
        ).squeeze(0)
        
        # ===================================================================
        # Fast-Slow Dynamics (Meta-Learning)
        # ===================================================================
        # Two timescales of plasticity:
        #
        # 1. Fast weights (τ ~ seconds):
        #    - Rapid adaptation for working memory
        #    - Quick learning of task-specific patterns
        #    - Decay quickly if not reinforced
        #
        # 2. Slow weights (τ ~ hours/days):
        #    - Long-term memory consolidation
        #    - Structural changes
        #    - Only updated from fast weights gradually
        #
        # This implements meta-learning: fast weights learn quickly,
        # slow weights store what's repeatedly useful!
        # ===================================================================
        
        # Update fast weights (rapid plasticity)
        self.w_fast = self.w_fast + dw_stdp + dw_bcm
        
        # Decay fast weights (working memory fade)
        decay_rate = dt / (self.tau_fast * 1000.0)  # Convert s to ms
        self.w_fast = self.w_fast * (1 - decay_rate)
        
        # Consolidate fast → slow (very gradual)
        consolidation_rate = 0.0001
        with torch.no_grad():
            self.w_slow.data = (
                self.w_slow.data +
                consolidation_rate * self.w_fast
            )
        
        # Update BCM threshold (sliding)
        # θ moves towards ⟨post²⟩
        threshold_rate = 0.001
        self.threshold = (
            self.threshold +
            threshold_rate * (post_rate ** 2 - self.threshold)
        )
        
        # Enforce sparsity (biological constraint)
        with torch.no_grad():
            self.w_slow.data *= self.connection_mask
            self.w_fast *= self.connection_mask
    
    def reset_traces(self) -> None:
        """Reset spike traces (between episodes/sequences)."""
        self.pre_trace.zero_()
        self.post_trace.zero_()
    
    def get_effective_weights(self) -> Tensor:
        """Get current effective weights (slow + fast)."""
        return (self.w_slow + self.w_fast) * self.connection_mask
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
            f'tau_fast={self.tau_fast}s, tau_slow={self.tau_slow}s'
        )


__all__ = [
    'HebbianSynapse',
]
