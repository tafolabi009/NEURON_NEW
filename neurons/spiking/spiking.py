"""
Spiking primitives - core building blocks.

Implements the surrogate gradient trick and basic LIF neurons.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient for spiking neurons.
    
    Forward: Heaviside step function (true spiking)
    Backward: Smooth surrogate (fast sigmoid) for gradient flow
    """
    
    @staticmethod
    def forward(ctx, input: Tensor, threshold: float = 1.0) -> Tensor:
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        input, = ctx.saved_tensors
        alpha = 10.0
        grad_input = grad_output / (
            1.0 + alpha * torch.abs(input - ctx.threshold)
        ) ** 2
        return grad_input, None


def spike_function(x: Tensor, threshold: float = 1.0) -> Tensor:
    """Apply spiking nonlinearity with surrogate gradient."""
    return SurrogateGradient.apply(x, threshold)


class SpikingNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron.
    
    The fundamental building block - NO linear layers!
    Information is encoded in spike timing, not continuous values.
    """
    
    def __init__(
        self,
        size: int,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        refractory_period: float = 2.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.size = size
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.refractory_period = refractory_period
        self.dt = dt
        
        self.register_buffer('membrane', torch.zeros(1, size))
        self.register_buffer('refractory', torch.zeros(1, size))
        
        self.leak = nn.Parameter(torch.ones(size))
        self.threshold_adapt = nn.Parameter(torch.zeros(size))
    
    def forward(self, input_current: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = input_current.shape[0]
        
        if self.membrane.shape[0] != batch_size:
            self.membrane = torch.zeros(
                batch_size, self.size,
                device=input_current.device,
                dtype=input_current.dtype
            )
            self.refractory = torch.zeros(
                batch_size, self.size,
                device=input_current.device,
                dtype=input_current.dtype
            )
        
        # Membrane dynamics
        leak_term = -self.membrane * self.leak / self.tau_mem
        input_term = input_current / self.tau_mem
        dV = (leak_term + input_term) * self.dt
        
        not_refractory = (self.refractory <= 0).float()
        self.membrane = self.membrane + dV * not_refractory
        
        effective_threshold = self.threshold + self.threshold_adapt
        spikes = spike_function(self.membrane, effective_threshold)
        
        self.membrane = self.membrane * (1 - spikes)
        
        self.refractory = torch.where(
            spikes.bool(),
            torch.full_like(self.refractory, self.refractory_period),
            torch.maximum(self.refractory - self.dt, torch.zeros_like(self.refractory))
        )
        
        return spikes, self.membrane
    
    def reset_state(self) -> None:
        self.membrane.zero_()
        self.refractory.zero_()


__all__ = ['SurrogateGradient', 'spike_function', 'SpikingNeuron']
