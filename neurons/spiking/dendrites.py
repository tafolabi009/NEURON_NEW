"""
Dendritic computation modules for NEURONSv2.

This module implements neurons with spatial structure - NOT point neurons!

Key insight: Real neurons have dendrites that perform local, nonlinear computation
before integrating at the soma. This gives exponentially more representational
capacity than traditional artificial neurons.

Implements:
- Dendritic branches with local nonlinearities
- NMDA-like voltage-dependent gating
- Multi-compartment neurons (basal + apical dendrites)
- Predictive coding through dendritic structure
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch import Tensor
import logging

from .spiking import SpikingNeuron

logger = logging.getLogger(__name__)


class DendriticBranch(nn.Module):
    """
    Single dendritic branch with local nonlinear computation.
    
    Real neurons have spatial structure - different dendritic branches
    compute different functions before integration at the soma.
    This provides 2^(n_branches) representational capacity vs 2^n
    for traditional point neurons.
    
    Features:
    - Sparse synaptic connectivity (biological realism)
    - Temporal integration with time constant
    - Optional NMDA-like voltage-dependent gating
    - Local nonlinear activation (calcium spikes)
    
    Args:
        input_dim: Dimension of input
        branch_dim: Dimension of branch output
        tau: Branch integration time constant (ms). Default: 10.0
        use_nmda: Whether to use NMDA-like voltage gating. Default: True
        sparsity: Connection sparsity (0-1). Default: 0.1
    """
    
    def __init__(
        self,
        input_dim: int,
        branch_dim: int,
        tau: float = 10.0,
        use_nmda: bool = True,
        sparsity: float = 0.1,
    ):
        """Initialize dendritic branch."""
        super().__init__()
        self.tau = tau
        self.use_nmda = use_nmda
        self.sparsity = sparsity
        
        # Synaptic weights (sparse connectivity)
        self.synapse = nn.Linear(input_dim, branch_dim, bias=False)
        
        # Initialize with sparse connectivity
        with torch.no_grad():
            mask = (torch.rand_like(self.synapse.weight) < sparsity).float()
            self.synapse.weight.data *= mask
            self.register_buffer('connection_mask', mask)
        
        # Branch state (temporal integration)
        self.register_buffer('activation', torch.zeros(1, branch_dim))
        
        logger.debug(f"Initialized DendriticBranch: {input_dim}→{branch_dim}, "
                    f"sparsity={sparsity*100:.0f}%")
    
    def forward(
        self,
        input_spikes: Tensor,
        voltage: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute branch activation.
        
        Args:
            input_spikes: (batch, input_dim) input spikes (binary or rate)
            voltage: Optional (batch, branch_dim) membrane voltage for NMDA gating
            
        Returns:
            branch_current: (batch, branch_dim) current injected to soma
        """
        batch_size = input_spikes.shape[0]
        
        # Resize activation buffer if needed
        if self.activation.shape[0] != batch_size:
            self.activation = torch.zeros(
                batch_size, self.activation.shape[1],
                device=input_spikes.device,
                dtype=input_spikes.dtype
            )
        
        # Synaptic transmission (through sparse connections)
        synaptic_current = self.synapse(input_spikes.float())
        
        # NMDA-like voltage-dependent nonlinearity
        # Models Mg2+ block removal at depolarized potentials
        if self.use_nmda and voltage is not None:
            voltage_gate = torch.sigmoid(voltage)
            synaptic_current = synaptic_current * voltage_gate
        
        # Temporal integration on branch
        # dA/dt = (-A + I) / τ_branch
        self.activation = self.activation + (
            -self.activation + synaptic_current
        ) / self.tau
        
        # Nonlinear activation (models dendritic calcium spikes)
        branch_current = torch.tanh(self.activation)
        
        return branch_current
    
    def reset_state(self) -> None:
        """Reset branch state."""
        self.activation.zero_()
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f'synapse={self.synapse.in_features}→{self.synapse.out_features}, '
            f'tau={self.tau}ms, nmda={self.use_nmda}, sparsity={self.sparsity}'
        )


class DendriticNeuron(nn.Module):
    """
    Multi-compartment neuron with basal and apical dendrites.
    
    Architecture implements predictive coding naturally:
        Basal dendrites ← bottom-up sensory input
        Apical dendrites ← top-down prediction/context
        Soma ← integration + prediction error computation
        
    The soma fires when there's a mismatch between bottom-up input and
    top-down prediction - this IS predictive coding, no extra machinery needed!
    
    This is fundamentally different from transformers. No attention,
    no feedforward layers - just biological neural computation.
    
    Args:
        input_dim: Dimension of bottom-up input
        hidden_dim: Dimension of neuron output
        num_basal: Number of basal dendrites. Default: 12
        num_apical: Number of apical dendrites. Default: 6
        tau_mem: Membrane time constant (ms). Default: 20.0
        use_nmda: Use NMDA-like gating. Default: True
        sparsity: Synaptic sparsity. Default: 0.1
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_basal: int = 12,
        num_apical: int = 6,
        tau_mem: float = 20.0,
        use_nmda: bool = True,
        sparsity: float = 0.1,
    ):
        """Initialize dendritic neuron."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_basal = num_basal
        self.num_apical = num_apical
        
        # Basal dendrites (bottom-up sensory processing)
        self.branch_dim_basal = max(1, hidden_dim // num_basal)
        self.basal_branches = nn.ModuleList([
            DendriticBranch(input_dim, self.branch_dim_basal, use_nmda=use_nmda, sparsity=sparsity)
            for _ in range(num_basal)
        ])
        
        # Apical dendrites (top-down prediction/context)
        self.branch_dim_apical = max(1, hidden_dim // num_apical)
        self.apical_branches = nn.ModuleList([
            DendriticBranch(hidden_dim, self.branch_dim_apical, use_nmda=use_nmda, sparsity=sparsity)
            for _ in range(num_apical)
        ])
        
        # Projection layers to ensure correct output dimensions
        self.basal_proj = nn.Linear(
            self.branch_dim_basal * num_basal,
            hidden_dim,
            bias=False
        )
        self.apical_proj = nn.Linear(
            self.branch_dim_apical * num_apical,
            hidden_dim,
            bias=False
        )
        
        # Soma (integrate-and-fire)
        self.soma = SpikingNeuron(
            size=hidden_dim,
            tau_mem=tau_mem,
            threshold=1.0,
        )
        
        # Lateral inhibition (winner-take-all competition)
        self.register_buffer('inhibition', torch.zeros(1, hidden_dim))
        self.inhibition_strength = 0.5
        
        logger.debug(f"Initialized DendriticNeuron: input={input_dim}, hidden={hidden_dim}, "
                    f"basal_branches={num_basal}, apical_branches={num_apical}")
    
    def forward(
        self,
        bottom_up: Tensor,
        top_down: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward one timestep through dendritic neuron.
        
        Args:
            bottom_up: (batch, input_dim) bottom-up input
            top_down: Optional (batch, hidden_dim) top-down prediction
            
        Returns:
            spikes: (batch, hidden_dim) output spikes
            info: Dict containing:
                - membrane: membrane potential
                - basal_current: basal dendritic current
                - apical_current: apical dendritic current
                - spikes: spike output
        """
        batch_size = bottom_up.shape[0]
        device = bottom_up.device
        
        # Process basal dendrites (bottom-up input)
        basal_currents = []
        for branch in self.basal_branches:
            current = branch(bottom_up, voltage=None)
            basal_currents.append(current)
        
        basal_concat = torch.cat(basal_currents, dim=1)
        basal_total = self.basal_proj(basal_concat)
        
        # Process apical dendrites (top-down prediction)
        apical_total = torch.zeros(batch_size, self.hidden_dim, device=device)
        if top_down is not None:
            apical_currents = []
            for branch in self.apical_branches:
                current = branch(top_down, voltage=None)
                apical_currents.append(current)
            
            apical_concat = torch.cat(apical_currents, dim=1)
            apical_total = self.apical_proj(apical_concat)
        
        # Predictive coding at soma: basal - apical = prediction error
        # Neuron fires when bottom-up input doesn't match top-down prediction
        soma_current = basal_total - 0.3 * apical_total
        
        # Lateral inhibition (winner-take-all dynamics)
        if self.inhibition.shape[0] != batch_size:
            self.inhibition = torch.zeros(
                batch_size, self.hidden_dim,
                device=device,
                dtype=bottom_up.dtype
            )
        
        soma_current = soma_current - self.inhibition_strength * self.inhibition
        
        # Soma spike generation
        spikes, membrane = self.soma(soma_current)
        
        # Update lateral inhibition based on recent activity
        self.inhibition = 0.9 * self.inhibition + 0.1 * spikes
        
        info = {
            'membrane': membrane,
            'basal_current': basal_total,
            'apical_current': apical_total,
            'spikes': spikes,
        }
        
        return spikes, info
    
    def reset_state(self) -> None:
        """Reset all neuron state."""
        self.soma.reset_state()
        for branch in self.basal_branches:
            branch.reset_state()
        for branch in self.apical_branches:
            branch.reset_state()
        self.inhibition.zero_()
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
            f'num_basal={self.num_basal}, num_apical={self.num_apical}'
        )


__all__ = [
    'DendriticBranch',
    'DendriticNeuron',
]
