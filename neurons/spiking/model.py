"""
NEURONSv2: A novel spiking neural network architecture.

This module implements a biologically-inspired neural network with:
- Leaky Integrate-and-Fire spiking neurons
- Multi-compartment dendritic computation
- Oscillatory dynamics (Kuramoto model) for inter-layer communication
- Hebbian plasticity (STDP, BCM) for local learning rules
- Predictive coding through dendritic error computation
"""

from typing import List, Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import logging

from .spiking import SpikingLayer, SurrogateGradient
from .dendrites import DendriticNeuron, MultiCompartmentLayer
from .oscillators import KuramotoOscillator, OscillatoryPopulation
from .plasticity import STDPSynapse, BCMLearning, HebbianPlasticityLayer
from .config import NEURONSv2Config

logger = logging.getLogger(__name__)


class NEURONSv2Layer(nn.Module):
    """
    A single layer in the NEURONSv2 architecture.
    
    This layer combines:
    - Spiking neurons (LIF dynamics)
    - Dendritic computation (spatial processing)
    - Oscillatory population (phase-based communication)
    - Hebbian plasticity (local learning)
    
    Args:
        in_features: Number of input features
        out_features: Number of output neurons
        config: Configuration for the layer
        use_dendrites: Whether to use dendritic computation. Default: True
        use_oscillators: Whether to use oscillatory dynamics. Default: True
        use_plasticity: Whether to use Hebbian plasticity. Default: True
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[NEURONSv2Config] = None,
        use_dendrites: bool = True,
        use_oscillators: bool = True,
        use_plasticity: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or NEURONSv2Config()
        
        self.use_dendrites = use_dendrites
        self.use_oscillators = use_oscillators
        self.use_plasticity = use_plasticity
        
        # Core spiking layer
        self.spiking_layer = SpikingLayer(
            in_features=in_features,
            out_features=out_features,
            tau_mem=self.config.tau_mem,
            tau_syn=self.config.tau_syn,
            threshold=self.config.threshold,
            reset_mechanism=self.config.reset_mechanism
        )
        
        # Optional dendritic computation
        if self.use_dendrites:
            self.dendritic_layer = MultiCompartmentLayer(
                in_features=in_features,
                out_features=out_features,
                n_basal=self.config.n_basal_dendrites,
                n_apical=self.config.n_apical_dendrites,
                tau_dendrite=self.config.tau_dendrite,
                coupling_strength=self.config.dendritic_coupling
            )
        
        # Optional oscillatory dynamics
        if self.use_oscillators:
            self.oscillators = OscillatoryPopulation(
                n_oscillators=out_features,
                natural_freq=self.config.natural_frequency,
                coupling_strength=self.config.coupling_strength,
                dt=self.config.dt
            )
        
        # Optional Hebbian plasticity
        if self.use_plasticity:
            self.plasticity = HebbianPlasticityLayer(
                in_features=in_features,
                out_features=out_features,
                learning_rate_fast=self.config.learning_rate_fast,
                learning_rate_slow=self.config.learning_rate_slow,
                stdp_tau_plus=self.config.stdp_tau_plus,
                stdp_tau_minus=self.config.stdp_tau_minus,
                bcm_tau=self.config.bcm_tau,
                use_stdp=self.config.use_stdp,
                use_bcm=self.config.use_bcm
            )
        
        # State variables
        self.register_buffer('membrane_potential', torch.zeros(1, out_features))
        self.register_buffer('synaptic_current', torch.zeros(1, out_features))
        self.register_buffer('phases', torch.zeros(1, out_features))
        
        logger.info(
            f"Created NEURONSv2Layer: {in_features}→{out_features} "
            f"(dendrites={use_dendrites}, oscillators={use_oscillators}, "
            f"plasticity={use_plasticity})"
        )
    
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset internal states for new sequence."""
        device = self.membrane_potential.device
        self.membrane_potential = torch.zeros(batch_size, self.out_features, device=device)
        self.synaptic_current = torch.zeros(batch_size, self.out_features, device=device)
        self.phases = torch.zeros(batch_size, self.out_features, device=device)
        
        if self.use_dendrites:
            self.dendritic_layer.reset_state(batch_size)
        
        if self.use_oscillators:
            self.oscillators.reset_phases(batch_size)
    
    def forward(
        self,
        x: torch.Tensor,
        external_phases: Optional[torch.Tensor] = None,
        return_spikes: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the NEURONSv2 layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features) or 
               (batch_size, time_steps, in_features)
            external_phases: External oscillatory phases for coupling
            return_spikes: Whether to return spike trains
            
        Returns:
            Dictionary containing:
                - 'output': Spike output or membrane potential
                - 'spikes': Binary spike indicators (if return_spikes=True)
                - 'phases': Oscillatory phases
                - 'membrane': Membrane potential
                - 'dendritic': Dendritic computation (if use_dendrites=True)
        """
        batch_size = x.size(0)
        has_time_dim = x.dim() == 3
        
        if has_time_dim:
            time_steps = x.size(1)
            self.reset_state(batch_size)
        else:
            time_steps = 1
            x = x.unsqueeze(1)  # Add time dimension
        
        # Storage for outputs
        spikes_over_time = []
        phases_over_time = []
        membrane_over_time = []
        dendritic_over_time = []
        
        for t in range(time_steps):
            x_t = x[:, t, :]
            
            # 1. Dendritic computation (if enabled)
            if self.use_dendrites:
                dendritic_output = self.dendritic_layer(x_t)
                processed_input = dendritic_output['total']
                dendritic_over_time.append(dendritic_output)
            else:
                processed_input = x_t
            
            # 2. Spiking dynamics
            spike_output = self.spiking_layer(
                processed_input,
                self.membrane_potential,
                self.synaptic_current
            )
            
            spikes = spike_output['spikes']
            self.membrane_potential = spike_output['membrane']
            self.synaptic_current = spike_output['synaptic']
            
            # 3. Oscillatory dynamics (if enabled)
            if self.use_oscillators:
                if external_phases is not None:
                    phases = self.oscillators.step(spikes, external_phases[:, t, :])
                else:
                    phases = self.oscillators.step(spikes)
                self.phases = phases
            else:
                phases = torch.zeros_like(spikes)
            
            # 4. Hebbian plasticity (if enabled and in training mode)
            if self.use_plasticity and self.training:
                self.plasticity.update(x_t, spikes)
            
            spikes_over_time.append(spikes)
            phases_over_time.append(phases)
            membrane_over_time.append(self.membrane_potential)
        
        # Stack time dimension
        result = {
            'spikes': torch.stack(spikes_over_time, dim=1),
            'phases': torch.stack(phases_over_time, dim=1),
            'membrane': torch.stack(membrane_over_time, dim=1),
        }
        
        if self.use_dendrites:
            result['dendritic'] = dendritic_over_time
        
        # Remove time dimension if input didn't have it
        if not has_time_dim:
            result['spikes'] = result['spikes'].squeeze(1)
            result['phases'] = result['phases'].squeeze(1)
            result['membrane'] = result['membrane'].squeeze(1)
        
        # Set output based on what we want to return
        if return_spikes:
            result['output'] = result['spikes']
        else:
            result['output'] = result['membrane']
        
        return result


class NEURONSv2(nn.Module):
    """
    Complete NEURONSv2 neural network.
    
    A multi-layer spiking neural network with:
    - Biologically plausible neuron dynamics
    - Dendritic computation for spatial processing
    - Oscillatory communication (replaces attention)
    - Hebbian plasticity (local learning, minimal backprop)
    - Predictive coding capabilities
    
    Args:
        layer_sizes: List of layer sizes (e.g., [784, 256, 128, 10])
        config: Global configuration for all layers
        use_dendrites: Enable dendritic computation
        use_oscillators: Enable oscillatory dynamics
        use_plasticity: Enable Hebbian plasticity
        output_mode: How to generate output ('spikes', 'rate', 'membrane')
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        config: Optional[NEURONSv2Config] = None,
        use_dendrites: bool = True,
        use_oscillators: bool = True,
        use_plasticity: bool = True,
        output_mode: str = 'rate'
    ):
        super().__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layer sizes")
        
        self.layer_sizes = layer_sizes
        self.config = config or NEURONSv2Config()
        self.use_dendrites = use_dendrites
        self.use_oscillators = use_oscillators
        self.use_plasticity = use_plasticity
        self.output_mode = output_mode
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = NEURONSv2Layer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                config=self.config,
                use_dendrites=use_dendrites,
                use_oscillators=use_oscillators,
                use_plasticity=use_plasticity
            )
            self.layers.append(layer)
        
        self.n_layers = len(self.layers)
        
        logger.info(f"Created NEURONSv2 with {self.n_layers} layers: {layer_sizes}")
        logger.info(f"Features: dendrites={use_dendrites}, oscillators={use_oscillators}, "
                   f"plasticity={use_plasticity}")
    
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all layer states."""
        for layer in self.layers:
            layer.reset_state(batch_size)
    
    def forward(
        self,
        x: torch.Tensor,
        time_steps: Optional[int] = None,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, features) or (batch_size, time_steps, features)
            time_steps: Number of time steps to simulate (if x is static)
            return_all_layers: Return outputs from all layers
            
        Returns:
            Output tensor or dict of layer outputs
        """
        batch_size = x.size(0)
        has_time_dim = x.dim() == 3
        
        # If no time dimension, expand and repeat for time_steps
        if not has_time_dim and time_steps is not None:
            x = x.unsqueeze(1).expand(-1, time_steps, -1)
        elif not has_time_dim:
            x = x.unsqueeze(1)  # Single time step
        
        # Reset states for new sequence
        self.reset_state(batch_size)
        
        # Forward through layers
        layer_outputs = []
        current_input = x
        external_phases = None
        
        for i, layer in enumerate(self.layers):
            output_dict = layer(
                current_input,
                external_phases=external_phases,
                return_spikes=True
            )
            
            layer_outputs.append(output_dict)
            
            # Pass spikes to next layer
            current_input = output_dict['spikes']
            
            # Pass phases for oscillatory coupling
            if self.use_oscillators:
                external_phases = output_dict['phases']
        
        # Final output processing
        final_output = layer_outputs[-1]
        
        if self.output_mode == 'spikes':
            output = final_output['spikes']
        elif self.output_mode == 'rate':
            # Spike rate: average over time
            output = final_output['spikes'].mean(dim=1)
        elif self.output_mode == 'membrane':
            # Membrane potential at last time step
            if final_output['membrane'].dim() == 3:
                output = final_output['membrane'][:, -1, :]
            else:
                output = final_output['membrane']
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
        
        if return_all_layers:
            return {
                'output': output,
                'layers': layer_outputs
            }
        
        return output
    
    def get_firing_rates(self, x: torch.Tensor, time_steps: int = 100) -> torch.Tensor:
        """
        Get firing rates for input.
        
        Args:
            x: Input tensor
            time_steps: Number of time steps to simulate
            
        Returns:
            Firing rates for output layer
        """
        with torch.no_grad():
            x_expanded = x.unsqueeze(1).expand(-1, time_steps, -1)
            output = self.forward(x_expanded)
            if output.dim() == 3:
                return output.mean(dim=1)
            return output
    
    def predict(self, x: torch.Tensor, time_steps: int = 100) -> torch.Tensor:
        """
        Make predictions (for classification).
        
        Args:
            x: Input tensor
            time_steps: Number of time steps to simulate
            
        Returns:
            Class predictions
        """
        self.eval()
        with torch.no_grad():
            rates = self.get_firing_rates(x, time_steps)
            return torch.argmax(rates, dim=1)
    
    def save(self, filepath: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'layer_sizes': self.layer_sizes,
            'config': self.config.__dict__,
            'use_dendrites': self.use_dendrites,
            'use_oscillators': self.use_oscillators,
            'use_plasticity': self.use_plasticity,
            'output_mode': self.output_mode,
            'state_dict': self.state_dict(),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> 'NEURONSv2':
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device or 'cpu')
        
        config = NEURONSv2Config(**checkpoint['config'])
        model = cls(
            layer_sizes=checkpoint['layer_sizes'],
            config=config,
            use_dendrites=checkpoint['use_dendrites'],
            use_oscillators=checkpoint['use_oscillators'],
            use_plasticity=checkpoint['use_plasticity'],
            output_mode=checkpoint['output_mode']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def __repr__(self) -> str:
        """String representation."""
        total_params = sum(p.numel() for p in self.parameters())
        return (
            f"NEURONSv2(\n"
            f"  layers={self.layer_sizes},\n"
            f"  dendrites={self.use_dendrites},\n"
            f"  oscillators={self.use_oscillators},\n"
            f"  plasticity={self.use_plasticity},\n"
            f"  output_mode='{self.output_mode}',\n"
            f"  parameters={total_params:,}\n"
            f")"
        )
