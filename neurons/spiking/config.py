"""
NEURONSv2 configuration.

Centralized configuration for the revolutionary spiking neural architecture.
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class NEURONSv2Config:
    """
    Configuration for NEURONSv2 spiking neural architecture.
    
    This is NOT a transformer configuration! It defines parameters for:
    - Spiking neuron dynamics
    - Dendritic computation
    - Oscillatory dynamics
    - Hebbian plasticity
    - Temporal coding
    """
    
    # ===================================================================
    # Model Architecture
    # ===================================================================
    vocab_size: int = 50257
    input_dim: int = 768
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 1024, 1024, 768])
    
    # ===================================================================
    # Spiking Dynamics
    # ===================================================================
    membrane_tau: float = 20.0  # Membrane time constant (ms)
    threshold: float = 1.0  # Spiking threshold
    refractory_period: float = 2.0  # Refractory period (ms)
    spike_function: str = "surrogate_gradient"
    
    # ===================================================================
    # Dendritic Computation
    # ===================================================================
    num_basal_branches: int = 12  # Basal dendrites (bottom-up input)
    num_apical_branches: int = 6  # Apical dendrites (top-down prediction)
    branch_tau: float = 10.0  # Dendritic integration time constant (ms)
    nmda_activation: bool = True  # NMDA-like nonlinearities
    
    # ===================================================================
    # Oscillatory Dynamics - NO ATTENTION!
    # ===================================================================
    theta_freq: float = 6.0  # Theta (4-8 Hz): working memory, sequences
    gamma_freq: float = 60.0  # Gamma (30-100 Hz): binding, local processing
    beta_freq: float = 20.0  # Beta (15-30 Hz): top-down predictions
    coupling_strength: float = 0.3  # Phase coupling strength
    
    # ===================================================================
    # Hebbian Plasticity - NO BACKPROP!
    # ===================================================================
    use_stdp: bool = True  # Spike-timing dependent plasticity
    use_bcm: bool = True  # BCM homeostatic rule
    use_predictive: bool = True  # Predictive coding
    tau_fast: float = 1.0  # Fast plasticity timescale (s)
    tau_slow: float = 10000.0  # Slow consolidation timescale (s)
    
    # ===================================================================
    # Temporal Coding
    # ===================================================================
    temporal_resolution_ms: float = 1.0  # Spike timing precision
    max_firing_rate: float = 200.0  # Hz
    use_phase_coding: bool = True  # Encode in oscillation phase
    use_latency_coding: bool = True  # Encode in first spike latency
    
    # ===================================================================
    # Network Topology (Biological Realism)
    # ===================================================================
    sparsity: float = 0.1  # Connection sparsity (brain is ~10% connected)
    lateral_inhibition: float = 0.5  # Lateral inhibition strength
    feedback_strength: float = 0.3  # Top-down feedback strength
    
    # ===================================================================
    # Context Handling (200K+ tokens!)
    # ===================================================================
    max_sequence_length: int = 200000  # 200K token context
    temporal_compression: int = 8  # Compression per layer
    num_timescales: int = 6  # Hierarchical temporal scales
    
    # ===================================================================
    # Training
    # ===================================================================
    learning_rate_fast: float = 0.01  # Fast synaptic learning
    learning_rate_slow: float = 0.0001  # Slow structural learning
    homeostasis_rate: float = 0.001  # BCM threshold adaptation
    
    # ===================================================================
    # Modality Support
    # ===================================================================
    modality: str = "text"  # "text", "vision", "audio", "multimodal"
    vision_patch_size: int = 16
    audio_sample_rate: int = 16000
    
    # ===================================================================
    # Optimization
    # ===================================================================
    use_sparse_computation: bool = True  # Event-driven (only active neurons)
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.membrane_tau > 0, "Membrane tau must be positive"
        assert self.threshold > 0, "Threshold must be positive"
        assert 0 < self.sparsity <= 1, "Sparsity must be in (0, 1]"
        assert len(self.hidden_dims) > 0, "Must have at least one hidden layer"


__all__ = ['NEURONSv2Config']
