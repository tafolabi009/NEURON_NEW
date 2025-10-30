"""
NEURONSv2 - Revolutionary Neural Architecture
==============================================

A completely novel approach to neural computation based on biological principles:
- Spiking neural dynamics (not continuous activations)
- Hebbian plasticity (not backpropagation)
- Oscillatory synchronization (not attention mechanisms)
- Dendritic computation (not linear layers)
- Predictive coding (not supervised learning)

This is NOT a transformer. This is something entirely different.

Author: Oluwatosin Abioye Afolabi
License: MIT
"""

"""
NEURONSv2: A biologically-inspired spiking neural network library.

This library implements a novel neural network architecture with:
- Leaky Integrate-and-Fire spiking neurons
- Multi-compartment dendritic computation
- Oscillatory dynamics (Kuramoto model) for temporal binding
- Hebbian plasticity (STDP, BCM) for local learning
- Predictive coding through feedback connections

This is NOT a standard feedforward network - it's a temporal, event-driven
architecture that processes information through spike timing and phase synchronization.
"""

from .spiking import (
    NEURONSv2,
    NEURONSv2Layer,
    NEURONSv2Config,
    SpikingNeuron,
    SpikingLayer,
    DendriticNeuron,
    MultiCompartmentLayer,
    KuramotoOscillator,
    OscillatoryPopulation,
    STDPSynapse,
    BCMLearning,
    HebbianPlasticityLayer,
)

from .utils import (
    setup_logging,
    create_data_loader,
    prepare_data,
    calculate_accuracy,
    calculate_metrics,
    plot_training_history,
    plot_confusion_matrix,
)

__version__ = "2.0.0"
__all__ = [
    # Main model
    'NEURONSv2',
    'NEURONSv2Layer',
    'NEURONSv2Config',
    # Core components
    'SpikingNeuron',
    'SpikingLayer',
    'DendriticNeuron',
    'MultiCompartmentLayer',
    'KuramotoOscillator',
    'OscillatoryPopulation',
    'STDPSynapse',
    'BCMLearning',
    'HebbianPlasticityLayer',
    # Utilities
    'setup_logging',
    'create_data_loader',
    'prepare_data',
    'calculate_accuracy',
    'calculate_metrics',
    'plot_training_history',
    'plot_confusion_matrix',
]

# Import spiking architecture (the novel one!)
from .spiking import (
    NEURONSv2,
    NEURONSv2Layer,
    NEURONSv2Config,
    SpikingNeuron,
    DendriticNeuron,
    KuramotoOscillator,
    STDPSynapse,
    BCMLearning,
)

# Import standard architecture (for comparison)
from .core import NeuralNetwork, Trainer

# Import utilities
from .activations import get_activation, ACTIVATION_REGISTRY
from .losses import get_loss, LOSS_REGISTRY
from .optimizers import get_optimizer, OPTIMIZER_REGISTRY
from .utils import (
    setup_logging,
    create_data_loader,
    prepare_data,
    calculate_accuracy,
    calculate_metrics,
    plot_training_history,
    plot_confusion_matrix,
)

__version__ = "2.0.0"
__all__ = [
    # ===== NOVEL SPIKING ARCHITECTURE =====
    'NEURONSv2',  # Main spiking model
    'NEURONSv2Layer',
    'NEURONSv2Config',
    'SpikingNeuron',
    'DendriticNeuron',
    'KuramotoOscillator',
    'STDPSynapse',
    'BCMLearning',
    
    # ===== STANDARD ARCHITECTURE (for comparison) =====
    'NeuralNetwork',
    'Trainer',
    
    # ===== UTILITIES =====
    'get_activation',
    'get_loss',
    'get_optimizer',
    'ACTIVATION_REGISTRY',
    'LOSS_REGISTRY',
    'OPTIMIZER_REGISTRY',
    'setup_logging',
    'create_data_loader',
    'prepare_data',
    'calculate_accuracy',
    'calculate_metrics',
    'plot_training_history',
    'plot_confusion_matrix',
]
