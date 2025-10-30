"""
Spiking neural network components.

This module implements biologically-inspired spiking neurons with:
- Leaky Integrate-and-Fire (LIF) dynamics
- Multi-compartment dendritic computation
- Oscillatory dynamics (Kuramoto model)
- Hebbian plasticity (STDP, BCM)
"""

from .config import NEURONSv2Config
from .spiking import SpikingNeuron, SpikingLayer, SurrogateGradient
from .dendrites import DendriticNeuron, MultiCompartmentLayer
from .oscillators import KuramotoOscillator, OscillatoryPopulation
from .plasticity import (
    STDPSynapse,
    BCMLearning,
    HebbianPlasticityLayer,
    FastSlowLearning
)
from .model import NEURONSv2Layer, NEURONSv2

__all__ = [
    'NEURONSv2Config',
    'SpikingNeuron',
    'SpikingLayer',
    'SurrogateGradient',
    'DendriticNeuron',
    'MultiCompartmentLayer',
    'KuramotoOscillator',
    'OscillatoryPopulation',
    'STDPSynapse',
    'BCMLearning',
    'HebbianPlasticityLayer',
    'FastSlowLearning',
    'NEURONSv2Layer',
    'NEURONSv2',
]

from .spiking import SurrogateGradient, spike_function, SpikingNeuron
from .dendrites import DendriticBranch, DendriticNeuron
from .oscillators import OscillatoryPopulation, MultiScaleOscillator
from .plasticity import HebbianSynapse

__all__ = [
    # Spiking primitives
    'SurrogateGradient',
    'spike_function',
    'SpikingNeuron',
    # Dendritic computation
    'DendriticBranch',
    'DendriticNeuron',
    # Oscillatory dynamics
    'OscillatoryPopulation',
    'MultiScaleOscillator',
    # Hebbian plasticity
    'HebbianSynapse',
]
