"""
NEURONS Core Package
Biologically-Inspired Neural Architecture
"""

__version__ = "1.0.0"
__author__ = "Oluwatosin Abioye Afolabi"
__email__ = "afolabi@genovotech.com"

# Removed old network.py import (file deleted)
from neurons.core.neuron import LIFNeuron, AdaptiveLIFNeuron
from neurons.core.plasticity import TripletSTDP, VoltageSTDP, SynapticScaling
from neurons.core.neuromodulation import NeuromodulatorSystem
from neurons.core.oscillations import NeuralOscillations
from neurons.core.ewc import ElasticWeightConsolidation

__all__ = [
    "LIFNeuron",
    "AdaptiveLIFNeuron",
    "TripletSTDP",
    "VoltageSTDP",
    "SynapticScaling",
    "NeuromodulatorSystem",
    "NeuralOscillations",
    "ElasticWeightConsolidation",
]
