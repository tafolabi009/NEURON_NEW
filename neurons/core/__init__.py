"""
Core components for NEURONS architecture
"""

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
