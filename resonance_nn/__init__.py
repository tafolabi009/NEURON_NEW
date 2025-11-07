"""
Resonance Neural Networks Package
Frequency-Domain Information Processing with Holographic Memory

This package implements the Resonance Neural Networks architecture 
from the paper "Resonance Neural Networks: Frequency-Domain Information 
Processing with Holographic Memory and Provable Efficiency Guarantees"
by Oluwatosin A. Afolabi (Genovo Technologies, 2025).

Key Features:
- O(n log n) computational complexity through frequency-domain processing
- Stable gradient computation for oscillatory parameters
- Holographic memory with provable capacity guarantees
- 4-6x parameter efficiency compared to transformers
- Ultra-long context support (260K-300K tokens)
- Large vocabulary support (500K-1M tokens)
- Multimodal capabilities (vision, audio, text)
- NO attention mechanism - pure frequency processing

Example:
    >>> from resonance_nn import ResonanceNet
    >>> from resonance_nn.models.specialized import ResonanceLanguageModel
    >>> from resonance_nn.models.long_context import LongContextResonanceNet
    >>> import torch
    >>> 
    >>> # Create standard model
    >>> model = ResonanceNet(
    ...     input_dim=512,
    ...     num_frequencies=64,
    ...     hidden_dim=256,
    ...     num_layers=4,
    ... )
    >>> 
    >>> # Create long context language model
    >>> lm = ResonanceLanguageModel(
    ...     vocab_size=50000,
    ...     max_seq_length=262144,  # 256K tokens
    ... )
    >>> 
    >>> # Process sequence
    >>> x = torch.randn(32, 128, 512)
    >>> output = model(x)
"""

__version__ = "2.0.0"
__author__ = "Oluwatosin A. Afolabi"
__license__ = "MIT"

# Core layers
from resonance_nn.layers.resonance import (
    ResonanceLayer,
    MultiScaleResonanceLayer,
    AdaptiveResonanceLayer,
    ComplexWeight,
)
from resonance_nn.layers.holographic import (
    HolographicMemory,
)
from resonance_nn.layers.embeddings import (
    HierarchicalVocabularyEmbedding,
    FrequencyCompressedEmbedding,
    AdaptiveEmbedding,
    ResonanceHashEmbedding,
    FrequencyPositionalEncoding,
)

# Base models
from resonance_nn.models.resonance_net import (
    ResonanceNet,
    ResonanceEncoder,
    ResonanceAutoencoder,
    ResonanceClassifier,
)

# Long context
from resonance_nn.models.long_context import (
    LongContextResonanceNet,
    StreamingLongContextNet,
)

# Specialized models
from resonance_nn.models.specialized import (
    ResonanceLanguageModel,
    ResonanceCausalLM,
    ResonanceCodeModel,
    ResonanceVisionModel,
    ResonanceAudioModel,
)

# Multimodal
from resonance_nn.multimodal import (
    ResonanceVisionEncoder,
    ResonanceAudioEncoder,
    MultiModalResonanceFusion,
    CrossModalResonance,
    HolographicModalityBinder,
)

# Training
from resonance_nn.training.trainer import (
    ResonanceTrainer,
    ResonanceAutoEncoderTrainer,
    ResonanceClassifierTrainer,
    create_criterion,
    create_trainer,
)

__version__ = "0.1.0"
__author__ = "Oluwatosin A. Afolabi"
__email__ = "afolabi@genovotech.com"
__license__ = "MIT"

__all__ = [
    # Layers
    "ResonanceLayer",
    "MultiScaleResonanceLayer",
    "AdaptiveResonanceLayer",
    "ComplexWeight",
    # Holographic Memory
    "HolographicMemory",
    "MultiModalHolographicMemory",
    "AdaptiveHolographicMemory",
    # Models
    "ResonanceNet",
    "ResonanceEncoder",
    "ResonanceAutoencoder",
    "ResonanceClassifier",
    # Training
    "ResonanceTrainer",
    "ResonanceAutoEncoderTrainer",
    "ResonanceClassifierTrainer",
    "create_criterion",
    "create_trainer",
]
