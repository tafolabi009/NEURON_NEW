"""
Specialized Resonance Models
Domain-specific architectures for different use cases

Available models:
- Language Model: Text generation and understanding
- Code Model: Code generation and analysis
- Vision Model: Image classification and understanding
- Audio Model: Audio processing and recognition
- Multimodal Model: Cross-modal understanding

All models can be exported for integration with other applications
"""

from resonance_nn.models.specialized.language_model import (
    ResonanceLanguageModel,
    ResonanceCausalLM,
)
from resonance_nn.models.specialized.code_model import (
    ResonanceCodeModel,
)
from resonance_nn.models.specialized.vision_model import (
    ResonanceVisionModel,
)
from resonance_nn.models.specialized.audio_model import (
    ResonanceAudioModel,
)

__all__ = [
    'ResonanceLanguageModel',
    'ResonanceCausalLM',
    'ResonanceCodeModel',
    'ResonanceVisionModel',
    'ResonanceAudioModel',
]
