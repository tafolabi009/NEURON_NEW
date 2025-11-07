"""
Multimodal Resonance Neural Networks
Frequency-domain processing for vision, audio, and cross-modal fusion

Unlike CNNs which use spatial convolutions, we use:
- 2D/3D FFT for spatial/temporal frequency analysis
- Resonance chambers for feature extraction
- Holographic patterns for cross-modal binding
"""

from resonance_nn.multimodal.vision import (
    ResonanceVisionEncoder,
    ResonancePatchProcessor,
    SpatialFrequencyProcessor,
)
from resonance_nn.multimodal.audio import (
    ResonanceAudioEncoder,
    SpectrogramResonance,
    TemporalFrequencyProcessor,
)
from resonance_nn.multimodal.fusion import (
    MultiModalResonanceFusion,
    CrossModalResonance,
    HolographicModalityBinder,
)

__all__ = [
    'ResonanceVisionEncoder',
    'ResonancePatchProcessor',
    'SpatialFrequencyProcessor',
    'ResonanceAudioEncoder',
    'SpectrogramResonance',
    'TemporalFrequencyProcessor',
    'MultiModalResonanceFusion',
    'CrossModalResonance',
    'HolographicModalityBinder',
]
