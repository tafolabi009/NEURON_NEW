"""
Multimodal Fusion Architecture
Unified sensory processing for vision, audio, text, and other modalities

This enables NEURONSv2 to process multiple modalities like humans do:
1. Modality-specific encoders
2. Cross-modal predictive coding
3. Unified sensory representations
4. Modality-agnostic processing

Key Innovation: Different modalities share the same neural substrate!
Just like cortex processes visual, auditory, and somatosensory information
using the same basic architecture.

References:
- Ghazanfar & Schroeder (2006): Is neocortex essentially multisensory?
- Driver & Noesselt (2008): Multisensory interplay
- Hawkins & Blakeslee (2004): On Intelligence (cortical uniformity)
- Ramesh et al. (2021): CLIP (multimodal learning)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class Modality(Enum):
    """Supported modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PROPRIOCEPTION = "proprioception"  # For robotics
    MOLECULAR = "molecular"  # For chemistry/biology
    GRAPH = "graph"  # For graph data
    TIME_SERIES = "time_series"


@dataclass
class ModalityConfig:
    """Configuration for a specific modality"""
    modality: Modality
    input_shape: Tuple[int, ...]
    embedding_dim: int
    preprocessing: str = "standard"  # 'standard', 'normalize', 'whitening'
    encoding_method: str = "temporal"  # 'temporal', 'spatial', 'spectral'


class ModalityEncoder:
    """
    Base class for modality-specific encoders
    
    Each modality has unique input characteristics but maps to the same
    unified representation space.
    
    Philosophy: The brain uses the same cortical architecture for all senses!
    Vision, audio, touch all use similar neural circuits - just different input patterns.
    """
    
    def __init__(self, config: ModalityConfig):
        self.config = config
        self.embedding_dim = config.embedding_dim
        
    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """
        Encode modality-specific input to unified representation
        
        Args:
            inputs: Raw input in modality-specific format
            
        Returns:
            embedding: (embedding_dim,) unified representation
        """
        raise NotImplementedError
    
    def decode(self, embedding: np.ndarray) -> np.ndarray:
        """
        Decode unified representation back to modality-specific format
        
        Args:
            embedding: (embedding_dim,) unified representation
            
        Returns:
            outputs: Reconstructed input
        """
        raise NotImplementedError


class TextEncoder(ModalityEncoder):
    """
    Text Encoder
    
    Converts text tokens → unified temporal representation
    Uses the rank-order and phase coding from our temporal system!
    """
    
    def __init__(self, config: ModalityConfig, vocab_size: int):
        super().__init__(config)
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, config.embedding_dim) * 0.1
        
        # Positional encoding (temporal-based)
        self.max_seq_length = 200000  # Support long context!
        
    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Encode text tokens
        
        Args:
            token_ids: (seq_length,) token indices
            
        Returns:
            embedding: (seq_length, embedding_dim) temporal embedding
        """
        # Get token embeddings
        embeddings = self.token_embeddings[token_ids]
        
        # Add positional information (temporal phase encoding)
        seq_length = len(token_ids)
        positions = np.arange(seq_length)
        
        # Temporal position encoding
        freq = 1.0 / (10000 ** (np.arange(0, self.embedding_dim, 2) / self.embedding_dim))
        pos_encoding = np.zeros((seq_length, self.embedding_dim))
        pos_encoding[:, 0::2] = np.sin(positions[:, None] * freq)
        pos_encoding[:, 1::2] = np.cos(positions[:, None] * freq)
        
        # Combine
        embeddings = embeddings + 0.1 * pos_encoding
        
        return embeddings
    
    def decode(self, embedding: np.ndarray) -> np.ndarray:
        """Decode embedding to token probabilities"""
        # Project back to vocabulary
        logits = embedding @ self.token_embeddings.T
        return logits


class ImageEncoder(ModalityEncoder):
    """
    Image Encoder
    
    Converts images → unified spatial-temporal representation
    Uses hierarchical processing like V1 → V2 → V4 → IT
    """
    
    def __init__(self, config: ModalityConfig, patch_size: int = 16):
        super().__init__(config)
        self.patch_size = patch_size
        
        # Expect input shape: (height, width, channels)
        if len(config.input_shape) == 3:
            self.height, self.width, self.channels = config.input_shape
        else:
            self.height = config.input_shape[0]
            self.width = config.input_shape[0]
            self.channels = 3
        
        # Patch embedding projection
        self.n_patches_h = self.height // patch_size
        self.n_patches_w = self.width // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        patch_dim = patch_size * patch_size * self.channels
        self.patch_projection = np.random.randn(patch_dim, config.embedding_dim) * 0.01
        
        # Spatial position embeddings (like retinotopic maps!)
        self.position_embeddings = np.random.randn(self.n_patches, config.embedding_dim) * 0.01
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image as patches
        
        Args:
            image: (height, width, channels) image
            
        Returns:
            embedding: (n_patches, embedding_dim) patch embeddings
        """
        # Extract patches
        patches = []
        for i in range(self.n_patches_h):
            for j in range(self.n_patches_w):
                h_start = i * self.patch_size
                h_end = h_start + self.patch_size
                w_start = j * self.patch_size
                w_end = w_start + self.patch_size
                
                if len(image.shape) == 3:
                    patch = image[h_start:h_end, w_start:w_end, :]
                else:
                    patch = image[h_start:h_end, w_start:w_end]
                    patch = np.expand_dims(patch, -1)
                
                patches.append(patch.flatten())
        
        patches = np.array(patches)  # (n_patches, patch_dim)
        
        # Project to embedding space
        embeddings = patches @ self.patch_projection
        
        # Add positional encoding (retinotopic)
        embeddings = embeddings + self.position_embeddings
        
        return embeddings
    
    def decode(self, embeddings: np.ndarray) -> np.ndarray:
        """Decode patch embeddings back to image"""
        # Project back to patch space
        patches = embeddings @ self.patch_projection.T
        
        # Reshape to image
        if self.channels == 1:
            image = np.zeros((self.height, self.width))
        else:
            image = np.zeros((self.height, self.width, self.channels))
        
        patch_idx = 0
        for i in range(self.n_patches_h):
            for j in range(self.n_patches_w):
                h_start = i * self.patch_size
                h_end = h_start + self.patch_size
                w_start = j * self.patch_size
                w_end = w_start + self.patch_size
                
                patch = patches[patch_idx].reshape(self.patch_size, self.patch_size, -1)
                if self.channels == 1:
                    image[h_start:h_end, w_start:w_end] = patch[:, :, 0]
                else:
                    image[h_start:h_end, w_start:w_end, :] = patch
                
                patch_idx += 1
        
        return image


class AudioEncoder(ModalityEncoder):
    """
    Audio Encoder
    
    Converts audio → unified spectral-temporal representation
    Uses cochlear-like frequency decomposition
    """
    
    def __init__(self, config: ModalityConfig, sample_rate: int = 16000, n_mels: int = 80):
        super().__init__(config)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Mel filterbank (cochlear frequency decomposition)
        self.mel_filterbank = self._create_mel_filterbank()
        
        # Temporal projection
        self.temporal_projection = np.random.randn(n_mels, config.embedding_dim) * 0.01
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel-scale filterbank (like cochlea!)"""
        # Simple mel filterbank (in practice would use librosa)
        n_fft = 512
        mel_filters = np.random.rand(self.n_mels, n_fft // 2 + 1) * 0.1
        
        # Make triangular filters
        for i in range(self.n_mels):
            center = int((i / self.n_mels) * (n_fft // 2))
            width = n_fft // (2 * self.n_mels)
            
            start = max(0, center - width)
            end = min(n_fft // 2 + 1, center + width)
            
            mel_filters[i, :] = 0
            for j in range(start, center):
                mel_filters[i, j] = (j - start) / (center - start)
            for j in range(center, end):
                mel_filters[i, j] = (end - j) / (end - center)
        
        return mel_filters
    
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """
        Encode audio waveform
        
        Args:
            audio: (n_samples,) audio waveform
            
        Returns:
            embedding: (n_frames, embedding_dim) temporal embedding
        """
        # STFT
        n_fft = 512
        hop_length = 160
        
        # Simple STFT (in practice would use scipy/librosa)
        n_frames = (len(audio) - n_fft) // hop_length + 1
        spectrogram = np.zeros((n_fft // 2 + 1, n_frames))
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            
            fft = np.fft.rfft(frame * np.hanning(n_fft))
            spectrogram[:, i] = np.abs(fft)
        
        # Apply mel filterbank
        mel_spec = self.mel_filterbank @ spectrogram  # (n_mels, n_frames)
        
        # Log compression (like human hearing)
        mel_spec = np.log(mel_spec + 1e-8)
        
        # Project to embedding space
        embeddings = mel_spec.T @ self.temporal_projection  # (n_frames, embedding_dim)
        
        return embeddings
    
    def decode(self, embeddings: np.ndarray) -> np.ndarray:
        """Decode embeddings to mel spectrogram"""
        # Project back to mel space
        mel_spec = embeddings @ self.temporal_projection.T
        return mel_spec


class CrossModalBindingNetwork:
    """
    Cross-Modal Binding Network
    
    Binds representations across modalities using predictive coding.
    
    Key Idea: Different modalities should predict each other!
        - See a dog → predict bark sound
        - Hear "cat" → predict cat image
        - See text "hot" → predict warmth sensation
        
    This is how the brain creates unified percepts!
    
    Mathematical Model:
        Prediction: X_pred^(modality_j) = f(X^(modality_i))
        Error: E = X^(modality_j) - X_pred^(modality_j)
        Learning: Minimize E across all modality pairs
    """
    
    def __init__(self, modalities: List[Modality], embedding_dim: int):
        self.modalities = modalities
        self.embedding_dim = embedding_dim
        self.n_modalities = len(modalities)
        
        # Cross-modal prediction matrices
        # For each pair (i, j): predict modality j from modality i
        self.cross_modal_predictors = {}
        
        for i, mod_i in enumerate(modalities):
            for j, mod_j in enumerate(modalities):
                if i != j:
                    key = (mod_i.value, mod_j.value)
                    # Initialize prediction matrix
                    self.cross_modal_predictors[key] = np.random.randn(embedding_dim, embedding_dim) * 0.01
        
        # Unified representation (multimodal fusion)
        self.unified_projection = np.random.randn(embedding_dim * self.n_modalities, embedding_dim) * 0.01
    
    def predict_cross_modal(self, source_modality: str, target_modality: str,
                           source_embedding: np.ndarray) -> np.ndarray:
        """
        Predict target modality from source modality
        
        Args:
            source_modality: Source modality name
            target_modality: Target modality name
            source_embedding: Source embedding
            
        Returns:
            predicted_embedding: Predicted target embedding
        """
        key = (source_modality, target_modality)
        if key not in self.cross_modal_predictors:
            return np.zeros_like(source_embedding)
        
        predictor = self.cross_modal_predictors[key]
        
        # Predict
        if source_embedding.ndim == 1:
            predicted = source_embedding @ predictor
        else:
            # Batch prediction
            predicted = source_embedding @ predictor
        
        return predicted
    
    def compute_cross_modal_error(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute prediction errors across all modality pairs
        
        Args:
            embeddings: Dictionary of embeddings per modality
            
        Returns:
            errors: Dictionary of prediction errors
        """
        errors = {}
        
        for source_mod in embeddings:
            for target_mod in embeddings:
                if source_mod != target_mod:
                    # Get embeddings (handle multi-dimensional)
                    source_embed = embeddings[source_mod]
                    target_embed = embeddings[target_mod]
                    
                    # Average if multi-dimensional
                    if source_embed.ndim > 1:
                        source_embed = np.mean(source_embed, axis=0)
                    if target_embed.ndim > 1:
                        target_embed = np.mean(target_embed, axis=0)
                    
                    # Predict target from source
                    predicted = self.predict_cross_modal(source_mod, target_mod, source_embed)
                    
                    # Compute error
                    error = np.mean((target_embed - predicted) ** 2)
                    errors[f"{source_mod}->{target_mod}"] = error
        
        return errors
    
    def fuse_modalities(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse multiple modalities into unified representation
        
        Args:
            embeddings: Dictionary of embeddings per modality
            
        Returns:
            unified: Unified multimodal embedding
        """
        # Concatenate all modality embeddings
        concat_embeddings = []
        for modality in self.modalities:
            if modality.value in embeddings:
                embed = embeddings[modality.value]
                # Ensure 1D
                if embed.ndim > 1:
                    embed = np.mean(embed, axis=0)
                concat_embeddings.append(embed)
            else:
                # Missing modality - use zeros
                concat_embeddings.append(np.zeros(self.embedding_dim))
        
        concat = np.concatenate(concat_embeddings)
        
        # Project to unified space
        unified = concat @ self.unified_projection
        
        return unified
    
    def learn_cross_modal_predictions(self, embeddings: Dict[str, np.ndarray],
                                     learning_rate: float = 0.001):
        """
        Learn to predict each modality from others
        
        This is the key learning mechanism for multimodal binding!
        """
        for source_mod in embeddings:
            for target_mod in embeddings:
                if source_mod != target_mod:
                    key = (source_mod, target_mod)
                    if key not in self.cross_modal_predictors:
                        continue
                    
                    # Get embeddings
                    source_embed = embeddings[source_mod]
                    target_embed = embeddings[target_mod]
                    
                    # Handle multi-dimensional embeddings
                    if source_embed.ndim > 1:
                        source_embed = np.mean(source_embed, axis=0)
                    if target_embed.ndim > 1:
                        target_embed = np.mean(target_embed, axis=0)
                    
                    # Predict
                    predicted = source_embed @ self.cross_modal_predictors[key]
                    
                    # Compute gradient
                    error = target_embed - predicted
                    gradient = np.outer(source_embed, error)
                    
                    # Update
                    self.cross_modal_predictors[key] += learning_rate * gradient


class MultimodalNEURONSv2:
    """
    Complete Multimodal NEURONSv2 System
    
    This is the main interface for multimodal learning!
    
    Features:
        1. Multiple modality encoders
        2. Cross-modal prediction and binding
        3. Unified sensory representation
        4. Task-specific decoders
    """
    
    def __init__(self, modality_configs: List[ModalityConfig], vocab_size: int = 50000):
        self.modality_configs = {config.modality: config for config in modality_configs}
        self.embedding_dim = modality_configs[0].embedding_dim
        
        # Create encoders
        self.encoders = {}
        for config in modality_configs:
            if config.modality == Modality.TEXT:
                self.encoders[Modality.TEXT] = TextEncoder(config, vocab_size)
            elif config.modality == Modality.IMAGE:
                self.encoders[Modality.IMAGE] = ImageEncoder(config)
            elif config.modality == Modality.AUDIO:
                self.encoders[Modality.AUDIO] = AudioEncoder(config)
            # Add more modalities as needed
        
        # Cross-modal binding
        self.binding_network = CrossModalBindingNetwork(
            list(self.encoders.keys()),
            self.embedding_dim
        )
        
        print(f"MultimodalNEURONSv2 initialized:")
        print(f"  - Modalities: {[m.value for m in self.encoders.keys()]}")
        print(f"  - Embedding dim: {self.embedding_dim}")
        print(f"  - Cross-modal predictors: {len(self.binding_network.cross_modal_predictors)}")
    
    def encode(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Encode inputs from multiple modalities
        
        Args:
            inputs: Dictionary of inputs per modality
            
        Returns:
            embeddings: Dictionary of embeddings per modality
        """
        embeddings = {}
        
        for modality_name, data in inputs.items():
            # Find matching modality
            modality = None
            for mod in self.encoders.keys():
                if mod.value == modality_name:
                    modality = mod
                    break
            
            if modality is not None:
                embeddings[modality_name] = self.encoders[modality].encode(data)
        
        return embeddings
    
    def forward(self, inputs: Dict[str, np.ndarray], learn_cross_modal: bool = True) -> Dict[str, Any]:
        """
        Full forward pass with multimodal fusion
        
        Args:
            inputs: Dictionary of inputs per modality
            learn_cross_modal: Whether to update cross-modal predictions
            
        Returns:
            outputs: Dictionary containing embeddings, unified representation, and errors
        """
        # Encode each modality
        embeddings = self.encode(inputs)
        
        # Fuse modalities
        unified = self.binding_network.fuse_modalities(embeddings)
        
        # Compute cross-modal errors
        cross_modal_errors = self.binding_network.compute_cross_modal_error(embeddings)
        
        # Learn cross-modal predictions
        if learn_cross_modal:
            self.binding_network.learn_cross_modal_predictions(embeddings)
        
        return {
            'embeddings': embeddings,
            'unified': unified,
            'cross_modal_errors': cross_modal_errors
        }
    
    def predict_modality(self, source_modality: str, target_modality: str,
                        source_input: np.ndarray) -> np.ndarray:
        """
        Predict one modality from another
        
        Example: predict_modality('text', 'image', "a cat") → cat image
        """
        # Encode source
        source_mod = None
        for mod in self.encoders.keys():
            if mod.value == source_modality:
                source_mod = mod
                break
        
        if source_mod is None:
            raise ValueError(f"Unknown source modality: {source_modality}")
        
        source_embedding = self.encoders[source_mod].encode(source_input)
        
        # Predict target
        predicted_embedding = self.binding_network.predict_cross_modal(
            source_modality,
            target_modality,
            source_embedding
        )
        
        # Decode target
        target_mod = None
        for mod in self.encoders.keys():
            if mod.value == target_modality:
                target_mod = mod
                break
        
        if target_mod is None:
            raise ValueError(f"Unknown target modality: {target_modality}")
        
        predicted_output = self.encoders[target_mod].decode(predicted_embedding)
        
        return predicted_output


# Quick test
if __name__ == "__main__":
    print("Testing Multimodal Fusion Architecture...")
    
    # Define modalities
    configs = [
        ModalityConfig(Modality.TEXT, (512,), embedding_dim=512),
        ModalityConfig(Modality.IMAGE, (224, 224, 3), embedding_dim=512),
        ModalityConfig(Modality.AUDIO, (16000,), embedding_dim=512),
    ]
    
    # Create system
    system = MultimodalNEURONSv2(configs, vocab_size=10000)
    
    # Test inputs
    inputs = {
        'text': np.random.randint(0, 10000, size=50),  # 50 tokens
        'image': np.random.randn(224, 224, 3),
        'audio': np.random.randn(16000),
    }
    
    print("\nProcessing multimodal inputs...")
    outputs = system.forward(inputs)
    
    print(f"\nResults:")
    print(f"  - Text embedding shape: {outputs['embeddings']['text'].shape}")
    print(f"  - Image embedding shape: {outputs['embeddings']['image'].shape}")
    print(f"  - Audio embedding shape: {outputs['embeddings']['audio'].shape}")
    print(f"  - Unified embedding shape: {outputs['unified'].shape}")
    print(f"  - Cross-modal errors: {len(outputs['cross_modal_errors'])}")
    for key, error in outputs['cross_modal_errors'].items():
        print(f"      {key}: {error:.6f}")
    
    print("\n✓ Multimodal Fusion working!")
