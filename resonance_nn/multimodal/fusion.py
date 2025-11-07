"""
Cross-Modal Resonance Fusion
Bind and fuse different modalities using holographic interference patterns

Key innovations:
- Holographic binding of modalities (like binding problem in neuroscience)
- Frequency-domain cross-modal attention
- Multi-modal resonance chambers
- No traditional cross-attention (O(n²))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class HolographicModalityBinder(nn.Module):
    """
    Bind modalities using holographic interference patterns
    
    Inspired by neural binding: different modalities create interference
    patterns that encode their relationships without explicit alignment
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hologram_dim: int = 1024,
        binding_strength: float = 0.5,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.hologram_dim = hologram_dim
        self.binding_strength = binding_strength
        
        # Projection to hologram space for each modality
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, hologram_dim)
            for name, dim in modality_dims.items()
        })
        
        # Phase encoders for each modality (creates unique frequency signature)
        self.phase_encoders = nn.ModuleDict({
            name: nn.Parameter(torch.randn(hologram_dim) * 0.1)
            for name in modality_dims.keys()
        })
        
        # Holographic memory for bound patterns
        from resonance_nn.layers.holographic import HolographicMemory
        
        self.holographic_memory = HolographicMemory(
            pattern_dim=hologram_dim,
            hologram_dim=hologram_dim * 2,
            capacity=1000,
        )
        
    def bind(
        self,
        modality_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Bind multiple modalities into holographic pattern
        
        Args:
            modality_features: Dictionary of {modality_name: features}
                              Each feature: (batch, seq_len, modality_dim)
                              
        Returns:
            Bound holographic pattern (batch, hologram_dim)
        """
        batch_size = list(modality_features.values())[0].shape[0]
        
        # Project each modality to hologram space
        projected_modalities = {}
        for name, features in modality_features.items():
            # Pool sequence dimension
            pooled = features.mean(dim=1) if features.dim() == 3 else features
            
            # Project to hologram space
            projected = self.modality_projections[name](pooled)
            
            # Add phase encoding (unique signature per modality)
            phase = self.phase_encoders[name]
            projected_complex = torch.complex(
                projected * torch.cos(phase),
                projected * torch.sin(phase),
            )
            
            projected_modalities[name] = projected_complex
        
        # Create interference pattern (holographic binding)
        # Sum of all modality patterns
        bound_pattern = sum(projected_modalities.values())
        
        # Take magnitude (intensity pattern)
        bound_magnitude = torch.abs(bound_pattern)
        
        # Normalize
        bound_magnitude = F.normalize(bound_magnitude, p=2, dim=-1)
        
        return bound_magnitude
    
    def unbind(
        self,
        bound_pattern: torch.Tensor,
        target_modality: str,
    ) -> torch.Tensor:
        """
        Retrieve specific modality from bound pattern
        
        Args:
            bound_pattern: Bound holographic pattern (batch, hologram_dim)
            target_modality: Name of modality to retrieve
            
        Returns:
            Retrieved modality features (batch, hologram_dim)
        """
        # Create reference signal for target modality
        phase = self.phase_encoders[target_modality]
        reference = torch.complex(
            torch.cos(phase).unsqueeze(0),
            torch.sin(phase).unsqueeze(0),
        )
        
        # Multiply bound pattern by conjugate of reference
        bound_complex = torch.complex(bound_pattern, torch.zeros_like(bound_pattern))
        retrieved = bound_complex * torch.conj(reference)
        
        # Take real part
        retrieved_real = torch.real(retrieved)
        
        return retrieved_real


class CrossModalResonance(nn.Module):
    """
    Cross-modal interaction using frequency-domain resonance
    Replaces traditional cross-attention with O(n log n) complexity
    """
    
    def __init__(
        self,
        dim: int,
        num_frequencies: int = 64,
        num_modalities: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_frequencies = num_frequencies
        self.num_modalities = num_modalities
        
        # Modality-specific frequency filters
        self.modality_filters = nn.ModuleList([
            nn.Parameter(torch.randn(num_frequencies, dim, 2) * 0.1)
            for _ in range(num_modalities)
        ])
        
        # Cross-modal interaction weights
        self.cross_modal_weights = nn.Parameter(
            torch.randn(num_modalities, num_modalities, num_frequencies) * 0.1
        )
        
        # Output projection
        self.output_proj = nn.Linear(dim * num_modalities, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        modality_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Cross-modal resonance interaction
        
        Args:
            modality_features: List of tensors, each (batch, seq_len, dim)
            
        Returns:
            List of enhanced features for each modality
        """
        batch_size = modality_features[0].shape[0]
        
        # Transform each modality to frequency domain
        modality_ffts = []
        for features in modality_features:
            # Pad to power of 2
            seq_len = features.shape[1]
            padded_len = 2 ** int(torch.ceil(torch.log2(torch.tensor(seq_len, dtype=torch.float32))))
            
            if padded_len > seq_len:
                features_padded = F.pad(features, (0, 0, 0, padded_len - seq_len))
            else:
                features_padded = features
            
            # FFT
            fft = torch.fft.rfft(features_padded, dim=1)
            modality_ffts.append(fft)
        
        # Cross-modal interaction in frequency domain
        enhanced_ffts = []
        for i, fft_i in enumerate(modality_ffts):
            # Apply modality-specific filter
            filter_i = self.modality_filters[i]
            filter_complex = torch.complex(filter_i[..., 0], filter_i[..., 1])
            
            # Interact with other modalities
            interaction = torch.zeros_like(fft_i)
            for j, fft_j in enumerate(modality_ffts):
                # Cross-modal weights
                weight = self.cross_modal_weights[i, j]
                
                # Weighted combination
                # Average frequency components
                avg_fft = fft_j.mean(dim=1, keepdim=True)
                interaction = interaction + weight.view(1, -1, 1) * avg_fft
            
            enhanced = fft_i + 0.1 * interaction
            enhanced_ffts.append(enhanced)
        
        # Transform back to time domain
        enhanced_features = []
        for i, enhanced_fft in enumerate(enhanced_ffts):
            seq_len = modality_features[i].shape[1]
            
            # IFFT
            enhanced = torch.fft.irfft(enhanced_fft, dim=1)
            
            # Trim to original length
            enhanced = enhanced[:, :seq_len, :]
            
            # Normalize and dropout
            enhanced = self.norm(enhanced + modality_features[i])
            enhanced = self.dropout(enhanced)
            
            enhanced_features.append(enhanced)
        
        return enhanced_features


class MultiModalResonanceFusion(nn.Module):
    """
    Complete multi-modal fusion model
    
    Supports:
    - Text (language model features)
    - Vision (image features)
    - Audio (audio features)
    - Any other modality
    
    Architecture:
    1. Modality-specific encoding
    2. Cross-modal resonance layers
    3. Holographic binding
    4. Task-specific heads
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 768,
        num_cross_modal_layers: int = 4,
        num_frequencies: int = 64,
        hologram_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to input dimension
            hidden_dim: Hidden dimension for all modalities
            num_cross_modal_layers: Number of cross-modal interaction layers
            num_frequencies: Number of frequencies for resonance
            hologram_dim: Dimension of holographic binding space
            dropout: Dropout rate
            num_classes: Optional number of classes for classification
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.hidden_dim = hidden_dim
        self.num_modalities = len(self.modality_names)
        
        # Modality-specific projections to common space
        self.modality_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for name, dim in modality_dims.items()
        })
        
        # Cross-modal resonance layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalResonance(
                dim=hidden_dim,
                num_frequencies=num_frequencies,
                num_modalities=self.num_modalities,
                dropout=dropout,
            )
            for _ in range(num_cross_modal_layers)
        ])
        
        # Holographic binder
        self.holographic_binder = HolographicModalityBinder(
            modality_dims={name: hidden_dim for name in self.modality_names},
            hologram_dim=hologram_dim,
        )
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(hologram_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Optional classification head
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        
    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        return_modality_features: bool = False,
    ) -> torch.Tensor:
        """
        Fuse multiple modalities
        
        Args:
            modality_inputs: Dictionary of {modality_name: features}
                           Each: (batch, seq_len, modality_dim) or (batch, modality_dim)
            return_modality_features: Return intermediate modality features
            
        Returns:
            If num_classes: class logits (batch, num_classes)
            Otherwise: fused features (batch, hidden_dim)
        """
        # Project each modality to common space
        projected = {}
        for name, features in modality_inputs.items():
            if features.dim() == 2:
                features = features.unsqueeze(1)
            projected[name] = self.modality_projections[name](features)
        
        # Convert to list for cross-modal layers
        modality_list = [projected[name] for name in self.modality_names if name in projected]
        
        # Apply cross-modal resonance layers
        for layer in self.cross_modal_layers:
            modality_list = layer(modality_list)
        
        # Convert back to dictionary
        enhanced = {
            name: features
            for name, features in zip(self.modality_names, modality_list)
            if name in modality_inputs
        }
        
        # Holographic binding
        bound = self.holographic_binder.bind(enhanced)
        
        # Fusion projection
        fused = self.fusion_proj(bound)
        
        if return_modality_features:
            return fused, enhanced
        
        # Classification
        if self.num_classes is not None:
            logits = self.classifier(fused)
            return logits
        
        return fused
    
    def forward_with_missing_modalities(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        available_modalities: List[str],
    ) -> torch.Tensor:
        """
        Handle missing modalities gracefully
        Uses holographic unbinding to fill in missing modalities
        """
        # Process available modalities
        projected = {}
        for name in available_modalities:
            if name in modality_inputs:
                features = modality_inputs[name]
                if features.dim() == 2:
                    features = features.unsqueeze(1)
                projected[name] = self.modality_projections[name](features)
        
        # Create partial bound pattern
        if len(projected) > 0:
            partial_bound = self.holographic_binder.bind(projected)
        else:
            batch_size = 1
            partial_bound = torch.zeros(
                batch_size,
                self.holographic_binder.hologram_dim,
                device=next(self.parameters()).device,
            )
        
        # Retrieve from holographic memory (if training data established patterns)
        # This allows the model to "hallucinate" missing modalities
        retrieved = self.holographic_binder.holographic_memory.reconstruct(partial_bound)
        
        # Fusion
        fused = self.fusion_proj(retrieved)
        
        # Classification
        if self.num_classes is not None:
            logits = self.classifier(fused)
            return logits
        
        return fused


class MultiModalResonanceEncoder(nn.Module):
    """
    Encoder-only multimodal model for representation learning
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 768,
        output_dim: int = 512,
        num_cross_modal_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.fusion = MultiModalResonanceFusion(
            modality_dims=modality_dims,
            hidden_dim=hidden_dim,
            num_cross_modal_layers=num_cross_modal_layers,
            dropout=dropout,
            num_classes=None,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode multimodal inputs to fixed-size embedding
        
        Returns:
            Embedding (batch, output_dim)
        """
        fused = self.fusion(modality_inputs)
        embedding = self.output_proj(fused)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding


class MultiModalResonanceGenerator(nn.Module):
    """
    Generate one modality from others
    E.g., image from text, text from image+audio, etc.
    """
    
    def __init__(
        self,
        input_modality_dims: Dict[str, int],
        output_modality: str,
        output_dim: int,
        hidden_dim: int = 768,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_modality = output_modality
        
        # Encoder
        self.encoder = MultiModalResonanceFusion(
            modality_dims=input_modality_dims,
            hidden_dim=hidden_dim,
            num_cross_modal_layers=num_layers // 2,
            dropout=dropout,
        )
        
        # Decoder
        from resonance_nn.layers.resonance import ResonanceLayer
        
        self.decoder_layers = nn.ModuleList([
            ResonanceLayer(
                input_dim=hidden_dim,
                num_frequencies=64,
                dropout=dropout,
            )
            for _ in range(num_layers // 2)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        target_length: int,
    ) -> torch.Tensor:
        """
        Generate target modality from input modalities
        
        Args:
            modality_inputs: Input modalities
            target_length: Length of generated sequence
            
        Returns:
            Generated output (batch, target_length, output_dim)
        """
        # Encode inputs
        encoded = self.encoder(modality_inputs)  # (batch, hidden_dim)
        
        # Expand to sequence
        batch_size = encoded.shape[0]
        expanded = encoded.unsqueeze(1).expand(-1, target_length, -1)
        
        # Decode
        decoded = expanded
        for layer in self.decoder_layers:
            decoded = layer(decoded)
        
        # Project to output modality
        output = self.output_proj(decoded)
        
        return output
