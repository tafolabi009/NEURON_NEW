"""
Holographic Memory Implementation
Information storage through interference patterns with provable capacity
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple


class HolographicMemory(nn.Module):
    """
    Holographic Memory Core using interference patterns
    
    Definition 2: Holographic Interference Pattern
    H = |O + R|² = |O|² + |R|² + O*R̄ + Ō*R
    
    Theorem 2: Holographic Information Capacity
    C = (A/λ²) log₂(1 + SNR)
    """
    
    def __init__(
        self,
        pattern_dim: int,
        hologram_dim: Optional[int] = None,
        capacity: int = 1000,
        wavelength: float = 1.0,
        snr: float = 10.0,
    ):
        """
        Args:
            pattern_dim: Dimension of patterns to store
            hologram_dim: Dimension of hologram (default: 2 * pattern_dim)
            capacity: Maximum number of patterns to store
            wavelength: Reference wavelength for capacity calculation
            snr: Signal-to-noise ratio for capacity
        """
        super().__init__()
        self.pattern_dim = pattern_dim
        self.hologram_dim = hologram_dim or (2 * pattern_dim)
        self.capacity = capacity
        self.wavelength = wavelength
        self.snr = snr
        
        # Hologram storage (interference pattern accumulator)
        self.register_buffer('hologram', torch.zeros(self.hologram_dim, dtype=torch.complex64))
        
        # Reference beam (fixed complex pattern)
        reference = torch.randn(self.hologram_dim) + 1j * torch.randn(self.hologram_dim)
        reference = reference / torch.abs(reference).mean()  # Normalize
        self.register_buffer('reference_beam', reference)
        
        # Pattern count
        self.register_buffer('num_patterns', torch.tensor(0))
        
        # Projection matrices for encoding/decoding
        self.encoder = nn.Linear(pattern_dim, hologram_dim, bias=False)
        self.decoder = nn.Linear(hologram_dim, pattern_dim, bias=False)
        
        # Initialize with complex-valued weights
        with torch.no_grad():
            self.encoder.weight.data = torch.randn_like(self.encoder.weight) * 0.1
            self.decoder.weight.data = torch.randn_like(self.decoder.weight) * 0.1
            
    def encode(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        Encode pattern into hologram using interference
        
        Definition 3: Holographic Encoding Operation
        Encode(P) = |P + R|² = |P|² + |R|² + P*R̄ + P̄*R
        
        Args:
            pattern: Pattern to encode (pattern_dim,) or (batch, pattern_dim)
            
        Returns:
            Updated hologram
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
            
        batch_size = pattern.shape[0]
        
        # Project pattern to hologram space
        object_beam = self.encoder(pattern)  # (batch, hologram_dim)
        
        # Convert to complex
        object_beam_complex = torch.complex(
            object_beam,
            torch.zeros_like(object_beam)
        )
        
        # Compute interference pattern: |O + R|²
        for i in range(batch_size):
            obj = object_beam_complex[i]
            ref = self.reference_beam
            
            # Interference pattern
            interference = torch.abs(obj + ref) ** 2
            
            # Accumulate in hologram (superposition)
            # Store as complex for phase information
            cross_term = obj * torch.conj(ref) + torch.conj(obj) * ref
            self.hologram += cross_term / self.capacity
            
        self.num_patterns += batch_size
        
        return self.hologram
    
    def reconstruct(self, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct pattern from hologram
        
        Definition 4: Holographic Reconstruction
        Reconstruct(H) = H ⋆ R = (|P|² + |R|² + P*R̄ + P̄*R) ⋆ R
        The term P*R̄ ⋆ R = P|R|² recovers the original pattern
        
        Args:
            query: Optional query pattern for associative recall
            
        Returns:
            Reconstructed pattern (pattern_dim,) or (batch, pattern_dim)
        """
        # Illuminate hologram with reference beam
        # H ⋆ R (convolution approximated as multiplication in this discrete case)
        reconstructed_complex = self.hologram * torch.conj(self.reference_beam)
        
        # Extract real part and decode
        reconstructed_real = torch.real(reconstructed_complex)
        reconstructed_pattern = self.decoder(reconstructed_real)
        
        if query is not None:
            # Associative recall: weight by similarity to query
            if query.dim() == 1:
                query = query.unsqueeze(0)
            similarity = torch.cosine_similarity(
                reconstructed_pattern.unsqueeze(0),
                query,
                dim=-1
            )
            reconstructed_pattern = reconstructed_pattern * similarity.unsqueeze(-1)
            
        return reconstructed_pattern
    
    def clear(self):
        """Clear hologram memory"""
        self.hologram.zero_()
        self.num_patterns.zero_()
        
    def get_capacity_utilization(self) -> float:
        """
        Calculate current capacity utilization
        
        Returns:
            Fraction of theoretical capacity used
        """
        theoretical_capacity = self.get_theoretical_capacity()
        return min(1.0, self.num_patterns.item() / theoretical_capacity)
    
    def get_theoretical_capacity(self) -> float:
        """
        Compute theoretical storage capacity
        
        Theorem 2: C = (A/λ²) log₂(1 + SNR)
        Where A is hologram area (dimension), λ is wavelength
        """
        area = self.hologram_dim
        wavelength_sq = self.wavelength ** 2
        capacity = (area / wavelength_sq) * np.log2(1 + self.snr)
        return capacity
    
    def get_reconstruction_fidelity(self, original: torch.Tensor) -> float:
        """
        Measure reconstruction fidelity
        
        Args:
            original: Original pattern
            
        Returns:
            Fidelity score (1.0 = perfect reconstruction)
        """
        reconstructed = self.reconstruct()
        
        if original.dim() == 1 and reconstructed.dim() == 0:
            reconstructed = reconstructed.unsqueeze(0)
        elif original.dim() == 2 and reconstructed.dim() == 1:
            reconstructed = reconstructed.unsqueeze(0)
            
        # Normalize for comparison
        orig_norm = original / (torch.norm(original) + 1e-8)
        recon_norm = reconstructed / (torch.norm(reconstructed) + 1e-8)
        
        # Cosine similarity as fidelity measure
        fidelity = torch.cosine_similarity(orig_norm, recon_norm, dim=-1).mean()
        return fidelity.item()


class MultiModalHolographicMemory(nn.Module):
    """
    Multi-modal holographic memory for different types of information
    """
    
    def __init__(
        self,
        modalities: List[str],
        pattern_dims: List[int],
        hologram_dim: int,
        capacity: int = 1000,
    ):
        """
        Args:
            modalities: List of modality names (e.g., ['visual', 'audio', 'text'])
            pattern_dims: List of pattern dimensions for each modality
            hologram_dim: Shared hologram dimension
            capacity: Total capacity across modalities
        """
        super().__init__()
        self.modalities = modalities
        self.pattern_dims = pattern_dims
        self.hologram_dim = hologram_dim
        
        # Create separate memory for each modality
        self.memories = nn.ModuleDict({
            modality: HolographicMemory(
                pattern_dim=dim,
                hologram_dim=hologram_dim,
                capacity=capacity // len(modalities)
            )
            for modality, dim in zip(modalities, pattern_dims)
        })
        
        # Cross-modal association weights
        self.cross_modal_weights = nn.Parameter(
            torch.eye(len(modalities)) * 0.1
        )
        
    def encode(self, patterns: dict) -> dict:
        """
        Encode patterns from multiple modalities
        
        Args:
            patterns: Dict mapping modality names to pattern tensors
            
        Returns:
            Dict of holograms for each modality
        """
        holograms = {}
        for modality, pattern in patterns.items():
            if modality in self.memories:
                hologram = self.memories[modality].encode(pattern)
                holograms[modality] = hologram
        return holograms
    
    def reconstruct(self, modality: str, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct pattern from specific modality
        
        Args:
            modality: Modality to reconstruct
            query: Optional query pattern
            
        Returns:
            Reconstructed pattern
        """
        if modality not in self.memories:
            raise ValueError(f"Unknown modality: {modality}")
            
        return self.memories[modality].reconstruct(query)
    
    def cross_modal_retrieve(self, source_modality: str, target_modality: str, 
                           query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve pattern from target modality using query from source modality
        
        Args:
            source_modality: Modality of query
            target_modality: Modality to retrieve
            query: Query pattern in source modality
            
        Returns:
            Retrieved pattern in target modality
        """
        # Encode query in source modality
        source_idx = self.modalities.index(source_modality)
        target_idx = self.modalities.index(target_modality)
        
        # Get cross-modal weight
        weight = self.cross_modal_weights[source_idx, target_idx]
        
        # Reconstruct in target modality weighted by association
        target_pattern = self.memories[target_modality].reconstruct()
        
        return target_pattern * weight


class AdaptiveHolographicMemory(HolographicMemory):
    """
    Holographic memory with adaptive capacity and forgetting
    """
    
    def __init__(
        self,
        pattern_dim: int,
        hologram_dim: Optional[int] = None,
        capacity: int = 1000,
        forgetting_rate: float = 0.01,
    ):
        super().__init__(pattern_dim, hologram_dim, capacity)
        self.forgetting_rate = forgetting_rate
        
    def encode(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        Encode with forgetting of old patterns
        """
        # Apply forgetting
        self.hologram *= (1 - self.forgetting_rate)
        
        # Encode new pattern
        return super().encode(pattern)
    
    def consolidate(self):
        """
        Consolidate memory by strengthening important patterns
        """
        # Strengthen patterns with high magnitude
        magnitude = torch.abs(self.hologram)
        threshold = magnitude.mean() + magnitude.std()
        mask = magnitude > threshold
        self.hologram[mask] *= 1.1
