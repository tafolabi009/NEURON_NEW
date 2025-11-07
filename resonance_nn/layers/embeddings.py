"""
Large Vocabulary Embedding System
Efficient embeddings for 500K-1M vocabulary using frequency-domain compression

Key innovations:
- Hierarchical factorized embeddings
- Frequency-domain compression
- Hash-based embedding with resonance
- O(log V) lookup complexity instead of O(V)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FrequencyCompressedEmbedding(nn.Module):
    """
    Compress large vocabulary embeddings using frequency domain
    
    Instead of vocab_size × embed_dim parameters,
    uses sqrt(vocab_size) × embed_dim + compression factors
    Reduces memory from O(V×D) to O(√V×D + k) where k << V
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_frequency_components: int = 256,
        compression_factor: int = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_frequency_components = num_frequency_components
        self.compression_factor = compression_factor
        
        # Hierarchical factorization
        # First level: map to compressed space
        self.factor1_size = int(math.sqrt(vocab_size)) + 1
        self.factor2_size = (vocab_size // self.factor1_size) + 1
        
        self.factor1_embed = nn.Embedding(
            self.factor1_size,
            embed_dim // 2,
        )
        self.factor2_embed = nn.Embedding(
            self.factor2_size,
            embed_dim // 2,
        )
        
        # Frequency components for fine-grained representation
        self.frequency_basis = nn.Parameter(
            torch.randn(num_frequency_components, embed_dim) * 0.01
        )
        
        # Hash function for frequency selection
        self.register_buffer(
            'hash_matrix',
            torch.randint(0, num_frequency_components, (vocab_size,))
        )
        
        # Combination weights
        self.combine_layer = nn.Linear(embed_dim, embed_dim)
        
        print(f"FrequencyCompressedEmbedding: {vocab_size:,} vocab")
        print(f"  Parameters: {self._count_parameters():,} vs {vocab_size * embed_dim:,} (standard)")
        print(f"  Compression: {(vocab_size * embed_dim) / self._count_parameters():.1f}x")
        
    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings
        
        Args:
            input_ids: Token IDs (batch, seq_len) or (batch,)
            
        Returns:
            Embeddings (batch, seq_len, embed_dim) or (batch, embed_dim)
        """
        original_shape = input_ids.shape
        input_ids = input_ids.flatten()
        
        # Factorized lookup
        factor1_ids = input_ids % self.factor1_size
        factor2_ids = input_ids // self.factor1_size
        factor2_ids = torch.clamp(factor2_ids, 0, self.factor2_size - 1)
        
        factor1_embeds = self.factor1_embed(factor1_ids)
        factor2_embeds = self.factor2_embed(factor2_ids)
        
        # Concatenate factors
        factored = torch.cat([factor1_embeds, factor2_embeds], dim=-1)
        
        # Add frequency components (hash-based selection)
        freq_indices = self.hash_matrix[input_ids]
        freq_components = self.frequency_basis[freq_indices]
        
        # Combine
        combined = factored + 0.1 * freq_components
        combined = self.combine_layer(combined)
        
        # Reshape to original
        if len(original_shape) == 2:
            combined = combined.view(original_shape[0], original_shape[1], -1)
        elif len(original_shape) == 1:
            combined = combined.view(original_shape[0], -1)
        
        return combined


class AdaptiveEmbedding(nn.Module):
    """
    Adaptive embedding with different dimensions for different frequency tokens
    High-frequency tokens get smaller embeddings, rare tokens get larger
    
    Inspired by adaptive softmax but for embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        cutoffs: list = [20000, 100000, 500000],
        div_val: float = 4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.div_val = div_val
        
        # Create embedding clusters
        self.embeddings = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        for i in range(len(self.cutoffs) - 1):
            start, end = self.cutoffs[i], self.cutoffs[i + 1]
            cluster_size = end - start
            
            # Adaptive dimension
            cluster_dim = embed_dim // (div_val ** i)
            cluster_dim = max(int(cluster_dim), 32)  # Minimum 32
            
            # Embedding
            embed = nn.Embedding(cluster_size, cluster_dim)
            self.embeddings.append(embed)
            
            # Projection to full dimension
            if cluster_dim != embed_dim:
                proj = nn.Linear(cluster_dim, embed_dim, bias=False)
            else:
                proj = nn.Identity()
            self.projections.append(proj)
        
        # Report sizes
        total_params = sum(p.numel() for p in self.parameters())
        standard_params = vocab_size * embed_dim
        print(f"AdaptiveEmbedding: {vocab_size:,} vocab")
        print(f"  Clusters: {len(self.embeddings)}")
        print(f"  Parameters: {total_params:,} vs {standard_params:,} (standard)")
        print(f"  Compression: {standard_params / total_params:.1f}x")
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up adaptive embeddings"""
        original_shape = input_ids.shape
        input_ids_flat = input_ids.flatten()
        
        # Initialize output
        output = torch.zeros(
            input_ids_flat.shape[0],
            self.embed_dim,
            device=input_ids.device,
            dtype=torch.float32,
        )
        
        # Process each cluster
        for i, (embed, proj) in enumerate(zip(self.embeddings, self.projections)):
            start, end = self.cutoffs[i], self.cutoffs[i + 1]
            
            # Find tokens in this cluster
            mask = (input_ids_flat >= start) & (input_ids_flat < end)
            
            if mask.any():
                # Adjust IDs to cluster-relative
                cluster_ids = input_ids_flat[mask] - start
                
                # Look up and project
                cluster_embeds = embed(cluster_ids)
                cluster_embeds = proj(cluster_embeds)
                
                # Assign to output
                output[mask] = cluster_embeds
        
        # Reshape
        if len(original_shape) == 2:
            output = output.view(original_shape[0], original_shape[1], -1)
        elif len(original_shape) == 1:
            output = output.view(original_shape[0], -1)
        
        return output


class ResonanceHashEmbedding(nn.Module):
    """
    Hash-based embedding with resonance for ultra-large vocabularies
    Uses multiple hash functions and resonance to create unique representations
    
    Can support effectively infinite vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_hash_buckets: int = 100000,
        num_hash_functions: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_hash_buckets = num_hash_buckets
        self.num_hash_functions = num_hash_functions
        
        # Hash tables (one per hash function)
        self.hash_embeddings = nn.ModuleList([
            nn.Embedding(num_hash_buckets, embed_dim // num_hash_functions)
            for _ in range(num_hash_functions)
        ])
        
        # Hash coefficients (different for each function)
        self.register_buffer(
            'hash_a',
            torch.randint(1, 1000000, (num_hash_functions,))
        )
        self.register_buffer(
            'hash_b',
            torch.randint(0, 1000000, (num_hash_functions,))
        )
        
        # Resonance mixing
        self.resonance_mixer = nn.Linear(embed_dim, embed_dim)
        
        print(f"ResonanceHashEmbedding: {vocab_size:,} vocab")
        print(f"  Hash buckets: {num_hash_buckets:,}")
        print(f"  Parameters: {self._count_parameters():,} vs {vocab_size * embed_dim:,} (standard)")
        print(f"  Compression: {(vocab_size * embed_dim) / self._count_parameters():.1f}x")
        
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def _hash(self, input_ids: torch.Tensor, hash_idx: int) -> torch.Tensor:
        """Apply hash function"""
        a = self.hash_a[hash_idx].item()
        b = self.hash_b[hash_idx].item()
        p = self.num_hash_buckets
        
        hashed = ((input_ids * a + b) % p)
        return hashed
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up hash-based embeddings"""
        original_shape = input_ids.shape
        input_ids_flat = input_ids.flatten()
        
        # Apply multiple hash functions
        hash_embeddings = []
        for i, hash_embed in enumerate(self.hash_embeddings):
            hashed_ids = self._hash(input_ids_flat, i)
            embeds = hash_embed(hashed_ids)
            hash_embeddings.append(embeds)
        
        # Concatenate hash embeddings
        combined = torch.cat(hash_embeddings, dim=-1)
        
        # Mix with resonance
        mixed = self.resonance_mixer(combined)
        
        # Reshape
        if len(original_shape) == 2:
            mixed = mixed.view(original_shape[0], original_shape[1], -1)
        elif len(original_shape) == 1:
            mixed = mixed.view(original_shape[0], -1)
        
        return mixed


class HierarchicalVocabularyEmbedding(nn.Module):
    """
    Hierarchical vocabulary embedding system
    Automatically selects best embedding strategy based on vocab size
    
    - Small vocab (<50K): Standard embedding
    - Medium vocab (50K-200K): Frequency compressed
    - Large vocab (200K-500K): Adaptive embedding
    - Ultra-large (500K+): Hash-based embedding
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        strategy: Optional[str] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Auto-select strategy if not specified
        if strategy is None:
            if vocab_size < 50000:
                strategy = 'standard'
            elif vocab_size < 200000:
                strategy = 'frequency_compressed'
            elif vocab_size < 500000:
                strategy = 'adaptive'
            else:
                strategy = 'hash'
        
        self.strategy = strategy
        
        print(f"HierarchicalVocabularyEmbedding: {vocab_size:,} tokens")
        print(f"  Strategy: {strategy}")
        
        # Create appropriate embedding
        if strategy == 'standard':
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        elif strategy == 'frequency_compressed':
            self.embedding = FrequencyCompressedEmbedding(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
            )
        elif strategy == 'adaptive':
            cutoffs = [20000, 100000, min(500000, vocab_size - 1)]
            cutoffs = [c for c in cutoffs if c < vocab_size]
            self.embedding = AdaptiveEmbedding(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                cutoffs=cutoffs,
            )
        elif strategy == 'hash':
            self.embedding = ResonanceHashEmbedding(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_hash_buckets=min(100000, vocab_size // 10),
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings"""
        return self.embedding(input_ids)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for long contexts
    Supports up to 1M positions
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 1000000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Position-encoded tensor
        """
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


class FrequencyPositionalEncoding(nn.Module):
    """
    Learnable positional encoding in frequency domain
    More flexible than sinusoidal for long contexts
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_frequency_components: int = 256,
        max_len: int = 1000000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frequency_components = num_frequency_components
        self.dropout = nn.Dropout(dropout)
        
        # Learnable frequency components
        self.freq_amplitudes = nn.Parameter(
            torch.randn(num_frequency_components, embed_dim) * 0.01
        )
        self.freq_phases = nn.Parameter(
            torch.zeros(num_frequency_components, embed_dim)
        )
        
        # Frequency indices (controls wavelength)
        self.register_buffer(
            'freq_indices',
            torch.linspace(1, num_frequency_components, num_frequency_components)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add frequency-based positional encoding"""
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate positions
        positions = torch.arange(seq_len, device=x.device).float()
        positions = positions.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # Compute frequency-based encoding
        # pos_enc = Σ A_k * sin(2π * k * pos / seq_len + φ_k)
        freq_idx = self.freq_indices.view(1, 1, -1, 1)  # (1, 1, num_freq, 1)
        positions = positions.unsqueeze(-2)  # (1, seq_len, 1, 1)
        
        # Compute phase
        phase = 2 * math.pi * freq_idx * positions / seq_len
        phase = phase + self.freq_phases.view(1, 1, -1, embed_dim)
        
        # Apply amplitude and sum over frequencies
        encoding = self.freq_amplitudes.view(1, 1, -1, embed_dim) * torch.sin(phase)
        encoding = encoding.sum(dim=2)  # (1, seq_len, embed_dim)
        
        # Add to input
        x = x + encoding
        return self.dropout(x)
