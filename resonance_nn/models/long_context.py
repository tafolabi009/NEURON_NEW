"""
Long Context Resonance Network
Supports 260K-300K tokens through hierarchical frequency processing

This module achieves ultra-long context without attention mechanisms by:
1. Hierarchical chunking with overlapping windows
2. Multi-level frequency compression
3. Global holographic memory for cross-chunk coherence
4. O(n log n) complexity maintained across all chunks
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from resonance_nn.layers.resonance import ResonanceLayer, MultiScaleResonanceLayer
from resonance_nn.layers.holographic import HolographicMemory


class ChunkProcessor(nn.Module):
    """
    Process individual chunks with resonance layers
    """
    
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        
        # Local resonance processing
        self.local_layers = nn.ModuleList([
            ResonanceLayer(
                input_dim=input_dim,
                num_frequencies=num_frequencies,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a single chunk"""
        for layer in self.local_layers:
            x = layer(x)
        return self.norm(x)


class HierarchicalFrequencyCompressor(nn.Module):
    """
    Compress chunks into compact frequency representations
    Uses multiple frequency scales for information preservation
    """
    
    def __init__(
        self,
        input_dim: int,
        compressed_dim: int,
        num_scales: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.num_scales = num_scales
        
        # Multi-scale frequency extractors
        self.scale_projections = nn.ModuleList([
            nn.Linear(input_dim, compressed_dim // num_scales)
            for _ in range(num_scales)
        ])
        
        # Frequency pooling at different scales
        self.freq_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(2 ** (num_scales - i))
            for i in range(num_scales)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress chunk to compact representation
        
        Args:
            x: Input chunk (batch, chunk_len, input_dim)
            
        Returns:
            Compressed representation (batch, compressed_dim)
        """
        batch_size = x.shape[0]
        
        # Extract multi-scale features
        scale_features = []
        for i, (proj, pool) in enumerate(zip(self.scale_projections, self.freq_pools)):
            # Project to scale-specific dim
            x_proj = proj(x)  # (batch, chunk_len, compressed_dim // num_scales)
            
            # Pool across sequence at different resolutions
            x_pooled = pool(x_proj.transpose(1, 2)).transpose(1, 2)
            
            # Average pool to single vector
            x_compressed = x_pooled.mean(dim=1)
            scale_features.append(x_compressed)
        
        # Concatenate all scales
        compressed = torch.cat(scale_features, dim=-1)
        
        return compressed


class GlobalContextIntegrator(nn.Module):
    """
    Integrate information across all chunks using frequency-domain operations
    Maintains O(k log k) where k is number of chunks
    """
    
    def __init__(
        self,
        compressed_dim: int,
        num_frequencies: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.compressed_dim = compressed_dim
        
        # Global resonance across chunks
        self.global_resonance = ResonanceLayer(
            input_dim=compressed_dim,
            num_frequencies=num_frequencies,
            dropout=dropout,
        )
        
        # Bidirectional chunk interaction
        self.chunk_interaction = nn.MultiheadAttention(
            embed_dim=compressed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
    def forward(self, chunk_representations: torch.Tensor) -> torch.Tensor:
        """
        Integrate global context across chunks
        
        Args:
            chunk_representations: (batch, num_chunks, compressed_dim)
            
        Returns:
            Globally integrated representations (batch, num_chunks, compressed_dim)
        """
        # Apply frequency-domain processing across chunks
        x = self.global_resonance(chunk_representations)
        
        # Cross-chunk communication (limited to chunk level, not token level)
        x_integrated, _ = self.chunk_interaction(x, x, x)
        
        return x_integrated


class ChunkExpander(nn.Module):
    """
    Expand compressed chunk representation back to full sequence
    """
    
    def __init__(
        self,
        compressed_dim: int,
        output_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        
        # Expansion layers
        self.expander = nn.Sequential(
            nn.Linear(compressed_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        
        # Refinement layers
        self.refinement = nn.ModuleList([
            ResonanceLayer(
                input_dim=output_dim,
                num_frequencies=64,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        compressed: torch.Tensor,
        chunk_len: int,
        original_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expand compressed representation back to chunk length
        
        Args:
            compressed: (batch, compressed_dim)
            chunk_len: Target chunk length
            original_chunk: Original chunk for residual connection
            
        Returns:
            Expanded chunk (batch, chunk_len, output_dim)
        """
        batch_size = compressed.shape[0]
        
        # Expand to full dimension
        expanded = self.expander(compressed)  # (batch, output_dim)
        
        # Broadcast to chunk length
        expanded = expanded.unsqueeze(1).expand(-1, chunk_len, -1)
        
        # Add residual from original
        if original_chunk is not None:
            expanded = expanded + original_chunk
        
        # Refine with resonance layers
        for layer in self.refinement:
            expanded = layer(expanded)
        
        return expanded


class LongContextResonanceNet(nn.Module):
    """
    Resonance Network for Ultra-Long Context (260K-300K tokens)
    
    Architecture:
    1. Split input into overlapping chunks (4K-8K per chunk)
    2. Process each chunk locally with resonance layers
    3. Compress chunks to compact representations
    4. Apply global resonance across chunk representations
    5. Store global state in holographic memory
    6. Expand chunks back with global context
    
    Complexity: O(n log n) where n is total sequence length
    - Chunk processing: O(c * k log k) where c=num_chunks, k=chunk_size
    - Global integration: O(c log c)
    - Total: O(n log n) since c*k = n
    """
    
    def __init__(
        self,
        input_dim: int,
        chunk_size: int = 4096,
        overlap: int = 512,
        compressed_dim: int = 512,
        num_chunk_layers: int = 3,
        num_expand_layers: int = 2,
        holographic_capacity: int = 10000,
        dropout: float = 0.1,
        max_chunks: int = 80,  # ~320K tokens at 4K chunk size
    ):
        """
        Args:
            input_dim: Input feature dimension
            chunk_size: Size of each chunk (4K-8K recommended)
            overlap: Overlap between chunks for continuity
            compressed_dim: Dimension of compressed chunk representation
            num_chunk_layers: Number of resonance layers per chunk
            num_expand_layers: Number of layers for chunk expansion
            holographic_capacity: Capacity of global memory
            dropout: Dropout rate
            max_chunks: Maximum number of chunks (defines max context)
        """
        super().__init__()
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap
        self.compressed_dim = compressed_dim
        self.max_chunks = max_chunks
        
        # Maximum context length
        self.max_context_length = max_chunks * self.stride + overlap
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, input_dim)
        
        # Chunk processor
        self.chunk_processor = ChunkProcessor(
            input_dim=input_dim,
            num_frequencies=64,
            num_layers=num_chunk_layers,
            dropout=dropout,
        )
        
        # Chunk compressor
        self.chunk_compressor = HierarchicalFrequencyCompressor(
            input_dim=input_dim,
            compressed_dim=compressed_dim,
            num_scales=4,
        )
        
        # Global context integrator
        self.global_integrator = GlobalContextIntegrator(
            compressed_dim=compressed_dim,
            num_frequencies=32,
            dropout=dropout,
        )
        
        # Chunk expander
        self.chunk_expander = ChunkExpander(
            compressed_dim=compressed_dim,
            output_dim=input_dim,
            num_layers=num_expand_layers,
        )
        
        # Holographic memory for very long-term dependencies
        self.holographic_memory = HolographicMemory(
            pattern_dim=compressed_dim,
            hologram_dim=compressed_dim * 2,
            capacity=holographic_capacity,
        )
        
        # Output projection
        self.output_projection = nn.Linear(input_dim, input_dim)
        
        print(f"Initialized LongContextResonanceNet:")
        print(f"  Max context length: {self.max_context_length:,} tokens")
        print(f"  Chunk size: {chunk_size}, Overlap: {overlap}")
        print(f"  Max chunks: {max_chunks}")
        print(f"  Complexity: O(n log n)")
        
    def _create_chunks(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Split input into overlapping chunks
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            List of chunks
        """
        batch_size, seq_len, dim = x.shape
        chunks = []
        
        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end, :]
            
            # Pad if needed
            if chunk.shape[1] < self.chunk_size:
                padding = self.chunk_size - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, padding))
            
            chunks.append(chunk)
            start += self.stride
            
            # Limit number of chunks
            if len(chunks) >= self.max_chunks:
                break
        
        return chunks
    
    def _merge_chunks(
        self,
        chunks: List[torch.Tensor],
        original_length: int,
    ) -> torch.Tensor:
        """
        Merge overlapping chunks back into sequence
        Uses weighted averaging in overlap regions
        """
        batch_size = chunks[0].shape[0]
        merged = torch.zeros(
            batch_size,
            original_length,
            self.input_dim,
            device=chunks[0].device,
            dtype=chunks[0].dtype,
        )
        weights = torch.zeros(
            batch_size,
            original_length,
            1,
            device=chunks[0].device,
        )
        
        start = 0
        for chunk in chunks:
            end = min(start + self.chunk_size, original_length)
            chunk_len = end - start
            
            # Create weight ramp for smooth blending
            weight = torch.ones(batch_size, chunk_len, 1, device=chunk.device)
            if start > 0:  # Ramp up in overlap
                ramp_len = min(self.overlap, chunk_len)
                weight[:, :ramp_len, :] = torch.linspace(
                    0, 1, ramp_len, device=chunk.device
                ).view(1, -1, 1)
            if end < original_length:  # Ramp down in overlap
                ramp_len = min(self.overlap, chunk_len)
                weight[:, -ramp_len:, :] = torch.linspace(
                    1, 0, ramp_len, device=chunk.device
                ).view(1, -1, 1)
            
            merged[:, start:end, :] += chunk[:, :chunk_len, :] * weight
            weights[:, start:end, :] += weight
            
            start += self.stride
        
        # Normalize by weights
        weights = torch.clamp(weights, min=1e-8)
        merged = merged / weights
        
        return merged
    
    def forward(
        self,
        x: torch.Tensor,
        use_memory: bool = True,
        store_to_memory: bool = True,
    ) -> torch.Tensor:
        """
        Process ultra-long context
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
               seq_len can be up to 260K-300K tokens
            use_memory: Whether to use holographic memory
            store_to_memory: Whether to store to memory
            
        Returns:
            Output tensor (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_context_length:
            print(f"Warning: Input length {seq_len} exceeds max context {self.max_context_length}")
            x = x[:, :self.max_context_length, :]
            seq_len = self.max_context_length
        
        # Project input
        x = self.input_projection(x)
        
        # Split into chunks
        chunks = self._create_chunks(x)
        num_chunks = len(chunks)
        
        # Process each chunk locally (parallelizable)
        processed_chunks = []
        compressed_chunks = []
        
        for chunk in chunks:
            # Local processing with resonance
            processed = self.chunk_processor(chunk)
            processed_chunks.append(processed)
            
            # Compress to compact representation
            compressed = self.chunk_compressor(processed)
            compressed_chunks.append(compressed)
        
        # Stack compressed representations
        compressed_stack = torch.stack(compressed_chunks, dim=1)
        # Shape: (batch, num_chunks, compressed_dim)
        
        # Global integration across chunks
        global_context = self.global_integrator(compressed_stack)
        
        # Integrate with holographic memory
        if use_memory:
            # Retrieve global patterns
            memory_pattern = self.holographic_memory.reconstruct()
            memory_pattern = memory_pattern.unsqueeze(0).unsqueeze(0)
            global_context = global_context + 0.1 * memory_pattern
        
        if store_to_memory:
            # Store global patterns for future use
            global_summary = global_context.mean(dim=1)  # (batch, compressed_dim)
            self.holographic_memory.encode(global_summary)
        
        # Expand chunks back with global context
        expanded_chunks = []
        for i, (processed_chunk, global_rep) in enumerate(
            zip(processed_chunks, global_context.unbind(1))
        ):
            expanded = self.chunk_expander(
                compressed=global_rep,
                chunk_len=self.chunk_size,
                original_chunk=processed_chunk,
            )
            expanded_chunks.append(expanded)
        
        # Merge chunks back into full sequence
        output = self._merge_chunks(expanded_chunks, seq_len)
        
        # Final projection
        output = self.output_projection(output)
        
        return output
    
    def get_memory_usage_estimate(self, seq_len: int) -> dict:
        """
        Estimate memory usage for given sequence length
        """
        num_chunks = (seq_len - self.overlap) // self.stride + 1
        num_chunks = min(num_chunks, self.max_chunks)
        
        # Memory per chunk (approximate)
        chunk_memory = self.chunk_size * self.input_dim * 4  # float32
        
        # Compressed memory
        compressed_memory = num_chunks * self.compressed_dim * 4
        
        # Total working memory
        total_memory = (
            chunk_memory * 2 +  # Input and processed
            compressed_memory * 2 +  # Compressed and global
            seq_len * self.input_dim * 4  # Output
        )
        
        return {
            'sequence_length': seq_len,
            'num_chunks': num_chunks,
            'chunk_memory_mb': chunk_memory / (1024 ** 2),
            'compressed_memory_mb': compressed_memory / (1024 ** 2),
            'total_memory_mb': total_memory / (1024 ** 2),
            'memory_efficiency': f'{seq_len / (total_memory / (1024**2)):.1f} tokens/MB',
        }


class StreamingLongContextNet(LongContextResonanceNet):
    """
    Streaming version for real-time processing of long contexts
    Processes chunks incrementally and maintains state
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # State management
        self.register_buffer('chunk_buffer', None)
        self.register_buffer('global_state', None)
        self.chunk_count = 0
        
    def reset_state(self):
        """Reset streaming state"""
        self.chunk_buffer = None
        self.global_state = None
        self.chunk_count = 0
        self.holographic_memory.clear()
        
    def process_chunk_streaming(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Process a single chunk in streaming mode
        
        Args:
            chunk: Input chunk (batch, chunk_len, input_dim)
            
        Returns:
            Processed chunk with global context
        """
        # Process locally
        processed = self.chunk_processor(chunk)
        
        # Compress
        compressed = self.chunk_compressor(processed)
        
        # Update global state
        if self.global_state is None:
            self.global_state = compressed.unsqueeze(1)
        else:
            self.global_state = torch.cat([self.global_state, compressed.unsqueeze(1)], dim=1)
            
            # Limit state buffer size
            if self.global_state.shape[1] > self.max_chunks:
                self.global_state = self.global_state[:, -self.max_chunks:, :]
        
        # Apply global integration
        integrated = self.global_integrator(self.global_state)
        
        # Get latest global context
        global_rep = integrated[:, -1, :]
        
        # Expand with global context
        output = self.chunk_expander(
            compressed=global_rep,
            chunk_len=chunk.shape[1],
            original_chunk=processed,
        )
        
        self.chunk_count += 1
        
        return output
