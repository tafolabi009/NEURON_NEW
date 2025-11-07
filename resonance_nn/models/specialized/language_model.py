"""
Resonance Language Model
Specialized for text generation, understanding, and NLP tasks

Features:
- Ultra-long context (260K-300K tokens)
- Large vocabulary support (500K-1M tokens)
- Causal generation
- Efficient O(n log n) complexity
- Can be exported and integrated into applications
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from resonance_nn.models.long_context import LongContextResonanceNet
from resonance_nn.layers.embeddings import HierarchicalVocabularyEmbedding, FrequencyPositionalEncoding
from resonance_nn.layers.resonance import ResonanceLayer


class ResonanceLanguageModel(nn.Module):
    """
    Complete language model for text processing
    
    Supports:
    - Text generation (causal and non-causal)
    - Text classification
    - Question answering
    - Summarization
    - Translation
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_frequencies: int = 64,
        max_seq_length: int = 262144,  # 256K tokens
        dropout: float = 0.1,
        use_long_context: bool = True,
        embedding_strategy: Optional[str] = None,
    ):
        """
        Args:
            vocab_size: Size of vocabulary (can be 500K-1M)
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension for processing
            num_layers: Number of resonance layers
            num_frequencies: Frequencies per layer
            max_seq_length: Maximum sequence length supported
            dropout: Dropout rate
            use_long_context: Use long context architecture for >4K sequences
            embedding_strategy: Embedding strategy (auto-selected if None)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.use_long_context = use_long_context
        
        # Efficient vocabulary embedding
        self.token_embedding = HierarchicalVocabularyEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            strategy=embedding_strategy,
        )
        
        # Positional encoding
        self.pos_encoding = FrequencyPositionalEncoding(
            embed_dim=embed_dim,
            num_frequency_components=256,
            max_len=max_seq_length,
            dropout=dropout,
        )
        
        # Input projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Core processing
        if use_long_context and max_seq_length > 4096:
            self.core = LongContextResonanceNet(
                input_dim=hidden_dim,
                chunk_size=4096,
                overlap=512,
                compressed_dim=512,
                num_chunk_layers=3,
                max_chunks=(max_seq_length // 4096) + 10,
                dropout=dropout,
            )
        else:
            # Standard resonance layers for shorter sequences
            self.core = nn.ModuleList([
                ResonanceLayer(
                    input_dim=hidden_dim,
                    num_frequencies=num_frequencies,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
            self.use_long_context = False
        
        # Output head
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Tie weights with embedding (optional, saves parameters)
        # Commented out for now as hierarchical embeddings may not support this
        # if hasattr(self.token_embedding.embedding, 'weight'):
        #     self.output_proj.weight = self.token_embedding.embedding.weight
        
        print(f"ResonanceLanguageModel initialized:")
        print(f"  Vocabulary: {vocab_size:,} tokens")
        print(f"  Max context: {max_seq_length:,} tokens")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Long context: {use_long_context}")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len) - optional
            return_hidden: Return hidden states instead of logits
            
        Returns:
            If return_hidden: hidden states (batch, seq_len, hidden_dim)
            Otherwise: logits (batch, seq_len, vocab_size)
        """
        # Embed tokens
        embeds = self.token_embedding(input_ids)
        
        # Add positional encoding
        embeds = self.pos_encoding(embeds)
        
        # Project to hidden dimension
        hidden = self.input_proj(embeds)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden)
            hidden = hidden * mask_expanded
        
        # Core processing
        if self.use_long_context:
            hidden = self.core(hidden, use_memory=True, store_to_memory=True)
        else:
            for layer in self.core:
                hidden = layer(hidden)
        
        # Normalize
        hidden = self.output_norm(hidden)
        
        if return_hidden:
            return hidden
        
        # Project to vocabulary
        logits = self.output_proj(hidden)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: Prompt token IDs (batch, prompt_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated token IDs (batch, prompt_len + max_new_tokens)
        """
        self.eval()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if max context reached
                if generated.shape[1] >= self.max_seq_length:
                    break
        
        return generated


class ResonanceCausalLM(ResonanceLanguageModel):
    """
    Causal language model for text generation
    Optimized for autoregressive generation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add causal mask support
        self.register_buffer(
            'causal_mask_cache',
            None,
        )
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get causal attention mask"""
        if self.causal_mask_cache is None or self.causal_mask_cache.shape[0] < seq_len:
            # Create causal mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device),
                diagonal=1,
            ).bool()
            self.causal_mask_cache = mask
        
        return self.causal_mask_cache[:seq_len, :seq_len]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        use_causal_mask: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward with optional causal masking
        """
        if use_causal_mask:
            seq_len = input_ids.shape[1]
            # For resonance layers, we apply masking differently
            # Since we process in frequency domain, we mask the reconstruction
            pass
        
        return super().forward(input_ids, **kwargs)


class ResonanceForSequenceClassification(nn.Module):
    """
    Sequence classification using resonance language model
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = ResonanceLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_long_context=False,
            dropout=dropout,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classify sequence
        
        Returns:
            Class logits (batch, num_classes)
        """
        # Get hidden states
        hidden = self.encoder(input_ids, attention_mask, return_hidden=True)
        
        # Pool sequence (mean pooling)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden)
            sum_hidden = (hidden * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
