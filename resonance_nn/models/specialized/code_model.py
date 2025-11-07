"""
Resonance Code Model
Specialized for code generation, analysis, and understanding

Features:
- Long context for code files (up to 100K tokens)
- Code-specific tokenization support
- Multi-language support
- Bug detection and code completion
"""

import torch
import torch.nn as nn
from typing import Optional
from resonance_nn.models.specialized.language_model import ResonanceLanguageModel


class ResonanceCodeModel(ResonanceLanguageModel):
    """
    Code model extending language model with code-specific features
    """
    
    def __init__(
        self,
        vocab_size: int = 100000,  # Larger vocab for code tokens
        embed_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        max_seq_length: int = 100000,  # 100K tokens for large files
        dropout: float = 0.1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_length=max_seq_length,
            dropout=dropout,
            use_long_context=True,
        )
        
        # Code-specific features
        self.syntax_analyzer = nn.Linear(hidden_dim, hidden_dim)
        
        print(f"ResonanceCodeModel initialized:")
        print(f"  Max file length: {max_seq_length:,} tokens")
        print(f"  Optimized for code generation and analysis")
    
    def analyze_syntax(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Analyze code syntax"""
        hidden = self.forward(input_ids, return_hidden=True)
        syntax_features = self.syntax_analyzer(hidden)
        return syntax_features
    
    def fill_mask(
        self,
        input_ids: torch.Tensor,
        mask_token_id: int,
        top_k: int = 5,
    ) -> torch.Tensor:
        """Fill in masked code tokens (for code completion)"""
        logits = self.forward(input_ids)
        
        # Find masked positions
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[0]) == 0:
            return input_ids
        
        # Get top-k predictions for masked positions
        masked_logits = logits[mask_positions]
        top_tokens = torch.topk(masked_logits, k=top_k, dim=-1).indices
        
        return top_tokens
