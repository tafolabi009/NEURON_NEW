"""
Unified NEURONSv2 PyTorch Model
Complete trainable model integrating all components

This is the main model you'll use for training and inference!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from neurons.core.spectral_temporal_torch import SpectralTemporalProcessor
from neurons.core.dendrites_torch import DendriticNetwork
from neurons.kernels.triton_kernels import has_triton


@dataclass
class NEURONSv2Config:
    """Complete configuration for NEURONSv2"""
    # Architecture
    vocab_size: int = 50257  # GPT-2 vocab size
    hidden_size: int = 768
    num_layers: int = 12
    
    # Context and compression
    max_seq_length: int = 200000  # 200K tokens!
    compression_ratio: float = 8.0
    n_hierarchy_levels: int = 6
    
    # Spectral processing
    n_frequency_bands: int = 32
    n_wavelet_scales: int = 8
    
    # Dendritic computation
    n_basal_branches: int = 5
    n_apical_branches: int = 3
    synapses_per_branch: int = 20
    use_dendrites: bool = True
    
    # Routing and modulation
    n_pathways: int = 4
    use_adaptive_routing: bool = True
    
    # Training
    dropout: float = 0.1
    use_gradient_checkpointing: bool = False
    
    # Optimization
    use_triton: bool = has_triton()
    use_mixed_precision: bool = True


class NEURONSv2Layer(nn.Module):
    """
    Single NEURONSv2 transformer-like layer
    
    But instead of self-attention, uses:
    1. Spectral-temporal processing
    2. Dendritic computation
    3. Adaptive routing
    """
    
    def __init__(self, config: NEURONSv2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Layer norm (pre-norm architecture like GPT)
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        
        # Spectral processing (replaces attention)
        self.spectral_processor = SpectralTemporalProcessor(
            feature_dim=config.hidden_size,
            max_seq_length=config.max_seq_length,
            n_frequency_bands=config.n_frequency_bands,
            n_wavelet_scales=config.n_wavelet_scales,
            compression_ratio=config.compression_ratio,
        )
        
        # Dendritic network (replaces FFN)
        if config.use_dendrites:
            self.dendritic_net = DendriticNetwork(
                n_neurons=config.hidden_size,
                n_basal_branches=config.n_basal_branches,
                n_apical_branches=config.n_apical_branches,
                synapses_per_branch=config.synapses_per_branch,
            )
        else:
            # Standard FFN as fallback
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, 4 * config.hidden_size),
                nn.GELU(),
                nn.Linear(4 * config.hidden_size, config.hidden_size),
                nn.Dropout(config.dropout),
            )
        
        # Routing network (dynamic pathway selection)
        if config.use_adaptive_routing:
            self.routing = nn.Sequential(
                nn.Linear(config.hidden_size, config.n_pathways),
                nn.Softmax(dim=-1),
            )
        
        # Projection back to hidden size
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through layer
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional mask
            use_cache: Whether to cache for generation
            
        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        # Save residual
        residual = hidden_states
        
        # Pre-norm
        hidden_states = self.ln_1(hidden_states)
        
        # Spectral-temporal processing (replaces self-attention)
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Process with spectral module
        results = self.spectral_processor(hidden_states)
        
        # Get compressed representation
        compressed = results['compressed']  # (batch, compressed_len, hidden_size)
        
        # Decompress back to original length
        # (In practice, you'd use the hierarchy for efficient attention)
        hidden_states = self.spectral_processor.decompress(compressed, seq_len)
        
        # Apply routing if enabled
        if self.config.use_adaptive_routing:
            routing_weights = self.routing(hidden_states)  # (batch, seq_len, n_pathways)
            # Weight the hidden states by routing
            hidden_states = hidden_states * routing_weights.sum(dim=-1, keepdim=True)
        
        # Projection and residual
        hidden_states = self.output_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # FFN/Dendritic block
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        
        if self.config.use_dendrites:
            # Dendritic computation
            # Reshape for dendritic network: (batch * seq_len, hidden_size)
            hidden_flat = hidden_states.reshape(-1, hidden_size)
            
            # Create dummy basal and apical inputs
            # In practice, these would come from different sources
            basal_inputs = hidden_flat.unsqueeze(1).unsqueeze(2).expand(
                -1, self.config.n_basal_branches, self.config.synapses_per_branch
            )
            apical_inputs = torch.zeros_like(basal_inputs) * 0.1  # Minimal apical
            
            # Forward through dendritic network
            spikes = self.dendritic_net(basal_inputs, apical_inputs)
            
            # Reshape back
            hidden_states = spikes.reshape(batch_size, seq_len, hidden_size)
        else:
            # Standard FFN
            hidden_states = self.ffn(hidden_states)
        
        # Residual
        hidden_states = residual + hidden_states
        
        return hidden_states


class NEURONSv2Model(nn.Module):
    """
    Complete NEURONSv2 Model
    
    This is the main model class - similar interface to Hugging Face transformers!
    """
    
    def __init__(self, config: NEURONSv2Config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Positional embeddings (learnable)
        # We use learnable instead of sinusoidal for flexibility
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NEURONSv2Layer(config, i)
            for i in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # LM head for language modeling
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights (like GPT-2)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"NEURONSv2Model initialized:")
        print(f"  - Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  - Layers: {config.num_layers}")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Max context: {config.max_seq_length:,}")
        print(f"  - Using Triton: {config.use_triton}")
        print(f"  - Using dendrites: {config.use_dendrites}")
    
    def _init_weights(self, module):
        """Initialize weights (like GPT-2)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: Optional (batch, seq_len) mask
            labels: Optional (batch, seq_len) labels for loss
            use_cache: Whether to cache for generation
            
        Returns:
            Dictionary with 'logits', 'loss' (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Check sequence length
        if seq_len > self.config.max_seq_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max {self.config.max_seq_length}")
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)  # (batch, seq_len, hidden_size)
        
        # Positional embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        
        # Combine
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Apply layers
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                # Gradient checkpointing to save memory
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_cache,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, use_cache)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: (batch, seq_len) prompt tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            generated_ids: (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for last token
                outputs = self.forward(generated)
                logits = outputs['logits'][:, -1, :]  # (batch, vocab_size)
                
                # Temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if (next_token == 50256).all():  # GPT-2 EOS token
                    break
        
        return generated
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters
        
        Args:
            non_embedding: If True, exclude embedding parameters
            
        Returns:
            n_params: Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
            n_params -= self.token_embedding.weight.numel()
        return n_params


# Helper function to create models
def create_neuronsv2_small() -> NEURONSv2Model:
    """Create small model (similar to GPT-2 small)"""
    config = NEURONSv2Config(
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        max_seq_length=200000,
    )
    return NEURONSv2Model(config)


def create_neuronsv2_medium() -> NEURONSv2Model:
    """Create medium model (similar to GPT-2 medium)"""
    config = NEURONSv2Config(
        vocab_size=50257,
        hidden_size=1024,
        num_layers=24,
        max_seq_length=200000,
    )
    return NEURONSv2Model(config)


def create_neuronsv2_large() -> NEURONSv2Model:
    """Create large model (similar to GPT-2 large)"""
    config = NEURONSv2Config(
        vocab_size=50257,
        hidden_size=1280,
        num_layers=36,
        max_seq_length=200000,
    )
    return NEURONSv2Model(config)


# Quick test
if __name__ == "__main__":
    print("Testing Unified NEURONSv2 Model...\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create small model
    model = create_neuronsv2_small()
    model = model.to(device)
    model.eval()
    
    print(f"\nModel statistics:")
    print(f"  - Total parameters: {model.get_num_params():,}")
    print(f"  - Non-embedding parameters: {model.get_num_params(non_embedding=True):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    
    print(f"  - Logits shape: {outputs['logits'].shape}")
    print(f"  - Loss: {outputs['loss'].item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, 50257, (1, 10), device=device)
    
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20)
    
    print(f"  - Prompt length: {prompt.shape[1]}")
    print(f"  - Generated length: {generated.shape[1]}")
    print(f"  - Generated tokens: {generated[0, :15].tolist()}")
    
    print("\n✓ Unified NEURONSv2 Model working!")
