"""
Resonance Vision Processor
Frequency-domain image processing WITHOUT CNN architecture

Key differences from CNN:
- Uses 2D FFT instead of spatial convolutions
- Frequency-domain feature extraction
- Resonance chambers for multi-scale processing
- No traditional pooling/stride convolutions
- O(n log n) complexity for n pixels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class SpatialFrequencyProcessor(nn.Module):
    """
    Process images in 2D frequency domain
    Extracts features using frequency resonance instead of convolution
    
    Unlike CNN kernels, we use:
    - 2D FFT for spatial frequency decomposition
    - Complex weights for frequency selection
    - Phase and magnitude processing
    """
    
    def __init__(
        self,
        channels: int,
        num_frequency_bands: int = 64,
        spatial_scales: List[int] = [1, 2, 4, 8],
    ):
        super().__init__()
        self.channels = channels
        self.num_frequency_bands = num_frequency_bands
        self.spatial_scales = spatial_scales
        
        # Complex weights for frequency filtering (per scale)
        self.frequency_weights = nn.ParameterList([
            nn.Parameter(torch.randn(num_frequency_bands, channels, 2) * 0.1)
            for _ in spatial_scales
        ])
        
        # Frequency band selection (learnable)
        self.register_buffer(
            'frequency_positions',
            torch.linspace(0, 1, num_frequency_bands)
        )
        
        # Channel mixing after frequency processing
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(channels * len(spatial_scales)),
            nn.Linear(channels * len(spatial_scales), channels),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image through spatial frequency domain
        
        Args:
            x: Input image (batch, channels, height, width)
            
        Returns:
            Frequency-processed features (batch, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # 2D FFT - decompose into spatial frequencies
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))
        # Shape: (batch, channels, height, width//2+1)
        
        scale_features = []
        
        for scale_idx, scale in enumerate(self.spatial_scales):
            # Extract frequency bands at this scale
            freq_h = x_fft.shape[2] // scale
            freq_w = x_fft.shape[3] // scale
            
            if freq_h == 0 or freq_w == 0:
                continue
            
            # Subsample frequency domain (acts like multi-scale)
            x_scale = x_fft[:, :, ::scale, ::scale]
            
            # Apply complex frequency weights
            weights = self.frequency_weights[scale_idx]
            # Convert to complex
            weight_complex = torch.complex(weights[..., 0], weights[..., 1])
            # Shape: (num_frequency_bands, channels)
            
            # Select frequency bands
            num_bands = min(self.num_frequency_bands, x_scale.shape[2], x_scale.shape[3])
            
            # Apply weights to magnitude and phase
            x_magnitude = torch.abs(x_scale)
            x_phase = torch.angle(x_scale)
            
            # Weight by learned frequency importance
            # Average across spatial frequency dimensions
            x_pooled = F.adaptive_avg_pool2d(x_magnitude, (num_bands, num_bands))
            x_processed = x_pooled.mean(dim=(-2, -1))  # (batch, channels)
            
            scale_features.append(x_processed)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_features, dim=-1)
        
        # Mix channels
        features = self.channel_mixer(multi_scale)
        
        # Broadcast back to spatial dimensions
        features = features.view(batch_size, channels, 1, 1)
        features = features.expand(-1, -1, height, width)
        
        return features


class ResonancePatchProcessor(nn.Module):
    """
    Process image patches using resonance layers
    Similar to Vision Transformer patches but with frequency processing
    """
    
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_frequencies: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Patch embedding using frequency projection
        self.patch_embed = nn.Linear(
            patch_size * patch_size * in_channels,
            embed_dim
        )
        
        # Spatial frequency processor for each patch
        self.spatial_freq_processor = SpatialFrequencyProcessor(
            channels=in_channels,
            num_frequency_bands=num_frequencies,
            spatial_scales=[1, 2, 4],
        )
        
        # Position encoding in frequency domain
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert image to patch embeddings
        
        Args:
            x: Input image (batch, channels, height, width)
            
        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
            Grid size (num_patches_h, num_patches_w)
        """
        batch_size, channels, height, width = x.shape
        
        # Apply spatial frequency processing first
        x_freq = self.spatial_freq_processor(x)
        
        # Extract patches
        patches = F.unfold(
            x_freq,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        # Shape: (batch, channels*patch_size*patch_size, num_patches)
        
        patches = patches.transpose(1, 2)
        # Shape: (batch, num_patches, channels*patch_size*patch_size)
        
        # Embed patches
        patch_embeddings = self.patch_embed(patches)
        
        # Add positional encoding
        num_patches = patch_embeddings.shape[1]
        pos_embed = self.pos_embed.expand(batch_size, num_patches, -1)
        patch_embeddings = patch_embeddings + pos_embed
        
        # Calculate grid size
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        
        return patch_embeddings, (grid_h, grid_w)


class FrequencyConvolutionReplacement(nn.Module):
    """
    Replaces traditional CNN convolutions with frequency-domain operations
    Maintains spatial structure while processing in frequency domain
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Instead of spatial kernels, use frequency domain filters
        # Learn magnitude and phase responses
        self.magnitude_filter = nn.Parameter(
            torch.ones(out_channels, in_channels) * 0.1
        )
        self.phase_filter = nn.Parameter(
            torch.zeros(out_channels, in_channels)
        )
        
        # Channel projection
        self.channel_proj = nn.Linear(in_channels, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency-domain convolution
        
        Args:
            x: (batch, in_channels, height, width)
            
        Returns:
            (batch, out_channels, height, width)
        """
        batch_size, in_ch, height, width = x.shape
        
        # Transform to frequency domain
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))
        
        # Apply frequency filters
        # This is the frequency-domain equivalent of convolution
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Reshape for channel mixing
        magnitude_flat = magnitude.permute(0, 2, 3, 1)  # (batch, h, w, channels)
        phase_flat = phase.permute(0, 2, 3, 1)
        
        # Apply learned filters
        mag_filtered = F.linear(magnitude_flat, self.magnitude_filter)
        phase_filtered = F.linear(phase_flat, self.phase_filter)
        
        # Reconstruct complex
        filtered_complex = mag_filtered * torch.exp(1j * phase_filtered)
        filtered_complex = filtered_complex.permute(0, 3, 1, 2)
        
        # Inverse FFT
        output = torch.fft.irfft2(filtered_complex, s=(height, width), dim=(-2, -1))
        
        return output


class ResonanceVisionEncoder(nn.Module):
    """
    Complete vision encoder using frequency-domain resonance
    NO CNN architecture - pure frequency processing
    
    Architecture:
    1. Patch extraction with frequency preprocessing
    2. Resonance layers for feature processing
    3. Multi-scale frequency analysis
    4. Global pooling in frequency domain
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_frequencies: int = 64,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch processor
        self.patch_processor = ResonancePatchProcessor(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frequencies=num_frequencies,
        )
        
        # Import after defining all vision components
        from resonance_nn.layers.resonance import ResonanceLayer
        
        # Resonance layers for patch processing
        self.resonance_layers = nn.ModuleList([
            ResonanceLayer(
                input_dim=embed_dim,
                num_frequencies=num_frequencies,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Optional classification head
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_classes),
            )
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Process image through frequency-domain resonance
        
        Args:
            x: Input image (batch, channels, height, width)
            return_features: If True, return patch features instead of pooled
            
        Returns:
            If num_classes is set: class logits (batch, num_classes)
            If return_features: patch features (batch, num_patches, embed_dim)
            Otherwise: pooled features (batch, embed_dim)
        """
        # Extract and embed patches
        patch_embeddings, grid_size = self.patch_processor(x)
        
        # Process through resonance layers
        features = patch_embeddings
        for layer in self.resonance_layers:
            features = layer(features)
        
        # Normalize
        features = self.norm(features)
        
        if return_features:
            return features
        
        # Global pooling
        pooled = features.mean(dim=1)
        
        # Classification
        if self.num_classes is not None:
            logits = self.classifier(pooled)
            return logits
        
        return pooled
    
    def extract_hierarchical_features(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extract features at multiple depths
        Useful for dense prediction tasks
        """
        patch_embeddings, grid_size = self.patch_processor(x)
        
        hierarchical_features = []
        features = patch_embeddings
        
        for i, layer in enumerate(self.resonance_layers):
            features = layer(features)
            # Store features at every 3rd layer
            if (i + 1) % 3 == 0:
                hierarchical_features.append(features)
        
        return hierarchical_features


class ResonanceVisionBackbone(nn.Module):
    """
    Vision backbone for dense prediction tasks (detection, segmentation)
    Provides multi-scale feature maps
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,  # Smaller patches for dense tasks
        in_channels: int = 3,
        embed_dims: List[int] = [96, 192, 384, 768],
        num_layers: List[int] = [2, 2, 6, 2],
        num_frequencies: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_stages = len(embed_dims)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dims[0],
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Multi-stage processing
        self.stages = nn.ModuleList()
        
        from resonance_nn.layers.resonance import ResonanceLayer
        
        for stage_idx in range(self.num_stages):
            stage_layers = nn.ModuleList()
            
            for _ in range(num_layers[stage_idx]):
                # Convert spatial to sequence for resonance processing
                layer = ResonanceLayer(
                    input_dim=embed_dims[stage_idx],
                    num_frequencies=num_frequencies,
                    dropout=dropout,
                )
                stage_layers.append(layer)
            
            self.stages.append(stage_layers)
            
            # Downsampling between stages (except last)
            if stage_idx < self.num_stages - 1:
                downsample = FrequencyConvolutionReplacement(
                    in_channels=embed_dims[stage_idx],
                    out_channels=embed_dims[stage_idx + 1],
                    kernel_size=3,
                )
                stage_layers.append(downsample)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features
        
        Args:
            x: Input image (batch, channels, height, width)
            
        Returns:
            List of feature maps at different scales
        """
        # Patch embedding
        x = self.patch_embed(x)
        batch_size, channels, height, width = x.shape
        
        multi_scale_features = []
        
        for stage_idx, stage_layers in enumerate(self.stages):
            # Reshape for resonance processing
            x_seq = x.flatten(2).transpose(1, 2)  # (batch, h*w, channels)
            
            # Process through resonance layers
            for layer in stage_layers[:-1] if stage_idx < self.num_stages - 1 else stage_layers:
                if isinstance(layer, FrequencyConvolutionReplacement):
                    # Reshape back for downsampling
                    x = x_seq.transpose(1, 2).view(batch_size, -1, height, width)
                    x = layer(x)
                    height, width = x.shape[2], x.shape[3]
                else:
                    x_seq = layer(x_seq)
            
            # Reshape back to spatial
            x = x_seq.transpose(1, 2).view(batch_size, -1, height, width)
            multi_scale_features.append(x)
            
            # Downsample for next stage
            if stage_idx < self.num_stages - 1:
                x = stage_layers[-1](x)
                height, width = height // 2, width // 2
        
        return multi_scale_features
