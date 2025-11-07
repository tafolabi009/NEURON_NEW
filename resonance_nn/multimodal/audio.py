"""
Resonance Audio Processor
Frequency-domain audio processing using resonance layers

Key features:
- Native frequency processing (audio is naturally frequency-domain)
- Temporal-spectral resonance
- Multi-resolution time-frequency analysis
- Efficient for long audio sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class SpectrogramResonance(nn.Module):
    """
    Process audio spectrograms using frequency resonance
    Treats spectrogram as 2D frequency-time representation
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        sample_rate: int = 22050,
        num_frequency_bands: int = 64,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        
        # Mel filterbank (learn able refinement of mel scale)
        mel_basis = torch.from_numpy(
            self._create_mel_filterbank()
        ).float()
        self.register_buffer('mel_basis', mel_basis)
        
        # Learnable frequency band weights
        self.freq_band_weights = nn.Parameter(
            torch.randn(num_frequency_bands, n_mels) * 0.1
        )
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(
            n_mels,
            n_mels,
            kernel_size=3,
            padding=1,
            groups=n_mels,  # Depthwise
        )
        
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank matrix"""
        # Simplified mel filterbank creation
        n_freqs = self.n_fft // 2 + 1
        mel_basis = np.zeros((self.n_mels, n_freqs))
        
        # Linear frequency to mel scale
        fmin = 0
        fmax = self.sample_rate / 2
        
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            for j in range(left, center):
                if center != left:
                    mel_basis[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    mel_basis[i, j] = (right - j) / (right - center)
        
        return mel_basis
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to enhanced spectrogram
        
        Args:
            audio: Raw audio (batch, samples) or spectrogram (batch, n_mels, time)
            
        Returns:
            Processed spectrogram (batch, n_mels, time)
        """
        if audio.dim() == 2:
            # Convert raw audio to spectrogram
            audio = audio.unsqueeze(1)  # (batch, 1, samples)
            
            # STFT
            stft = torch.stft(
                audio.squeeze(1),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
            )
            
            # Magnitude spectrogram
            magnitude = torch.abs(stft)  # (batch, n_fft//2+1, time)
            
            # Apply mel filterbank
            mel_spec = torch.matmul(self.mel_basis, magnitude)
            # Shape: (batch, n_mels, time)
        else:
            mel_spec = audio
        
        # Apply temporal processing
        mel_spec = self.temporal_conv(mel_spec)
        
        return mel_spec


class TemporalFrequencyProcessor(nn.Module):
    """
    Process audio in both temporal and frequency dimensions
    Uses resonance layers for efficient long-sequence processing
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 512,
        num_frequencies: int = 64,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        
        # Project mel features to embedding dimension
        self.mel_projection = nn.Linear(n_mels, embed_dim)
        
        # Import after vision module is defined
        from resonance_nn.layers.resonance import ResonanceLayer
        
        # Temporal resonance layers (process time dimension)
        self.temporal_layers = nn.ModuleList([
            ResonanceLayer(
                input_dim=embed_dim,
                num_frequencies=num_frequencies,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Process mel spectrogram temporally
        
        Args:
            mel_spec: Mel spectrogram (batch, n_mels, time)
            
        Returns:
            Temporal features (batch, time, embed_dim)
        """
        # Transpose to (batch, time, n_mels)
        x = mel_spec.transpose(1, 2)
        
        # Project to embedding dimension
        x = self.mel_projection(x)
        
        # Process through temporal resonance layers
        for layer in self.temporal_layers:
            x = layer(x)
        
        # Normalize
        x = self.norm(x)
        
        return x


class AudioPatchProcessor(nn.Module):
    """
    Process audio in patches similar to vision patches
    Each patch is a time-frequency region
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        patch_size: int = 16,  # Time steps per patch
        embed_dim: int = 512,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Linear(
            n_mels * patch_size,
            embed_dim,
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Extract audio patches
        
        Args:
            mel_spec: (batch, n_mels, time)
            
        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
        """
        batch_size, n_mels, time_steps = mel_spec.shape
        
        # Pad time dimension to multiple of patch_size
        if time_steps % self.patch_size != 0:
            pad_len = self.patch_size - (time_steps % self.patch_size)
            mel_spec = F.pad(mel_spec, (0, pad_len))
            time_steps = mel_spec.shape[2]
        
        # Reshape into patches
        num_patches = time_steps // self.patch_size
        patches = mel_spec.reshape(
            batch_size,
            n_mels,
            num_patches,
            self.patch_size,
        )
        patches = patches.permute(0, 2, 1, 3)  # (batch, num_patches, n_mels, patch_size)
        patches = patches.reshape(batch_size, num_patches, -1)
        
        # Embed patches
        patch_embeddings = self.patch_embed(patches)
        
        # Add positional encoding
        pos_embed = self.pos_embed.expand(batch_size, num_patches, -1)
        patch_embeddings = patch_embeddings + pos_embed
        
        return patch_embeddings


class ResonanceAudioEncoder(nn.Module):
    """
    Complete audio encoder using frequency-domain resonance
    
    Architecture:
    1. Convert audio to mel spectrogram
    2. Extract patches or process temporally
    3. Resonance layers for feature extraction
    4. Optional classification/embedding head
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        sample_rate: int = 22050,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_frequencies: int = 64,
        dropout: float = 0.1,
        use_patches: bool = True,
        patch_size: int = 16,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        self.use_patches = use_patches
        
        # Spectrogram processor
        self.spectrogram = SpectrogramResonance(
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        
        if use_patches:
            # Patch-based processing
            self.patch_processor = AudioPatchProcessor(
                n_mels=n_mels,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )
            input_dim = embed_dim
        else:
            # Direct temporal processing
            self.temporal_processor = TemporalFrequencyProcessor(
                n_mels=n_mels,
                embed_dim=embed_dim,
                num_frequencies=num_frequencies,
                num_layers=num_layers,
                dropout=dropout,
            )
            input_dim = embed_dim
        
        # Import after all dependencies
        from resonance_nn.layers.resonance import ResonanceLayer
        
        # Additional resonance layers
        self.resonance_layers = nn.ModuleList([
            ResonanceLayer(
                input_dim=input_dim,
                num_frequencies=num_frequencies,
                dropout=dropout,
            )
            for _ in range(num_layers if use_patches else num_layers // 2)
        ])
        
        # Norm
        self.norm = nn.LayerNorm(input_dim)
        
        # Optional classification head
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, num_classes),
            )
        
    def forward(
        self,
        audio: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Process audio through resonance encoder
        
        Args:
            audio: Raw audio (batch, samples) or spectrogram (batch, n_mels, time)
            return_features: Return sequence features instead of pooled
            
        Returns:
            If num_classes: class logits (batch, num_classes)
            If return_features: features (batch, seq_len, embed_dim)
            Otherwise: pooled embedding (batch, embed_dim)
        """
        # Convert to spectrogram if needed
        mel_spec = self.spectrogram(audio)
        
        # Process based on mode
        if self.use_patches:
            features = self.patch_processor(mel_spec)
        else:
            features = self.temporal_processor(mel_spec)
        
        # Apply resonance layers
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


class ResonanceAudioAutoencoder(nn.Module):
    """
    Audio autoencoder for compression and generation
    Uses resonance layers for efficient processing
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        latent_dim: int = 256,
        embed_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Encoder
        self.encoder = ResonanceAudioEncoder(
            n_mels=n_mels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_patches=False,
        )
        
        # Latent projection
        self.to_latent = nn.Linear(embed_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, embed_dim)
        
        # Decoder
        from resonance_nn.layers.resonance import ResonanceLayer
        
        self.decoder_layers = nn.ModuleList([
            ResonanceLayer(
                input_dim=embed_dim,
                num_frequencies=64,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Reconstruction head
        self.reconstruct = nn.Linear(embed_dim, n_mels)
        
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent"""
        features = self.encoder(audio, return_features=True)
        latent = self.to_latent(features.mean(dim=1))
        return latent
    
    def decode(self, latent: torch.Tensor, target_length: int) -> torch.Tensor:
        """Decode latent to audio spectrogram"""
        batch_size = latent.shape[0]
        
        # Expand latent
        features = self.from_latent(latent)
        features = features.unsqueeze(1).expand(-1, target_length, -1)
        
        # Decode
        for layer in self.decoder_layers:
            features = layer(features)
        
        # Reconstruct mel spectrogram
        mel_spec = self.reconstruct(features)
        mel_spec = mel_spec.transpose(1, 2)
        
        return mel_spec
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and decode audio
        
        Returns:
            Tuple of (reconstruction, latent)
        """
        # Get features
        features = self.encoder(audio, return_features=True)
        seq_len = features.shape[1]
        
        # Encode to latent
        latent = self.to_latent(features.mean(dim=1))
        
        # Decode
        reconstruction = self.decode(latent, seq_len)
        
        return reconstruction, latent
