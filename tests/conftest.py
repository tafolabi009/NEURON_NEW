"""Pytest configuration"""
import pytest
import torch


@pytest.fixture
def device():
    """Get available device"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def small_model():
    """Create a small model for testing"""
    from resonance_nn import ResonanceNet
    return ResonanceNet(
        input_dim=32,
        num_frequencies=8,
        num_layers=2,
        holographic_capacity=10,
    )


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing"""
    return {
        'input': torch.randn(4, 16, 32),
        'target': torch.randn(4, 16, 32),
    }
