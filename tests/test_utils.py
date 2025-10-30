"""
Tests for utility functions.
"""

import pytest
import torch
import numpy as np
from neurons.utils import (
    prepare_data,
    create_data_loader,
    calculate_accuracy,
    calculate_metrics
)


class TestDataUtils:
    """Test suite for data utilities."""
    
    def test_prepare_data_numpy(self):
        """Test preparing numpy arrays."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        
        X_tensor, y_tensor = prepare_data(X, y)
        
        assert isinstance(X_tensor, torch.Tensor)
        assert isinstance(y_tensor, torch.Tensor)
        assert X_tensor.shape == (100, 10)
        assert y_tensor.shape == (100,)
        assert X_tensor.dtype == torch.float32
        assert y_tensor.dtype == torch.long
    
    def test_prepare_data_tensors(self):
        """Test preparing torch tensors."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        
        X_tensor, y_tensor = prepare_data(X, y)
        
        assert isinstance(X_tensor, torch.Tensor)
        assert isinstance(y_tensor, torch.Tensor)
        assert X_tensor.shape == (100, 10)
        assert y_tensor.shape == (100,)
    
    def test_create_data_loader(self):
        """Test creating data loader."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        
        loader = create_data_loader(X, y, batch_size=16, shuffle=True)
        
        assert len(loader) == 7  # 100 / 16 = 6.25, rounded up = 7
        
        # Test iteration
        for batch_X, batch_y in loader:
            assert isinstance(batch_X, torch.Tensor)
            assert isinstance(batch_y, torch.Tensor)
            assert batch_X.shape[1] == 10
            break


class TestMetrics:
    """Test suite for metrics."""
    
    def test_calculate_accuracy_perfect(self):
        """Test accuracy calculation with perfect predictions."""
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 0, 1])
        
        acc = calculate_accuracy(predictions, targets)
        assert acc == 1.0
    
    def test_calculate_accuracy_half(self):
        """Test accuracy calculation with 50% accuracy."""
        predictions = torch.tensor([0, 0, 0, 0])
        targets = torch.tensor([0, 0, 1, 1])
        
        acc = calculate_accuracy(predictions, targets)
        assert acc == 0.5
    
    def test_calculate_accuracy_with_probs(self):
        """Test accuracy calculation with probability outputs."""
        predictions = torch.tensor([
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.1, 0.9]
        ])
        targets = torch.tensor([0, 1, 2])
        
        acc = calculate_accuracy(predictions, targets)
        assert acc == 1.0
    
    def test_calculate_metrics(self):
        """Test comprehensive metrics calculation."""
        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        
        metrics = calculate_metrics(predictions, targets)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_calculate_metrics_binary(self):
        """Test metrics for binary classification."""
        predictions = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])
        
        metrics = calculate_metrics(predictions, targets, average='binary')
        
        assert 'accuracy' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
