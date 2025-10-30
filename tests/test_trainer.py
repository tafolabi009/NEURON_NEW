"""
Tests for the Trainer class.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from neurons.core import NeuralNetwork, Trainer
from neurons.losses import MSELoss


class TestTrainer:
    """Test suite for Trainer class."""
    
    @pytest.fixture
    def simple_dataset(self):
        """Create a simple dataset for testing."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        return X, y
    
    @pytest.fixture
    def data_loaders(self, simple_dataset):
        """Create train and validation data loaders."""
        X, y = simple_dataset
        train_dataset = TensorDataset(X[:80], y[:80])
        val_dataset = TensorDataset(X[80:], y[80:])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        return train_loader, val_loader
    
    def test_trainer_initialization(self, data_loaders):
        """Test trainer initialization."""
        model = NeuralNetwork(layer_sizes=[10, 20, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(model, optimizer, loss_fn)
        assert trainer.model == model
        assert trainer.optimizer == optimizer
        assert trainer.loss_fn == loss_fn
    
    def test_trainer_fit(self, data_loaders):
        """Test basic training."""
        train_loader, val_loader = data_loaders
        
        model = NeuralNetwork(layer_sizes=[10, 20, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(model, optimizer, loss_fn)
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=5,
            verbose=0
        )
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
    
    def test_trainer_fit_without_validation(self, data_loaders):
        """Test training without validation."""
        train_loader, _ = data_loaders
        
        model = NeuralNetwork(layer_sizes=[10, 20, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(model, optimizer, loss_fn)
        history = trainer.fit(
            train_loader,
            epochs=5,
            verbose=0
        )
        
        assert 'train_loss' in history
        assert 'val_loss' not in history
    
    def test_trainer_early_stopping(self, data_loaders):
        """Test early stopping."""
        train_loader, val_loader = data_loaders
        
        model = NeuralNetwork(layer_sizes=[10, 20, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(model, optimizer, loss_fn)
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=100,
            patience=3,
            verbose=0
        )
        
        # Should stop before 100 epochs
        assert len(history['train_loss']) < 100
    
    def test_trainer_checkpointing(self, data_loaders):
        """Test model checkpointing."""
        train_loader, val_loader = data_loaders
        
        model = NeuralNetwork(layer_sizes=[10, 20, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(model, optimizer, loss_fn)
            trainer.fit(
                train_loader,
                val_loader,
                epochs=5,
                patience=3,
                checkpoint_dir=tmpdir,
                verbose=0
            )
            
            # Check if best model was saved
            import os
            assert os.path.exists(os.path.join(tmpdir, 'best_model.pth'))
    
    def test_trainer_gradient_clipping(self, data_loaders):
        """Test gradient clipping."""
        train_loader, _ = data_loaders
        
        model = NeuralNetwork(layer_sizes=[10, 20, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(model, optimizer, loss_fn, grad_clip_value=1.0)
        history = trainer.fit(
            train_loader,
            epochs=3,
            verbose=0
        )
        
        assert len(history['train_loss']) == 3
    
    def test_trainer_evaluate(self, data_loaders):
        """Test evaluation method."""
        _, val_loader = data_loaders
        
        model = NeuralNetwork(layer_sizes=[10, 20, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(model, optimizer, loss_fn)
        results = trainer.evaluate(val_loader)
        
        assert 'loss' in results
        assert isinstance(results['loss'], float)
    
    def test_trainer_evaluate_with_predictions(self, data_loaders):
        """Test evaluation with predictions."""
        _, val_loader = data_loaders
        
        model = NeuralNetwork(layer_sizes=[10, 20, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = Trainer(model, optimizer, loss_fn)
        results = trainer.evaluate(val_loader, return_predictions=True)
        
        assert 'loss' in results
        assert 'predictions' in results
        assert 'targets' in results
        assert results['predictions'].shape[0] == 20  # Validation set size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
