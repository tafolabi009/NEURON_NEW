"""
Tests for the core NeuralNetwork model.
"""

import pytest
import torch
import tempfile
import os
from neurons.core import NeuralNetwork


class TestNeuralNetwork:
    """Test suite for NeuralNetwork class."""
    
    def test_model_initialization(self):
        """Test basic model initialization."""
        model = NeuralNetwork(
            layer_sizes=[10, 20, 5],
            activation='relu'
        )
        assert model.layer_sizes == [10, 20, 5]
        assert model.activation_name == 'relu'
    
    def test_model_forward_pass(self):
        """Test forward pass through the network."""
        model = NeuralNetwork(layer_sizes=[10, 20, 5])
        x = torch.randn(32, 10)
        output = model(x)
        assert output.shape == (32, 5)
    
    def test_model_with_single_sample(self):
        """Test model with single sample input."""
        model = NeuralNetwork(layer_sizes=[10, 5])
        x = torch.randn(10)
        output = model(x)
        assert output.shape == (1, 5)
    
    def test_model_with_dropout(self):
        """Test model with dropout."""
        model = NeuralNetwork(
            layer_sizes=[10, 20, 5],
            dropout_rate=0.5
        )
        x = torch.randn(32, 10)
        
        # Training mode - dropout active
        model.train()
        output_train = model(x)
        
        # Eval mode - dropout inactive
        model.eval()
        output_eval = model(x)
        
        assert output_train.shape == output_eval.shape == (32, 5)
    
    def test_model_with_batch_norm(self):
        """Test model with batch normalization."""
        model = NeuralNetwork(
            layer_sizes=[10, 20, 5],
            use_batch_norm=True
        )
        x = torch.randn(32, 10)
        output = model(x)
        assert output.shape == (32, 5)
    
    def test_model_predict(self):
        """Test prediction method."""
        model = NeuralNetwork(layer_sizes=[10, 20, 3])
        x = torch.randn(32, 10)
        
        # Get class predictions
        predictions = model.predict(x, return_probs=False)
        assert predictions.shape == (32,)
        assert predictions.dtype == torch.int64
        
        # Get probabilities
        probs = model.predict(x, return_probs=True)
        assert probs.shape == (32, 3)
    
    def test_model_num_parameters(self):
        """Test getting number of parameters."""
        model = NeuralNetwork(layer_sizes=[10, 20, 5])
        num_params = model.get_num_parameters()
        assert num_params > 0
        
        # Manually calculate expected parameters
        # Layer 1: 10*20 + 20 = 220
        # Layer 2: 20*5 + 5 = 105
        expected = 220 + 105
        assert num_params == expected
    
    def test_model_save_load(self):
        """Test saving and loading model."""
        model = NeuralNetwork(layer_sizes=[10, 20, 5])
        x = torch.randn(8, 10)
        output_before = model(x)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            filepath = f.name
        
        try:
            # Save model
            model.save(filepath)
            assert os.path.exists(filepath)
            
            # Load model
            loaded_model = NeuralNetwork.load(filepath)
            output_after = loaded_model(x)
            
            # Outputs should be identical
            assert torch.allclose(output_before, output_after)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_model_invalid_layer_sizes(self):
        """Test model with invalid layer sizes."""
        with pytest.raises(ValueError, match="at least 2 elements"):
            NeuralNetwork(layer_sizes=[10])
        
        with pytest.raises(ValueError, match="positive integers"):
            NeuralNetwork(layer_sizes=[10, -5, 3])
    
    def test_model_invalid_dropout(self):
        """Test model with invalid dropout rate."""
        with pytest.raises(ValueError, match="dropout_rate must be"):
            NeuralNetwork(layer_sizes=[10, 5], dropout_rate=1.5)
    
    def test_model_input_size_mismatch(self):
        """Test model with mismatched input size."""
        model = NeuralNetwork(layer_sizes=[10, 5])
        x = torch.randn(32, 15)  # Wrong input size
        
        with pytest.raises(RuntimeError, match="Input size mismatch"):
            model(x)
    
    def test_model_with_different_activations(self):
        """Test model with different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
        
        for act in activations:
            model = NeuralNetwork(
                layer_sizes=[10, 20, 5],
                activation=act
            )
            x = torch.randn(16, 10)
            output = model(x)
            assert output.shape == (16, 5)
    
    def test_model_with_output_activation(self):
        """Test model with output activation."""
        model = NeuralNetwork(
            layer_sizes=[10, 20, 5],
            output_activation='softmax'
        )
        x = torch.randn(16, 10)
        output = model(x)
        
        # Softmax output should sum to 1 for each sample
        assert torch.allclose(output.sum(dim=1), torch.ones(16))
    
    def test_model_gpu_cpu_transfer(self):
        """Test model device transfer."""
        if torch.cuda.is_available():
            # Test on GPU
            model_gpu = NeuralNetwork(layer_sizes=[10, 5], device='cuda')
            assert str(model_gpu.device) == 'cuda'
            
            x_gpu = torch.randn(8, 10, device='cuda')
            output = model_gpu(x_gpu)
            assert output.device.type == 'cuda'
        
        # Test on CPU
        model_cpu = NeuralNetwork(layer_sizes=[10, 5], device='cpu')
        assert str(model_cpu.device) == 'cpu'
        
        x_cpu = torch.randn(8, 10)
        output = model_cpu(x_cpu)
        assert output.device.type == 'cpu'
    
    def test_model_repr(self):
        """Test model string representation."""
        model = NeuralNetwork(layer_sizes=[10, 20, 5])
        repr_str = repr(model)
        assert 'NeuralNetwork' in repr_str
        assert '[10, 20, 5]' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
