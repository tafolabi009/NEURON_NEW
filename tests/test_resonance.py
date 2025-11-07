"""
Test suite for Resonance Neural Networks
"""

import pytest
import torch
from resonance_nn import ResonanceNet, ResonanceLayer
from resonance_nn.layers.holographic import HolographicMemory
from resonance_nn.training import ResonanceTrainer, create_criterion


class TestResonanceLayer:
    """Test resonance layer functionality"""
    
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape"""
        layer = ResonanceLayer(input_dim=64, num_frequencies=16)
        x = torch.randn(8, 32, 64)  # (batch, seq, dim)
        output = layer(x)
        assert output.shape == x.shape
        
    def test_complex_weights(self):
        """Test complex weight initialization"""
        layer = ResonanceLayer(input_dim=64, num_frequencies=16)
        weights = layer.weights()
        assert weights.dtype == torch.complex64 or weights.dtype == torch.complex128
        
    def test_gradient_computation(self):
        """Test that gradients are computed and bounded"""
        layer = ResonanceLayer(input_dim=64, num_frequencies=16)
        x = torch.randn(4, 32, 64)
        target = torch.randn(4, 32, 64)
        
        output = layer(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Check gradients exist
        assert layer.weights.magnitude.grad is not None
        assert layer.weights.phase.grad is not None
        
        # Check gradients are bounded
        mag_grad_norm = torch.norm(layer.weights.magnitude.grad).item()
        phase_grad_norm = torch.norm(layer.weights.phase.grad).item()
        
        assert mag_grad_norm < 1000, "Magnitude gradient exploded"
        assert phase_grad_norm < 1000, "Phase gradient exploded"


class TestHolographicMemory:
    """Test holographic memory functionality"""
    
    def test_encode_decode(self):
        """Test basic encoding and decoding"""
        memory = HolographicMemory(pattern_dim=128, capacity=10)
        pattern = torch.randn(128)
        
        memory.encode(pattern.unsqueeze(0))
        reconstructed = memory.reconstruct()
        
        assert reconstructed.shape[0] == 128
        
    def test_capacity(self):
        """Test capacity calculations"""
        memory = HolographicMemory(pattern_dim=128, capacity=100)
        
        theoretical = memory.get_theoretical_capacity()
        assert theoretical > 0
        
        utilization = memory.get_capacity_utilization()
        assert 0 <= utilization <= 1.0
        
    def test_multiple_patterns(self):
        """Test storing multiple patterns"""
        memory = HolographicMemory(pattern_dim=64, capacity=50)
        
        patterns = [torch.randn(64) for _ in range(5)]
        for pattern in patterns:
            memory.encode(pattern.unsqueeze(0))
            
        assert memory.num_patterns.item() == 5
        
    def test_reconstruction_fidelity(self):
        """Test reconstruction quality"""
        memory = HolographicMemory(pattern_dim=128, capacity=10)
        pattern = torch.randn(128)
        
        memory.encode(pattern.unsqueeze(0))
        fidelity = memory.get_reconstruction_fidelity(pattern.unsqueeze(0))
        
        assert 0 <= fidelity <= 1.0


class TestResonanceNet:
    """Test complete network"""
    
    def test_forward(self):
        """Test forward pass"""
        model = ResonanceNet(
            input_dim=64,
            num_frequencies=16,
            num_layers=2,
            holographic_capacity=10,
        )
        
        x = torch.randn(4, 32, 64)
        output = model(x, use_memory=False)
        
        assert output.shape == x.shape
        
    def test_memory_integration(self):
        """Test holographic memory integration"""
        model = ResonanceNet(
            input_dim=64,
            num_frequencies=16,
            num_layers=2,
            holographic_capacity=10,
        )
        
        x = torch.randn(4, 32, 64)
        
        # Without memory
        output1 = model(x, use_memory=False)
        
        # Store to memory
        model.encode_to_memory(x)
        
        # With memory
        output2 = model(x, use_memory=True)
        
        # Outputs should be different
        assert not torch.allclose(output1, output2)
        
    def test_complexity_estimate(self):
        """Test complexity estimation"""
        model = ResonanceNet(
            input_dim=64,
            num_frequencies=16,
            num_layers=2,
        )
        
        complexity = model.get_complexity_estimate(128)
        
        assert 'total' in complexity
        assert 'resonance' in complexity
        assert 'complexity_class' in complexity
        assert complexity['complexity_class'] == 'O(n log n + k²)'


class TestTrainer:
    """Test training functionality"""
    
    def test_train_step(self):
        """Test single training step"""
        model = ResonanceNet(
            input_dim=32,
            num_frequencies=8,
            num_layers=2,
        )
        
        trainer = ResonanceTrainer(model, learning_rate=1e-3)
        criterion = create_criterion('regression')
        
        batch = {
            'input': torch.randn(4, 16, 32),
            'target': torch.randn(4, 16, 32),
        }
        
        loss = trainer.train_step(batch, criterion)
        
        assert isinstance(loss, float)
        assert loss > 0
        
    def test_gradient_stability_check(self):
        """Test gradient stability checking"""
        model = ResonanceNet(
            input_dim=32,
            num_frequencies=8,
            num_layers=2,
        )
        
        trainer = ResonanceTrainer(model, learning_rate=1e-3)
        criterion = create_criterion('regression')
        
        # Run a few steps
        for _ in range(5):
            batch = {
                'input': torch.randn(4, 16, 32),
                'target': torch.randn(4, 16, 32),
            }
            trainer.train_step(batch, criterion)
            
        stability = trainer.check_gradient_stability()
        
        assert 'magnitude_stable' in stability
        assert 'phase_stable' in stability
        assert 'all_stable' in stability


class TestComplexity:
    """Test complexity properties"""
    
    def test_time_scaling(self):
        """Test that time scales approximately as O(n log n)"""
        import time
        import numpy as np
        
        model = ResonanceNet(
            input_dim=64,
            num_frequencies=16,
            num_layers=2,
        )
        model.eval()
        
        sequence_lengths = [32, 64, 128]
        times = []
        
        with torch.no_grad():
            for n in sequence_lengths:
                x = torch.randn(4, n, 64)
                
                # Warmup
                _ = model(x, use_memory=False)
                
                # Measure
                start = time.time()
                for _ in range(10):
                    _ = model(x, use_memory=False)
                end = time.time()
                
                times.append((end - start) / 10)
                
        # Check that time doesn't scale as O(n^2)
        # For O(n^2), ratio would be 4x, for O(n log n) it's ~2.2x
        ratio = times[-1] / times[0]
        
        # Should be closer to n*log(n) than n^2
        expected_nlogn = (128 * np.log2(128)) / (32 * np.log2(32))
        expected_n2 = (128 ** 2) / (32 ** 2)
        
        # Ratio should be closer to expected_nlogn than expected_n2
        assert abs(ratio - expected_nlogn) < abs(ratio - expected_n2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
