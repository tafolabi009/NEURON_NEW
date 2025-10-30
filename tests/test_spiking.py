"""
Tests for spiking neural network components.
"""

import pytest
import torch
from neurons.spiking import (
    SpikingNeuron,
    SpikingLayer,
    DendriticNeuron,
    KuramotoOscillator,
    OscillatoryPopulation,
    STDPSynapse,
    BCMLearning,
)


class TestSpikingNeuron:
    """Test spiking neuron dynamics."""
    
    def test_lif_initialization(self):
        """Test LIF neuron initialization."""
        neuron = SpikingNeuron(threshold=1.0, tau_mem=20.0, tau_syn=5.0)
        assert neuron.threshold == 1.0
        assert neuron.tau_mem == 20.0
        assert neuron.tau_syn == 5.0
    
    def test_lif_spike_generation(self):
        """Test that neuron spikes when threshold is reached."""
        neuron = SpikingNeuron(threshold=1.0, tau_mem=20.0)
        
        # Strong input should cause spike
        strong_input = torch.ones(1, 10) * 2.0
        output = neuron(strong_input)
        
        assert 'spikes' in output
        assert output['spikes'].sum() > 0  # At least some spikes
    
    def test_lif_subthreshold(self):
        """Test that weak input doesn't cause spike."""
        neuron = SpikingNeuron(threshold=1.0, tau_mem=20.0)
        
        # Weak input shouldn't cause immediate spike
        weak_input = torch.ones(1, 10) * 0.01
        output = neuron(weak_input)
        
        assert output['spikes'].sum() == 0  # No spikes


class TestSpikingLayer:
    """Test spiking layer."""
    
    def test_layer_initialization(self):
        """Test layer initialization."""
        layer = SpikingLayer(
            in_features=10,
            out_features=5,
            tau_mem=20.0,
            tau_syn=5.0,
            threshold=1.0
        )
        assert layer.in_features == 10
        assert layer.out_features == 5
    
    def test_layer_forward(self):
        """Test forward pass through layer."""
        layer = SpikingLayer(in_features=10, out_features=5)
        x = torch.randn(8, 10)
        
        output = layer(x)
        assert 'spikes' in output
        assert output['spikes'].shape == (8, 5)
        assert output['membrane'].shape == (8, 5)


class TestDendriticNeuron:
    """Test dendritic computation."""
    
    def test_dendritic_initialization(self):
        """Test dendritic neuron initialization."""
        neuron = DendriticNeuron(
            input_size=10,
            n_basal_branches=5,
            n_apical_branches=3
        )
        assert neuron.n_basal == 5
        assert neuron.n_apical == 3
    
    def test_dendritic_forward(self):
        """Test dendritic computation."""
        neuron = DendriticNeuron(
            input_size=10,
            n_basal_branches=5,
            n_apical_branches=3
        )
        
        basal_input = torch.randn(8, 10)
        apical_input = torch.randn(8, 10)
        
        output = neuron(basal_input, apical_input)
        
        assert 'total' in output
        assert 'basal' in output
        assert 'apical' in output
        assert output['total'].shape == (8, 1)


class TestKuramotoOscillator:
    """Test Kuramoto oscillator dynamics."""
    
    def test_oscillator_initialization(self):
        """Test oscillator initialization."""
        osc = KuramotoOscillator(natural_freq=40.0, coupling_strength=0.1)
        assert osc.natural_freq == 40.0
        assert osc.coupling_strength == 0.1
    
    def test_phase_evolution(self):
        """Test that phase evolves over time."""
        osc = KuramotoOscillator(natural_freq=40.0)
        
        initial_phase = torch.tensor([0.0])
        phase_t1 = osc.step(initial_phase, dt=1.0)
        phase_t2 = osc.step(phase_t1, dt=1.0)
        
        # Phase should evolve
        assert not torch.allclose(phase_t1, initial_phase)
        assert not torch.allclose(phase_t2, phase_t1)
    
    def test_phase_wrapping(self):
        """Test that phase wraps around 2π."""
        osc = KuramotoOscillator(natural_freq=1000.0)  # High frequency
        
        phase = torch.tensor([0.0])
        for _ in range(100):
            phase = osc.step(phase, dt=1.0)
        
        # Phase should be wrapped
        assert phase.item() >= -torch.pi and phase.item() <= torch.pi


class TestOscillatoryPopulation:
    """Test oscillatory population."""
    
    def test_population_initialization(self):
        """Test population initialization."""
        pop = OscillatoryPopulation(
            n_oscillators=10,
            natural_freq=40.0,
            coupling_strength=0.1
        )
        assert pop.n_oscillators == 10
    
    def test_population_step(self):
        """Test population step."""
        pop = OscillatoryPopulation(n_oscillators=10)
        spikes = torch.zeros(10)
        
        phases = pop.step(spikes, dt=1.0)
        assert phases.shape == (10,)
    
    def test_synchrony_computation(self):
        """Test synchrony computation."""
        pop = OscillatoryPopulation(n_oscillators=10)
        
        # Perfect synchrony
        pop.phases = torch.zeros(10)
        synchrony = pop.compute_synchrony()
        assert torch.isclose(synchrony, torch.tensor(1.0), atol=1e-5)
        
        # No synchrony (random phases)
        pop.phases = torch.rand(10) * 2 * torch.pi - torch.pi
        synchrony = pop.compute_synchrony()
        assert synchrony < 1.0


class TestSTDPSynapse:
    """Test STDP plasticity."""
    
    def test_stdp_initialization(self):
        """Test STDP synapse initialization."""
        synapse = STDPSynapse(
            in_features=10,
            out_features=5,
            tau_plus=20.0,
            tau_minus=20.0
        )
        assert synapse.tau_plus == 20.0
        assert synapse.tau_minus == 20.0
    
    def test_stdp_weight_update(self):
        """Test that STDP updates weights."""
        synapse = STDPSynapse(in_features=10, out_features=5)
        
        # Get initial weights
        initial_weights = synapse.weights.clone()
        
        # Apply STDP update
        pre_spikes = torch.rand(10) > 0.5
        post_spikes = torch.rand(5) > 0.5
        
        synapse.update(pre_spikes.float(), post_spikes.float(), dt=1.0)
        
        # Weights should change
        assert not torch.allclose(synapse.weights, initial_weights)


class TestBCMLearning:
    """Test BCM learning rule."""
    
    def test_bcm_initialization(self):
        """Test BCM initialization."""
        bcm = BCMLearning(n_units=10, tau_theta=1000.0)
        assert bcm.tau_theta == 1000.0
        assert bcm.theta.shape == (10,)
    
    def test_bcm_threshold_adaptation(self):
        """Test that BCM threshold adapts."""
        bcm = BCMLearning(n_units=10)
        
        initial_theta = bcm.theta.clone()
        
        # High activity should increase threshold
        high_activity = torch.ones(10)
        for _ in range(100):
            bcm.update(high_activity, dt=1.0)
        
        assert torch.all(bcm.theta > initial_theta)


class TestIntegration:
    """Test integration of components."""
    
    def test_spiking_with_dendrites(self):
        """Test spiking layer with dendritic computation."""
        from neurons.spiking import MultiCompartmentLayer
        
        layer = MultiCompartmentLayer(
            in_features=10,
            out_features=5,
            n_basal=3,
            n_apical=2
        )
        
        basal_input = torch.randn(8, 10)
        apical_input = torch.randn(8, 10)
        
        output = layer(basal_input, apical_input)
        assert output['total'].shape == (8, 5)
    
    def test_spikes_drive_oscillations(self):
        """Test that spikes can modulate oscillatory phases."""
        pop = OscillatoryPopulation(n_oscillators=10)
        
        # No spikes
        no_spikes = torch.zeros(10)
        phase1 = pop.step(no_spikes, dt=1.0)
        
        # With spikes
        with_spikes = torch.ones(10)
        phase2 = pop.step(with_spikes, dt=1.0)
        
        # Phases should be different
        assert not torch.allclose(phase1, phase2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
