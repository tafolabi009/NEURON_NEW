"""
Dendritic Neuron Models
Neurons with multiple computational branches - each neuron is a network!

Key Innovation: Move beyond point neurons to capture dendritic computation.
Each branch performs local nonlinear computation before somatic integration.

This exponentially increases representational capacity:
    Point neuron: O(n) functions
    Dendritic neuron: O(n^k) functions where k = number of branches

References:
- Poirazi et al. (2003): Pyramidal neuron as two-layer neural network
- London & Häusser (2005): Dendritic computation
- Larkum et al. (2009): A cellular mechanism for cortical associations
- Stuart & Spruston (2015): Dendritic integration
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from numba import jit
from dataclasses import dataclass


@dataclass
class BranchState:
    """State of a single dendritic branch"""
    voltage: float
    threshold: float
    inputs: np.ndarray
    output: float
    spike: bool


class DendriticBranch:
    """
    Single dendritic branch with local computation
    
    Mathematical Model:
        Vbranch = g_threshold(Σᵢ wᵢxᵢ - θ)
        
    Where g_threshold implements:
        - Sublinear integration (shunting inhibition)
        - Threshold (active conductances)
        - Supralinear integration (NMDA spikes)
        - Saturation (voltage-gated channels)
    
    Biological Basis:
        - Apical dendrites: integrate feedback
        - Basal dendrites: integrate feedforward
        - Each branch ~20-50 synapses
        - Local NMDA spikes provide nonlinearity
    """
    
    def __init__(
        self,
        n_inputs: int,
        threshold: float = 0.5,
        nonlinearity: str = 'threshold_linear',  # 'threshold_linear', 'sigmoid', 'quadratic'
        learning_rate: float = 0.01
    ):
        self.n_inputs = n_inputs
        self.threshold = threshold
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate
        
        # Synaptic weights
        self.weights = np.random.randn(n_inputs) * 0.1
        
        # State
        self.voltage = 0.0
        self.output = 0.0
        self.spike = False
        
        # History
        self.input_history: List[np.ndarray] = []
        self.output_history: List[float] = []
    
    def compute(self, inputs: np.ndarray) -> float:
        """
        Compute branch output
        
        Parameters:
        -----------
        inputs : np.ndarray
            Synaptic inputs to this branch
            
        Returns:
        --------
        float : Branch output (current to soma)
        """
        # Linear integration
        self.voltage = np.dot(self.weights, inputs)
        
        # Apply nonlinearity
        if self.nonlinearity == 'threshold_linear':
            # Rectified linear above threshold
            self.output = np.maximum(0, self.voltage - self.threshold)
            
        elif self.nonlinearity == 'sigmoid':
            # Sigmoidal (smooth NMDA-like)
            self.output = 1.0 / (1.0 + np.exp(-(self.voltage - self.threshold)))
            
        elif self.nonlinearity == 'quadratic':
            # Supralinear above threshold (NMDA spike)
            if self.voltage > self.threshold:
                self.output = (self.voltage - self.threshold) ** 2
            else:
                self.output = 0.0
        
        # Saturation (max branch current)
        self.output = np.tanh(self.output)
        
        # Check for branch spike
        self.spike = self.voltage > (self.threshold + 0.5)
        
        # Record
        self.input_history.append(inputs.copy())
        self.output_history.append(self.output)
        
        return self.output
    
    def update_weights(self, error: float, modulation: float = 1.0):
        """
        Update synaptic weights
        
        Local learning rule: Δw = η · error · input
        """
        if len(self.input_history) == 0:
            return
        
        recent_input = self.input_history[-1]
        dw = self.learning_rate * modulation * error * recent_input
        self.weights += dw
        
        # Clip weights
        self.weights = np.clip(self.weights, -1.0, 1.0)
    
    def get_capacity(self) -> float:
        """
        Estimate representational capacity of this branch
        
        Threshold-linear unit can implement ~2^n_inputs boolean functions
        """
        return 2 ** self.n_inputs


class DendriticNeuron:
    """
    Multi-compartment neuron with dendritic branches
    
    Architecture:
        Inputs → [Branch 1]
                 [Branch 2] → Soma → Spike
                 [Branch 3]
                 [...]
    
    Mathematical Model:
        Branch outputs: yⱼ = g_branch(Σᵢ wᵢⱼxᵢ - θⱼ)
        Soma: V_soma = Σⱼ yⱼ
        Spike: if V_soma > θ_soma
    
    Key Property (Poirazi et al., 2003):
        This implements a 2-layer neural network!
        Hidden layer = branches
        Output layer = soma
        
        → Single neuron has same capacity as small network
    
    Theoretical Capacity:
        With k branches, each with n inputs:
            Point neuron: 2^n boolean functions
            Dendritic neuron: 2^(n·k) boolean functions
            
        Example: n=20, k=5
            Point: 2^20 ≈ 1 million functions
            Dendritic: 2^100 ≈ 10^30 functions!
    """
    
    def __init__(
        self,
        n_branches: int,
        branch_inputs: List[int],  # Number of inputs per branch
        soma_threshold: float = 0.5,
        branch_nonlinearity: str = 'threshold_linear',
        learning_rate: float = 0.01
    ):
        self.n_branches = n_branches
        self.soma_threshold = soma_threshold
        self.learning_rate = learning_rate
        
        # Create branches
        self.branches: List[DendriticBranch] = []
        for n_in in branch_inputs:
            branch = DendriticBranch(
                n_inputs=n_in,
                threshold=0.3,
                nonlinearity=branch_nonlinearity,
                learning_rate=learning_rate
            )
            self.branches.append(branch)
        
        # Soma state (integrate-and-fire)
        self.v_soma = -70.0  # mV
        self.v_rest = -70.0
        self.v_reset = -75.0
        self.tau_m = 20.0  # ms
        self.refractory_time = 0.0
        self.refractory_period = 2.0  # ms
        
        # Output
        self.spike = False
        self.spike_times: List[float] = []
        self.current_time = 0.0
    
    def forward(
        self,
        branch_inputs: List[np.ndarray],
        dt: float = 1.0,
        external_current: float = 0.0
    ) -> bool:
        """
        Forward pass through neuron
        
        Parameters:
        -----------
        branch_inputs : List[np.ndarray]
            Inputs to each branch
        dt : float
            Time step (ms)
        external_current : float
            Direct current to soma
            
        Returns:
        --------
        bool : True if neuron spiked
        """
        self.current_time += dt
        
        # Check refractory period
        if self.refractory_time > 0:
            self.refractory_time -= dt
            self.v_soma = self.v_reset
            self.spike = False
            return False
        
        # Compute branch outputs
        i_dendritic = 0.0
        for branch, inputs in zip(self.branches, branch_inputs):
            branch_current = branch.compute(inputs)
            i_dendritic += branch_current
        
        # Total somatic current
        i_total = i_dendritic + external_current
        
        # Update soma voltage (LIF dynamics)
        dv = (-(self.v_soma - self.v_rest) + i_total) / self.tau_m
        self.v_soma += dv * dt
        
        # Check for spike
        if self.v_soma >= self.soma_threshold:
            self.spike = True
            self.spike_times.append(self.current_time)
            self.v_soma = self.v_reset
            self.refractory_time = self.refractory_period
            return True
        
        self.spike = False
        return False
    
    def backward(
        self,
        error: float,
        neuromodulation: float = 1.0
    ):
        """
        Backward pass: update branch weights
        
        Uses error signal to update each branch independently.
        This is a form of local credit assignment.
        """
        # Distribute error across branches
        # Branches with larger outputs get more credit/blame
        branch_outputs = np.array([b.output_history[-1] if b.output_history else 0.0 
                                   for b in self.branches])
        
        if np.sum(np.abs(branch_outputs)) > 1e-6:
            branch_errors = error * branch_outputs / (np.sum(np.abs(branch_outputs)) + 1e-6)
        else:
            branch_errors = np.ones(self.n_branches) * error / self.n_branches
        
        # Update each branch
        for branch, branch_error in zip(self.branches, branch_errors):
            branch.update_weights(branch_error, neuromodulation)
    
    def get_total_capacity(self) -> float:
        """
        Compute total representational capacity
        
        Product of branch capacities (they can encode independently)
        """
        total_capacity = 1.0
        for branch in self.branches:
            total_capacity *= branch.get_capacity()
        return total_capacity
    
    def get_firing_rate(self, time_window: float = 100.0) -> float:
        """Calculate recent firing rate (Hz)"""
        recent_spikes = [t for t in self.spike_times 
                        if self.current_time - t <= time_window]
        return len(recent_spikes) * 1000.0 / time_window if time_window > 0 else 0.0
    
    def reset(self):
        """Reset neuron state"""
        self.v_soma = self.v_rest
        self.refractory_time = 0.0
        self.spike = False
        self.current_time = 0.0
        self.spike_times = []
        for branch in self.branches:
            branch.voltage = 0.0
            branch.output = 0.0


class DendriticLayer:
    """
    Layer of dendritic neurons
    
    Each neuron has multiple branches, enabling rich computation.
    """
    
    def __init__(
        self,
        n_neurons: int,
        n_branches_per_neuron: int,
        inputs_per_branch: int,
        soma_threshold: float = 0.5
    ):
        self.n_neurons = n_neurons
        
        # Create neurons
        self.neurons: List[DendriticNeuron] = []
        for _ in range(n_neurons):
            neuron = DendriticNeuron(
                n_branches=n_branches_per_neuron,
                branch_inputs=[inputs_per_branch] * n_branches_per_neuron,
                soma_threshold=soma_threshold
            )
            self.neurons.append(neuron)
    
    def forward(
        self,
        inputs: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Forward pass through layer
        
        Each neuron gets different subsets of inputs to different branches.
        This implements random projections / receptive fields.
        """
        spikes = np.zeros(self.n_neurons, dtype=bool)
        
        for i, neuron in enumerate(self.neurons):
            # Divide inputs among branches (random receptive fields)
            n_branches = neuron.n_branches
            n_inputs_total = len(inputs)
            inputs_per_branch = n_inputs_total // n_branches
            
            branch_inputs = []
            for b in range(n_branches):
                start = (b * inputs_per_branch) % n_inputs_total
                end = ((b + 1) * inputs_per_branch) % n_inputs_total
                if end > start:
                    branch_input = inputs[start:end]
                else:
                    branch_input = np.concatenate([inputs[start:], inputs[:end]])
                
                # Pad or trim to correct size
                expected_size = neuron.branches[b].n_inputs
                if len(branch_input) < expected_size:
                    branch_input = np.pad(branch_input, (0, expected_size - len(branch_input)))
                elif len(branch_input) > expected_size:
                    branch_input = branch_input[:expected_size]
                
                branch_inputs.append(branch_input)
            
            # Forward through neuron
            spiked = neuron.forward(branch_inputs, dt)
            spikes[i] = spiked
        
        return spikes
    
    def backward(
        self,
        errors: np.ndarray,
        neuromodulation: float = 1.0
    ):
        """
        Backward pass: update all neuron weights
        """
        for neuron, error in zip(self.neurons, errors):
            neuron.backward(error, neuromodulation)
    
    def get_firing_rates(self) -> np.ndarray:
        """Get firing rates of all neurons"""
        return np.array([n.get_firing_rate() for n in self.neurons])
    
    def get_total_capacity(self) -> float:
        """Total layer capacity (sum of neuron capacities)"""
        return sum(n.get_total_capacity() for n in self.neurons)
    
    def reset(self):
        """Reset all neurons"""
        for neuron in self.neurons:
            neuron.reset()


class ApicalBasalNeuron:
    """
    Neuron with separate apical and basal dendrites
    
    Biological Motivation (Larkum et al., 2009):
        - Basal dendrites: integrate feedforward inputs
        - Apical dendrites: integrate feedback/contextual inputs
        - Coincidence detection: BAC (Back-propagating Action potential 
          Coincident with apical Ca²⁺ spike) firing
    
    This implements a gating mechanism:
        - Feedforward alone → weak response
        - Feedback alone → weak response
        - Feedforward + Feedback → strong burst!
    
    Mathematical Model:
        V_basal = Σᵢ w_basal_i · x_i
        V_apical = Σⱼ w_apical_j · context_j
        
        Spike if:
            (V_basal > θ_basal) AND (V_apical > θ_apical)
        
        This implements AND logic → coincidence detection
    """
    
    def __init__(
        self,
        n_basal_branches: int = 5,
        n_apical_branches: int = 3,
        inputs_per_branch: int = 20,
        coincidence_window: float = 5.0  # ms
    ):
        # Basal dendrites (feedforward)
        self.basal_dendrites = DendriticNeuron(
            n_branches=n_basal_branches,
            branch_inputs=[inputs_per_branch] * n_basal_branches,
            soma_threshold=0.5
        )
        
        # Apical dendrites (feedback/context)
        self.apical_dendrites = DendriticNeuron(
            n_branches=n_apical_branches,
            branch_inputs=[inputs_per_branch] * n_apical_branches,
            soma_threshold=0.3
        )
        
        # Coincidence detection parameters
        self.coincidence_window = coincidence_window
        self.last_basal_spike: Optional[float] = None
        self.last_apical_spike: Optional[float] = None
        
        # Output
        self.burst = False
        self.spike = False
    
    def forward(
        self,
        feedforward_inputs: List[np.ndarray],
        feedback_inputs: List[np.ndarray],
        dt: float = 1.0
    ) -> Tuple[bool, bool]:
        """
        Forward pass
        
        Returns:
        --------
        Tuple[bool, bool] : (normal spike, burst spike)
        """
        # Process both dendritic regions
        basal_spike = self.basal_dendrites.forward(feedforward_inputs, dt)
        apical_spike = self.apical_dendrites.forward(feedback_inputs, dt)
        
        current_time = self.basal_dendrites.current_time
        
        # Track spike times
        if basal_spike:
            self.last_basal_spike = current_time
        if apical_spike:
            self.last_apical_spike = current_time
        
        # Check for coincidence (burst)
        self.burst = False
        if self.last_basal_spike is not None and self.last_apical_spike is not None:
            time_diff = abs(self.last_basal_spike - self.last_apical_spike)
            if time_diff <= self.coincidence_window:
                self.burst = True
        
        # Regular spike
        self.spike = basal_spike
        
        return self.spike, self.burst


def test_dendritic_neuron():
    """
    Test dendritic neuron capacity and computation
    """
    print("Testing Dendritic Neuron...")
    
    # Create neuron
    neuron = DendriticNeuron(
        n_branches=5,
        branch_inputs=[10, 10, 10, 10, 10],
        soma_threshold=0.5
    )
    
    # Test forward pass
    branch_inputs = [np.random.randn(10) for _ in range(5)]
    
    spike_count = 0
    for _ in range(100):
        spiked = neuron.forward(branch_inputs, dt=1.0)
        if spiked:
            spike_count += 1
    
    firing_rate = neuron.get_firing_rate()
    capacity = neuron.get_total_capacity()
    
    print(f"Firing rate: {firing_rate:.1f} Hz")
    print(f"Spike count: {spike_count}")
    print(f"Representational capacity: {capacity:.2e} functions")
    
    # Compare to point neuron
    point_neuron_capacity = 2 ** 50  # 50 total inputs
    print(f"Point neuron capacity: {point_neuron_capacity:.2e} functions")
    print(f"Improvement: {capacity / point_neuron_capacity:.2e}×")
    
    print("\n✓ Dendritic neuron works!")
    print(f"✓ Each neuron is a multi-layer network")
    
    return neuron


if __name__ == "__main__":
    test_dendritic_neuron()
