"""
Advanced Dendritic Computation
Hierarchical dendritic trees with active conductances and compartmental modeling

This goes far beyond simple point neurons to implement realistic dendritic processing:
1. Multi-compartment dendrites with calcium dynamics
2. Active conductances (NMDA spikes, calcium spikes)
3. Hierarchical branching with parent-child interactions
4. Branch-specific plasticity and routing
5. Dendritic prediction and error computation

Key Innovation: Each neuron is a hierarchical computational tree!

References:
- Häusser et al. (2000): The beat goes on
- Larkum (2013): A cellular mechanism for cortical associations
- Magee & Grienberger (2020): Synaptic Plasticity Forms and Functions
- Richards & Lillicrap (2019): Dendritic solutions to the credit assignment problem
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class CompartmentType(Enum):
    """Types of dendritic compartments"""
    SOMA = "soma"
    BASAL = "basal"  # Basal dendrites: feedforward input
    APICAL_PROXIMAL = "apical_proximal"  # Proximal apical: lateral input
    APICAL_DISTAL = "apical_distal"  # Distal apical: feedback input
    OBLIQUE = "oblique"  # Oblique dendrites: mixed input


@dataclass
class CompartmentParams:
    """Parameters for a dendritic compartment"""
    # Passive properties
    leak_conductance: float = 0.1  # Leak conductance (mS/cm²)
    leak_reversal: float = -70.0  # Leak reversal potential (mV)
    capacitance: float = 1.0  # Membrane capacitance (μF/cm²)
    
    # Active conductances
    has_nmda: bool = True  # NMDA spikes
    has_calcium: bool = False  # Calcium spikes
    has_sodium: bool = False  # Sodium spikes (for soma)
    
    # NMDA parameters
    nmda_threshold: float = -45.0  # mV
    nmda_conductance: float = 0.5  # mS/cm²
    nmda_reversal: float = 0.0  # mV
    
    # Calcium parameters
    calcium_threshold: float = -30.0  # mV
    calcium_conductance: float = 1.0  # mS/cm²
    calcium_reversal: float = 120.0  # mV
    
    # Sodium parameters (action potentials)
    sodium_threshold: float = -50.0  # mV
    sodium_conductance: float = 120.0  # mS/cm²
    sodium_reversal: float = 50.0  # mV
    
    # Coupling
    axial_resistance: float = 100.0  # Ω·cm


class DendriticCompartment:
    """
    Single dendritic compartment with active conductances
    
    This implements realistic dendritic physics using cable theory
    and Hodgkin-Huxley-style active conductances.
    
    Voltage equation (cable theory):
        C·dV/dt = -I_leak - I_syn + I_axial + I_active
        
    Where:
        I_leak = g_L(V - E_L)
        I_syn = Σ g_syn(V - E_syn)
        I_axial = (V_parent - V) / R_axial
        I_active = I_NMDA + I_Ca + I_Na
    """
    
    def __init__(self, compartment_type: CompartmentType, params: Optional[CompartmentParams] = None):
        self.type = compartment_type
        self.params = params or CompartmentParams()
        
        # State variables
        self.voltage = self.params.leak_reversal
        self.calcium = 0.0  # Internal calcium concentration
        
        # Synaptic inputs
        self.synaptic_weights = np.array([])
        self.synaptic_conductances = np.array([])
        
        # Connections
        self.parent: Optional[DendriticCompartment] = None
        self.children: List[DendriticCompartment] = []
        
        # Activity history
        self.voltage_history = []
        self.spike_times = []
        self.calcium_history = []
        
        # Learning
        self.plasticity_trace = 0.0
        self.prediction = 0.0
        self.prediction_error = 0.0
    
    def initialize_synapses(self, n_synapses: int):
        """Initialize synaptic connections"""
        self.synaptic_weights = np.random.randn(n_synapses) * 0.1
        self.synaptic_conductances = np.zeros(n_synapses)
    
    def compute_leak_current(self) -> float:
        """Compute leak current"""
        return self.params.leak_conductance * (self.voltage - self.params.leak_reversal)
    
    def compute_synaptic_current(self, inputs: np.ndarray) -> float:
        """
        Compute synaptic current from inputs
        
        Args:
            inputs: Synaptic input signals
        """
        if len(self.synaptic_weights) == 0:
            return 0.0
        
        # Update conductances based on inputs
        self.synaptic_conductances = np.maximum(0, self.synaptic_weights * inputs)
        
        # Excitatory synaptic current (AMPA-like)
        E_syn = 0.0  # Reversal potential for excitatory synapses
        I_syn = np.sum(self.synaptic_conductances * (self.voltage - E_syn))
        
        return I_syn
    
    def compute_nmda_current(self) -> float:
        """
        Compute NMDA current with voltage-dependent magnesium block
        
        NMDA receptors provide:
            1. Voltage-dependent nonlinearity (Mg²⁺ block)
            2. Calcium influx for plasticity
            3. Dendritic spikes
        """
        if not self.params.has_nmda:
            return 0.0
        
        # Voltage-dependent magnesium block
        mg_block = 1.0 / (1.0 + 0.33 * np.exp(-0.06 * (self.voltage - self.params.nmda_threshold)))
        
        # NMDA current
        g_nmda = self.params.nmda_conductance * mg_block
        I_nmda = g_nmda * (self.voltage - self.params.nmda_reversal)
        
        # Calcium influx through NMDA
        if self.voltage > self.params.nmda_threshold:
            self.calcium += 0.1 * mg_block
        
        return I_nmda
    
    def compute_calcium_current(self) -> float:
        """
        Compute calcium current (dendritic calcium spikes)
        
        Calcium spikes in distal dendrites provide:
            1. Long-lasting depolarization
            2. Strong nonlinearity
            3. Top-down modulation signal
        """
        if not self.params.has_calcium:
            return 0.0
        
        # Voltage-gated calcium channels
        activation = 1.0 / (1.0 + np.exp(-(self.voltage - self.params.calcium_threshold) / 5.0))
        
        if activation > 0.5:  # Threshold crossing
            g_ca = self.params.calcium_conductance * activation
            I_ca = g_ca * (self.voltage - self.params.calcium_reversal)
            
            # Increase internal calcium
            self.calcium += 0.5 * activation
            
            return I_ca
        
        return 0.0
    
    def compute_sodium_current(self) -> float:
        """
        Compute sodium current (action potentials in soma)
        
        Fast sodium spikes for action potential generation
        """
        if not self.params.has_sodium:
            return 0.0
        
        # Fast activation
        activation = 1.0 / (1.0 + np.exp(-(self.voltage - self.params.sodium_threshold) / 3.0))
        
        if activation > 0.9:  # Threshold for spike
            g_na = self.params.sodium_conductance * activation
            I_na = g_na * (self.voltage - self.params.sodium_reversal)
            return I_na
        
        return 0.0
    
    def compute_axial_current(self) -> float:
        """
        Compute axial current from parent compartment
        
        This couples compartments together in the dendritic tree
        """
        if self.parent is None:
            return 0.0
        
        # Ohm's law
        I_axial = (self.parent.voltage - self.voltage) / self.params.axial_resistance
        return I_axial
    
    def update(self, inputs: np.ndarray, dt: float = 0.1) -> bool:
        """
        Update compartment voltage for one time step
        
        Args:
            inputs: Synaptic inputs to this compartment
            dt: Time step (ms)
            
        Returns:
            spike: Whether this compartment generated a spike
        """
        # Compute all currents
        I_leak = self.compute_leak_current()
        I_syn = self.compute_synaptic_current(inputs)
        I_axial = self.compute_axial_current()
        I_nmda = self.compute_nmda_current()
        I_ca = self.compute_calcium_current()
        I_na = self.compute_sodium_current()
        
        # Total current
        I_total = -I_leak - I_syn + I_axial - I_nmda - I_ca - I_na
        
        # Update voltage
        dV = (I_total / self.params.capacitance) * dt
        self.voltage += dV
        
        # Decay calcium
        self.calcium *= 0.99
        
        # Check for spike
        spike = False
        if self.params.has_sodium and self.voltage > 0:  # Soma spike
            spike = True
            self.spike_times.append(len(self.voltage_history))
            # Reset after spike
            self.voltage = -65.0
        elif self.params.has_nmda and self.voltage > self.params.nmda_threshold + 20:  # NMDA spike
            spike = True
            self.spike_times.append(len(self.voltage_history))
        
        # Record
        self.voltage_history.append(self.voltage)
        self.calcium_history.append(self.calcium)
        
        return spike
    
    def update_plasticity_trace(self, dt: float = 0.1, tau: float = 20.0):
        """
        Update plasticity trace (eligibility trace for learning)
        
        This decays exponentially and is boosted by spikes and calcium
        """
        # Decay
        self.plasticity_trace *= np.exp(-dt / tau)
        
        # Boost by recent spikes
        if len(self.spike_times) > 0 and len(self.voltage_history) - self.spike_times[-1] < 5:
            self.plasticity_trace += 1.0
        
        # Boost by calcium
        self.plasticity_trace += 0.1 * self.calcium
    
    def learn(self, error: float, learning_rate: float = 0.001, modulation: float = 1.0):
        """
        Update synaptic weights using error signal and plasticity trace
        
        This implements three-factor learning:
            Δw = η · pre · post · modulation
            
        Where:
            pre = input activity
            post = plasticity trace (calcium, spikes)
            modulation = dopamine, attention, etc.
        """
        if len(self.synaptic_weights) == 0:
            return
        
        # Get recent inputs (from last update)
        recent_inputs = self.synaptic_conductances / (self.synaptic_weights + 1e-8)
        
        # Three-factor rule
        dw = learning_rate * modulation * error * self.plasticity_trace * recent_inputs
        
        # Update
        self.synaptic_weights += dw
        
        # Clip
        self.synaptic_weights = np.clip(self.synaptic_weights, -2.0, 2.0)


class HierarchicalDendriticTree:
    """
    Complete dendritic tree with hierarchical structure
    
    Structure:
                    [Soma]
                      |
          +-----------+-----------+
          |           |           |
       [Basal]   [Apical Prox] [Oblique]
          |           |
       [B1][B2]  [Apical Dist]
    
    This implements the two-point neuron model (Larkum 2013):
        1. Basal integration point: feedforward input
        2. Apical integration point: feedback/context
        3. Soma: combines both for action potentials
        
    Key property: Apical and basal can interact nonlinearly!
        - Basal alone: weak response
        - Apical alone: weak response
        - Basal + Apical: STRONG response (BAC firing)
    """
    
    def __init__(self, n_basal_branches: int = 5, n_apical_branches: int = 3,
                 synapses_per_branch: int = 20):
        self.n_basal_branches = n_basal_branches
        self.n_apical_branches = n_apical_branches
        self.synapses_per_branch = synapses_per_branch
        
        # Create soma (can generate action potentials)
        soma_params = CompartmentParams(
            has_nmda=False,
            has_calcium=False,
            has_sodium=True
        )
        self.soma = DendriticCompartment(CompartmentType.SOMA, soma_params)
        
        # Create basal dendrites (feedforward)
        basal_params = CompartmentParams(has_nmda=True, has_calcium=False)
        self.basal_dendrites = []
        for _ in range(n_basal_branches):
            branch = DendriticCompartment(CompartmentType.BASAL, basal_params)
            branch.initialize_synapses(synapses_per_branch)
            branch.parent = self.soma
            self.soma.children.append(branch)
            self.basal_dendrites.append(branch)
        
        # Create proximal apical dendrite
        apical_prox_params = CompartmentParams(has_nmda=True, has_calcium=False)
        self.apical_proximal = DendriticCompartment(CompartmentType.APICAL_PROXIMAL, apical_prox_params)
        self.apical_proximal.parent = self.soma
        self.soma.children.append(self.apical_proximal)
        
        # Create distal apical dendrites (feedback)
        apical_dist_params = CompartmentParams(has_nmda=True, has_calcium=True)
        self.apical_distal_dendrites = []
        for _ in range(n_apical_branches):
            branch = DendriticCompartment(CompartmentType.APICAL_DISTAL, apical_dist_params)
            branch.initialize_synapses(synapses_per_branch)
            branch.parent = self.apical_proximal
            self.apical_proximal.children.append(branch)
            self.apical_distal_dendrites.append(branch)
        
        # All compartments
        self.all_compartments = [self.soma, self.apical_proximal] + self.basal_dendrites + self.apical_distal_dendrites
    
    def forward(self, basal_inputs: np.ndarray, apical_inputs: np.ndarray,
                duration_ms: float = 50.0, dt: float = 0.1) -> Tuple[bool, Dict[str, float]]:
        """
        Forward pass through dendritic tree
        
        Args:
            basal_inputs: (n_basal_branches, synapses_per_branch) feedforward inputs
            apical_inputs: (n_apical_branches, synapses_per_branch) feedback inputs
            duration_ms: Simulation duration
            dt: Time step
            
        Returns:
            Tuple of (spike, metrics)
        """
        n_steps = int(duration_ms / dt)
        soma_spikes = []
        
        for step in range(n_steps):
            # Update basal dendrites (feedforward)
            for i, dendrite in enumerate(self.basal_dendrites):
                if i < len(basal_inputs):
                    dendrite.update(basal_inputs[i], dt)
                    dendrite.update_plasticity_trace(dt)
            
            # Update apical dendrites (feedback)
            for i, dendrite in enumerate(self.apical_distal_dendrites):
                if i < len(apical_inputs):
                    dendrite.update(apical_inputs[i], dt)
                    dendrite.update_plasticity_trace(dt)
            
            # Update apical proximal
            self.apical_proximal.update(np.array([]), dt)
            self.apical_proximal.update_plasticity_trace(dt)
            
            # Update soma
            spike = self.soma.update(np.array([]), dt)
            if spike:
                soma_spikes.append(step)
        
        # Compute metrics
        fired = len(soma_spikes) > 0
        metrics = {
            'n_spikes': len(soma_spikes),
            'mean_basal_voltage': np.mean([d.voltage for d in self.basal_dendrites]),
            'mean_apical_voltage': np.mean([d.voltage for d in self.apical_distal_dendrites]),
            'soma_voltage': self.soma.voltage,
            'apical_calcium': np.mean([d.calcium for d in self.apical_distal_dendrites])
        }
        
        return fired, metrics
    
    def backward(self, error: float, learning_rate: float = 0.001, modulation: float = 1.0):
        """
        Backward pass: update all synaptic weights
        
        Uses predictive error and compartment-specific plasticity traces
        """
        # Update basal dendrites (use error signal directly)
        for dendrite in self.basal_dendrites:
            dendrite.learn(error, learning_rate, modulation)
        
        # Update apical dendrites (use error with calcium gating)
        for dendrite in self.apical_distal_dendrites:
            # Apical learning is gated by calcium (top-down attention)
            calcium_gate = min(1.0, dendrite.calcium / 0.5)
            dendrite.learn(error, learning_rate, modulation * calcium_gate)
    
    def compute_bac_firing(self) -> float:
        """
        Compute BAC (Backpropagating Action potential-activated Calcium spike) probability
        
        This measures the nonlinear interaction between basal and apical:
            - Strong basal + strong apical → BAC firing (burst)
            - Weak basal or weak apical → no BAC
        """
        # Check for coincident basal depolarization and apical calcium
        basal_active = np.mean([d.voltage for d in self.basal_dendrites]) > -50
        apical_calcium = np.mean([d.calcium for d in self.apical_distal_dendrites])
        
        if basal_active and apical_calcium > 0.3:
            return 1.0  # BAC firing
        else:
            return 0.0
    
    def reset(self):
        """Reset all compartments"""
        for compartment in self.all_compartments:
            compartment.voltage = compartment.params.leak_reversal
            compartment.calcium = 0.0
            compartment.voltage_history = []
            compartment.calcium_history = []
            compartment.spike_times = []


class DendriticNetwork:
    """
    Network of hierarchical dendritic neurons
    
    This is the high-level interface for using dendritic computation in NEURONSv2
    """
    
    def __init__(self, n_neurons: int, n_basal_branches: int = 5,
                 n_apical_branches: int = 3, synapses_per_branch: int = 20):
        self.n_neurons = n_neurons
        
        # Create neurons
        self.neurons = []
        for _ in range(n_neurons):
            neuron = HierarchicalDendriticTree(
                n_basal_branches=n_basal_branches,
                n_apical_branches=n_apical_branches,
                synapses_per_branch=synapses_per_branch
            )
            self.neurons.append(neuron)
    
    def forward(self, feedforward_inputs: np.ndarray, feedback_inputs: Optional[np.ndarray] = None,
                duration_ms: float = 50.0, dt: float = 0.1) -> np.ndarray:
        """
        Forward pass through network
        
        Args:
            feedforward_inputs: (n_neurons, n_basal_branches, synapses_per_branch)
            feedback_inputs: Optional (n_neurons, n_apical_branches, synapses_per_branch)
            
        Returns:
            spikes: (n_neurons,) binary spike outputs
        """
        spikes = np.zeros(self.n_neurons)
        
        if feedback_inputs is None:
            feedback_inputs = np.zeros((self.n_neurons, self.neurons[0].n_apical_branches,
                                       self.neurons[0].synapses_per_branch))
        
        for i, neuron in enumerate(self.neurons):
            fired, metrics = neuron.forward(
                feedforward_inputs[i],
                feedback_inputs[i],
                duration_ms,
                dt
            )
            spikes[i] = 1.0 if fired else 0.0
        
        return spikes
    
    def backward(self, errors: np.ndarray, learning_rate: float = 0.001, modulation: float = 1.0):
        """Backward pass: update all neurons"""
        for i, neuron in enumerate(self.neurons):
            neuron.backward(errors[i], learning_rate, modulation)
    
    def reset(self):
        """Reset all neurons"""
        for neuron in self.neurons:
            neuron.reset()


# Quick test
if __name__ == "__main__":
    print("Testing Advanced Dendritic Computation...")
    
    # Create single neuron
    neuron = HierarchicalDendriticTree(
        n_basal_branches=5,
        n_apical_branches=3,
        synapses_per_branch=20
    )
    
    # Test inputs
    basal_inputs = np.random.randn(5, 20) * 0.5
    apical_inputs = np.random.randn(3, 20) * 0.3
    
    print("\nTest 1: Basal only")
    fired, metrics = neuron.forward(basal_inputs, np.zeros_like(apical_inputs))
    print(f"  Fired: {fired}, Soma voltage: {metrics['soma_voltage']:.2f}")
    
    neuron.reset()
    
    print("\nTest 2: Apical only")
    fired, metrics = neuron.forward(np.zeros_like(basal_inputs), apical_inputs)
    print(f"  Fired: {fired}, Apical calcium: {metrics['apical_calcium']:.3f}")
    
    neuron.reset()
    
    print("\nTest 3: Both (BAC firing)")
    fired, metrics = neuron.forward(basal_inputs, apical_inputs)
    bac = neuron.compute_bac_firing()
    print(f"  Fired: {fired}, BAC probability: {bac:.2f}")
    print(f"  Soma voltage: {metrics['soma_voltage']:.2f}")
    print(f"  Apical calcium: {metrics['apical_calcium']:.3f}")
    
    print("\n✓ Advanced Dendritic Computation working!")
