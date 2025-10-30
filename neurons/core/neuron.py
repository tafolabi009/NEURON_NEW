"""
Leaky Integrate-and-Fire Neuron Models
Implements biologically-realistic neurons with adaptive thresholds
"""

import numpy as np
from typing import Optional, Dict, List
from numba import jit


class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron model
    
    Implements the equation:
    τₘ dV/dt = -(V - V_rest) + R·I_syn + I_ext
    
    Parameters:
    -----------
    tau_m : float
        Membrane time constant (ms), default=20.0
    v_rest : float
        Resting membrane potential (mV), default=-70.0
    v_threshold : float
        Spike threshold (mV), default=-55.0
    v_reset : float
        Reset potential after spike (mV), default=-75.0
    refractory_period : float
        Refractory period (ms), default=2.0
    resistance : float
        Membrane resistance (MΩ), default=10.0
    dt : float
        Time step for integration (ms), default=1.0
    """
    
    def __init__(
        self,
        tau_m: float = 20.0,
        v_rest: float = -70.0,
        v_threshold: float = -55.0,
        v_reset: float = -75.0,
        refractory_period: float = 2.0,
        resistance: float = 10.0,
        dt: float = 1.0
    ):
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.refractory_period = refractory_period
        self.resistance = resistance
        self.dt = dt
        
        # State variables
        self.v_membrane = v_rest
        self.refractory_time = 0.0
        self.spike_times: List[float] = []
        self.current_time = 0.0
        
    def reset(self):
        """Reset neuron to resting state"""
        self.v_membrane = self.v_rest
        self.refractory_time = 0.0
        self.spike_times = []
        self.current_time = 0.0
        
    def step(self, i_synaptic: float, i_external: float = 0.0) -> bool:
        """
        Simulate one time step
        
        Parameters:
        -----------
        i_synaptic : float
            Synaptic current input
        i_external : float
            External current input
            
        Returns:
        --------
        bool : True if neuron spiked, False otherwise
        """
        self.current_time += self.dt
        spiked = False
        
        # Check if in refractory period
        if self.refractory_time > 0:
            self.refractory_time -= self.dt
            self.v_membrane = self.v_reset
            return False
        
        # Compute total current
        i_total = i_synaptic + i_external
        
        # Update membrane potential using Euler integration
        # dV/dt = -(V - V_rest) / tau_m + R * I / tau_m
        dv = (-(self.v_membrane - self.v_rest) + self.resistance * i_total) / self.tau_m
        self.v_membrane += dv * self.dt
        
        # Check for spike
        if self.v_membrane >= self.v_threshold:
            spiked = True
            self.spike_times.append(self.current_time)
            self.v_membrane = self.v_reset
            self.refractory_time = self.refractory_period
            
        return spiked
    
    def get_firing_rate(self, time_window: float = 100.0) -> float:
        """
        Calculate firing rate over recent time window
        
        Parameters:
        -----------
        time_window : float
            Time window in ms
            
        Returns:
        --------
        float : Firing rate in Hz
        """
        recent_spikes = [t for t in self.spike_times 
                        if self.current_time - t <= time_window]
        if time_window == 0:
            return 0.0
        return len(recent_spikes) * 1000.0 / time_window
    
    def get_state(self) -> Dict:
        """Get current neuron state"""
        return {
            'v_membrane': self.v_membrane,
            'refractory_time': self.refractory_time,
            'firing_rate': self.get_firing_rate(),
            'spike_count': len(self.spike_times)
        }


class AdaptiveLIFNeuron(LIFNeuron):
    """
    Adaptive LIF neuron with spike frequency adaptation
    
    Implements adaptive threshold:
    V_threshold(t) = V_threshold_base + β·spike_count(t-100ms)
    
    This models the spike frequency adaptation observed in cortical pyramidal neurons.
    
    Parameters:
    -----------
    adaptation_beta : float
        Adaptation strength, default=0.1
    adaptation_window : float
        Time window for adaptation (ms), default=100.0
    """
    
    def __init__(
        self,
        tau_m: float = 20.0,
        v_rest: float = -70.0,
        v_threshold_base: float = -55.0,
        v_reset: float = -75.0,
        refractory_period: float = 2.0,
        resistance: float = 10.0,
        dt: float = 1.0,
        adaptation_beta: float = 0.1,
        adaptation_window: float = 100.0
    ):
        super().__init__(
            tau_m=tau_m,
            v_rest=v_rest,
            v_threshold=v_threshold_base,
            v_reset=v_reset,
            refractory_period=refractory_period,
            resistance=resistance,
            dt=dt
        )
        self.v_threshold_base = v_threshold_base
        self.adaptation_beta = adaptation_beta
        self.adaptation_window = adaptation_window
        
    def get_adaptive_threshold(self) -> float:
        """
        Calculate adaptive threshold based on recent spike history
        
        Returns:
        --------
        float : Current adaptive threshold
        """
        recent_spikes = [t for t in self.spike_times 
                        if self.current_time - t <= self.adaptation_window]
        spike_count = len(recent_spikes)
        return self.v_threshold_base + self.adaptation_beta * spike_count
    
    def step(self, i_synaptic: float, i_external: float = 0.0) -> bool:
        """
        Simulate one time step with adaptive threshold
        
        Parameters:
        -----------
        i_synaptic : float
            Synaptic current input
        i_external : float
            External current input
            
        Returns:
        --------
        bool : True if neuron spiked, False otherwise
        """
        # Update threshold based on adaptation
        self.v_threshold = self.get_adaptive_threshold()
        
        # Call parent step method
        return super().step(i_synaptic, i_external)


class LIFNeuronPopulation:
    """
    Population of LIF neurons for efficient batch processing
    
    Parameters:
    -----------
    n_neurons : int
        Number of neurons in population
    adaptive : bool
        Use adaptive neurons if True
    """
    
    def __init__(
        self,
        n_neurons: int,
        adaptive: bool = True,
        **neuron_params
    ):
        self.n_neurons = n_neurons
        self.adaptive = adaptive
        
        # Vectorized state variables
        self.v_membrane = np.full(n_neurons, neuron_params.get('v_rest', -70.0))
        self.refractory_time = np.zeros(n_neurons)
        self.spike_counts = np.zeros(n_neurons)
        self.firing_rates = np.zeros(n_neurons)
        
        # Parameters
        self.tau_m = neuron_params.get('tau_m', 20.0)
        self.v_rest = neuron_params.get('v_rest', -70.0)
        self.v_threshold_base = neuron_params.get('v_threshold', -55.0)
        self.v_reset = neuron_params.get('v_reset', -75.0)
        self.refractory_period = neuron_params.get('refractory_period', 2.0)
        self.resistance = neuron_params.get('resistance', 10.0)
        self.dt = neuron_params.get('dt', 1.0)
        
        if adaptive:
            self.adaptation_beta = neuron_params.get('adaptation_beta', 0.1)
            self.adaptation_window = neuron_params.get('adaptation_window', 100.0)
            self.recent_spikes = [[] for _ in range(n_neurons)]
        
        self.current_time = 0.0
        
    def reset(self):
        """Reset all neurons to resting state"""
        self.v_membrane[:] = self.v_rest
        self.refractory_time[:] = 0.0
        self.spike_counts[:] = 0.0
        self.firing_rates[:] = 0.0
        self.current_time = 0.0
        if self.adaptive:
            self.recent_spikes = [[] for _ in range(self.n_neurons)]
    
    def step(self, i_synaptic: np.ndarray, i_external: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate one time step for all neurons
        
        Parameters:
        -----------
        i_synaptic : np.ndarray
            Synaptic current input for each neuron (shape: n_neurons)
        i_external : Optional[np.ndarray]
            External current input for each neuron
            
        Returns:
        --------
        np.ndarray : Boolean array indicating which neurons spiked
        """
        self.current_time += self.dt
        
        if i_external is None:
            i_external = np.zeros(self.n_neurons)
        
        # Update refractory periods
        in_refractory = self.refractory_time > 0
        self.refractory_time[in_refractory] -= self.dt
        self.v_membrane[in_refractory] = self.v_reset
        
        # Compute membrane potential update for non-refractory neurons
        active = ~in_refractory
        i_total = i_synaptic + i_external
        
        dv = (-(self.v_membrane[active] - self.v_rest) + 
              self.resistance * i_total[active]) / self.tau_m
        self.v_membrane[active] += dv * self.dt
        
        # Determine adaptive thresholds
        if self.adaptive:
            v_threshold = self._get_adaptive_thresholds()
        else:
            v_threshold = np.full(self.n_neurons, self.v_threshold_base)
        
        # Check for spikes
        spikes = (self.v_membrane >= v_threshold) & active
        
        # Process spikes
        self.v_membrane[spikes] = self.v_reset
        self.refractory_time[spikes] = self.refractory_period
        self.spike_counts[spikes] += 1
        
        # Update spike history for adaptive neurons
        if self.adaptive:
            spike_indices = np.where(spikes)[0]
            for idx in spike_indices:
                self.recent_spikes[idx].append(self.current_time)
        
        return spikes
    
    def _get_adaptive_thresholds(self) -> np.ndarray:
        """Calculate adaptive thresholds for all neurons"""
        thresholds = np.full(self.n_neurons, self.v_threshold_base)
        
        for i in range(self.n_neurons):
            # Count recent spikes
            self.recent_spikes[i] = [t for t in self.recent_spikes[i]
                                     if self.current_time - t <= self.adaptation_window]
            spike_count = len(self.recent_spikes[i])
            thresholds[i] += self.adaptation_beta * spike_count
        
        return thresholds
    
    def get_firing_rates(self, time_window: float = 100.0) -> np.ndarray:
        """
        Calculate firing rates for all neurons
        
        Parameters:
        -----------
        time_window : float
            Time window in ms
            
        Returns:
        --------
        np.ndarray : Firing rates in Hz
        """
        if not self.adaptive:
            # Approximate from spike counts
            return self.spike_counts * 1000.0 / self.current_time if self.current_time > 0 else np.zeros(self.n_neurons)
        
        rates = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            recent = [t for t in self.recent_spikes[i]
                     if self.current_time - t <= time_window]
            rates[i] = len(recent) * 1000.0 / time_window if time_window > 0 else 0.0
        
        return rates
    
    def get_sparsity(self) -> float:
        """
        Calculate current sparsity (fraction of silent neurons)
        
        Returns:
        --------
        float : Sparsity value (0-1)
        """
        firing_rates = self.get_firing_rates()
        silent = np.sum(firing_rates < 1.0)  # < 1 Hz considered silent
        return silent / self.n_neurons


@jit(nopython=True)
def _vectorized_lif_step(
    v_membrane: np.ndarray,
    refractory_time: np.ndarray,
    i_synaptic: np.ndarray,
    i_external: np.ndarray,
    tau_m: float,
    v_rest: float,
    v_threshold: float,
    v_reset: float,
    refractory_period: float,
    resistance: float,
    dt: float
) -> np.ndarray:
    """
    Optimized vectorized LIF neuron step using Numba JIT compilation
    
    This provides significant speedup for large populations
    """
    n_neurons = len(v_membrane)
    spikes = np.zeros(n_neurons, dtype=np.bool_)
    
    for i in range(n_neurons):
        # Check refractory period
        if refractory_time[i] > 0:
            refractory_time[i] -= dt
            v_membrane[i] = v_reset
            continue
        
        # Update membrane potential
        i_total = i_synaptic[i] + i_external[i]
        dv = (-(v_membrane[i] - v_rest) + resistance * i_total) / tau_m
        v_membrane[i] += dv * dt
        
        # Check for spike
        if v_membrane[i] >= v_threshold:
            spikes[i] = True
            v_membrane[i] = v_reset
            refractory_time[i] = refractory_period
    
    return spikes
