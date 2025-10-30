"""
Neural Oscillations
Implements theta and gamma oscillations with phase-amplitude coupling
"""

import numpy as np
from typing import Tuple, Optional


class NeuralOscillations:
    """
    Neural Oscillations with Theta-Gamma Coupling
    
    Implements:
    θ(t) = A_θ · sin(2π·f_θ·t)
    γ(t) = A_γ · sin(2π·f_γ·t + φ_couple·θ(t))
    
    This enables:
    - Temporal segmentation of information
    - Attention through gamma synchronization
    - Working memory maintenance in theta cycles
    
    Parameters:
    -----------
    theta_freq : float
        Theta frequency (Hz), default=6.0 (4-8 Hz range)
    gamma_freq : float
        Gamma frequency (Hz), default=60.0 (30-100 Hz range)
    theta_amplitude : float
        Theta amplitude (mV), default=5.0
    gamma_amplitude : float
        Gamma amplitude (mV), default=2.0
    coupling_strength : float
        Phase-amplitude coupling strength, default=0.3
    dt : float
        Time step (ms), default=1.0
    """
    
    def __init__(
        self,
        theta_freq: float = 6.0,
        gamma_freq: float = 60.0,
        theta_amplitude: float = 5.0,
        gamma_amplitude: float = 2.0,
        coupling_strength: float = 0.3,
        dt: float = 1.0
    ):
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        self.theta_amplitude = theta_amplitude
        self.gamma_amplitude = gamma_amplitude
        self.coupling_strength = coupling_strength
        self.dt = dt
        
        self.current_time = 0.0
        self.theta_phase = 0.0
        self.gamma_phase = 0.0
        
        # History for analysis
        self.theta_history = []
        self.gamma_history = []
        
    def reset(self):
        """Reset oscillations to initial state"""
        self.current_time = 0.0
        self.theta_phase = 0.0
        self.gamma_phase = 0.0
        self.theta_history = []
        self.gamma_history = []
    
    def step(self) -> Tuple[float, float]:
        """
        Compute one time step of oscillations
        
        Returns:
        --------
        Tuple[float, float] : (theta_signal, gamma_signal)
        """
        self.current_time += self.dt
        
        # Update theta phase
        self.theta_phase = 2 * np.pi * self.theta_freq * self.current_time / 1000.0
        theta_signal = self.theta_amplitude * np.sin(self.theta_phase)
        
        # Update gamma phase with coupling to theta
        phase_coupling = self.coupling_strength * theta_signal
        self.gamma_phase = 2 * np.pi * self.gamma_freq * self.current_time / 1000.0 + phase_coupling
        gamma_signal = self.gamma_amplitude * np.sin(self.gamma_phase)
        
        # Store history
        self.theta_history.append(theta_signal)
        self.gamma_history.append(gamma_signal)
        
        return theta_signal, gamma_signal
    
    def get_theta_phase_normalized(self) -> float:
        """
        Get normalized theta phase (0-1)
        
        Returns:
        --------
        float : Normalized theta phase
        """
        return (np.sin(self.theta_phase) + 1.0) / 2.0
    
    def get_gamma_phase_normalized(self) -> float:
        """
        Get normalized gamma phase (0-1)
        
        Returns:
        --------
        float : Normalized gamma phase
        """
        return (np.sin(self.gamma_phase) + 1.0) / 2.0
    
    def modulate_input(self, inputs: np.ndarray) -> np.ndarray:
        """
        Modulate inputs by oscillations
        
        Theta modulation: slow temporal gating
        Gamma modulation: fast attention gating
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input signals
            
        Returns:
        --------
        np.ndarray : Modulated inputs
        """
        theta_signal, gamma_signal = self.theta_history[-1], self.gamma_history[-1]
        
        # Theta provides slow gating (0.5-1.5 range)
        theta_gate = 0.5 + 0.5 * (theta_signal / self.theta_amplitude + 1.0)
        
        # Gamma provides fast modulation (0.8-1.2 range)
        gamma_gate = 0.8 + 0.2 * (gamma_signal / self.gamma_amplitude + 1.0)
        
        # Combined modulation
        modulated = inputs * theta_gate * gamma_gate
        return modulated
    
    def get_plasticity_window(self) -> float:
        """
        Get plasticity window based on theta phase
        
        Peak theta phase -> maximum plasticity
        Trough theta phase -> minimum plasticity
        
        Returns:
        --------
        float : Plasticity multiplier (0-2)
        """
        # Maximum plasticity at theta peak
        phase_factor = (np.sin(self.theta_phase) + 1.0) / 2.0
        return 0.5 + 1.5 * phase_factor
    
    def synchronize_gamma(self, attention_signal: np.ndarray) -> np.ndarray:
        """
        Synchronize gamma oscillations based on attention
        
        High attention -> strong gamma synchronization
        
        Parameters:
        -----------
        attention_signal : np.ndarray
            Attention values (0-1)
            
        Returns:
        --------
        np.ndarray : Gamma-modulated attention
        """
        gamma_signal = self.gamma_history[-1] if self.gamma_history else 0.0
        gamma_factor = (gamma_signal / self.gamma_amplitude + 1.0) / 2.0
        
        # Amplify attention during gamma peak
        synchronized = attention_signal * (1.0 + gamma_factor)
        return synchronized
    
    def get_working_memory_capacity(self) -> float:
        """
        Estimate working memory capacity based on theta cycles
        
        Theta cycles segment information into chunks
        Typical capacity: 4-7 items (matches human working memory)
        
        Returns:
        --------
        float : Estimated capacity
        """
        # One theta cycle (125-250ms) can hold one item
        # 4-7 items per second matches 4-8 Hz theta
        cycles_per_second = self.theta_freq
        capacity = np.clip(cycles_per_second, 4, 7)
        return capacity


class OscillationPopulation:
    """
    Population-level oscillations
    
    Different neurons can have different oscillation phases,
    creating traveling waves and phase coding.
    """
    
    def __init__(
        self,
        n_neurons: int,
        theta_freq: float = 6.0,
        gamma_freq: float = 60.0,
        theta_amplitude: float = 5.0,
        gamma_amplitude: float = 2.0,
        coupling_strength: float = 0.3,
        phase_spread: float = 0.0,  # Phase variability across neurons
        dt: float = 1.0
    ):
        self.n_neurons = n_neurons
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        self.theta_amplitude = theta_amplitude
        self.gamma_amplitude = gamma_amplitude
        self.coupling_strength = coupling_strength
        self.dt = dt
        
        # Initialize phase offsets for each neuron
        if phase_spread > 0:
            self.theta_phase_offsets = np.random.uniform(
                -phase_spread, phase_spread, n_neurons
            )
            self.gamma_phase_offsets = np.random.uniform(
                -phase_spread, phase_spread, n_neurons
            )
        else:
            self.theta_phase_offsets = np.zeros(n_neurons)
            self.gamma_phase_offsets = np.zeros(n_neurons)
        
        self.current_time = 0.0
        
    def reset(self):
        """Reset oscillations"""
        self.current_time = 0.0
    
    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one time step for all neurons
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (theta_signals, gamma_signals)
        """
        self.current_time += self.dt
        
        # Compute theta for all neurons
        theta_phases = (2 * np.pi * self.theta_freq * self.current_time / 1000.0 + 
                       self.theta_phase_offsets)
        theta_signals = self.theta_amplitude * np.sin(theta_phases)
        
        # Compute gamma with theta coupling
        phase_coupling = self.coupling_strength * theta_signals
        gamma_phases = (2 * np.pi * self.gamma_freq * self.current_time / 1000.0 + 
                       self.gamma_phase_offsets + phase_coupling)
        gamma_signals = self.gamma_amplitude * np.sin(gamma_phases)
        
        return theta_signals, gamma_signals
    
    def modulate_currents(
        self,
        currents: np.ndarray,
        use_theta: bool = True,
        use_gamma: bool = True
    ) -> np.ndarray:
        """
        Modulate neural currents by oscillations
        
        Parameters:
        -----------
        currents : np.ndarray
            Input currents for each neuron
        use_theta : bool
            Apply theta modulation
        use_gamma : bool
            Apply gamma modulation
            
        Returns:
        --------
        np.ndarray : Modulated currents
        """
        theta_signals, gamma_signals = self.step()
        modulated = currents.copy()
        
        if use_theta:
            theta_gates = 0.5 + 0.5 * (theta_signals / self.theta_amplitude + 1.0)
            modulated *= theta_gates
        
        if use_gamma:
            gamma_gates = 0.8 + 0.2 * (gamma_signals / self.gamma_amplitude + 1.0)
            modulated *= gamma_gates
        
        return modulated
    
    def get_phase_synchrony(self) -> float:
        """
        Calculate phase synchrony across population
        
        Higher synchrony -> more coordinated activity
        
        Returns:
        --------
        float : Synchrony measure (0-1)
        """
        theta_phases = 2 * np.pi * self.theta_freq * self.current_time / 1000.0 + self.theta_phase_offsets
        
        # Compute phase coherence
        complex_phases = np.exp(1j * theta_phases)
        coherence = np.abs(np.mean(complex_phases))
        
        return coherence


class CrossFrequencyCoupling:
    """
    Cross-Frequency Coupling Analysis
    
    Measures phase-amplitude coupling between oscillations
    """
    
    def __init__(self):
        self.theta_history = []
        self.gamma_history = []
        
    def add_samples(self, theta: float, gamma: float):
        """Add oscillation samples"""
        self.theta_history.append(theta)
        self.gamma_history.append(gamma)
    
    def compute_pac(self, window_size: int = 1000) -> float:
        """
        Compute Phase-Amplitude Coupling (PAC)
        
        Measures how gamma amplitude is modulated by theta phase
        
        Parameters:
        -----------
        window_size : int
            Number of samples to analyze
            
        Returns:
        --------
        float : PAC strength (0-1)
        """
        if len(self.theta_history) < window_size:
            return 0.0
        
        # Get recent samples
        theta = np.array(self.theta_history[-window_size:])
        gamma = np.array(self.gamma_history[-window_size:])
        
        # Compute theta phase
        theta_phase = np.angle(theta + 1j * np.imag(np.fft.fft(theta)))
        
        # Compute gamma amplitude
        gamma_amp = np.abs(gamma)
        
        # Bin gamma amplitude by theta phase
        n_bins = 18  # 20-degree bins
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        binned_amp = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (theta_phase >= bins[i]) & (theta_phase < bins[i + 1])
            if np.any(mask):
                binned_amp[i] = np.mean(gamma_amp[mask])
        
        # Compute modulation index (normalized entropy)
        if np.sum(binned_amp) > 0:
            p = binned_amp / np.sum(binned_amp)
            entropy = -np.sum(p * np.log(p + 1e-10))
            max_entropy = np.log(n_bins)
            modulation_index = 1.0 - entropy / max_entropy
        else:
            modulation_index = 0.0
        
        return modulation_index
    
    def reset(self):
        """Reset history"""
        self.theta_history = []
        self.gamma_history = []
