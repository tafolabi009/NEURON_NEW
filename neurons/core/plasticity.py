"""
Synaptic Plasticity Mechanisms
Implements STDP, voltage-dependent plasticity, and synaptic scaling
"""

import numpy as np
from typing import Optional, Tuple
from numba import jit


class TripletSTDP:
    """
    Triplet Spike-Timing-Dependent Plasticity (Pfister & Gerstner, 2006)
    
    Implements frequency-dependent plasticity:
    dw/dt = A₊·x·y + A₋·x·y²
    
    Where x and y are pre- and post-synaptic traces with different time constants.
    This captures both pair-based and triplet interactions for stable learning.
    
    Parameters:
    -----------
    tau_plus : float
        Time constant for pre-synaptic trace (ms), default=16.8
    tau_minus : float
        Time constant for post-synaptic trace (ms), default=33.7
    tau_x : float
        Time constant for triplet pre trace (ms), default=101.0
    tau_y : float
        Time constant for triplet post trace (ms), default=125.0
    A_plus : float
        Potentiation amplitude, default=0.01
    A_minus : float
        Depression amplitude, default=-0.012
    w_min : float
        Minimum synaptic weight, default=0.0
    w_max : float
        Maximum synaptic weight, default=1.0
    """
    
    def __init__(
        self,
        tau_plus: float = 16.8,
        tau_minus: float = 33.7,
        tau_x: float = 101.0,
        tau_y: float = 125.0,
        A_plus: float = 0.01,
        A_minus: float = -0.012,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0
    ):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_min = w_min
        self.w_max = w_max
        self.dt = dt
        
    def initialize_traces(self, n_pre: int, n_post: int) -> Tuple[np.ndarray, ...]:
        """
        Initialize synaptic traces
        
        Parameters:
        -----------
        n_pre : int
            Number of pre-synaptic neurons
        n_post : int
            Number of post-synaptic neurons
            
        Returns:
        --------
        Tuple of trace arrays (x1, x2, y1, y2)
        """
        x1 = np.zeros(n_pre)  # Fast pre trace
        x2 = np.zeros(n_pre)  # Slow pre trace (triplet)
        y1 = np.zeros(n_post)  # Fast post trace
        y2 = np.zeros(n_post)  # Slow post trace (triplet)
        return x1, x2, y1, y2
    
    def update_traces(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """
        Update all synaptic traces
        
        Parameters:
        -----------
        x1, x2 : np.ndarray
            Pre-synaptic traces
        y1, y2 : np.ndarray
            Post-synaptic traces
        pre_spikes : np.ndarray
            Boolean array of pre-synaptic spikes
        post_spikes : np.ndarray
            Boolean array of post-synaptic spikes
            
        Returns:
        --------
        Tuple of updated traces
        """
        # Decay traces
        x1 *= np.exp(-self.dt / self.tau_plus)
        x2 *= np.exp(-self.dt / self.tau_x)
        y1 *= np.exp(-self.dt / self.tau_minus)
        y2 *= np.exp(-self.dt / self.tau_y)
        
        # Increment traces on spikes
        x1[pre_spikes] += 1.0
        x2[pre_spikes] += 1.0
        y1[post_spikes] += 1.0
        y2[post_spikes] += 1.0
        
        return x1, x2, y1, y2
    
    def compute_weight_update(
        self,
        weights: np.ndarray,
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray
    ) -> np.ndarray:
        """
        Compute weight updates based on triplet STDP rule
        
        Parameters:
        -----------
        weights : np.ndarray
            Current synaptic weights (shape: n_pre × n_post)
        x1, x2 : np.ndarray
            Pre-synaptic traces
        y1, y2 : np.ndarray
            Post-synaptic traces
        pre_spikes : np.ndarray
            Boolean array of pre-synaptic spikes
        post_spikes : np.ndarray
            Boolean array of post-synaptic spikes
            
        Returns:
        --------
        np.ndarray : Weight updates (same shape as weights)
        """
        dw = np.zeros_like(weights)
        
        # Potentiation: post spike coincides with elevated pre traces
        if np.any(post_spikes):
            post_idx = np.where(post_spikes)[0]
            for j in post_idx:
                # LTP depends on x1 (fast pre trace)
                dw[:, j] += self.A_plus * x1
        
        # Depression: pre spike coincides with elevated post traces
        if np.any(pre_spikes):
            pre_idx = np.where(pre_spikes)[0]
            for i in pre_idx:
                # LTD depends on y1 (fast post trace) and y2 (slow post trace)
                dw[i, :] += self.A_minus * y1 * (1.0 + y2)
        
        return dw
    
    def apply_update(
        self,
        weights: np.ndarray,
        dw: np.ndarray,
        learning_rate: float = 1.0
    ) -> np.ndarray:
        """
        Apply weight updates with bounds
        
        Parameters:
        -----------
        weights : np.ndarray
            Current synaptic weights
        dw : np.ndarray
            Weight updates
        learning_rate : float
            Learning rate multiplier
            
        Returns:
        --------
        np.ndarray : Updated weights
        """
        weights = weights + learning_rate * dw
        weights = np.clip(weights, self.w_min, self.w_max)
        return weights


class VoltageSTDP:
    """
    Voltage-Dependent STDP (Clopath et al., 2010)
    
    Synaptic changes depend on post-synaptic voltage:
    dw/dt = A_LTP·θ(V - θ₊)·x + A_LTD·θ(θ₋ - V)·x
    
    This provides more stable learning than timing-only STDP and naturally
    integrates with dendritic computation.
    
    Parameters:
    -----------
    theta_plus : float
        Potentiation voltage threshold (mV), default=-45.0
    theta_minus : float
        Depression voltage threshold (mV), default=-70.0
    A_LTP : float
        LTP amplitude, default=0.008
    A_LTD : float
        LTD amplitude, default=0.010
    tau_lowpass : float
        Time constant for voltage low-pass filter (ms), default=10.0
    """
    
    def __init__(
        self,
        theta_plus: float = -45.0,
        theta_minus: float = -70.0,
        A_LTP: float = 0.008,
        A_LTD: float = 0.010,
        tau_lowpass: float = 10.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0
    ):
        self.theta_plus = theta_plus
        self.theta_minus = theta_minus
        self.A_LTP = A_LTP
        self.A_LTD = A_LTD
        self.tau_lowpass = tau_lowpass
        self.w_min = w_min
        self.w_max = w_max
        self.dt = dt
        
    def initialize_traces(self, n_pre: int, n_post: int) -> Tuple[np.ndarray, ...]:
        """Initialize voltage-dependent traces"""
        x_pre = np.zeros(n_pre)  # Pre-synaptic trace
        v_lowpass = np.zeros(n_post)  # Low-pass filtered voltage
        return x_pre, v_lowpass
    
    def update_traces(
        self,
        x_pre: np.ndarray,
        v_lowpass: np.ndarray,
        pre_spikes: np.ndarray,
        v_membrane: np.ndarray,
        tau_pre: float = 20.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update voltage-dependent traces
        
        Parameters:
        -----------
        x_pre : np.ndarray
            Pre-synaptic trace
        v_lowpass : np.ndarray
            Low-pass filtered post-synaptic voltage
        pre_spikes : np.ndarray
            Boolean array of pre-synaptic spikes
        v_membrane : np.ndarray
            Post-synaptic membrane voltages
        tau_pre : float
            Pre-synaptic trace time constant
            
        Returns:
        --------
        Tuple of updated traces
        """
        # Decay pre-synaptic trace
        x_pre *= np.exp(-self.dt / tau_pre)
        x_pre[pre_spikes] += 1.0
        
        # Update low-pass filtered voltage
        alpha = self.dt / self.tau_lowpass
        v_lowpass = (1 - alpha) * v_lowpass + alpha * v_membrane
        
        return x_pre, v_lowpass
    
    def compute_weight_update(
        self,
        weights: np.ndarray,
        x_pre: np.ndarray,
        v_lowpass: np.ndarray,
        v_membrane: np.ndarray,
        pre_spikes: np.ndarray
    ) -> np.ndarray:
        """
        Compute weight updates based on voltage-dependent STDP
        
        Parameters:
        -----------
        weights : np.ndarray
            Current weights (shape: n_pre × n_post)
        x_pre : np.ndarray
            Pre-synaptic traces
        v_lowpass : np.ndarray
            Low-pass filtered post-synaptic voltages
        v_membrane : np.ndarray
            Current post-synaptic voltages
        pre_spikes : np.ndarray
            Boolean array of pre-synaptic spikes
            
        Returns:
        --------
        np.ndarray : Weight updates
        """
        dw = np.zeros_like(weights)
        
        # LTP: when post-synaptic voltage exceeds theta_plus
        ltp_active = v_lowpass > self.theta_plus
        if np.any(ltp_active) and np.any(pre_spikes):
            for i in np.where(pre_spikes)[0]:
                dw[i, ltp_active] += self.A_LTP * x_pre[i]
        
        # LTD: when post-synaptic voltage is near theta_minus
        ltd_active = v_membrane < self.theta_minus
        if np.any(ltd_active) and np.any(pre_spikes):
            for i in np.where(pre_spikes)[0]:
                dw[i, ltd_active] += -self.A_LTD * x_pre[i]
        
        return dw
    
    def apply_update(
        self,
        weights: np.ndarray,
        dw: np.ndarray,
        learning_rate: float = 1.0
    ) -> np.ndarray:
        """Apply weight updates with bounds"""
        weights = weights + learning_rate * dw
        weights = np.clip(weights, self.w_min, self.w_max)
        return weights


class SynapticScaling:
    """
    Homeostatic Synaptic Scaling (Turrigiano, 2008)
    
    Maintains target firing rates through slow weight adjustments:
    τ_scale dw/dt = (r_target - r_actual) · w
    
    This prevents runaway excitation and maintains network stability.
    
    Parameters:
    -----------
    tau_scale : float
        Scaling time constant (s), default=3600.0 (1 hour)
    target_rate : float
        Target firing rate (Hz), default=10.0
    """
    
    def __init__(
        self,
        tau_scale: float = 3600.0,
        target_rate: float = 10.0,
        dt: float = 1.0
    ):
        self.tau_scale = tau_scale * 1000.0  # Convert to ms
        self.target_rate = target_rate
        self.dt = dt
        
    def compute_scaling(
        self,
        weights: np.ndarray,
        firing_rates: np.ndarray
    ) -> np.ndarray:
        """
        Compute synaptic scaling updates
        
        Parameters:
        -----------
        weights : np.ndarray
            Current synaptic weights (shape: n_pre × n_post)
        firing_rates : np.ndarray
            Post-synaptic firing rates (Hz)
            
        Returns:
        --------
        np.ndarray : Weight scaling factors
        """
        # Compute rate error for each post-synaptic neuron
        rate_error = self.target_rate - firing_rates
        
        # Scale weights proportional to rate error
        # Positive error (too low firing) -> increase weights
        # Negative error (too high firing) -> decrease weights
        scaling = 1.0 + (rate_error / self.target_rate) * (self.dt / self.tau_scale)
        
        # Apply scaling to weights (broadcast across pre-synaptic dimension)
        scaled_weights = weights * scaling[np.newaxis, :]
        
        return scaled_weights
    
    def apply_scaling(
        self,
        weights: np.ndarray,
        firing_rates: np.ndarray,
        w_min: float = 0.0,
        w_max: float = 1.0
    ) -> np.ndarray:
        """
        Apply homeostatic scaling to weights
        
        Parameters:
        -----------
        weights : np.ndarray
            Current weights
        firing_rates : np.ndarray
            Post-synaptic firing rates
        w_min, w_max : float
            Weight bounds
            
        Returns:
        --------
        np.ndarray : Scaled weights
        """
        scaled_weights = self.compute_scaling(weights, firing_rates)
        scaled_weights = np.clip(scaled_weights, w_min, w_max)
        return scaled_weights


class CombinedPlasticity:
    """
    Combines triplet STDP, voltage-dependent STDP, and synaptic scaling
    
    This provides the complete plasticity system used in NEURONS.
    """
    
    def __init__(
        self,
        triplet_stdp: Optional[TripletSTDP] = None,
        voltage_stdp: Optional[VoltageSTDP] = None,
        synaptic_scaling: Optional[SynapticScaling] = None,
        use_triplet: bool = True,
        use_voltage: bool = True,
        use_scaling: bool = True
    ):
        self.use_triplet = use_triplet
        self.use_voltage = use_voltage
        self.use_scaling = use_scaling
        
        self.triplet_stdp = triplet_stdp if triplet_stdp else TripletSTDP()
        self.voltage_stdp = voltage_stdp if voltage_stdp else VoltageSTDP()
        self.synaptic_scaling = synaptic_scaling if synaptic_scaling else SynapticScaling()
        
        # Traces
        self.triplet_traces = None
        self.voltage_traces = None
        
    def initialize(self, n_pre: int, n_post: int):
        """Initialize all plasticity traces"""
        if self.use_triplet:
            self.triplet_traces = self.triplet_stdp.initialize_traces(n_pre, n_post)
        if self.use_voltage:
            self.voltage_traces = self.voltage_stdp.initialize_traces(n_pre, n_post)
    
    def update(
        self,
        weights: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        v_membrane: np.ndarray,
        firing_rates: np.ndarray,
        learning_rate: float = 1.0,
        neuromodulation: float = 1.0
    ) -> np.ndarray:
        """
        Apply all plasticity mechanisms
        
        Parameters:
        -----------
        weights : np.ndarray
            Current synaptic weights
        pre_spikes : np.ndarray
            Pre-synaptic spikes
        post_spikes : np.ndarray
            Post-synaptic spikes
        v_membrane : np.ndarray
            Post-synaptic membrane voltages
        firing_rates : np.ndarray
            Post-synaptic firing rates
        learning_rate : float
            Base learning rate
        neuromodulation : float
            Neuromodulatory factor (from dopamine, etc.)
            
        Returns:
        --------
        np.ndarray : Updated weights
        """
        dw_total = np.zeros_like(weights)
        
        # Triplet STDP
        if self.use_triplet and self.triplet_traces is not None:
            x1, x2, y1, y2 = self.triplet_traces
            self.triplet_traces = self.triplet_stdp.update_traces(
                x1, x2, y1, y2, pre_spikes, post_spikes
            )
            dw_triplet = self.triplet_stdp.compute_weight_update(
                weights, *self.triplet_traces, pre_spikes, post_spikes
            )
            dw_total += dw_triplet
        
        # Voltage-dependent STDP
        if self.use_voltage and self.voltage_traces is not None:
            x_pre, v_lowpass = self.voltage_traces
            self.voltage_traces = self.voltage_stdp.update_traces(
                x_pre, v_lowpass, pre_spikes, v_membrane
            )
            dw_voltage = self.voltage_stdp.compute_weight_update(
                weights, *self.voltage_traces, v_membrane, pre_spikes
            )
            dw_total += dw_voltage
        
        # Apply weight updates with neuromodulation
        weights = weights + learning_rate * neuromodulation * dw_total
        weights = np.clip(weights, 0.0, 1.0)
        
        # Synaptic scaling (slower timescale)
        if self.use_scaling:
            weights = self.synaptic_scaling.apply_scaling(weights, firing_rates)
        
        return weights


@jit(nopython=True)
def _fast_stdp_update(
    weights: np.ndarray,
    pre_trace: np.ndarray,
    post_trace: np.ndarray,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    A_plus: float,
    A_minus: float,
    learning_rate: float
) -> np.ndarray:
    """
    Fast STDP update using Numba JIT compilation
    
    This provides optimized performance for large networks.
    """
    n_pre, n_post = weights.shape
    dw = np.zeros_like(weights)
    
    # Process post-synaptic spikes (LTP)
    for j in range(n_post):
        if post_spikes[j]:
            for i in range(n_pre):
                dw[i, j] += A_plus * pre_trace[i] * learning_rate
    
    # Process pre-synaptic spikes (LTD)
    for i in range(n_pre):
        if pre_spikes[i]:
            for j in range(n_post):
                dw[i, j] += A_minus * post_trace[j] * learning_rate
    
    return weights + dw
