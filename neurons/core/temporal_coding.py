"""
Temporal Spike Coding Systems
Implements phase-of-firing, rank-order, and latency codes

These codes exploit millisecond-precision timing to encode
1000× more information than rate codes.

References:
- O'Keefe & Recce (1993): Phase-of-firing codes in hippocampus
- Thorpe & Gautrais (1998): Rank-order coding
- Gollisch & Meister (2008): Latency codes in retina
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from numba import jit


class PhaseCode:
    """
    Phase-of-Firing Code
    
    Information encoded in WHEN a spike occurs relative to an oscillation phase.
    
    Key insight: If neurons fire at different phases of a theta oscillation (4-8Hz),
    they can encode 360° of information per spike cycle.
    
    Information capacity:
        I = log₂(2π / Δφ) bits per spike
        With 1ms precision and 6Hz theta: I ≈ 8 bits per spike
        Compare to rate code: ~0.01 bits per spike
    
    Mathematical foundation (O'Keefe & Recce, 1993):
        φ(x) = φ₀ + 2π · (x - x_min) / (x_max - x_min)
        
    Where x is the encoded value and φ is the spike phase.
    
    Parameters:
    -----------
    theta_freq : float
        Theta oscillation frequency (Hz), default=6.0
    phase_precision : float
        Phase precision in radians, default=0.1 (corresponds to ~5° or 2.8ms)
    """
    
    def __init__(
        self,
        theta_freq: float = 6.0,
        phase_precision: float = 0.1
    ):
        self.theta_freq = theta_freq
        self.phase_precision = phase_precision
        self.theta_period = 1000.0 / theta_freq  # ms
        
        # Information capacity (bits per spike)
        self.bits_per_spike = np.log2(2 * np.pi / phase_precision)
    
    def encode(
        self,
        values: np.ndarray,
        theta_phase: float,
        value_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        Encode values as spike times relative to theta phase
        
        Parameters:
        -----------
        values : np.ndarray
            Values to encode (each in value_range)
        theta_phase : float
            Current theta phase in radians [0, 2π)
        value_range : Tuple[float, float]
            Min and max of value range
            
        Returns:
        --------
        np.ndarray : Spike times in ms (relative to current time)
        """
        v_min, v_max = value_range
        
        # Normalize values to [0, 1]
        normalized = (values - v_min) / (v_max - v_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Map to phase [0, 2π]
        target_phases = normalized * 2 * np.pi
        
        # Compute phase difference from current theta
        phase_diff = (target_phases - theta_phase) % (2 * np.pi)
        
        # Convert phase to time
        # Time within current theta cycle
        spike_times = (phase_diff / (2 * np.pi)) * self.theta_period
        
        return spike_times
    
    def decode(
        self,
        spike_times: np.ndarray,
        theta_phase: float,
        value_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        Decode values from spike phases
        
        Parameters:
        -----------
        spike_times : np.ndarray
            Spike times in ms (relative to current time)
        theta_phase : float
            Current theta phase in radians
        value_range : Tuple[float, float]
            Min and max of value range
            
        Returns:
        --------
        np.ndarray : Decoded values
        """
        v_min, v_max = value_range
        
        # Convert times to phases
        phases = (spike_times / self.theta_period) * 2 * np.pi
        phases = (phases + theta_phase) % (2 * np.pi)
        
        # Map phases to values
        normalized = phases / (2 * np.pi)
        values = v_min + normalized * (v_max - v_min)
        
        return values
    
    def decode_circular_mean(
        self,
        spike_phases: np.ndarray,
        value_range: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Decode using circular mean (for multiple spikes encoding same value)
        
        More robust to noise than simple mean.
        """
        # Circular mean
        mean_phase = np.angle(np.mean(np.exp(1j * spike_phases)))
        mean_phase = (mean_phase + 2 * np.pi) % (2 * np.pi)
        
        # Map to value
        v_min, v_max = value_range
        normalized = mean_phase / (2 * np.pi)
        value = v_min + normalized * (v_max - v_min)
        
        return value


class RankOrderCode:
    """
    Rank-Order Coding
    
    Information encoded in the ORDER in which neurons spike, not their firing rates.
    
    Key insight: If 100 neurons spike in a specific order, there are 100! ≈ 10^157
    possible patterns. This is vastly more than rate codes can achieve.
    
    Mathematical foundation (Thorpe & Gautrais, 1998):
        For n neurons:
            Rate code capacity: ~O(n) patterns
            Rank-order capacity: ~O(n!) patterns
        
        For n=100:
            Rate code: ~10^2 patterns
            Rank-order: ~10^157 patterns
    
    Biological evidence:
        - Visual cortex can discriminate images in <150ms (1-2 spike volleys)
        - First-to-fire neurons carry most information
        - Rank order preserved across cortical layers
    
    Parameters:
    -----------
    temporal_precision : float
        Minimum time between distinguishable spikes (ms), default=1.0
    """
    
    def __init__(self, temporal_precision: float = 1.0):
        self.temporal_precision = temporal_precision
    
    def encode(
        self,
        values: np.ndarray,
        latency_range: Tuple[float, float] = (0.0, 50.0)
    ) -> np.ndarray:
        """
        Encode values as rank-order spike times
        
        Larger values spike first (shorter latency).
        
        Parameters:
        -----------
        values : np.ndarray
            Values to encode (can be any scale)
        latency_range : Tuple[float, float]
            Min and max latency in ms
            
        Returns:
        --------
        np.ndarray : Spike times where earlier = higher value
        """
        # Get ranking (descending order)
        ranks = np.argsort(-values)  # Negative for descending
        
        # Convert ranks to spike times
        n = len(values)
        spike_times = np.zeros(n)
        
        # Assign times based on rank
        latency_min, latency_max = latency_range
        latency_step = (latency_max - latency_min) / n
        
        for i, rank in enumerate(ranks):
            spike_times[rank] = latency_min + i * latency_step
        
        return spike_times
    
    def decode(
        self,
        spike_times: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Decode values from spike order
        
        Parameters:
        -----------
        spike_times : np.ndarray
            Spike times in ms
        normalize : bool
            If True, return normalized values [0, 1]
            
        Returns:
        --------
        np.ndarray : Decoded values (earlier spikes = higher values)
        """
        # Get ranks from spike times (earlier = higher rank)
        ranks = np.argsort(spike_times)
        
        # Convert ranks to values
        n = len(spike_times)
        values = np.zeros(n)
        
        for i, rank in enumerate(ranks):
            values[rank] = n - i  # Higher value for earlier spike
        
        if normalize:
            values = values / n
        
        return values
    
    def compute_kendall_distance(
        self,
        spike_times_1: np.ndarray,
        spike_times_2: np.ndarray
    ) -> float:
        """
        Compute Kendall tau distance between two rank orders
        
        This measures how different two spike patterns are.
        
        Returns:
        --------
        float : Distance in [0, 1] where 0 = identical, 1 = opposite
        """
        n = len(spike_times_1)
        
        # Get ranks
        ranks_1 = np.argsort(spike_times_1)
        ranks_2 = np.argsort(spike_times_2)
        
        # Count discordant pairs
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Check if relative order differs
                order_1 = ranks_1[i] < ranks_1[j]
                order_2 = ranks_2[i] < ranks_2[j]
                if order_1 != order_2:
                    discordant += 1
        
        # Normalize by total pairs
        total_pairs = n * (n - 1) // 2
        distance = discordant / total_pairs if total_pairs > 0 else 0.0
        
        return distance


class LatencyCode:
    """
    Latency Coding
    
    Information encoded in absolute spike timing (latency from stimulus onset).
    
    Key insight: Stronger stimuli evoke faster spikes. The latency of the first
    spike carries more information than subsequent firing rate.
    
    Mathematical foundation (Gollisch & Meister, 2008):
        Latency-intensity relationship:
            t(I) = t₀ + k/I
        
        Where:
            t = spike latency
            I = stimulus intensity
            t₀ = minimum latency
            k = constant
    
    Information capacity:
        With 1ms precision over 50ms window: log₂(50) ≈ 5.6 bits per spike
        With population of 100 neurons: 100 × 5.6 = 560 bits
        Compare to rate code over 50ms: ~100 bits
    
    Biological evidence:
        - Retinal ganglion cells use latency codes
        - Auditory brainstem: <1ms timing precision
        - Fast decision making relies on first spike timing
    
    Parameters:
    -----------
    min_latency : float
        Minimum possible latency (ms), default=2.0
    max_latency : float
        Maximum latency (ms), default=50.0
    """
    
    def __init__(
        self,
        min_latency: float = 2.0,
        max_latency: float = 50.0
    ):
        self.min_latency = min_latency
        self.max_latency = max_latency
        self.latency_range = max_latency - min_latency
        
        # Information capacity
        self.bits_per_spike = np.log2(self.latency_range)
    
    def encode(
        self,
        intensities: np.ndarray,
        nonlinear: bool = True
    ) -> np.ndarray:
        """
        Encode stimulus intensities as spike latencies
        
        Parameters:
        -----------
        intensities : np.ndarray
            Stimulus intensities in [0, 1]
        nonlinear : bool
            If True, use realistic nonlinear relationship t ~ 1/I
            If False, use linear relationship
            
        Returns:
        --------
        np.ndarray : Spike latencies in ms
        """
        # Clip to valid range
        intensities = np.clip(intensities, 1e-6, 1.0)
        
        if nonlinear:
            # Realistic nonlinear relationship: t = t₀ + k/I
            k = self.latency_range
            latencies = self.min_latency + k * (1.0 / intensities - 1.0)
        else:
            # Linear relationship: higher intensity = shorter latency
            latencies = self.max_latency - intensities * self.latency_range
        
        # Clip to valid latency range
        latencies = np.clip(latencies, self.min_latency, self.max_latency)
        
        return latencies
    
    def decode(
        self,
        latencies: np.ndarray,
        nonlinear: bool = True
    ) -> np.ndarray:
        """
        Decode intensities from spike latencies
        
        Parameters:
        -----------
        latencies : np.ndarray
            Spike latencies in ms
        nonlinear : bool
            Must match encoding mode
            
        Returns:
        --------
        np.ndarray : Decoded intensities in [0, 1]
        """
        if nonlinear:
            # Invert: I = k / (t - t₀)
            k = self.latency_range
            intensities = k / (latencies - self.min_latency + k)
        else:
            # Linear inversion
            intensities = (self.max_latency - latencies) / self.latency_range
        
        # Clip to valid range
        intensities = np.clip(intensities, 0.0, 1.0)
        
        return intensities
    
    def compute_first_spike_time(
        self,
        spike_trains: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract first spike time from each neuron
        
        This is the most informative part of the response.
        
        Parameters:
        -----------
        spike_trains : List[np.ndarray]
            List of spike time arrays (one per neuron)
            
        Returns:
        --------
        np.ndarray : First spike time for each neuron (or max_latency if no spike)
        """
        first_spikes = np.full(len(spike_trains), self.max_latency)
        
        for i, spikes in enumerate(spike_trains):
            if len(spikes) > 0:
                first_spikes[i] = np.min(spikes)
        
        return first_spikes


class TemporalPopulationCode:
    """
    Combined temporal coding using all three mechanisms
    
    This achieves maximum information density by exploiting:
    1. Phase: When in theta cycle (coarse timing)
    2. Rank: Order of neuron spikes (relative timing)
    3. Latency: Absolute spike times (precise timing)
    
    Total information capacity:
        Phase: ~8 bits per spike
        Rank: ~log₂(n!) bits for n neurons
        Latency: ~6 bits per spike
        
        For 100 neurons in 50ms window:
            Phase: 800 bits
            Rank: ~158 bits (log₂(100!))
            Latency: 600 bits
            Total: ~1558 bits vs ~100 bits for rate code
            
        Improvement: 15× information density!
    """
    
    def __init__(
        self,
        theta_freq: float = 6.0,
        min_latency: float = 2.0,
        max_latency: float = 50.0
    ):
        self.phase_code = PhaseCode(theta_freq=theta_freq)
        self.rank_code = RankOrderCode()
        self.latency_code = LatencyCode(
            min_latency=min_latency,
            max_latency=max_latency
        )
        
        self.theta_freq = theta_freq
        self.theta_period = 1000.0 / theta_freq
    
    def encode_complete(
        self,
        values: np.ndarray,
        theta_phase: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode values using all three temporal codes
        
        Returns:
        --------
        Tuple of (phase_times, rank_times, latency_times)
        """
        # Phase encoding (within theta cycle)
        phase_times = self.phase_code.encode(values, theta_phase)
        
        # Rank-order encoding
        rank_times = self.rank_code.encode(values)
        
        # Latency encoding
        latency_times = self.latency_code.encode(values)
        
        return phase_times, rank_times, latency_times
    
    def encode_hybrid(
        self,
        values: np.ndarray,
        theta_phase: float = 0.0,
        use_phase: bool = True,
        use_rank: bool = True,
        use_latency: bool = True
    ) -> np.ndarray:
        """
        Encode using selected combination of codes
        
        The spike times combine information from multiple codes.
        """
        spike_times = np.zeros(len(values))
        
        if use_phase:
            phase_times = self.phase_code.encode(values, theta_phase)
            spike_times += phase_times
        
        if use_rank:
            rank_times = self.rank_code.encode(values, latency_range=(0, 20))
            spike_times += rank_times
        
        if use_latency:
            latency_times = self.latency_code.encode(values)
            spike_times += latency_times
        
        # Normalize
        n_codes = sum([use_phase, use_rank, use_latency])
        spike_times /= n_codes
        
        return spike_times
    
    def decode_hybrid(
        self,
        spike_times: np.ndarray,
        theta_phase: float = 0.0,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Decode values from spike times using multiple codes
        
        Parameters:
        -----------
        spike_times : np.ndarray
            Observed spike times
        theta_phase : float
            Current theta phase
        weights : Optional[np.ndarray]
            Weights for combining different codes [w_phase, w_rank, w_latency]
            If None, use equal weights
            
        Returns:
        --------
        np.ndarray : Decoded values
        """
        if weights is None:
            weights = np.array([1.0, 1.0, 1.0])
        weights = weights / np.sum(weights)
        
        # Decode using each code
        phase_values = self.phase_code.decode(spike_times, theta_phase)
        rank_values = self.rank_code.decode(spike_times)
        latency_values = self.latency_code.decode(spike_times)
        
        # Weighted combination
        values = (
            weights[0] * phase_values +
            weights[1] * rank_values +
            weights[2] * latency_values
        )
        
        return values
    
    def information_capacity(self, n_neurons: int) -> Dict[str, float]:
        """
        Compute total information capacity
        
        Returns:
        --------
        Dict with bits per code type and total
        """
        # Phase code
        phase_bits = n_neurons * self.phase_code.bits_per_spike
        
        # Rank-order code (combinatorial)
        rank_bits = np.log2(np.math.factorial(min(n_neurons, 20)))  # Cap at 20 for computation
        if n_neurons > 20:
            # Stirling approximation: log(n!) ≈ n·log(n) - n
            rank_bits = n_neurons * np.log2(n_neurons) - n_neurons * np.log2(np.e)
        
        # Latency code
        latency_bits = n_neurons * self.latency_code.bits_per_spike
        
        return {
            'phase_bits': phase_bits,
            'rank_bits': rank_bits,
            'latency_bits': latency_bits,
            'total_bits': phase_bits + rank_bits + latency_bits,
            'rate_code_bits': n_neurons * 0.01,  # For comparison
            'improvement_factor': (phase_bits + rank_bits + latency_bits) / (n_neurons * 0.01)
        }


@jit(nopython=True)
def fast_phase_encode(
    values: np.ndarray,
    theta_phase: float,
    theta_period: float
) -> np.ndarray:
    """
    JIT-compiled fast phase encoding
    
    For real-time applications requiring maximum speed.
    """
    n = len(values)
    spike_times = np.zeros(n)
    
    for i in range(n):
        # Normalize and map to phase
        v = np.clip(values[i], 0.0, 1.0)
        target_phase = v * 2 * np.pi
        
        # Phase difference
        phase_diff = (target_phase - theta_phase) % (2 * np.pi)
        
        # Convert to time
        spike_times[i] = (phase_diff / (2 * np.pi)) * theta_period
    
    return spike_times


@jit(nopython=True)
def fast_rank_encode(values: np.ndarray, latency_min: float, latency_max: float) -> np.ndarray:
    """
    JIT-compiled fast rank-order encoding
    """
    n = len(values)
    spike_times = np.zeros(n)
    
    # Get ranks (descending)
    ranks = np.argsort(-values)
    
    # Assign times
    latency_step = (latency_max - latency_min) / n
    for i in range(n):
        spike_times[ranks[i]] = latency_min + i * latency_step
    
    return spike_times
