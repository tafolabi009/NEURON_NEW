"""
Optimization Utilities
SIMD vectorization, INT8 quantization, sparse computation helpers
"""

import numpy as np
from typing import Tuple, Optional
from numba import jit
import time


class EnergyMonitor:
    """
    Monitor energy consumption
    
    Estimates energy based on:
    - Number of active neurons
    - Synaptic operations
    - Time elapsed
    """
    
    def __init__(self, base_power_w: float = 0.1):
        """
        Parameters:
        -----------
        base_power_w : float
            Base power consumption in Watts
        """
        self.base_power_w = base_power_w
        self.start_time: Optional[float] = None
        self.total_operations = 0
        self.active_neuron_count = 0
        
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.total_operations = 0
        self.active_neuron_count = 0
        
    def record_activity(self, n_active_neurons: int, n_synaptic_ops: int):
        """
        Record neural activity
        
        Parameters:
        -----------
        n_active_neurons : int
            Number of active neurons in this step
        n_synaptic_ops : int
            Number of synaptic operations
        """
        self.active_neuron_count += n_active_neurons
        self.total_operations += n_synaptic_ops
        
    def stop(self) -> float:
        """
        Stop monitoring and return energy consumed
        
        Returns:
        --------
        float : Energy in Watt-hours (Wh)
        """
        if self.start_time is None:
            return 0.0
        
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        
        # Energy = Power × Time
        # Power scales with activity
        avg_power = self.base_power_w * (1 + self.total_operations / 1e6)
        energy_wh = avg_power * elapsed_hours
        
        return energy_wh
    
    def get_power(self) -> float:
        """
        Get current power consumption estimate
        
        Returns:
        --------
        float : Power in Watts
        """
        if self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return self.base_power_w
        
        ops_per_second = self.total_operations / elapsed
        power = self.base_power_w * (1 + ops_per_second / 1e6)
        
        return power


class INT8Quantizer:
    """
    INT8 quantization for weights
    
    Reduces memory 4× with <1% accuracy loss
    """
    
    def __init__(self, symmetric: bool = True):
        """
        Parameters:
        -----------
        symmetric : bool
            Use symmetric quantization (-127 to 127)
        """
        self.symmetric = symmetric
        self.scales = {}
        self.zero_points = {}
        
    def quantize(self, weights: np.ndarray, name: str = "weights") -> np.ndarray:
        """
        Quantize weights to INT8
        
        Parameters:
        -----------
        weights : np.ndarray
            Float32 weights
        name : str
            Name for storing scale/zero_point
            
        Returns:
        --------
        np.ndarray : INT8 weights
        """
        w_min, w_max = weights.min(), weights.max()
        
        if self.symmetric:
            # Symmetric: use max absolute value
            w_abs_max = max(abs(w_min), abs(w_max))
            scale = w_abs_max / 127.0
            zero_point = 0
            
            quantized = np.round(weights / scale).astype(np.int8)
        else:
            # Asymmetric: use full range
            scale = (w_max - w_min) / 255.0
            zero_point = -128 - int(w_min / scale)
            
            quantized = np.round(weights / scale + zero_point).astype(np.int8)
        
        self.scales[name] = scale
        self.zero_points[name] = zero_point
        
        return quantized
    
    def dequantize(self, quantized: np.ndarray, name: str = "weights") -> np.ndarray:
        """
        Dequantize INT8 weights back to float32
        
        Parameters:
        -----------
        quantized : np.ndarray
            INT8 weights
        name : str
            Name to retrieve scale/zero_point
            
        Returns:
        --------
        np.ndarray : Float32 weights
        """
        if name not in self.scales:
            raise ValueError(f"No scale found for {name}")
        
        scale = self.scales[name]
        zero_point = self.zero_points[name]
        
        if self.symmetric:
            dequantized = quantized.astype(np.float32) * scale
        else:
            dequantized = (quantized.astype(np.float32) - zero_point) * scale
        
        return dequantized


class SparseMatrix:
    """
    Compressed Sparse Row (CSR) matrix for efficient sparse computation
    
    Only stores non-zero elements
    """
    
    def __init__(self, dense_matrix: Optional[np.ndarray] = None, threshold: float = 1e-6):
        """
        Parameters:
        -----------
        dense_matrix : Optional[np.ndarray]
            Dense matrix to convert
        threshold : float
            Values below threshold are treated as zero
        """
        if dense_matrix is not None:
            self.from_dense(dense_matrix, threshold)
        else:
            self.data = np.array([])
            self.indices = np.array([], dtype=np.int32)
            self.indptr = np.array([0], dtype=np.int32)
            self.shape = (0, 0)
    
    def from_dense(self, dense: np.ndarray, threshold: float = 1e-6):
        """Convert dense matrix to CSR format"""
        self.shape = dense.shape
        
        # Find non-zero elements
        mask = np.abs(dense) > threshold
        
        self.data = dense[mask]
        self.indices = np.where(mask)[1].astype(np.int32)
        
        # Build row pointers
        self.indptr = np.zeros(self.shape[0] + 1, dtype=np.int32)
        for i in range(self.shape[0]):
            self.indptr[i + 1] = self.indptr[i] + np.sum(mask[i])
    
    def to_dense(self) -> np.ndarray:
        """Convert CSR to dense matrix"""
        dense = np.zeros(self.shape, dtype=np.float32)
        
        for i in range(self.shape[0]):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            dense[i, self.indices[start:end]] = self.data[start:end]
        
        return dense
    
    def matvec(self, vec: np.ndarray) -> np.ndarray:
        """
        Sparse matrix-vector multiplication
        
        Parameters:
        -----------
        vec : np.ndarray
            Input vector
            
        Returns:
        --------
        np.ndarray : Result vector
        """
        result = np.zeros(self.shape[0], dtype=np.float32)
        
        for i in range(self.shape[0]):
            start = self.indptr[i]
            end = self.indptr[i + 1]
            result[i] = np.dot(self.data[start:end], vec[self.indices[start:end]])
        
        return result
    
    def get_sparsity(self) -> float:
        """
        Get sparsity (fraction of zero elements)
        
        Returns:
        --------
        float : Sparsity (0-1)
        """
        total_elements = self.shape[0] * self.shape[1]
        if total_elements == 0:
            return 0.0
        return 1.0 - len(self.data) / total_elements


@jit(nopython=True)
def sparse_matvec_jit(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    vec: np.ndarray,
    result: np.ndarray
):
    """
    JIT-compiled sparse matrix-vector multiplication
    
    Much faster than pure Python
    """
    n_rows = len(indptr) - 1
    
    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i + 1]
        
        acc = 0.0
        for j in range(start, end):
            acc += data[j] * vec[indices[j]]
        
        result[i] = acc


@jit(nopython=True)
def apply_sparse_mask(
    weights: np.ndarray,
    sparsity: float
) -> np.ndarray:
    """
    Apply sparsity mask to weights
    
    Keeps only top (1-sparsity)% of weights by magnitude
    
    Parameters:
    -----------
    weights : np.ndarray
        Weight matrix
    sparsity : float
        Target sparsity (0-1)
        
    Returns:
    --------
    np.ndarray : Sparse weights
    """
    if sparsity <= 0:
        return weights.copy()
    
    flat = weights.flatten()
    n_keep = int(len(flat) * (1 - sparsity))
    
    if n_keep == 0:
        return np.zeros_like(weights)
    
    # Get threshold
    abs_flat = np.abs(flat)
    threshold = np.partition(abs_flat, -n_keep)[-n_keep]
    
    # Apply mask
    mask = abs_flat >= threshold
    sparse = flat * mask
    
    return sparse.reshape(weights.shape)


class EventQueue:
    """
    Event-driven processing queue
    
    Only processes spikes when they occur
    """
    
    def __init__(self):
        self.events = []
        self.current_time = 0.0
        
    def add_spike(self, neuron_id: int, time: float):
        """Add spike event"""
        self.events.append((time, neuron_id))
        
    def get_events_at_time(self, time: float, tolerance: float = 0.1) -> list:
        """
        Get all events within tolerance of time
        
        Parameters:
        -----------
        time : float
            Query time
        tolerance : float
            Time tolerance
            
        Returns:
        --------
        list : List of (time, neuron_id) tuples
        """
        events = [(t, nid) for t, nid in self.events 
                 if abs(t - time) <= tolerance]
        return events
    
    def clear_old_events(self, time: float, window: float = 100.0):
        """Remove events older than window"""
        self.events = [(t, nid) for t, nid in self.events 
                      if time - t <= window]
        
    def get_event_count(self) -> int:
        """Get total number of events"""
        return len(self.events)


def compute_sparsity(values: np.ndarray, threshold: float = 1e-6) -> float:
    """
    Compute sparsity of array
    
    Parameters:
    -----------
    values : np.ndarray
        Array to analyze
    threshold : float
        Values below threshold are considered zero
        
    Returns:
    --------
    float : Sparsity (0-1)
    """
    if values.size == 0:
        return 0.0
    
    zeros = np.sum(np.abs(values) <= threshold)
    return zeros / values.size


def prune_weights(
    weights: np.ndarray,
    importance: np.ndarray,
    prune_fraction: float
) -> np.ndarray:
    """
    Prune weights based on importance
    
    Parameters:
    -----------
    weights : np.ndarray
        Weight matrix
    importance : np.ndarray
        Importance scores (same shape as weights)
    prune_fraction : float
        Fraction of weights to prune (0-1)
        
    Returns:
    --------
    np.ndarray : Pruned weights
    """
    if prune_fraction <= 0:
        return weights.copy()
    
    # Find threshold
    flat_importance = importance.flatten()
    n_prune = int(len(flat_importance) * prune_fraction)
    
    if n_prune == 0:
        return weights.copy()
    
    threshold = np.partition(flat_importance, n_prune)[n_prune]
    
    # Apply pruning mask
    mask = importance >= threshold
    pruned = weights * mask
    
    return pruned


@jit(nopython=True)
def fast_relu(x: np.ndarray) -> np.ndarray:
    """Fast ReLU using Numba"""
    return np.maximum(0, x)


@jit(nopython=True)
def fast_sigmoid(x: np.ndarray) -> np.ndarray:
    """Fast sigmoid using Numba"""
    return 1.0 / (1.0 + np.exp(-x))


@jit(nopython=True)
def fast_softmax(x: np.ndarray) -> np.ndarray:
    """Fast softmax using Numba"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
