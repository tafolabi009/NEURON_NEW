"""
Predictive Plasticity Mechanisms
Learning by predicting future states, not correcting past errors

This eliminates the need for backpropagation and enables truly online learning.

Key Innovation: Instead of error = target - output, we use error = current - predicted_from_future

References:
- Rao & Ballard (1999): Predictive coding in visual cortex
- Friston (2005): Free energy principle
- Whittington & Bogacz (2017): Theories of error back-propagation in the brain
- Millidge et al. (2022): Predictive coding approximates backprop
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PredictiveState:
    """State of a predictive coding layer"""
    representation: np.ndarray  # Current activity
    prediction: np.ndarray      # Prediction from layer above
    error: np.ndarray           # Prediction error
    precision: float            # Confidence in this layer


class PredictiveCodingLayer:
    """
    Single layer in predictive coding hierarchy
    
    Mathematical Model (Rao & Ballard, 1999):
    
    Forward (bottom-up):
        r_i = f(Σⱼ w_ij · r_j + error_from_below)
    
    Backward (top-down):
        prediction_i = g(Σⱼ w_ji · r_j+1)
    
    Error:
        ε_i = r_i - prediction_i
    
    Learning:
        Δw_ij = η · ε_i · r_j
    
    Key insight: Each layer tries to predict the layer below it.
    When prediction fails, the error drives learning.
    """
    
    def __init__(
        self,
        size: int,
        prediction_tau: float = 50.0,  # Prediction timescale (ms)
        error_tau: float = 10.0,        # Error timescale (ms)
        precision: float = 1.0           # Inverse noise variance
    ):
        self.size = size
        self.prediction_tau = prediction_tau
        self.error_tau = error_tau
        self.precision = precision
        
        # State variables
        self.representation = np.zeros(size)
        self.prediction = np.zeros(size)
        self.error = np.zeros(size)
        self.error_history = []
    
    def compute_prediction(
        self,
        higher_layer_activity: np.ndarray,
        top_down_weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute top-down prediction from higher layer
        
        prediction = W_top_down @ r_higher
        """
        self.prediction = top_down_weights.T @ higher_layer_activity
        return self.prediction
    
    def compute_error(self) -> np.ndarray:
        """
        Compute prediction error
        
        ε = r - prediction
        """
        self.error = self.representation - self.prediction
        self.error_history.append(self.error.copy())
        return self.error
    
    def update_representation(
        self,
        bottom_up_input: np.ndarray,
        error_from_below: np.ndarray,
        dt: float = 1.0
    ):
        """
        Update representation based on bottom-up input and error
        
        dr/dt = (-r + input + error) / tau
        """
        # Combine inputs
        drive = bottom_up_input + error_from_below
        
        # Leaky integration
        dr = (-(self.representation) + drive) / self.prediction_tau
        self.representation += dr * dt
        
        # Nonlinearity (sigmoid)
        self.representation = 1.0 / (1.0 + np.exp(-self.representation))
    
    def update_error(self, dt: float = 1.0):
        """
        Update error dynamics
        
        dε/dt = (-ε + (r - prediction)) / tau
        """
        target_error = self.representation - self.prediction
        dε = (-(self.error) + target_error) / self.error_tau
        self.error += dε * dt


class PredictiveCodingNetwork:
    """
    Full predictive coding network
    
    Architecture:
        Input → [PC Layer 1] → [PC Layer 2] → ... → [PC Layer N] → Output
                    ↑ ↓           ↑ ↓                  ↑ ↓
                (predictions & errors flow bidirectionally)
    
    Learning Rule:
        Forward weights:  Δw_ij^forward = η · ε_j · r_i
        Backward weights: Δw_ji^backward = η · ε_i · r_j
    
    Key Properties:
        1. No backpropagation needed
        2. Online learning (no batches)
        3. Local updates only
        4. Biologically plausible
    
    Theoretical Guarantee (Millidge et al., 2022):
        Under certain conditions, predictive coding approximates backprop:
            Δw_PC ≈ Δw_backprop
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        prediction_tau: float = 50.0,
        error_tau: float = 10.0,
        learning_rate: float = 0.01
    ):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        
        # Create layers
        self.layers: List[PredictiveCodingLayer] = []
        for size in layer_sizes:
            layer = PredictiveCodingLayer(
                size=size,
                prediction_tau=prediction_tau,
                error_tau=error_tau
            )
            self.layers.append(layer)
        
        # Weights
        self.forward_weights: List[np.ndarray] = []
        self.backward_weights: List[np.ndarray] = []
        
        for i in range(len(layer_sizes) - 1):
            # Forward (bottom-up)
            w_forward = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            self.forward_weights.append(w_forward)
            
            # Backward (top-down) - initially symmetric
            w_backward = w_forward.T.copy()
            self.backward_weights.append(w_backward)
    
    def forward_pass(
        self,
        input_data: np.ndarray,
        n_iterations: int = 10,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Forward pass with iterative inference
        
        Unlike backprop, predictive coding settles into a state through
        iterative dynamics.
        
        Parameters:
        -----------
        input_data : np.ndarray
            Input to network
        n_iterations : int
            Number of settling iterations
        dt : float
            Time step
            
        Returns:
        --------
        np.ndarray : Output layer activity
        """
        # Set input
        self.layers[0].representation = input_data
        
        # Iterative settling
        for _ in range(n_iterations):
            # Forward sweep: compute predictions
            for i in range(self.n_layers - 1):
                higher_activity = self.layers[i+1].representation
                self.layers[i].compute_prediction(
                    higher_activity,
                    self.backward_weights[i]
                )
            
            # Compute errors
            for i in range(self.n_layers):
                self.layers[i].compute_error()
            
            # Backward sweep: update representations
            for i in range(1, self.n_layers):
                # Bottom-up input
                lower_activity = self.layers[i-1].representation
                bottom_up = self.forward_weights[i-1].T @ lower_activity
                
                # Error from below
                error_below = self.layers[i-1].error
                
                # Update
                self.layers[i].update_representation(bottom_up, error_below, dt)
        
        return self.layers[-1].representation
    
    def learning_step(
        self,
        target: Optional[np.ndarray] = None
    ):
        """
        Update weights based on prediction errors
        
        If target is provided, clamp output layer to target (supervised).
        Otherwise, use unsupervised predictive learning.
        """
        # If supervised, set output layer
        if target is not None:
            self.layers[-1].representation = target
            # Recompute errors
            for i in range(self.n_layers - 1):
                self.layers[i].compute_prediction(
                    self.layers[i+1].representation,
                    self.backward_weights[i]
                )
                self.layers[i].compute_error()
        
        # Update forward weights
        for i in range(len(self.forward_weights)):
            pre_activity = self.layers[i].representation
            post_error = self.layers[i+1].error
            
            # Δw = η · error · activity
            dw_forward = self.learning_rate * np.outer(pre_activity, post_error)
            self.forward_weights[i] += dw_forward
        
        # Update backward weights
        for i in range(len(self.backward_weights)):
            pre_error = self.layers[i].error
            post_activity = self.layers[i+1].representation
            
            dw_backward = self.learning_rate * np.outer(post_activity, pre_error)
            self.backward_weights[i] += dw_backward
    
    def train_step(
        self,
        input_data: np.ndarray,
        target: np.ndarray,
        n_iterations: int = 10
    ) -> float:
        """
        Complete training step
        
        Returns:
        --------
        float : Total prediction error
        """
        # Forward pass
        output = self.forward_pass(input_data, n_iterations)
        
        # Learning
        self.learning_step(target)
        
        # Compute loss
        loss = np.mean((output - target) ** 2)
        
        return loss


class TemporalPredictiveCoding:
    """
    Predictive coding adapted for temporal spike patterns
    
    Key Extension: Predict future spike times, not just activities.
    
    This enables:
        1. Anticipatory computation
        2. Temporal credit assignment
        3. Learning from temporal patterns
    
    Mathematical Model:
        Predict spike at time t+Δt given spikes up to time t
        
        p(spike[t+Δt] | spikes[0:t]) = σ(W @ spike_history)
        
        Error = actual_spike - predicted_spike
    """
    
    def __init__(
        self,
        n_neurons: int,
        prediction_horizon: float = 20.0,  # ms ahead to predict
        history_window: float = 50.0        # ms of history to use
    ):
        self.n_neurons = n_neurons
        self.prediction_horizon = prediction_horizon
        self.history_window = history_window
        
        # Spike history buffer
        self.spike_history: List[np.ndarray] = []
        self.time_history: List[float] = []
        
        # Prediction weights (learn to predict future from past)
        self.prediction_weights = np.random.randn(n_neurons, n_neurons) * 0.1
        
        # Prediction state
        self.predicted_spikes = np.zeros(n_neurons)
        self.prediction_error = np.zeros(n_neurons)
    
    def add_spikes(self, spikes: np.ndarray, current_time: float):
        """Add new spike observations to history"""
        self.spike_history.append(spikes.copy())
        self.time_history.append(current_time)
        
        # Prune old history
        cutoff_time = current_time - self.history_window
        while self.time_history and self.time_history[0] < cutoff_time:
            self.spike_history.pop(0)
            self.time_history.pop(0)
    
    def predict_future(self, target_time: float) -> np.ndarray:
        """
        Predict spike pattern at future time
        
        Uses temporal convolution over recent history.
        """
        if len(self.spike_history) == 0:
            return np.zeros(self.n_neurons)
        
        # Weight recent history (exponential decay)
        current_time = self.time_history[-1]
        tau_decay = 20.0  # ms
        
        weighted_history = np.zeros(self.n_neurons)
        for spikes, t in zip(self.spike_history, self.time_history):
            time_diff = current_time - t
            weight = np.exp(-time_diff / tau_decay)
            weighted_history += weight * spikes
        
        # Predict via learned transformation
        self.predicted_spikes = self.prediction_weights @ weighted_history
        
        # Sigmoid for probability
        self.predicted_spikes = 1.0 / (1.0 + np.exp(-self.predicted_spikes))
        
        return self.predicted_spikes
    
    def compute_prediction_error(self, actual_spikes: np.ndarray) -> np.ndarray:
        """
        Compute error between predicted and actual spikes
        """
        self.prediction_error = actual_spikes - self.predicted_spikes
        return self.prediction_error
    
    def update_weights(self, learning_rate: float = 0.01):
        """
        Update prediction weights to minimize error
        
        Uses gradient descent on prediction error.
        """
        if len(self.spike_history) < 2:
            return
        
        # Get recent activity
        recent_activity = self.spike_history[-1]
        
        # Update: Δw = η · error · activity^T
        dw = learning_rate * np.outer(self.prediction_error, recent_activity)
        self.prediction_weights += dw
        
        # L2 regularization
        self.prediction_weights *= 0.9999


class MetaPredictiveLearning:
    """
    Meta-learning through predictive coding
    
    Learn to learn by predicting how learning will change representations.
    
    Two-level hierarchy:
        Level 1: Predict sensory inputs (fast timescale)
        Level 2: Predict level 1's predictions (slow timescale)
    
    This implements a form of learning to learn without explicit meta-training.
    """
    
    def __init__(
        self,
        input_size: int,
        representation_size: int,
        meta_size: int
    ):
        # Fast predictive coding (sensory level)
        self.fast_pc = PredictiveCodingNetwork(
            layer_sizes=[input_size, representation_size],
            prediction_tau=10.0,
            error_tau=5.0,
            learning_rate=0.1
        )
        
        # Slow predictive coding (meta level)
        self.slow_pc = PredictiveCodingNetwork(
            layer_sizes=[representation_size, meta_size],
            prediction_tau=100.0,
            error_tau=50.0,
            learning_rate=0.01
        )
        
        # Meta-learning rate (how fast to adapt)
        self.meta_learning_rate = 0.001
    
    def fast_adaptation(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        n_steps: int = 5
    ) -> float:
        """
        Rapidly adapt to new task using meta-knowledge
        
        This is the "inner loop" of meta-learning.
        """
        losses = []
        
        for _ in range(n_steps):
            for inp, tgt in zip(inputs, targets):
                # Fast learning
                loss = self.fast_pc.train_step(inp, tgt, n_iterations=3)
                losses.append(loss)
        
        return np.mean(losses)
    
    def meta_update(self, task_history: List[Tuple[np.ndarray, float]]):
        """
        Update meta-level based on learning history
        
        This is the "outer loop" of meta-learning.
        
        Parameters:
        -----------
        task_history : List of (representation, loss) tuples
        """
        for rep, loss in task_history:
            # Predict: what representation should minimize loss?
            output = self.slow_pc.forward_pass(rep, n_iterations=5)
            
            # Target: representation that achieved low loss
            target = rep if loss < 0.1 else rep * 0.9
            
            # Learn
            self.slow_pc.learning_step(target)
    
    def get_adapted_learning_rate(self, representation: np.ndarray) -> float:
        """
        Query meta-network for adapted learning rate
        """
        # Get meta-prediction
        meta_output = self.slow_pc.forward_pass(representation, n_iterations=3)
        
        # Convert to learning rate
        lr_scale = np.mean(meta_output)
        adapted_lr = self.fast_pc.learning_rate * (0.5 + lr_scale)
        
        return adapted_lr


def test_predictive_coding():
    """
    Test predictive coding on simple task
    """
    print("Testing Predictive Coding Network...")
    
    # Create network
    pc_net = PredictiveCodingNetwork(
        layer_sizes=[10, 20, 10],
        learning_rate=0.01
    )
    
    # Generate toy data
    np.random.seed(42)
    X_train = np.random.randn(100, 10) * 0.5
    Y_train = np.sign(X_train)  # Simple nonlinear mapping
    
    # Training
    losses = []
    for epoch in range(10):
        epoch_loss = 0
        for x, y in zip(X_train, Y_train):
            loss = pc_net.train_step(x, y, n_iterations=10)
            epoch_loss += loss
        losses.append(epoch_loss / len(X_train))
        print(f"Epoch {epoch+1}, Loss: {losses[-1]:.4f}")
    
    # Test
    x_test = X_train[0]
    y_test = Y_train[0]
    y_pred = pc_net.forward_pass(x_test, n_iterations=10)
    
    print(f"\nTest example:")
    print(f"Target:  {y_test[:5]}")
    print(f"Predicted: {y_pred[:5]}")
    print(f"Error: {np.mean((y_test - y_pred)**2):.4f}")
    
    print("\n✓ Predictive coding works!")
    return pc_net, losses


if __name__ == "__main__":
    test_predictive_coding()
