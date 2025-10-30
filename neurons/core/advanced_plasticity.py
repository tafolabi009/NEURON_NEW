"""
Advanced Plasticity Mechanisms
Synaptic tagging, consolidation, and catastrophic forgetting prevention

This implements state-of-the-art continual learning mechanisms:
1. Synaptic tagging and capture (Frey & Morris, 1997)
2. Synaptic consolidation (slow → permanent weights)
3. Elastic Weight Consolidation (EWC) for preventing forgetting
4. Complementary Learning Systems (hippocampal replay)
5. Meta-plasticity (plasticity of plasticity)

Key Innovation: Learn continuously without catastrophic forgetting!

References:
- Frey & Morris (1997): Synaptic tagging and long-term potentiation
- Kirkpatrick et al. (2017): Overcoming catastrophic forgetting (EWC)
- McClelland et al. (1995): Complementary learning systems
- Abraham & Bear (1996): Metaplasticity
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class PlasticityConfig:
    """Configuration for plasticity mechanisms"""
    # Synaptic tagging
    tag_threshold: float = 0.5  # Threshold for setting tag
    tag_decay_tau: float = 3600000.0  # Tag decay time (1 hour in ms)
    capture_window: float = 60000.0  # Capture window (1 minute in ms)
    
    # Consolidation
    consolidation_tau: float = 86400000.0  # Consolidation time (24 hours in ms)
    consolidation_threshold: float = 0.3  # Minimum tag for consolidation
    
    # EWC
    ewc_lambda: float = 0.4  # EWC regularization strength
    fisher_samples: int = 100  # Samples for Fisher computation
    
    # Replay
    replay_buffer_size: int = 10000
    replay_frequency: int = 100  # Replay every N updates
    replay_batch_size: int = 32
    
    # Meta-plasticity
    metaplasticity_enabled: bool = True
    bcm_threshold_tau: float = 10000.0  # BCM threshold adaptation


class SynapticTag:
    """
    Synaptic Tag
    
    Mathematical Model (Frey & Morris, 1997):
        Tag(t) = Tag(t-1)·exp(-dt/τ) + δ(strong_activity)
        
    Tags mark synapses for consolidation:
        - Strong activity sets tag
        - Tag decays over hours
        - Tagged synapses are preferentially consolidated
        
    This explains how brief strong experiences can be remembered long-term!
    """
    
    def __init__(self, shape: Tuple[int, ...], config: PlasticityConfig):
        self.shape = shape
        self.config = config
        
        # Tag values (0 = no tag, 1 = strong tag)
        self.tags = np.zeros(shape)
        
        # Tag eligibility (has this synapse been active recently?)
        self.eligibility = np.zeros(shape)
        
        # Consolidation state (0 = labile, 1 = consolidated)
        self.consolidation = np.zeros(shape)
        
        # History
        self.tag_history = []
    
    def set_tags(self, pre_activity: np.ndarray, post_error: np.ndarray):
        """
        Set synaptic tags based on activity and error
        
        Strong activity + high error → set tag
        
        Args:
            pre_activity: Pre-synaptic activity
            post_error: Post-synaptic error signal
        """
        # Compute tagging signal (outer product for weight matrix)
        if len(self.shape) == 2:
            tagging_signal = np.abs(np.outer(pre_activity, post_error))
        else:
            tagging_signal = np.abs(pre_activity * post_error)
        
        # Set tags where signal exceeds threshold
        new_tags = (tagging_signal > self.config.tag_threshold).astype(float)
        
        # Update tags (keep existing + add new)
        self.tags = np.maximum(self.tags, new_tags)
        
        # Update eligibility
        self.eligibility = np.maximum(self.eligibility, new_tags)
    
    def decay_tags(self, dt: float = 1.0):
        """
        Decay tags over time
        
        Args:
            dt: Time step (ms)
        """
        decay_factor = np.exp(-dt / self.config.tag_decay_tau)
        self.tags *= decay_factor
        self.eligibility *= decay_factor
    
    def consolidate(self, weights: np.ndarray, learning_rate: float = 0.001) -> np.ndarray:
        """
        Consolidate tagged synapses
        
        Tagged synapses gradually become permanent (consolidated)
        
        Args:
            weights: Current synaptic weights
            learning_rate: Consolidation rate
            
        Returns:
            consolidated_weights: Updated consolidated component
        """
        # Only consolidate synapses with strong tags
        consolidation_mask = self.tags > self.config.consolidation_threshold
        
        # Consolidation dynamics
        consolidation_rate = learning_rate / self.config.consolidation_tau
        
        # Move weights toward consolidated state
        delta_consolidation = consolidation_mask * weights * consolidation_rate
        self.consolidation += delta_consolidation
        
        # Record
        self.tag_history.append(np.mean(self.tags))
        
        return self.consolidation
    
    def get_protection_mask(self) -> np.ndarray:
        """
        Get protection mask for consolidated synapses
        
        Returns:
            mask: (shape) protection mask (1 = protected, 0 = plastic)
        """
        # Consolidated synapses are protected from change
        protection = np.clip(self.consolidation, 0.0, 1.0)
        return protection


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC)
    
    Prevents catastrophic forgetting by constraining updates to important weights.
    
    Mathematical Model (Kirkpatrick et al., 2017):
        L_EWC = L_task + (λ/2)·Σᵢ Fᵢ(θᵢ - θ*ᵢ)²
        
    Where:
        F = Fisher information (importance of weight)
        θ* = old weight value
        λ = regularization strength
        
    Key Idea: Important weights (high Fisher) are hard to change!
    """
    
    def __init__(self, shape: Tuple[int, ...], config: PlasticityConfig):
        self.shape = shape
        self.config = config
        
        # Store optimal weights from previous task
        self.optimal_weights = {}  # task_id → weights
        
        # Store Fisher information for previous tasks
        self.fisher_information = {}  # task_id → Fisher matrix
        
        # Current task ID
        self.current_task_id = 0
    
    def compute_fisher(self, weights: np.ndarray, gradients_history: List[np.ndarray]) -> np.ndarray:
        """
        Compute Fisher information matrix
        
        Fisher ≈ E[∇log p(y|x; θ)²] = E[gradient²]
        
        Args:
            weights: Current weights
            gradients_history: List of recent gradients
            
        Returns:
            fisher: Fisher information matrix
        """
        if len(gradients_history) == 0:
            return np.zeros_like(weights)
        
        # Sample gradients
        n_samples = min(self.config.fisher_samples, len(gradients_history))
        sampled_gradients = gradients_history[-n_samples:]
        
        # Compute Fisher as expectation of squared gradients
        fisher = np.mean([g ** 2 for g in sampled_gradients], axis=0)
        
        return fisher
    
    def store_task_parameters(self, task_id: int, weights: np.ndarray,
                             gradients_history: List[np.ndarray]):
        """
        Store parameters and Fisher for completed task
        
        Args:
            task_id: Task identifier
            weights: Final weights for this task
            gradients_history: Gradients collected during training
        """
        self.optimal_weights[task_id] = weights.copy()
        self.fisher_information[task_id] = self.compute_fisher(weights, gradients_history)
        self.current_task_id = task_id + 1
    
    def compute_ewc_loss(self, current_weights: np.ndarray) -> float:
        """
        Compute EWC regularization loss
        
        Args:
            current_weights: Current weight values
            
        Returns:
            ewc_loss: EWC penalty
        """
        ewc_loss = 0.0
        
        for task_id in self.optimal_weights:
            optimal = self.optimal_weights[task_id]
            fisher = self.fisher_information[task_id]
            
            # EWC penalty: (λ/2)·Σ F·(θ - θ*)²
            penalty = 0.5 * self.config.ewc_lambda * np.sum(fisher * (current_weights - optimal) ** 2)
            ewc_loss += penalty
        
        return ewc_loss
    
    def compute_ewc_gradient(self, current_weights: np.ndarray) -> np.ndarray:
        """
        Compute gradient of EWC loss
        
        Args:
            current_weights: Current weights
            
        Returns:
            ewc_gradient: Gradient of EWC penalty
        """
        ewc_gradient = np.zeros_like(current_weights)
        
        for task_id in self.optimal_weights:
            optimal = self.optimal_weights[task_id]
            fisher = self.fisher_information[task_id]
            
            # Gradient: λ·F·(θ - θ*)
            ewc_gradient += self.config.ewc_lambda * fisher * (current_weights - optimal)
        
        return ewc_gradient


class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Stores past experiences for replay during learning.
    
    Key Idea: Replay old experiences to prevent forgetting (hippocampal replay!)
    
    This is how the brain consolidates memories during sleep.
    """
    
    def __init__(self, capacity: int, input_dim: int, output_dim: int):
        self.capacity = capacity
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Circular buffer
        self.inputs = np.zeros((capacity, input_dim))
        self.outputs = np.zeros((capacity, output_dim))
        self.errors = np.zeros((capacity, output_dim))
        self.importance = np.ones(capacity)  # For prioritized replay
        
        self.size = 0
        self.position = 0
    
    def add(self, input_data: np.ndarray, output_data: np.ndarray,
            error: np.ndarray, importance: float = 1.0):
        """
        Add experience to buffer
        
        Args:
            input_data: Input
            output_data: Target output
            error: Prediction error
            importance: Importance weight (higher = more important)
        """
        self.inputs[self.position] = input_data
        self.outputs[self.position] = output_data
        self.errors[self.position] = error
        self.importance[self.position] = importance
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, prioritized: bool = True) -> Dict[str, np.ndarray]:
        """
        Sample batch from buffer
        
        Args:
            batch_size: Size of batch
            prioritized: Use importance weighting
            
        Returns:
            batch: Dictionary with inputs, outputs, errors
        """
        if self.size == 0:
            return None
        
        batch_size = min(batch_size, self.size)
        
        if prioritized:
            # Sample proportional to importance
            probs = self.importance[:self.size]
            probs = probs / np.sum(probs)
            indices = np.random.choice(self.size, size=batch_size, p=probs)
        else:
            # Uniform sampling
            indices = np.random.choice(self.size, size=batch_size)
        
        return {
            'inputs': self.inputs[indices],
            'outputs': self.outputs[indices],
            'errors': self.errors[indices],
            'importance': self.importance[indices]
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics"""
        if self.size == 0:
            return {'size': 0, 'mean_error': 0.0, 'mean_importance': 0.0}
        
        return {
            'size': self.size,
            'mean_error': np.mean(np.abs(self.errors[:self.size])),
            'mean_importance': np.mean(self.importance[:self.size])
        }


class BCMPlasticity:
    """
    BCM (Bienenstock-Cooper-Munro) Metaplasticity
    
    Key Idea: The threshold for plasticity adapts based on average activity!
    
    Mathematical Model (Abraham & Bear, 1996):
        dw/dt = η·x·(y - θ)·y
        dθ/dt = (y² - θ)/τ
        
    Where:
        x = pre-synaptic activity
        y = post-synaptic activity
        θ = modification threshold (adapts!)
        
    Effects:
        - High average activity → high threshold → harder to potentiate
        - Low average activity → low threshold → easier to potentiate
        
    This prevents runaway excitation and stabilizes learning!
    """
    
    def __init__(self, shape: Tuple[int, ...], config: PlasticityConfig):
        self.shape = shape
        self.config = config
        
        # Modification threshold (adapts over time)
        self.theta = np.ones(shape) * 0.5
        
        # Average activity (for threshold adaptation)
        self.avg_activity = np.zeros(shape)
        
    def update_threshold(self, post_activity: np.ndarray, dt: float = 1.0):
        """
        Update BCM threshold based on activity
        
        Args:
            post_activity: Post-synaptic activity
            dt: Time step
        """
        # Running average of squared activity
        activity_squared = post_activity ** 2
        
        # Threshold tracks average activity²
        tau = self.config.bcm_threshold_tau
        alpha = dt / tau
        self.avg_activity = (1 - alpha) * self.avg_activity + alpha * activity_squared
        
        # Update threshold
        self.theta = self.avg_activity
        
        # Clip to reasonable range
        self.theta = np.clip(self.theta, 0.1, 2.0)
    
    def compute_plasticity_factor(self, post_activity: np.ndarray) -> np.ndarray:
        """
        Compute BCM plasticity factor
        
        Args:
            post_activity: Post-synaptic activity
            
        Returns:
            factor: Plasticity factor (y - θ)·y
        """
        # BCM rule: φ(y) = (y - θ)·y
        factor = (post_activity - self.theta) * post_activity
        return factor


class AdvancedPlasticitySystem:
    """
    Complete Advanced Plasticity System
    
    Integrates:
        1. Synaptic tagging and consolidation
        2. EWC for preventing forgetting
        3. Replay for memory consolidation
        4. BCM metaplasticity for stability
        
    This is the complete system for continual learning!
    """
    
    def __init__(self, weight_shape: Tuple[int, ...], config: Optional[PlasticityConfig] = None):
        self.weight_shape = weight_shape
        self.config = config or PlasticityConfig()
        
        # Components
        self.tags = SynapticTag(weight_shape, self.config)
        self.ewc = ElasticWeightConsolidation(weight_shape, self.config)
        self.bcm = BCMPlasticity(weight_shape, self.config)
        
        # Replay buffer
        input_dim = weight_shape[0] if len(weight_shape) > 0 else 1
        output_dim = weight_shape[1] if len(weight_shape) > 1 else 1
        self.replay_buffer = ReplayBuffer(
            self.config.replay_buffer_size,
            input_dim,
            output_dim
        )
        
        # Tracking
        self.update_counter = 0
        self.gradient_history = []
        
        print(f"AdvancedPlasticitySystem initialized:")
        print(f"  - Weight shape: {weight_shape}")
        print(f"  - EWC lambda: {self.config.ewc_lambda}")
        print(f"  - Replay buffer: {self.config.replay_buffer_size}")
        print(f"  - Metaplasticity: {self.config.metaplasticity_enabled}")
    
    def update_weights(self, weights: np.ndarray, pre_activity: np.ndarray,
                      post_activity: np.ndarray, error: np.ndarray,
                      learning_rate: float = 0.001, dt: float = 1.0) -> np.ndarray:
        """
        Update weights with advanced plasticity
        
        Args:
            weights: Current weights
            pre_activity: Pre-synaptic activity
            post_activity: Post-synaptic activity
            error: Prediction error
            learning_rate: Base learning rate
            dt: Time step
            
        Returns:
            updated_weights: New weight values
        """
        self.update_counter += 1
        
        # 1. Compute base gradient
        gradient = np.outer(pre_activity, error)
        self.gradient_history.append(gradient.copy())
        
        # Keep only recent gradients
        if len(self.gradient_history) > self.config.fisher_samples:
            self.gradient_history.pop(0)
        
        # 2. Apply BCM metaplasticity
        if self.config.metaplasticity_enabled:
            bcm_factor = self.bcm.compute_plasticity_factor(post_activity)
            gradient = gradient * (1.0 + 0.5 * bcm_factor)
            self.bcm.update_threshold(post_activity, dt)
        
        # 3. Set synaptic tags
        self.tags.set_tags(pre_activity, error)
        self.tags.decay_tags(dt)
        
        # 4. Get protection from consolidation
        protection = self.tags.get_protection_mask()
        
        # 5. Compute EWC gradient (prevent forgetting)
        ewc_gradient = self.ewc.compute_ewc_gradient(weights)
        
        # 6. Combined update
        # Protected synapses change less
        plasticity_mask = 1.0 - 0.9 * protection
        total_gradient = gradient - ewc_gradient
        weight_update = learning_rate * plasticity_mask * total_gradient * dt
        
        # Update weights
        updated_weights = weights + weight_update
        
        # 7. Consolidate tagged synapses
        consolidated = self.tags.consolidate(updated_weights, learning_rate)
        
        # 8. Add experience to replay buffer
        if pre_activity.shape == (self.replay_buffer.input_dim,):
            self.replay_buffer.add(
                pre_activity,
                post_activity,
                error,
                importance=np.mean(np.abs(error))
            )
        
        return updated_weights
    
    def replay_update(self, weights: np.ndarray, network_forward_fn,
                     learning_rate: float = 0.001) -> Optional[np.ndarray]:
        """
        Perform replay-based update
        
        Args:
            weights: Current weights
            network_forward_fn: Function to compute network output
            learning_rate: Learning rate
            
        Returns:
            weight_update: Weight update from replay (or None if no replay)
        """
        # Only replay periodically
        if self.update_counter % self.config.replay_frequency != 0:
            return None
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.config.replay_batch_size)
        
        if batch is None:
            return None
        
        # Compute updates from replayed experiences
        total_update = np.zeros_like(weights)
        
        for i in range(len(batch['inputs'])):
            input_data = batch['inputs'][i]
            target_error = batch['errors'][i]
            importance = batch['importance'][i]
            
            # Compute gradient
            gradient = np.outer(input_data, target_error)
            
            # Weight by importance
            total_update += importance * gradient
        
        # Average
        total_update /= len(batch['inputs'])
        
        # Apply
        weight_update = learning_rate * 0.1 * total_update  # Smaller learning rate for replay
        
        return weight_update
    
    def finish_task(self, task_id: int, weights: np.ndarray):
        """
        Mark task as complete and store parameters
        
        Args:
            task_id: Task identifier
            weights: Final weights
        """
        self.ewc.store_task_parameters(task_id, weights, self.gradient_history)
        print(f"Task {task_id} completed. Parameters stored for EWC.")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plasticity statistics"""
        stats = {
            'mean_tag': np.mean(self.tags.tags),
            'mean_consolidation': np.mean(self.tags.consolidation),
            'mean_bcm_threshold': np.mean(self.bcm.theta),
            'n_tasks_stored': len(self.ewc.optimal_weights),
            'replay_buffer': self.replay_buffer.get_statistics(),
            'updates': self.update_counter
        }
        return stats


# Quick test
if __name__ == "__main__":
    print("Testing Advanced Plasticity System...")
    
    # Create system
    weight_shape = (100, 50)
    system = AdvancedPlasticitySystem(weight_shape)
    
    # Initial weights
    weights = np.random.randn(*weight_shape) * 0.1
    
    print("\nSimulating learning...")
    
    # Simulate Task 1
    for step in range(500):
        pre = np.random.randn(100)
        post = np.random.randn(50)
        error = np.random.randn(50) * 0.1
        
        weights = system.update_weights(weights, pre, post, error, learning_rate=0.01)
        
        # Replay
        replay_update = system.replay_update(weights, lambda x: x, learning_rate=0.01)
        if replay_update is not None:
            weights += replay_update
    
    system.finish_task(0, weights)
    
    stats = system.get_statistics()
    print(f"\nAfter Task 1:")
    print(f"  - Mean tag: {stats['mean_tag']:.4f}")
    print(f"  - Mean consolidation: {stats['mean_consolidation']:.4f}")
    print(f"  - BCM threshold: {stats['mean_bcm_threshold']:.4f}")
    print(f"  - Replay buffer size: {stats['replay_buffer']['size']}")
    
    # Simulate Task 2
    print("\nLearning Task 2 (with EWC to prevent forgetting Task 1)...")
    for step in range(300):
        pre = np.random.randn(100)
        post = np.random.randn(50)
        error = np.random.randn(50) * 0.1
        
        weights = system.update_weights(weights, pre, post, error, learning_rate=0.01)
    
    system.finish_task(1, weights)
    
    stats = system.get_statistics()
    print(f"\nAfter Task 2:")
    print(f"  - Tasks stored: {stats['n_tasks_stored']}")
    print(f"  - Total updates: {stats['updates']}")
    
    print("\n✓ Advanced Plasticity System working!")
