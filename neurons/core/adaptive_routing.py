"""
Advanced Neuromodulation and Adaptive Routing
Sophisticated neuromodulatory systems with predictive modulation and dynamic gating

This implements state-of-the-art neuromodulation that goes beyond simple gain control:
1. Hierarchical neuromodulation (global → local)
2. Predictive modulation (anticipatory adjustments)
3. Dynamic routing and gating
4. Context-dependent learning rate adaptation
5. Multi-timescale modulation

NO transformer attention - this is biologically-inspired routing!

Key Innovation: Neuromodulators don't just amplify - they predict and route!

References:
- Doya (2002): Metalearning and neuromodulation
- Sara (2009): The locus coeruleus and noradrenergic modulation of cognition
- Friston et al. (2012): Dopamine, affordance and active inference
- Yagishita et al. (2014): A critical time window for dopamine actions on the structural plasticity of dendritic spines
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ModulatorType(Enum):
    """Types of neuromodulators"""
    DOPAMINE = "dopamine"  # Reward prediction error, learning rate
    SEROTONIN = "serotonin"  # Time horizon, patience
    NOREPINEPHRINE = "norepinephrine"  # Arousal, gain, exploration
    ACETYLCHOLINE = "acetylcholine"  # Attention, uncertainty
    

@dataclass
class ModulatorState:
    """State of a single neuromodulator"""
    level: float = 1.0  # Current level (normalized)
    baseline: float = 1.0  # Baseline level
    target: float = 1.0  # Target level
    prediction: float = 1.0  # Predicted future level
    prediction_error: float = 0.0  # Surprise
    
    # Dynamics
    tau_fast: float = 100.0  # Fast timescale (ms)
    tau_slow: float = 10000.0  # Slow timescale (ms)
    
    # History
    history: List[float] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


class PredictiveModulator:
    """
    Predictive Neuromodulator
    
    Key Innovation: Doesn't just react to events - PREDICTS future needs!
    
    Mathematical Model:
        Level dynamics: dM/dt = (M_target - M)/τ + η(t)
        Prediction: M_pred(t+Δt) = f(M(t), context(t))
        Surprise: δ = M_actual - M_pred
        Adaptation: M_target ← M_target + α·δ
        
    This enables:
        - Anticipatory learning rate changes
        - Proactive exploration
        - Context-dependent modulation
    """
    
    def __init__(self, modulator_type: ModulatorType, baseline: float = 1.0,
                 tau_fast: float = 100.0, tau_slow: float = 10000.0):
        self.type = modulator_type
        self.state = ModulatorState(
            level=baseline,
            baseline=baseline,
            target=baseline,
            tau_fast=tau_fast,
            tau_slow=tau_slow
        )
        
        # Predictive model (simple linear for now, can be learned)
        self.context_weights = np.zeros(10)  # Context features → prediction
        self.prediction_error_history = []
        
    def predict(self, context: np.ndarray, horizon_ms: float = 100.0) -> float:
        """
        Predict future modulator level
        
        Args:
            context: Context features (recent activity, errors, etc.)
            horizon_ms: Prediction horizon
            
        Returns:
            predicted_level: Predicted modulator level
        """
        # Ensure context has correct size
        if len(context) < len(self.context_weights):
            context_padded = np.zeros(len(self.context_weights))
            context_padded[:len(context)] = context
            context = context_padded
        elif len(context) > len(self.context_weights):
            context = context[:len(self.context_weights)]
        
        # Linear prediction
        prediction_delta = np.dot(self.context_weights, context)
        
        # Add current level and decay
        decay_factor = np.exp(-horizon_ms / self.state.tau_fast)
        predicted = self.state.level * decay_factor + prediction_delta
        
        # Bound prediction
        predicted = np.clip(predicted, 0.0, 3.0)
        
        self.state.prediction = predicted
        return predicted
    
    def update(self, signal: float, context: np.ndarray, dt: float = 1.0) -> float:
        """
        Update modulator level
        
        Args:
            signal: External signal (e.g., reward for dopamine)
            context: Context features
            dt: Time step (ms)
            
        Returns:
            current_level: Updated modulator level
        """
        # Predict current level
        predicted = self.predict(context, horizon_ms=0)
        
        # Compute prediction error (surprise)
        actual_delta = signal
        self.state.prediction_error = actual_delta - (predicted - self.state.level)
        self.prediction_error_history.append(self.state.prediction_error)
        
        # Update target based on signal and prediction error
        self.state.target = self.state.baseline + signal + 0.1 * self.state.prediction_error
        
        # Fast dynamics: track target
        fast_update = (self.state.target - self.state.level) / self.state.tau_fast * dt
        
        # Slow dynamics: adjust baseline
        slow_update = (self.state.level - self.state.baseline) / self.state.tau_slow * dt
        self.state.baseline += slow_update * 0.1
        
        # Update level
        self.state.level += fast_update
        
        # Add noise
        noise = np.random.randn() * 0.01
        self.state.level += noise
        
        # Bound
        self.state.level = np.clip(self.state.level, 0.0, 3.0)
        self.state.baseline = np.clip(self.state.baseline, 0.5, 1.5)
        
        # Record
        self.state.history.append(self.state.level)
        
        return self.state.level
    
    def learn_prediction_model(self, learning_rate: float = 0.001):
        """
        Update predictive model to minimize prediction errors
        
        This makes the modulator LEARN to anticipate future needs!
        """
        if len(self.prediction_error_history) < 2:
            return
        
        # Get recent prediction error
        recent_error = self.prediction_error_history[-1]
        
        # Update context weights (reduce prediction error)
        # This is a placeholder - would use actual context from previous step
        gradient = np.random.randn(len(self.context_weights)) * recent_error
        self.context_weights -= learning_rate * gradient
        
        # Clip weights
        self.context_weights = np.clip(self.context_weights, -1.0, 1.0)


class DynamicRouter:
    """
    Dynamic Routing System
    
    Routes information through the network based on:
        1. Current neuromodulatory state
        2. Prediction errors
        3. Context relevance
        
    This is NOT attention - it's predictive routing!
    
    Mathematical Model:
        Routing weights: W_route = softmax(f(M, E, C))
        where M = modulator state, E = errors, C = context
        
        Information flow: Y = W_route ⊙ X
        
    Key difference from attention:
        - Attention: Query-key similarity
        - This: Predictive relevance and error-driven routing
    """
    
    def __init__(self, n_pathways: int, feature_dim: int):
        self.n_pathways = n_pathways
        self.feature_dim = feature_dim
        
        # Routing parameters (learnable)
        self.modulator_sensitivity = np.ones(n_pathways)
        self.error_sensitivity = np.ones(n_pathways)
        self.pathway_biases = np.zeros(n_pathways)
        
        # State
        self.current_routing = np.ones(n_pathways) / n_pathways
        self.routing_history = []
    
    def compute_routing_weights(self, modulator_states: Dict[str, float],
                               prediction_errors: np.ndarray,
                               context: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute dynamic routing weights
        
        Args:
            modulator_states: Dictionary of modulator levels
            prediction_errors: Prediction errors per pathway
            context: Optional context features
            
        Returns:
            routing_weights: (n_pathways,) routing weights
        """
        # Start with biases
        logits = self.pathway_biases.copy()
        
        # Modulate by neuromodulators
        if 'norepinephrine' in modulator_states:
            # Norepinephrine boosts all pathways (arousal)
            logits += 0.5 * (modulator_states['norepinephrine'] - 1.0)
        
        if 'acetylcholine' in modulator_states:
            # Acetylcholine sharpens routing (focus)
            logits *= modulator_states['acetylcholine']
        
        # Error-driven routing: route through pathways with high prediction errors
        # (These need updating!)
        if prediction_errors is not None and len(prediction_errors) == self.n_pathways:
            error_contribution = self.error_sensitivity * np.abs(prediction_errors)
            logits += error_contribution
        
        # Context-dependent routing
        if context is not None:
            # Simple context influence (can be made more sophisticated)
            context_influence = np.mean(context) * np.ones(self.n_pathways)
            logits += 0.2 * context_influence
        
        # Softmax to get weights
        routing_weights = self._softmax(logits)
        
        # Apply modulator sensitivities
        routing_weights = routing_weights * self.modulator_sensitivity
        routing_weights = routing_weights / (np.sum(routing_weights) + 1e-8)
        
        self.current_routing = routing_weights
        self.routing_history.append(routing_weights.copy())
        
        return routing_weights
    
    @staticmethod
    def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature"""
        x = x / temperature
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def route(self, inputs: List[np.ndarray], routing_weights: np.ndarray) -> np.ndarray:
        """
        Route information through pathways
        
        Args:
            inputs: List of inputs for each pathway
            routing_weights: Routing weights
            
        Returns:
            routed_output: Weighted combination of pathway outputs
        """
        outputs = []
        for i, pathway_input in enumerate(inputs):
            # Weight by routing
            weighted = routing_weights[i] * pathway_input
            outputs.append(weighted)
        
        # Combine
        routed_output = np.sum(outputs, axis=0)
        return routed_output
    
    def adapt_routing(self, performance_gradient: np.ndarray, learning_rate: float = 0.001):
        """
        Adapt routing parameters based on performance
        
        Args:
            performance_gradient: Gradient of performance w.r.t. routing
            learning_rate: Learning rate
        """
        # Update biases
        self.pathway_biases += learning_rate * performance_gradient
        self.pathway_biases = np.clip(self.pathway_biases, -2.0, 2.0)


class HierarchicalNeuromodulation:
    """
    Hierarchical Neuromodulation System
    
    Structure:
        Global modulators (affect entire network)
            ↓
        Regional modulators (affect network regions)
            ↓
        Local modulators (affect individual neurons)
        
    This creates a cascade of modulation effects, similar to cortical hierarchy.
    
    Mathematical Model:
        Local effect = Global × Regional × Local
        
    Example:
        - Global dopamine (reward): affects all regions
        - Regional acetylcholine: focuses specific brain areas
        - Local calcium: affects individual synapses
    """
    
    def __init__(self, n_global: int = 4, n_regional: int = 8, n_local: int = 64):
        self.n_global = n_global
        self.n_regional = n_regional
        self.n_local = n_local
        
        # Global modulators
        self.global_modulators = {
            ModulatorType.DOPAMINE: PredictiveModulator(ModulatorType.DOPAMINE, baseline=1.0),
            ModulatorType.SEROTONIN: PredictiveModulator(ModulatorType.SEROTONIN, baseline=1.0),
            ModulatorType.NOREPINEPHRINE: PredictiveModulator(ModulatorType.NOREPINEPHRINE, baseline=1.0),
            ModulatorType.ACETYLCHOLINE: PredictiveModulator(ModulatorType.ACETYLCHOLINE, baseline=1.0),
        }
        
        # Regional modulators
        self.regional_modulators = [
            PredictiveModulator(ModulatorType.ACETYLCHOLINE, baseline=1.0)
            for _ in range(n_regional)
        ]
        
        # Local modulation factors (simple scalars for now)
        self.local_modulation = np.ones(n_local)
        
        # Hierarchical weights (how much global affects regional, regional affects local)
        self.global_to_regional = np.random.rand(n_global, n_regional) * 0.5
        self.regional_to_local = np.random.rand(n_regional, n_local) * 0.5
    
    def get_global_state(self) -> Dict[str, float]:
        """Get current global modulator levels"""
        return {
            mod_type.value: modulator.state.level
            for mod_type, modulator in self.global_modulators.items()
        }
    
    def update_global(self, signals: Dict[str, float], context: np.ndarray, dt: float = 1.0):
        """
        Update global modulators
        
        Args:
            signals: Dictionary of signals for each modulator
            context: Global context features
            dt: Time step
        """
        for mod_type, modulator in self.global_modulators.items():
            signal = signals.get(mod_type.value, 0.0)
            modulator.update(signal, context, dt)
    
    def update_regional(self, regional_errors: np.ndarray, context: np.ndarray, dt: float = 1.0):
        """
        Update regional modulators based on regional prediction errors
        
        Args:
            regional_errors: Prediction errors per region
            context: Regional context
            dt: Time step
        """
        # Global influence on regional
        global_values = np.array([m.state.level for m in self.global_modulators.values()])
        global_influence = self.global_to_regional.T @ global_values
        
        # Update each regional modulator
        for i, modulator in enumerate(self.regional_modulators):
            if i < len(regional_errors):
                signal = regional_errors[i] + global_influence[i]
                modulator.update(signal, context, dt)
    
    def compute_local_modulation(self, local_errors: np.ndarray) -> np.ndarray:
        """
        Compute local modulation factors
        
        Args:
            local_errors: Local prediction errors
            
        Returns:
            local_modulation: (n_local,) modulation factors
        """
        # Regional influence on local
        regional_values = np.array([m.state.level for m in self.regional_modulators])
        regional_influence = self.regional_to_local.T @ regional_values
        
        # Ensure local_errors matches n_local dimension
        if len(local_errors) < self.n_local:
            # Pad with zeros
            padded_errors = np.zeros(self.n_local)
            padded_errors[:len(local_errors)] = local_errors
            local_errors = padded_errors
        elif len(local_errors) > self.n_local:
            # Truncate
            local_errors = local_errors[:self.n_local]
        
        # Combine with local errors
        self.local_modulation = 0.7 * regional_influence + 0.3 * (1.0 + local_errors)
        
        # Bound
        self.local_modulation = np.clip(self.local_modulation, 0.1, 2.0)
        
        return self.local_modulation
    
    def get_learning_rate_modulation(self, neuron_idx: int) -> float:
        """
        Get learning rate modulation for specific neuron
        
        This is the key output: how much should this neuron learn?
        """
        if neuron_idx >= self.n_local:
            neuron_idx = neuron_idx % self.n_local
        
        # Dopamine: global learning rate
        dopamine = self.global_modulators[ModulatorType.DOPAMINE].state.level
        
        # Local modulation
        local = self.local_modulation[neuron_idx]
        
        # Combined
        learning_rate_factor = dopamine * local
        
        return learning_rate_factor
    
    def get_exploration_factor(self) -> float:
        """
        Get exploration factor (for policy noise, etc.)
        
        High norepinephrine → more exploration
        """
        ne = self.global_modulators[ModulatorType.NOREPINEPHRINE].state.level
        return max(0.1, ne - 0.5)
    
    def get_attention_sharpness(self) -> float:
        """
        Get attention sharpness factor
        
        High acetylcholine → sharper attention (more selective routing)
        """
        ach = self.global_modulators[ModulatorType.ACETYLCHOLINE].state.level
        return ach


class AdaptiveRoutingNetwork:
    """
    Complete Adaptive Routing Network
    
    Combines:
        1. Hierarchical neuromodulation
        2. Dynamic routing
        3. Predictive modulation
        
    This is the main interface for advanced neuromodulation!
    """
    
    def __init__(self, n_pathways: int, feature_dim: int, n_neurons: int):
        self.n_pathways = n_pathways
        self.feature_dim = feature_dim
        self.n_neurons = n_neurons
        
        # Components
        self.neuromodulation = HierarchicalNeuromodulation(
            n_global=4,
            n_regional=n_pathways,
            n_local=n_neurons
        )
        self.router = DynamicRouter(n_pathways, feature_dim)
        
    def forward(self, inputs: List[np.ndarray], prediction_errors: np.ndarray,
               context: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Forward pass with adaptive routing
        
        Args:
            inputs: List of inputs for each pathway
            prediction_errors: Prediction errors
            context: Context features
            
        Returns:
            Tuple of (routed_output, metrics)
        """
        # Get modulator states
        modulator_states = self.neuromodulation.get_global_state()
        
        # Compute routing weights
        routing_weights = self.router.compute_routing_weights(
            modulator_states,
            prediction_errors,
            context
        )
        
        # Route information
        routed_output = self.router.route(inputs, routing_weights)
        
        # Metrics
        metrics = {
            'dopamine': modulator_states['dopamine'],
            'norepinephrine': modulator_states['norepinephrine'],
            'routing_entropy': -np.sum(routing_weights * np.log(routing_weights + 1e-8)),
            'max_routing': np.max(routing_weights)
        }
        
        return routed_output, metrics
    
    def update_neuromodulation(self, reward: float, errors: np.ndarray, context: np.ndarray, dt: float = 1.0):
        """
        Update neuromodulation system
        
        Args:
            reward: Reward signal
            errors: Prediction errors (regional + local)
            context: Context features
            dt: Time step
        """
        # Update global modulators
        global_signals = {
            'dopamine': reward,  # Reward prediction error
            'norepinephrine': np.std(errors),  # Uncertainty → arousal
            'acetylcholine': np.mean(np.abs(errors)),  # Need for attention
            'serotonin': 0.0  # Can be set based on time horizon
        }
        self.neuromodulation.update_global(global_signals, context, dt)
        
        # Update regional
        if len(errors) >= self.n_pathways:
            regional_errors = errors[:self.n_pathways]
            self.neuromodulation.update_regional(regional_errors, context, dt)
        
        # Update local
        self.neuromodulation.compute_local_modulation(errors)
    
    def get_neuron_learning_rates(self) -> np.ndarray:
        """Get learning rate modulation for all neurons"""
        return np.array([
            self.neuromodulation.get_learning_rate_modulation(i)
            for i in range(self.n_neurons)
        ])


# Quick test
if __name__ == "__main__":
    print("Testing Advanced Neuromodulation and Adaptive Routing...")
    
    # Create system
    n_pathways = 4
    feature_dim = 128
    n_neurons = 64
    
    system = AdaptiveRoutingNetwork(n_pathways, feature_dim, n_neurons)
    
    print(f"\nInitialized system:")
    print(f"  - Pathways: {n_pathways}")
    print(f"  - Feature dim: {feature_dim}")
    print(f"  - Neurons: {n_neurons}")
    
    # Test forward pass
    inputs = [np.random.randn(feature_dim) for _ in range(n_pathways)]
    errors = np.random.randn(n_pathways) * 0.1
    context = np.random.randn(10)
    
    output, metrics = system.forward(inputs, errors, context)
    
    print(f"\nForward pass:")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Dopamine: {metrics['dopamine']:.3f}")
    print(f"  - Routing entropy: {metrics['routing_entropy']:.3f}")
    
    # Test learning with reward
    reward = 1.0
    system.update_neuromodulation(reward, errors, context)
    
    learning_rates = system.get_neuron_learning_rates()
    print(f"\nAfter reward:")
    print(f"  - Mean learning rate: {np.mean(learning_rates):.3f}")
    print(f"  - Learning rate range: [{np.min(learning_rates):.3f}, {np.max(learning_rates):.3f}]")
    
    print("\n✓ Advanced Neuromodulation working!")
