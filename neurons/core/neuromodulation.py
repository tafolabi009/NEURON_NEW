"""
Neuromodulation Systems
Implements dopamine, serotonin, norepinephrine, and acetylcholine
"""

import numpy as np
from typing import Dict, Optional


class Neuromodulator:
    """
    Base class for neuromodulator systems
    
    Parameters:
    -----------
    baseline : float
        Baseline neuromodulator level
    tau : float
        Time constant for decay (ms)
    dt : float
        Time step (ms)
    """
    
    def __init__(self, baseline: float, tau: float, dt: float = 1.0):
        self.baseline = baseline
        self.tau = tau
        self.dt = dt
        self.level = baseline
        self.history = []
        
    def update(self, signal: float = 0.0) -> float:
        """
        Update neuromodulator level
        
        Parameters:
        -----------
        signal : float
            External signal (positive or negative)
            
        Returns:
        --------
        float : Current neuromodulator level
        """
        # Decay toward baseline
        decay = (self.baseline - self.level) * (self.dt / self.tau)
        self.level += decay + signal
        
        # Prevent negative levels
        self.level = max(0.0, self.level)
        
        self.history.append(self.level)
        return self.level
    
    def reset(self):
        """Reset to baseline"""
        self.level = self.baseline
        self.history = []
    
    def get_modulation_factor(self) -> float:
        """
        Get modulation factor (0-2 range, 1 = baseline)
        
        Returns:
        --------
        float : Modulation factor
        """
        return self.level / self.baseline if self.baseline > 0 else 1.0


class DopamineSystem(Neuromodulator):
    """
    Dopamine (DA) - Reward Prediction Error Signals
    
    Implements:
    δ_DA = r(t) - V(s_t)
    DA(t) = DA_baseline + gain · δ_DA
    
    Modulates learning rate: η_eff = η · DA(t)
    
    Parameters:
    -----------
    baseline : float
        Baseline dopamine level (Hz), default=4.0
    gain : float
        Gain for reward prediction error, default=1.0
    tau : float
        Time constant (ms), default=200.0
    """
    
    def __init__(
        self,
        baseline: float = 4.0,
        gain: float = 1.0,
        tau: float = 200.0,
        dt: float = 1.0
    ):
        super().__init__(baseline, tau, dt)
        self.gain = gain
        self.reward_history = []
        self.value_history = []
        
    def compute_rpe(self, reward: float, value_estimate: float) -> float:
        """
        Compute Reward Prediction Error (RPE)
        
        Parameters:
        -----------
        reward : float
            Actual reward received
        value_estimate : float
            Predicted value
            
        Returns:
        --------
        float : RPE (δ)
        """
        rpe = reward - value_estimate
        self.reward_history.append(reward)
        self.value_history.append(value_estimate)
        return rpe
    
    def update_from_rpe(self, reward: float, value_estimate: float) -> float:
        """
        Update dopamine based on RPE
        
        Parameters:
        -----------
        reward : float
            Actual reward
        value_estimate : float
            Predicted value
            
        Returns:
        --------
        float : Updated dopamine level
        """
        rpe = self.compute_rpe(reward, value_estimate)
        signal = self.gain * rpe
        return self.update(signal)
    
    def get_learning_rate_modulation(self) -> float:
        """
        Get learning rate modulation factor
        
        High dopamine -> increased learning
        Low dopamine -> decreased learning
        
        Returns:
        --------
        float : Learning rate multiplier
        """
        return self.get_modulation_factor()


class SerotoninSystem(Neuromodulator):
    """
    Serotonin (5-HT) - Mood and Exploration
    
    - Reduces learning rate under stress
    - Promotes consolidation
    - Affects exploration-exploitation balance
    
    Parameters:
    -----------
    baseline : float
        Baseline serotonin level, default=2.0
    tau : float
        Time constant (ms), default=500.0
    """
    
    def __init__(
        self,
        baseline: float = 2.0,
        tau: float = 500.0,
        dt: float = 1.0
    ):
        super().__init__(baseline, tau, dt)
        self.stress_level = 0.0
        
    def update_from_stress(self, stress: float) -> float:
        """
        Update serotonin based on stress level
        
        Parameters:
        -----------
        stress : float
            Current stress level (0-1)
            
        Returns:
        --------
        float : Updated serotonin level
        """
        self.stress_level = stress
        # High stress decreases serotonin
        signal = -stress * self.baseline
        return self.update(signal)
    
    def get_consolidation_factor(self) -> float:
        """
        Get memory consolidation factor
        
        Higher serotonin -> more consolidation
        
        Returns:
        --------
        float : Consolidation multiplier
        """
        return self.get_modulation_factor()
    
    def get_exploration_factor(self) -> float:
        """
        Get exploration factor
        
        Lower serotonin -> more exploration
        Higher serotonin -> more exploitation
        
        Returns:
        --------
        float : Exploration probability (0-1)
        """
        # Inverse relationship
        return 1.0 / (1.0 + self.get_modulation_factor())


class NorepinephrineSystem(Neuromodulator):
    """
    Norepinephrine (NE) - Arousal and Attention
    
    - Increases plasticity for salient stimuli
    - Enhances signal-to-noise ratio
    - Promotes alertness and focus
    
    Parameters:
    -----------
    baseline : float
        Baseline norepinephrine level, default=3.0
    tau : float
        Time constant (ms), default=300.0
    """
    
    def __init__(
        self,
        baseline: float = 3.0,
        tau: float = 300.0,
        dt: float = 1.0
    ):
        super().__init__(baseline, tau, dt)
        self.arousal_level = 0.5
        
    def update_from_arousal(self, arousal: float) -> float:
        """
        Update norepinephrine based on arousal
        
        Parameters:
        -----------
        arousal : float
            Arousal level (0-1)
            
        Returns:
        --------
        float : Updated NE level
        """
        self.arousal_level = arousal
        signal = arousal * self.baseline
        return self.update(signal)
    
    def get_attention_modulation(self) -> float:
        """
        Get attention modulation factor
        
        High NE -> enhanced learning for salient stimuli
        
        Returns:
        --------
        float : Attention multiplier
        """
        return self.get_modulation_factor()
    
    def apply_attention_gate(self, inputs: np.ndarray, salience: np.ndarray) -> np.ndarray:
        """
        Apply attention gating to inputs
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input signals
        salience : np.ndarray
            Salience map (0-1)
            
        Returns:
        --------
        np.ndarray : Gated inputs
        """
        attention_factor = self.get_attention_modulation()
        # Amplify salient inputs
        gated = inputs * (1.0 + attention_factor * salience)
        return gated


class AcetylcholineSystem(Neuromodulator):
    """
    Acetylcholine (ACh) - Encoding vs Retrieval
    
    - High ACh: encoding mode (new learning)
    - Low ACh: retrieval mode (recall)
    - Switches between hippocampal and cortical learning
    
    Parameters:
    -----------
    baseline : float
        Baseline acetylcholine level, default=5.0
    tau : float
        Time constant (ms), default=400.0
    """
    
    def __init__(
        self,
        baseline: float = 5.0,
        tau: float = 400.0,
        dt: float = 1.0
    ):
        super().__init__(baseline, tau, dt)
        self.mode = "encoding"  # "encoding" or "retrieval"
        
    def set_encoding_mode(self):
        """Switch to encoding mode (high ACh)"""
        self.mode = "encoding"
        self.level = self.baseline * 1.5
        
    def set_retrieval_mode(self):
        """Switch to retrieval mode (low ACh)"""
        self.mode = "retrieval"
        self.level = self.baseline * 0.5
        
    def get_encoding_strength(self) -> float:
        """
        Get encoding strength
        
        High ACh -> strong encoding
        
        Returns:
        --------
        float : Encoding strength (0-2)
        """
        return self.get_modulation_factor()
    
    def get_retrieval_strength(self) -> float:
        """
        Get retrieval strength
        
        Low ACh -> strong retrieval
        
        Returns:
        --------
        float : Retrieval strength (0-2)
        """
        # Inverse relationship
        return 2.0 - self.get_modulation_factor()
    
    def modulate_plasticity(self, plasticity: float) -> float:
        """
        Modulate plasticity based on ACh level
        
        High ACh enhances plasticity for new learning
        
        Parameters:
        -----------
        plasticity : float
            Base plasticity value
            
        Returns:
        --------
        float : Modulated plasticity
        """
        return plasticity * self.get_encoding_strength()


class NeuromodulatorSystem:
    """
    Complete neuromodulator system with all four systems
    
    Coordinates dopamine, serotonin, norepinephrine, and acetylcholine
    to modulate learning and behavior.
    """
    
    def __init__(
        self,
        enable_dopamine: bool = True,
        enable_serotonin: bool = True,
        enable_norepinephrine: bool = True,
        enable_acetylcholine: bool = True,
        **kwargs
    ):
        self.enable_dopamine = enable_dopamine
        self.enable_serotonin = enable_serotonin
        self.enable_norepinephrine = enable_norepinephrine
        self.enable_acetylcholine = enable_acetylcholine
        
        # Initialize systems
        self.dopamine = DopamineSystem(**kwargs.get('dopamine', {})) if enable_dopamine else None
        self.serotonin = SerotoninSystem(**kwargs.get('serotonin', {})) if enable_serotonin else None
        self.norepinephrine = NorepinephrineSystem(**kwargs.get('norepinephrine', {})) if enable_norepinephrine else None
        self.acetylcholine = AcetylcholineSystem(**kwargs.get('acetylcholine', {})) if enable_acetylcholine else None
        
    def reset(self):
        """Reset all neuromodulator systems"""
        if self.dopamine:
            self.dopamine.reset()
        if self.serotonin:
            self.serotonin.reset()
        if self.norepinephrine:
            self.norepinephrine.reset()
        if self.acetylcholine:
            self.acetylcholine.reset()
    
    def update(
        self,
        reward: Optional[float] = None,
        value_estimate: Optional[float] = None,
        stress: Optional[float] = None,
        arousal: Optional[float] = None
    ):
        """
        Update all neuromodulator systems
        
        Parameters:
        -----------
        reward : Optional[float]
            Reward signal for dopamine
        value_estimate : Optional[float]
            Value estimate for dopamine RPE
        stress : Optional[float]
            Stress level for serotonin
        arousal : Optional[float]
            Arousal level for norepinephrine
        """
        if self.dopamine and reward is not None and value_estimate is not None:
            self.dopamine.update_from_rpe(reward, value_estimate)
        
        if self.serotonin and stress is not None:
            self.serotonin.update_from_stress(stress)
        
        if self.norepinephrine and arousal is not None:
            self.norepinephrine.update_from_arousal(arousal)
        
        if self.acetylcholine:
            self.acetylcholine.update()
    
    def get_learning_rate_modulation(self) -> float:
        """
        Get combined learning rate modulation
        
        Returns:
        --------
        float : Combined modulation factor
        """
        modulation = 1.0
        
        if self.dopamine:
            modulation *= self.dopamine.get_learning_rate_modulation()
        
        if self.serotonin:
            # Serotonin has mild effect on learning rate
            modulation *= (0.5 + 0.5 * self.serotonin.get_modulation_factor())
        
        if self.norepinephrine:
            # NE enhances learning for attended stimuli
            modulation *= self.norepinephrine.get_attention_modulation()
        
        if self.acetylcholine:
            # ACh controls encoding vs retrieval
            modulation *= self.acetylcholine.get_encoding_strength()
        
        return modulation
    
    def get_state(self) -> Dict[str, float]:
        """
        Get current state of all neuromodulators
        
        Returns:
        --------
        Dict : Current levels of all neuromodulators
        """
        state = {}
        
        if self.dopamine:
            state['dopamine'] = self.dopamine.level
        if self.serotonin:
            state['serotonin'] = self.serotonin.level
        if self.norepinephrine:
            state['norepinephrine'] = self.norepinephrine.level
        if self.acetylcholine:
            state['acetylcholine'] = self.acetylcholine.level
            state['ach_mode'] = self.acetylcholine.mode
        
        return state
    
    def apply_to_plasticity(self, base_learning_rate: float) -> float:
        """
        Apply neuromodulation to learning rate
        
        Parameters:
        -----------
        base_learning_rate : float
            Base learning rate
            
        Returns:
        --------
        float : Modulated learning rate
        """
        return base_learning_rate * self.get_learning_rate_modulation()
    
    def set_task_context(self, context: str):
        """
        Set task context (encoding, retrieval, exploration, etc.)
        
        Parameters:
        -----------
        context : str
            Task context ("encoding", "retrieval", "exploration", "exploitation")
        """
        if self.acetylcholine:
            if context == "encoding":
                self.acetylcholine.set_encoding_mode()
            elif context == "retrieval":
                self.acetylcholine.set_retrieval_mode()
        
        if self.serotonin and context == "exploration":
            # Low serotonin for exploration
            self.serotonin.level = self.serotonin.baseline * 0.5
        elif self.serotonin and context == "exploitation":
            # High serotonin for exploitation
            self.serotonin.level = self.serotonin.baseline * 1.5
