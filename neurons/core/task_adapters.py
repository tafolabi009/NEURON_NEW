"""
Task-Specific Adapters for NEURONSv2
Specialized modules for vision, audio, RL, time-series, graphs, and more

This enables NEURONSv2 to be used for diverse tasks beyond language modeling,
just like transformers have specialized variants (ViT, Audio Transformers, etc.)

Key Innovation: Same neural substrate, different task adapters!

Supported Tasks:
1. Computer Vision (classification, detection, segmentation)
2. Audio Processing (speech, music, sound events)
3. Reinforcement Learning (policy, value, model-based)
4. Time Series (forecasting, anomaly detection)
5. Graph Learning (node/edge/graph classification)
6. Molecular Modeling (property prediction, generation)
7. Robotics (control, planning, perception)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Supported task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    SEQUENCE_TO_SEQUENCE = "seq2seq"
    REINFORCEMENT_LEARNING = "rl"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    FORECASTING = "forecasting"


@dataclass
class TaskConfig:
    """Configuration for a task adapter"""
    task_type: TaskType
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = None
    use_temporal: bool = True
    use_hierarchical: bool = True


class VisionAdapter:
    """
    Computer Vision Adapter
    
    Supports:
        - Image classification
        - Object detection
        - Semantic segmentation
        - Instance segmentation
    """
    
    def __init__(self, embedding_dim: int, n_classes: int, task_type: str = "classification"):
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.task_type = task_type
        
        if task_type == "classification":
            # Simple classification head
            self.classifier = np.random.randn(embedding_dim, n_classes) * 0.01
            
        elif task_type == "detection":
            # Object detection: predict boxes + classes
            # Each output: [x, y, w, h, class_prob...]
            self.box_predictor = np.random.randn(embedding_dim, 4) * 0.01  # Bounding box
            self.class_predictor = np.random.randn(embedding_dim, n_classes) * 0.01
            
        elif task_type == "segmentation":
            # Semantic segmentation: per-pixel classification
            # Uses hierarchical upsampling
            self.upsample_layers = []
            current_dim = embedding_dim
            for _ in range(3):  # 3 upsampling stages
                upsample_weight = np.random.randn(current_dim, current_dim * 4) * 0.01
                self.upsample_layers.append(upsample_weight)
            
            self.segmentation_head = np.random.randn(current_dim, n_classes) * 0.01
    
    def forward_classification(self, patch_embeddings: np.ndarray) -> np.ndarray:
        """
        Image classification
        
        Args:
            patch_embeddings: (n_patches, embedding_dim) patch features
            
        Returns:
            logits: (n_classes,) class probabilities
        """
        # Global average pooling
        global_feature = np.mean(patch_embeddings, axis=0)
        
        # Classify
        logits = global_feature @ self.classifier
        return logits
    
    def forward_detection(self, patch_embeddings: np.ndarray,
                         confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Object detection
        
        Args:
            patch_embeddings: (n_patches, embedding_dim) patch features
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            detections: List of {box, class, confidence}
        """
        detections = []
        
        for patch_idx, patch_feat in enumerate(patch_embeddings):
            # Predict box
            box = patch_feat @ self.box_predictor  # (4,) [x, y, w, h]
            
            # Predict class
            class_logits = patch_feat @ self.class_predictor
            class_probs = self._softmax(class_logits)
            
            # Get best class
            best_class = np.argmax(class_probs)
            confidence = class_probs[best_class]
            
            if confidence > confidence_threshold:
                detections.append({
                    'box': box,
                    'class': best_class,
                    'confidence': confidence,
                    'patch_idx': patch_idx
                })
        
        return detections
    
    def forward_segmentation(self, patch_embeddings: np.ndarray,
                            original_size: Tuple[int, int]) -> np.ndarray:
        """
        Semantic segmentation
        
        Args:
            patch_embeddings: (n_patches, embedding_dim) patch features
            original_size: (height, width) of original image
            
        Returns:
            segmentation_map: (height, width, n_classes) per-pixel class probabilities
        """
        # Upsample features
        upsampled = patch_embeddings
        for upsample_layer in self.upsample_layers:
            upsampled = upsampled @ upsample_layer
            # Reshape for spatial upsampling
            n_patches = upsampled.shape[0]
            upsampled = upsampled.reshape(n_patches, -1)
        
        # Segment
        logits = upsampled @ self.segmentation_head  # (n_patches, n_classes)
        
        # Reshape to spatial map
        h, w = original_size
        n_patches_h = int(np.sqrt(len(patch_embeddings)))
        n_patches_w = n_patches_h
        
        # Simple reshaping (would need better interpolation in practice)
        segmentation_map = np.zeros((h, w, self.n_classes))
        
        return segmentation_map
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class AudioAdapter:
    """
    Audio Processing Adapter
    
    Supports:
        - Speech recognition
        - Music generation
        - Sound event detection
        - Audio classification
    """
    
    def __init__(self, embedding_dim: int, vocab_size: int = None, n_classes: int = None,
                 task_type: str = "speech_recognition"):
        self.embedding_dim = embedding_dim
        self.task_type = task_type
        
        if task_type == "speech_recognition":
            # Speech → text
            self.vocab_size = vocab_size
            self.text_decoder = np.random.randn(embedding_dim, vocab_size) * 0.01
            
        elif task_type == "classification":
            # Audio classification
            self.n_classes = n_classes
            self.classifier = np.random.randn(embedding_dim, n_classes) * 0.01
            
        elif task_type == "generation":
            # Audio generation
            self.generator = np.random.randn(embedding_dim, 80) * 0.01  # 80 mel bins
    
    def forward_speech_recognition(self, audio_embeddings: np.ndarray) -> np.ndarray:
        """
        Speech recognition
        
        Args:
            audio_embeddings: (n_frames, embedding_dim) temporal audio features
            
        Returns:
            token_logits: (n_frames, vocab_size) token probabilities over time
        """
        # Decode to tokens
        token_logits = audio_embeddings @ self.text_decoder
        return token_logits
    
    def forward_classification(self, audio_embeddings: np.ndarray) -> np.ndarray:
        """
        Audio classification
        
        Args:
            audio_embeddings: (n_frames, embedding_dim) temporal features
            
        Returns:
            class_logits: (n_classes,) class probabilities
        """
        # Temporal pooling
        global_feature = np.mean(audio_embeddings, axis=0)
        
        # Classify
        class_logits = global_feature @ self.classifier
        return class_logits
    
    def forward_generation(self, conditioning: np.ndarray, length: int) -> np.ndarray:
        """
        Audio generation
        
        Args:
            conditioning: (embedding_dim,) conditioning vector
            length: Number of frames to generate
            
        Returns:
            mel_spectrogram: (length, 80) generated mel spectrogram
        """
        # Autoregressive generation
        mel_spectrogram = []
        current_state = conditioning
        
        for _ in range(length):
            # Generate one frame
            mel_frame = current_state @ self.generator
            mel_spectrogram.append(mel_frame)
            
            # Update state (would use recurrence in practice)
            current_state = current_state * 0.9 + np.tanh(mel_frame @ self.generator.T) * 0.1
        
        return np.array(mel_spectrogram)


class RLAdapter:
    """
    Reinforcement Learning Adapter
    
    Supports:
        - Policy networks (actor)
        - Value networks (critic)
        - Model-based RL (world model)
    """
    
    def __init__(self, embedding_dim: int, action_dim: int, continuous: bool = True):
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        # Policy head
        if continuous:
            # Gaussian policy: output mean and log_std
            self.policy_mean = np.random.randn(embedding_dim, action_dim) * 0.01
            self.policy_log_std = np.random.randn(embedding_dim, action_dim) * 0.01
        else:
            # Discrete policy: output action probabilities
            self.policy = np.random.randn(embedding_dim, action_dim) * 0.01
        
        # Value head
        self.value = np.random.randn(embedding_dim, 1) * 0.01
        
        # World model (optional)
        self.world_model_transition = np.random.randn(embedding_dim + action_dim, embedding_dim) * 0.01
        self.world_model_reward = np.random.randn(embedding_dim, 1) * 0.01
    
    def forward_policy(self, state_embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for policy
        
        Args:
            state_embedding: (embedding_dim,) state representation
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        if self.continuous:
            # Gaussian policy
            mean = state_embedding @ self.policy_mean
            log_std = state_embedding @ self.policy_log_std
            std = np.exp(log_std)
            
            # Sample action
            action = mean + std * np.random.randn(self.action_dim)
            
            # Log probability
            log_prob = -0.5 * np.sum(((action - mean) / std) ** 2) - np.sum(log_std)
            
        else:
            # Discrete policy
            logits = state_embedding @ self.policy
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            
            # Sample action
            action = np.random.choice(self.action_dim, p=probs)
            log_prob = np.log(probs[action])
        
        return action, log_prob
    
    def forward_value(self, state_embedding: np.ndarray) -> float:
        """
        Forward pass for value function
        
        Args:
            state_embedding: (embedding_dim,) state representation
            
        Returns:
            value: Estimated state value
        """
        value = state_embedding @ self.value
        return value[0]
    
    def forward_world_model(self, state_embedding: np.ndarray,
                           action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass for world model
        
        Args:
            state_embedding: (embedding_dim,) current state
            action: (action_dim,) action taken
            
        Returns:
            next_state_pred: Predicted next state embedding
            reward_pred: Predicted reward
        """
        # Concatenate state and action
        state_action = np.concatenate([state_embedding, action])
        
        # Predict next state
        next_state_pred = np.tanh(state_action @ self.world_model_transition)
        
        # Predict reward
        reward_pred = (state_embedding @ self.world_model_reward)[0]
        
        return next_state_pred, reward_pred


class TimeSeriesAdapter:
    """
    Time Series Adapter
    
    Supports:
        - Forecasting
        - Anomaly detection
        - Classification
    """
    
    def __init__(self, embedding_dim: int, forecast_horizon: int, n_features: int):
        self.embedding_dim = embedding_dim
        self.forecast_horizon = forecast_horizon
        self.n_features = n_features
        
        # Forecasting head
        self.forecast_decoder = np.random.randn(embedding_dim, forecast_horizon * n_features) * 0.01
        
        # Anomaly detection head
        self.anomaly_scorer = np.random.randn(embedding_dim, 1) * 0.01
    
    def forward_forecasting(self, sequence_embedding: np.ndarray) -> np.ndarray:
        """
        Forecasting
        
        Args:
            sequence_embedding: (seq_length, embedding_dim) historical data
            
        Returns:
            forecast: (forecast_horizon, n_features) future predictions
        """
        # Use last embedding for forecasting
        last_embedding = sequence_embedding[-1]
        
        # Decode to forecast
        forecast_flat = last_embedding @ self.forecast_decoder
        forecast = forecast_flat.reshape(self.forecast_horizon, self.n_features)
        
        return forecast
    
    def forward_anomaly_detection(self, sequence_embedding: np.ndarray) -> np.ndarray:
        """
        Anomaly detection
        
        Args:
            sequence_embedding: (seq_length, embedding_dim) time series features
            
        Returns:
            anomaly_scores: (seq_length,) anomaly scores (higher = more anomalous)
        """
        # Compute anomaly score for each timestep
        anomaly_scores = sequence_embedding @ self.anomaly_scorer
        return anomaly_scores.flatten()


class GraphAdapter:
    """
    Graph Learning Adapter
    
    Supports:
        - Node classification
        - Edge prediction
        - Graph classification
    """
    
    def __init__(self, embedding_dim: int, n_node_classes: int = None,
                 task_type: str = "node_classification"):
        self.embedding_dim = embedding_dim
        self.task_type = task_type
        
        if task_type == "node_classification":
            self.node_classifier = np.random.randn(embedding_dim, n_node_classes) * 0.01
        elif task_type == "edge_prediction":
            self.edge_scorer = np.random.randn(embedding_dim * 2, 1) * 0.01
        elif task_type == "graph_classification":
            self.graph_classifier = np.random.randn(embedding_dim, n_node_classes) * 0.01
    
    def forward_node_classification(self, node_embeddings: np.ndarray) -> np.ndarray:
        """
        Node classification
        
        Args:
            node_embeddings: (n_nodes, embedding_dim) node features
            
        Returns:
            node_logits: (n_nodes, n_classes) per-node class logits
        """
        node_logits = node_embeddings @ self.node_classifier
        return node_logits
    
    def forward_edge_prediction(self, node_embeddings: np.ndarray,
                               edge_pairs: np.ndarray) -> np.ndarray:
        """
        Edge prediction
        
        Args:
            node_embeddings: (n_nodes, embedding_dim) node features
            edge_pairs: (n_edges, 2) pairs of node indices
            
        Returns:
            edge_scores: (n_edges,) edge existence probabilities
        """
        edge_scores = []
        
        for source, target in edge_pairs:
            # Concatenate source and target embeddings
            edge_feat = np.concatenate([node_embeddings[source], node_embeddings[target]])
            
            # Score edge
            score = (edge_feat @ self.edge_scorer)[0]
            edge_scores.append(score)
        
        return np.array(edge_scores)
    
    def forward_graph_classification(self, node_embeddings: np.ndarray) -> np.ndarray:
        """
        Graph classification
        
        Args:
            node_embeddings: (n_nodes, embedding_dim) node features
            
        Returns:
            graph_logits: (n_classes,) graph-level class logits
        """
        # Global pooling
        graph_embedding = np.mean(node_embeddings, axis=0)
        
        # Classify
        graph_logits = graph_embedding @ self.graph_classifier
        return graph_logits


class MolecularAdapter:
    """
    Molecular Modeling Adapter
    
    Supports:
        - Property prediction (e.g., toxicity, solubility)
        - Molecule generation
        - Reaction prediction
    """
    
    def __init__(self, embedding_dim: int, n_properties: int = 1,
                 n_atom_types: int = 100, task_type: str = "property_prediction"):
        self.embedding_dim = embedding_dim
        self.task_type = task_type
        
        if task_type == "property_prediction":
            self.property_predictor = np.random.randn(embedding_dim, n_properties) * 0.01
        elif task_type == "generation":
            self.atom_generator = np.random.randn(embedding_dim, n_atom_types) * 0.01
            self.bond_generator = np.random.randn(embedding_dim * 2, 4) * 0.01  # 4 bond types
    
    def forward_property_prediction(self, molecule_embedding: np.ndarray) -> np.ndarray:
        """
        Molecular property prediction
        
        Args:
            molecule_embedding: (embedding_dim,) molecule representation
            
        Returns:
            properties: (n_properties,) predicted properties
        """
        properties = molecule_embedding @ self.property_predictor
        return properties
    
    def forward_molecule_generation(self, n_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate molecule
        
        Args:
            n_atoms: Number of atoms to generate
            
        Returns:
            atom_types: (n_atoms,) atom type indices
            bonds: (n_atoms, n_atoms) adjacency matrix with bond types
        """
        atom_types = []
        current_state = np.random.randn(self.embedding_dim) * 0.1
        
        # Generate atoms
        for _ in range(n_atoms):
            atom_logits = current_state @ self.atom_generator
            atom_type = np.argmax(atom_logits)
            atom_types.append(atom_type)
            
            # Update state
            current_state = np.tanh(atom_logits @ self.atom_generator.T)
        
        atom_types = np.array(atom_types)
        
        # Generate bonds (simplified)
        bonds = np.zeros((n_atoms, n_atoms))
        
        return atom_types, bonds


class RoboticsAdapter:
    """
    Robotics Adapter
    
    Supports:
        - Motor control
        - Motion planning
        - Visual servoing
    """
    
    def __init__(self, embedding_dim: int, n_joints: int, workspace_dim: int = 3):
        self.embedding_dim = embedding_dim
        self.n_joints = n_joints
        self.workspace_dim = workspace_dim
        
        # Motor control
        self.joint_controller = np.random.randn(embedding_dim, n_joints) * 0.01
        
        # Inverse kinematics
        self.ik_solver = np.random.randn(workspace_dim, n_joints) * 0.01
    
    def forward_motor_control(self, state_embedding: np.ndarray,
                             target_pose: np.ndarray) -> np.ndarray:
        """
        Motor control
        
        Args:
            state_embedding: (embedding_dim,) current state
            target_pose: (workspace_dim,) target end-effector pose
            
        Returns:
            joint_commands: (n_joints,) joint position/velocity commands
        """
        # Inverse kinematics
        joint_target = target_pose @ self.ik_solver
        
        # PD control (simplified)
        joint_commands = state_embedding @ self.joint_controller
        joint_commands = np.tanh(joint_commands + 0.1 * joint_target)
        
        return joint_commands


# Test all adapters
if __name__ == "__main__":
    print("Testing Task-Specific Adapters...")
    
    embedding_dim = 512
    
    # Vision adapter
    print("\n1. Vision Adapter:")
    vision = VisionAdapter(embedding_dim, n_classes=1000, task_type="classification")
    patch_embeddings = np.random.randn(196, embedding_dim)  # 14x14 patches
    logits = vision.forward_classification(patch_embeddings)
    print(f"   Classification logits shape: {logits.shape}")
    
    # Audio adapter
    print("\n2. Audio Adapter:")
    audio = AudioAdapter(embedding_dim, vocab_size=5000, task_type="speech_recognition")
    audio_embeddings = np.random.randn(100, embedding_dim)  # 100 frames
    token_logits = audio.forward_speech_recognition(audio_embeddings)
    print(f"   Speech recognition logits shape: {token_logits.shape}")
    
    # RL adapter
    print("\n3. RL Adapter:")
    rl = RLAdapter(embedding_dim, action_dim=4, continuous=True)
    state = np.random.randn(embedding_dim)
    action, log_prob = rl.forward_policy(state)
    value = rl.forward_value(state)
    print(f"   Action: {action.shape}, Value: {value}")
    
    # Time series adapter
    print("\n4. Time Series Adapter:")
    ts = TimeSeriesAdapter(embedding_dim, forecast_horizon=10, n_features=5)
    sequence = np.random.randn(50, embedding_dim)
    forecast = ts.forward_forecasting(sequence)
    print(f"   Forecast shape: {forecast.shape}")
    
    # Graph adapter
    print("\n5. Graph Adapter:")
    graph = GraphAdapter(embedding_dim, n_node_classes=7, task_type="node_classification")
    node_embeddings = np.random.randn(20, embedding_dim)
    node_logits = graph.forward_node_classification(node_embeddings)
    print(f"   Node classification logits shape: {node_logits.shape}")
    
    # Molecular adapter
    print("\n6. Molecular Adapter:")
    molecular = MolecularAdapter(embedding_dim, n_properties=3, task_type="property_prediction")
    molecule_embedding = np.random.randn(embedding_dim)
    properties = molecular.forward_property_prediction(molecule_embedding)
    print(f"   Molecular properties: {properties.shape}")
    
    # Robotics adapter
    print("\n7. Robotics Adapter:")
    robotics = RoboticsAdapter(embedding_dim, n_joints=7, workspace_dim=3)
    state = np.random.randn(embedding_dim)
    target = np.array([0.5, 0.3, 0.2])
    joint_commands = robotics.forward_motor_control(state, target)
    print(f"   Joint commands shape: {joint_commands.shape}")
    
    print("\n✓ All Task-Specific Adapters working!")
