"""
Few-Shot Learning Benchmark
"""

import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import time

from neurons.network import NEURONSNetwork


def generate_few_shot_task(
    n_classes: int = 5,
    n_examples: int = 5,
    n_test: int = 20,
    input_dim: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic few-shot learning task
    
    Parameters:
    -----------
    n_classes : int
        Number of classes
    n_examples : int
        Examples per class (support set)
    n_test : int
        Test examples per class (query set)
    input_dim : int
        Input dimensionality
        
    Returns:
    --------
    Tuple : (support_x, support_y, query_x, query_y)
    """
    # Generate class prototypes
    prototypes = np.random.randn(n_classes, input_dim)
    
    # Generate support set
    support_x = []
    support_y = []
    for class_idx in range(n_classes):
        for _ in range(n_examples):
            # Sample near prototype with noise
            sample = prototypes[class_idx] + np.random.randn(input_dim) * 0.3
            support_x.append(sample)
            support_y.append(class_idx)
    
    support_x = np.array(support_x)
    support_y = np.array(support_y)
    
    # Generate query set
    query_x = []
    query_y = []
    for class_idx in range(n_classes):
        for _ in range(n_test):
            sample = prototypes[class_idx] + np.random.randn(input_dim) * 0.3
            query_x.append(sample)
            query_y.append(class_idx)
    
    query_x = np.array(query_x)
    query_y = np.array(query_y)
    
    return support_x, support_y, query_x, query_y


def train_fewshot(
    network: NEURONSNetwork = None,
    n_examples: int = 5,
    n_classes: int = 5,
    n_trials: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Train and evaluate few-shot learning
    
    Demonstrates human-like learning from minimal examples
    
    Parameters:
    -----------
    network : NEURONSNetwork
        Network to use (creates new if None)
    n_examples : int
        Number of examples per class
    n_classes : int
        Number of classes
    n_trials : int
        Number of few-shot tasks to average over
    verbose : bool
        Print progress
        
    Returns:
    --------
    Dict : Few-shot learning results
    """
    # Create network if not provided
    if network is None:
        network = NEURONSNetwork(
            input_size=100,
            hidden_sizes=[1000, 50],
            output_size=n_classes,
            enable_neuromodulation=True,
            enable_oscillations=True,
            enable_ewc=True
        )
    
    # Set encoding mode for fast learning
    if network.enable_neuromodulation:
        network.neuromodulation.set_task_context("encoding")
    
    results = {
        'accuracies': [],
        'trials_to_criterion': [],
        'training_times': []
    }
    
    if verbose:
        print(f"Few-Shot Learning: {n_examples} examples per class")
        pbar = tqdm(range(n_trials), desc="Trials")
    else:
        pbar = range(n_trials)
    
    for trial in pbar:
        # Generate task
        support_x, support_y, query_x, query_y = generate_few_shot_task(
            n_classes=n_classes,
            n_examples=n_examples,
            n_test=20,
            input_dim=network.input_size
        )
        
        # Reset network for new task
        network.reset()
        
        # Meta-learning: train on support set with high ACh (encoding mode)
        start_time = time.time()
        
        # Multiple passes through support set for consolidation
        max_passes = 10
        criterion = 0.8  # 80% accuracy threshold
        
        for pass_idx in range(max_passes):
            # Shuffle support set
            indices = np.random.permutation(len(support_x))
            
            for idx in indices:
                x = support_x[idx]
                y = support_y[idx]
                
                # One-hot target
                target = np.zeros(n_classes)
                target[y] = 1.0
                
                # Train with high learning signal
                metrics = network.train_step(
                    inputs=x,
                    targets=target,
                    reward=1.0  # High reward for encoding
                )
            
            # Check if learned
            correct = 0
            for idx in range(len(support_x)):
                output = network.predict(support_x[idx])
                if np.argmax(output) == support_y[idx]:
                    correct += 1
            
            support_accuracy = correct / len(support_x)
            
            if support_accuracy >= criterion:
                results['trials_to_criterion'].append(pass_idx + 1)
                break
        else:
            results['trials_to_criterion'].append(max_passes)
        
        training_time = time.time() - start_time
        results['training_times'].append(training_time)
        
        # Evaluate on query set
        correct = 0
        for idx in range(len(query_x)):
            output = network.predict(query_x[idx])
            if np.argmax(output) == query_y[idx]:
                correct += 1
        
        accuracy = correct / len(query_x)
        results['accuracies'].append(accuracy)
        
        if verbose and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'acc': f"{np.mean(results['accuracies']):.4f}",
                'trials': f"{np.mean(results['trials_to_criterion']):.1f}"
            })
    
    # Aggregate results
    summary = {
        'mean_accuracy': np.mean(results['accuracies']),
        'std_accuracy': np.std(results['accuracies']),
        'mean_trials_to_criterion': np.mean(results['trials_to_criterion']),
        'mean_training_time': np.mean(results['training_times']),
        'n_examples': n_examples,
        'n_classes': n_classes,
        'detailed_results': results
    }
    
    if verbose:
        print(f"\n=== Few-Shot Learning Results ===")
        print(f"Examples per class: {n_examples}")
        print(f"Accuracy: {summary['mean_accuracy']:.2%} ± {summary['std_accuracy']:.2%}")
        print(f"Trials to criterion: {summary['mean_trials_to_criterion']:.1f}")
        print(f"Avg training time: {summary['mean_training_time']:.3f}s")
        print(f"\nData efficiency: {n_examples * n_classes} total training examples")
    
    return summary


if __name__ == "__main__":
    print("Few-Shot Learning with NEURONS")
    print("=" * 50)
    
    # Test with different numbers of examples
    for n_ex in [1, 5, 10]:
        print(f"\n{n_ex}-shot learning:")
        results = train_fewshot(n_examples=n_ex, n_trials=10, verbose=True)
