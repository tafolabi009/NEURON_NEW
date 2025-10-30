"""
Continual Learning Benchmark
Tests catastrophic forgetting prevention
"""

import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from neurons.network import NEURONSNetwork


def generate_task_data(
    task_id: int,
    n_samples: int = 100,
    input_dim: int = 100,
    n_classes: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic task data
    
    Each task has different input distributions
    
    Parameters:
    -----------
    task_id : int
        Task identifier
    n_samples : int
        Number of samples
    input_dim : int
        Input dimensionality
    n_classes : int
        Number of classes
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : (inputs, labels)
    """
    # Set random seed for reproducibility
    np.random.seed(task_id * 1000)
    
    # Generate task-specific patterns
    task_patterns = np.random.randn(n_classes, input_dim) * (task_id + 1) * 0.5
    
    inputs = []
    labels = []
    
    for _ in range(n_samples):
        label = np.random.randint(0, n_classes)
        # Sample near task pattern with noise
        sample = task_patterns[label] + np.random.randn(input_dim) * 0.3
        inputs.append(sample)
        labels.append(label)
    
    return np.array(inputs), np.array(labels)


def evaluate_task(
    network: NEURONSNetwork,
    task_id: int,
    n_samples: int = 50
) -> float:
    """
    Evaluate network on a specific task
    
    Parameters:
    -----------
    network : NEURONSNetwork
        Network to evaluate
    task_id : int
        Task to evaluate on
    n_samples : int
        Number of test samples
        
    Returns:
    --------
    float : Accuracy on task
    """
    test_x, test_y = generate_task_data(task_id, n_samples)
    
    correct = 0
    for idx in range(len(test_x)):
        output = network.predict(test_x[idx])
        if np.argmax(output) == test_y[idx]:
            correct += 1
    
    return correct / len(test_x)


def train_continual(
    network: NEURONSNetwork = None,
    n_tasks: int = 3,
    n_samples_per_task: int = 100,
    epochs_per_task: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Train on sequential tasks and measure forgetting
    
    This tests the Elastic Weight Consolidation (EWC) mechanism
    
    Parameters:
    -----------
    network : NEURONSNetwork
        Network to use (creates new if None)
    n_tasks : int
        Number of sequential tasks
    n_samples_per_task : int
        Training samples per task
    epochs_per_task : int
        Training epochs per task
    verbose : bool
        Print progress
        
    Returns:
    --------
    Dict : Continual learning results
    """
    # Create network if not provided
    if network is None:
        network = NEURONSNetwork(
            input_size=100,
            hidden_sizes=[200, 100],
            output_size=10,
            enable_neuromodulation=True,
            enable_oscillations=True,
            enable_ewc=True
        )
    
    results = {
        'task_accuracies': [],  # Accuracy after learning each task
        'forgetting': [],  # Forgetting for each previous task
        'task_history': []  # Full accuracy matrix
    }
    
    # Store task data for later evaluation
    task_datasets = []
    
    if verbose:
        print("Continual Learning: Sequential Task Training")
        print("=" * 50)
    
    # Train on each task sequentially
    for task_idx in range(n_tasks):
        if verbose:
            print(f"\n=== Task {task_idx + 1}/{n_tasks} ===")
        
        # Generate task data
        train_x, train_y = generate_task_data(
            task_id=task_idx,
            n_samples=n_samples_per_task
        )
        task_datasets.append((task_idx, train_x, train_y))
        
        # Train on current task
        for epoch in range(epochs_per_task):
            indices = np.random.permutation(len(train_x))
            
            if verbose:
                pbar = tqdm(indices, desc=f"Epoch {epoch+1}/{epochs_per_task}")
            else:
                pbar = indices
            
            epoch_correct = 0
            for idx in pbar:
                x = train_x[idx]
                y = train_y[idx]
                
                # One-hot target
                target = np.zeros(10)
                target[y] = 1.0
                
                # Train step
                metrics = network.train_step(
                    inputs=x,
                    targets=target,
                    reward=0.5
                )
                
                # Check correctness
                if np.argmax(metrics['output']) == y:
                    epoch_correct += 1
            
            if verbose:
                print(f"  Accuracy: {epoch_correct/len(train_x):.2%}")
        
        # Consolidate task if EWC enabled
        if network.enable_ewc:
            network.consolidate_task(train_x, train_y)
            if verbose:
                print("  Task consolidated with EWC")
        
        # Evaluate on all tasks seen so far
        task_accuracies = []
        for prev_task_idx in range(task_idx + 1):
            acc = evaluate_task(network, prev_task_idx)
            task_accuracies.append(acc)
            
            if verbose:
                if prev_task_idx == task_idx:
                    print(f"  Current task accuracy: {acc:.2%}")
                else:
                    print(f"  Task {prev_task_idx + 1} accuracy: {acc:.2%} (after {task_idx - prev_task_idx} tasks)")
        
        results['task_history'].append(task_accuracies)
    
    # Compute forgetting metrics
    if verbose:
        print(f"\n=== Forgetting Analysis ===")
    
    for task_idx in range(n_tasks - 1):
        # Initial accuracy (right after learning)
        initial_acc = results['task_history'][task_idx][task_idx]
        
        # Final accuracy (after learning all subsequent tasks)
        final_acc = results['task_history'][-1][task_idx]
        
        # Forgetting = initial - final
        forgetting = (initial_acc - final_acc) * 100
        results['forgetting'].append(forgetting)
        
        if verbose:
            print(f"Task {task_idx + 1}: Initial={initial_acc:.2%}, "
                  f"Final={final_acc:.2%}, Forgetting={forgetting:.1f}%")
    
    # Summary statistics
    avg_forgetting = np.mean(results['forgetting']) if results['forgetting'] else 0.0
    final_accuracies = results['task_history'][-1]
    avg_final_accuracy = np.mean(final_accuracies)
    
    summary = {
        'n_tasks': n_tasks,
        'avg_forgetting': avg_forgetting,
        'avg_final_accuracy': avg_final_accuracy,
        'final_accuracies': final_accuracies,
        'forgetting_per_task': results['forgetting'],
        'full_history': results['task_history'],
        'ewc_enabled': network.enable_ewc
    }
    
    if verbose:
        print(f"\n=== Summary ===")
        print(f"Average forgetting: {avg_forgetting:.1f}%")
        print(f"Average final accuracy: {avg_final_accuracy:.2%}")
        print(f"EWC enabled: {network.enable_ewc}")
        
        # Get network state
        state = network.get_state()
        if 'forgetting_estimate' in state:
            print(f"EWC forgetting estimate: {state['forgetting_estimate']:.2%}")
    
    return summary


def compare_with_without_ewc(
    n_tasks: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Compare continual learning with and without EWC
    
    Parameters:
    -----------
    n_tasks : int
        Number of tasks
    verbose : bool
        Print progress
        
    Returns:
    --------
    Dict : Comparison results
    """
    if verbose:
        print("Comparing continual learning with/without EWC")
        print("=" * 50)
    
    # Train without EWC
    if verbose:
        print("\n>>> WITHOUT EWC <<<")
    network_no_ewc = NEURONSNetwork(
        input_size=100,
        hidden_sizes=[200, 100],
        output_size=10,
        enable_ewc=False
    )
    results_no_ewc = train_continual(
        network=network_no_ewc,
        n_tasks=n_tasks,
        verbose=verbose
    )
    
    # Train with EWC
    if verbose:
        print("\n>>> WITH EWC <<<")
    network_with_ewc = NEURONSNetwork(
        input_size=100,
        hidden_sizes=[200, 100],
        output_size=10,
        enable_ewc=True
    )
    results_with_ewc = train_continual(
        network=network_with_ewc,
        n_tasks=n_tasks,
        verbose=verbose
    )
    
    # Comparison
    comparison = {
        'without_ewc': results_no_ewc,
        'with_ewc': results_with_ewc,
        'forgetting_reduction': results_no_ewc['avg_forgetting'] - results_with_ewc['avg_forgetting']
    }
    
    if verbose:
        print(f"\n=== Comparison ===")
        print(f"Forgetting without EWC: {results_no_ewc['avg_forgetting']:.1f}%")
        print(f"Forgetting with EWC: {results_with_ewc['avg_forgetting']:.1f}%")
        print(f"Reduction: {comparison['forgetting_reduction']:.1f}%")
    
    return comparison


if __name__ == "__main__":
    # Test continual learning
    results = train_continual(n_tasks=3, verbose=True)
    
    # Compare with/without EWC
    print("\n" + "="*50)
    comparison = compare_with_without_ewc(n_tasks=3, verbose=True)
