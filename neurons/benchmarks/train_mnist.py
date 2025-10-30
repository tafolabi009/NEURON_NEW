"""
MNIST Training and Evaluation
"""

import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
import time

try:
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from neurons.network import NEURONSNetwork


def load_mnist(train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST dataset
    
    Parameters:
    -----------
    train : bool
        Load training set if True, test set otherwise
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : (images, labels)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torchvision required for MNIST. Install with: pip install torchvision")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    
    # Convert to numpy
    images = dataset.data.numpy().reshape(-1, 784) / 255.0
    labels = dataset.targets.numpy()
    
    return images, labels


def train_mnist(
    network: NEURONSNetwork = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    verbose: bool = True
) -> Dict:
    """
    Train NEURONS on MNIST
    
    Parameters:
    -----------
    network : NEURONSNetwork
        Network to train (creates new if None)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    verbose : bool
        Print progress
        
    Returns:
    --------
    Dict : Training results
    """
    # Create network if not provided
    if network is None:
        network = NEURONSNetwork(
            input_size=784,
            hidden_sizes=[500],
            output_size=10,
            enable_neuromodulation=True,
            enable_oscillations=True,
            enable_ewc=True
        )
    
    # Load data
    if verbose:
        print("Loading MNIST dataset...")
    train_images, train_labels = load_mnist(train=True)
    
    # Training loop
    start_time = time.time()
    history = {
        'loss': [],
        'accuracy': [],
        'energy': [],
        'sparsity': []
    }
    
    for epoch in range(epochs):
        epoch_loss = []
        epoch_correct = 0
        epoch_total = 0
        
        # Shuffle data
        indices = np.random.permutation(len(train_images))
        
        # Progress bar
        if verbose:
            pbar = tqdm(range(0, len(train_images), batch_size),
                       desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = range(0, len(train_images), batch_size)
        
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(train_images))
            batch_indices = indices[batch_start:batch_end]
            
            batch_loss = 0
            batch_correct = 0
            
            for idx in batch_indices:
                # Get sample
                image = train_images[idx]
                label = train_labels[idx]
                
                # Convert label to one-hot
                target = np.zeros(10)
                target[label] = 1.0
                
                # Train step
                metrics = network.train_step(
                    inputs=image,
                    targets=target,
                    reward=0.0  # Will be updated based on correctness
                )
                
                # Check if correct
                pred = np.argmax(metrics['output'])
                if pred == label:
                    batch_correct += 1
                    # Positive reward for correct prediction
                    network.neuromodulation.update(reward=1.0, value_estimate=0.5)
                else:
                    # Negative reward for incorrect
                    network.neuromodulation.update(reward=0.0, value_estimate=0.5)
                
                batch_loss += metrics['loss']
                epoch_total += 1
            
            epoch_loss.append(batch_loss / len(batch_indices))
            epoch_correct += batch_correct
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f"{np.mean(epoch_loss):.4f}",
                    'acc': f"{epoch_correct/epoch_total:.4f}"
                })
        
        # Epoch metrics
        avg_loss = np.mean(epoch_loss)
        accuracy = epoch_correct / epoch_total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        # Get network state
        state = network.get_state()
        history['sparsity'].append(state['weight_sparsity'])
        
        if verbose:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, "
                  f"Sparsity={np.mean(state['weight_sparsity']):.4f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    test_accuracy = evaluate_mnist(network, verbose=verbose)
    
    results = {
        'training_time': training_time,
        'final_accuracy': test_accuracy,
        'history': history,
        'network_state': network.get_state()
    }
    
    if verbose:
        print(f"\n=== MNIST Results ===")
        print(f"Training time: {training_time:.2f}s")
        print(f"Test accuracy: {test_accuracy:.4%}")
        print(f"Avg sparsity: {np.mean(history['sparsity']):.4%}")
    
    return results


def evaluate_mnist(
    network: NEURONSNetwork,
    verbose: bool = True
) -> float:
    """
    Evaluate NEURONS on MNIST test set
    
    Parameters:
    -----------
    network : NEURONSNetwork
        Trained network
    verbose : bool
        Print progress
        
    Returns:
    --------
    float : Test accuracy
    """
    # Load test data
    test_images, test_labels = load_mnist(train=False)
    
    correct = 0
    total = 0
    
    if verbose:
        pbar = tqdm(range(len(test_images)), desc="Evaluating")
    else:
        pbar = range(len(test_images))
    
    for idx in pbar:
        image = test_images[idx]
        label = test_labels[idx]
        
        # Predict
        output = network.predict(image)
        pred = np.argmax(output)
        
        if pred == label:
            correct += 1
        total += 1
        
        if verbose and isinstance(pbar, tqdm):
            pbar.set_postfix({'accuracy': f"{correct/total:.4f}"})
    
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    # Example usage
    print("Training NEURONS on MNIST...")
    
    network = NEURONSNetwork(
        input_size=784,
        hidden_sizes=[500],
        output_size=10
    )
    
    results = train_mnist(network, epochs=5, verbose=True)
    
    print(f"\nFinal test accuracy: {results['final_accuracy']:.2%}")
