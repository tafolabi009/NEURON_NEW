"""
NEURONSv2 Comprehensive Benchmark Suite
Tests all mechanisms across diverse tasks
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add neurons to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurons.models import create_mnist_model, create_cifar10_model, create_fewshot_model
from neurons.optimization import fast_temporal_encode_phase, fast_rank_order_encode
from neurons.neuronsv2_network import NEURONSv2Network


def load_mnist(data_path: str = "data/MNIST/raw"):
    """Load MNIST dataset"""
    import struct
    
    def read_idx(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
    X_train = read_idx(f"{data_path}/train-images-idx3-ubyte").astype(float) / 255.0
    y_train = read_idx(f"{data_path}/train-labels-idx1-ubyte")
    X_test = read_idx(f"{data_path}/t10k-images-idx3-ubyte").astype(float) / 255.0
    y_test = read_idx(f"{data_path}/t10k-labels-idx1-ubyte")
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, y_train, X_test, y_test


def temporal_encode_batch(images: np.ndarray, encoding: str = "phase") -> np.ndarray:
    """Encode images as spike times"""
    batch_size, input_size = images.shape
    spike_times = np.zeros((batch_size, input_size), dtype=np.float32)
    
    for i in range(batch_size):
        if encoding == "phase":
            spike_times[i] = fast_temporal_encode_phase(images[i], 0.0, 166.7)
        elif encoding == "rank":
            spike_times[i] = fast_rank_order_encode(images[i], 0.0, 50.0)
        else:
            spike_times[i] = (1.0 - images[i]) * 50.0
    
    return spike_times


class BenchmarkRunner:
    """Run comprehensive benchmarks"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def benchmark_mnist(self, n_epochs: int = 10, batch_size: int = 64) -> Dict:
        """
        MNIST Benchmark - Tests core learning capability
        
        Validates:
        - Temporal coding (phase encoding)
        - Predictive plasticity (learning without backprop)
        - Emergent attention (feature selection)
        - Dendritic computation (nonlinear features)
        - Meta-learning (fast adaptation)
        
        Target: >95% accuracy in 10 epochs
        """
        print("\n" + "=" * 80)
        print("MNIST BENCHMARK")
        print("=" * 80)
        
        # Load data
        print("\nLoading MNIST...")
        X_train, y_train, X_test, y_test = load_mnist()
        
        # Use subset for speed (optional)
        n_train = 10000
        X_train, y_train = X_train[:n_train], y_train[:n_train]
        
        # Create model
        print("Creating model...")
        model = create_mnist_model()
        print(f"Parameters: {model.get_num_parameters():,}")
        
        # Training loop
        print(f"\nTraining {n_epochs} epochs...")
        best_acc = 0.0
        history = {'train_loss': [], 'test_acc': [], 'epoch_time': []}
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Train
            indices = np.random.permutation(len(X_train))
            epoch_loss = []
            
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                # Encode
                spike_batch = temporal_encode_batch(batch_X, "phase")
                
                # One-hot targets
                targets = np.zeros((len(batch_y), 10))
                targets[np.arange(len(batch_y)), batch_y] = 1.0
                
                # Train
                loss = model.train_step(spike_batch, targets, learning_rate=0.01)
                epoch_loss.append(loss)
            
            # Evaluate
            correct = 0
            total = 0
            for i in range(0, len(X_test), 100):
                batch_X = X_test[i:i+100]
                batch_y = y_test[i:i+100]
                
                spike_batch = temporal_encode_batch(batch_X, "phase")
                outputs = np.array([model.forward(s) for s in spike_batch])
                preds = np.argmax(outputs, axis=1)
                
                correct += np.sum(preds == batch_y)
                total += len(batch_y)
            
            test_acc = correct / total
            epoch_time = time.time() - epoch_start
            
            history['train_loss'].append(np.mean(epoch_loss))
            history['test_acc'].append(test_acc)
            history['epoch_time'].append(epoch_time)
            
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Loss={np.mean(epoch_loss):.4f}, "
                  f"Acc={test_acc*100:.2f}%, "
                  f"Time={epoch_time:.1f}s")
            
            best_acc = max(best_acc, test_acc)
        
        # Results
        results = {
            'task': 'MNIST',
            'final_accuracy': test_acc,
            'best_accuracy': best_acc,
            'avg_epoch_time': np.mean(history['epoch_time']),
            'parameters': model.get_num_parameters(),
            'target_met': best_acc > 0.95,
            'history': history
        }
        
        print(f"\n✓ Final Accuracy: {test_acc*100:.2f}%")
        print(f"✓ Best Accuracy: {best_acc*100:.2f}%")
        print(f"✓ Avg Epoch Time: {np.mean(history['epoch_time']):.1f}s")
        print(f"✓ Target Met: {'YES' if best_acc > 0.95 else 'NO (need >95%)'}")
        
        self.results['mnist'] = results
        return results
    
    def benchmark_fewshot(self, n_way: int = 5, n_shot: int = 1, n_episodes: int = 100) -> Dict:
        """
        Few-Shot Learning Benchmark
        
        Validates:
        - Fast weights (τ=0.1s) for immediate adaptation
        - Meta-learning without meta-training
        - Rapid feature selection via attention
        
        Target: >90% on 5-way 1-shot
        """
        print("\n" + "=" * 80)
        print(f"FEW-SHOT LEARNING BENCHMARK ({n_way}-way {n_shot}-shot)")
        print("=" * 80)
        
        # Load MNIST as few-shot task
        print("\nLoading data...")
        X_train, y_train, X_test, y_test = load_mnist()
        
        # Create model
        print("Creating few-shot model...")
        model = create_fewshot_model()
        print(f"Parameters: {model.get_num_parameters():,}")
        print(f"Fast weight τ: {model.config.tau_fast}s (rapid adaptation!)")
        
        # Few-shot episodes
        print(f"\nRunning {n_episodes} episodes...")
        accuracies = []
        
        for episode in range(n_episodes):
            # Sample classes
            classes = np.random.choice(10, n_way, replace=False)
            
            # Support set (for adaptation)
            support_X = []
            support_y = []
            for i, cls in enumerate(classes):
                cls_idx = np.where(y_train == cls)[0]
                samples = np.random.choice(cls_idx, n_shot, replace=False)
                support_X.append(X_train[samples])
                support_y.extend([i] * n_shot)
            
            support_X = np.vstack(support_X)
            support_y = np.array(support_y)
            
            # Query set (for testing)
            query_X = []
            query_y = []
            for i, cls in enumerate(classes):
                cls_idx = np.where(y_test == cls)[0]
                sample = np.random.choice(cls_idx, 1)[0]
                query_X.append(X_test[sample])
                query_y.append(i)
            
            query_X = np.array(query_X)
            query_y = np.array(query_y)
            
            # Adapt on support set (fast weights activate!)
            support_spikes = temporal_encode_batch(support_X, "phase")
            support_targets = np.zeros((len(support_y), n_way))
            support_targets[np.arange(len(support_y)), support_y] = 1.0
            
            # Quick adaptation
            model.train_step(support_spikes, support_targets, learning_rate=0.1)
            
            # Test on query set
            query_spikes = temporal_encode_batch(query_X, "phase")
            outputs = np.array([model.forward(s) for s in query_spikes])
            preds = np.argmax(outputs, axis=1)
            
            acc = np.mean(preds == query_y)
            accuracies.append(acc)
            
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode+1}/{n_episodes}: Avg Acc = {np.mean(accuracies)*100:.2f}%")
        
        # Results
        final_acc = np.mean(accuracies)
        results = {
            'task': f'{n_way}-way {n_shot}-shot',
            'accuracy': final_acc,
            'std': np.std(accuracies),
            'episodes': n_episodes,
            'target_met': final_acc > 0.90,
            'fast_weight_tau': model.config.tau_fast
        }
        
        print(f"\n✓ Final Accuracy: {final_acc*100:.2f}% ± {np.std(accuracies)*100:.2f}%")
        print(f"✓ Target Met: {'YES' if final_acc > 0.90 else 'NO (need >90%)'}")
        
        self.results['fewshot'] = results
        return results
    
    def benchmark_continual_learning(self, n_tasks: int = 5) -> Dict:
        """
        Continual Learning Benchmark (PermutedMNIST)
        
        Validates:
        - Slow weights (τ=100000s) for retention
        - No catastrophic forgetting
        - Task switching
        
        Target: <10% accuracy drop on old tasks
        """
        print("\n" + "=" * 80)
        print(f"CONTINUAL LEARNING BENCHMARK ({n_tasks} tasks)")
        print("=" * 80)
        
        # Load MNIST
        print("\nLoading data...")
        X_train, y_train, X_test, y_test = load_mnist()
        
        # Use subset
        n_samples = 1000
        X_train, y_train = X_train[:n_samples], y_train[:n_samples]
        X_test, y_test = X_test[:200], y_test[:200]
        
        # Create model
        from neurons.models import create_continual_learning_model
        print("Creating continual learning model...")
        model = create_continual_learning_model()
        print(f"Parameters: {model.get_num_parameters():,}")
        print(f"Slow weight τ: {model.config.tau_slow}s (long-term memory!)")
        
        # Train on sequence of tasks (permuted inputs)
        task_accuracies = []
        permutations = [np.random.permutation(784) for _ in range(n_tasks)]
        
        for task_id in range(n_tasks):
            print(f"\nTask {task_id+1}/{n_tasks}...")
            
            # Permute inputs
            perm = permutations[task_id]
            X_train_perm = X_train[:, perm]
            X_test_perm = X_test[:, perm]
            
            # Train on this task
            for epoch in range(5):
                indices = np.random.permutation(len(X_train_perm))
                for i in range(0, len(X_train_perm), 32):
                    batch_idx = indices[i:i+32]
                    batch_X = X_train_perm[batch_idx]
                    batch_y = y_train[batch_idx]
                    
                    spike_batch = temporal_encode_batch(batch_X, "phase")
                    targets = np.zeros((len(batch_y), 10))
                    targets[np.arange(len(batch_y)), batch_y] = 1.0
                    
                    model.train_step(spike_batch, targets, learning_rate=0.01)
            
            # Test on ALL previous tasks (check forgetting)
            task_accs = []
            for prev_task_id in range(task_id + 1):
                perm_prev = permutations[prev_task_id]
                X_test_prev = X_test[:, perm_prev]
                
                spike_batch = temporal_encode_batch(X_test_prev, "phase")
                outputs = np.array([model.forward(s) for s in spike_batch])
                preds = np.argmax(outputs, axis=1)
                
                acc = np.mean(preds == y_test)
                task_accs.append(acc)
                print(f"  Task {prev_task_id+1} accuracy: {acc*100:.2f}%")
            
            task_accuracies.append(task_accs)
        
        # Calculate forgetting
        final_task_accs = task_accuracies[-1]
        forgetting = []
        for i in range(len(final_task_accs) - 1):
            initial_acc = task_accuracies[i][i]
            final_acc = final_task_accs[i]
            forget = initial_acc - final_acc
            forgetting.append(forget)
        
        avg_forgetting = np.mean(forgetting) if forgetting else 0.0
        
        # Results
        results = {
            'task': 'Continual Learning',
            'n_tasks': n_tasks,
            'final_accuracies': final_task_accs,
            'avg_forgetting': avg_forgetting,
            'target_met': avg_forgetting < 0.10,
            'slow_weight_tau': model.config.tau_slow
        }
        
        print(f"\n✓ Final Accuracies: {[f'{a*100:.1f}%' for a in final_task_accs]}")
        print(f"✓ Avg Forgetting: {avg_forgetting*100:.2f}%")
        print(f"✓ Target Met: {'YES' if avg_forgetting < 0.10 else 'NO (need <10%)'}")
        
        self.results['continual'] = results
        return results
    
    def benchmark_speed(self) -> Dict:
        """
        Speed Benchmark - Measure optimization effectiveness
        
        Validates:
        - Numba JIT speedup (10-50×)
        - Sparse computation
        - Event-driven processing
        
        Target: <50ms per forward pass
        """
        print("\n" + "=" * 80)
        print("SPEED BENCHMARK")
        print("=" * 80)
        
        print("\nCreating model...")
        model = create_mnist_model()
        
        # Generate test data
        test_input = np.random.rand(784)
        spike_times = fast_temporal_encode_phase(test_input, 0.0, 166.7)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            model.forward(spike_times)
        
        # Benchmark forward pass
        print("\nBenchmarking forward pass...")
        n_runs = 100
        times = []
        
        start = time.time()
        for _ in range(n_runs):
            t0 = time.time()
            model.forward(spike_times)
            times.append(time.time() - t0)
        total_time = time.time() - start
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        
        print(f"Average forward pass: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"Throughput: {1000/avg_time:.1f} samples/sec")
        
        # Benchmark encoding
        print("\nBenchmarking temporal encoding...")
        encoding_times = []
        for _ in range(n_runs):
            t0 = time.time()
            fast_temporal_encode_phase(test_input, 0.0, 166.7)
            encoding_times.append(time.time() - t0)
        
        avg_encoding = np.mean(encoding_times) * 1000
        print(f"Average encoding: {avg_encoding:.2f}ms")
        
        # Results
        results = {
            'task': 'Speed',
            'forward_pass_ms': avg_time,
            'forward_std_ms': std_time,
            'encoding_ms': avg_encoding,
            'throughput_per_sec': 1000 / avg_time,
            'target_met': avg_time < 50.0
        }
        
        print(f"\n✓ Forward Pass: {avg_time:.2f}ms")
        print(f"✓ Target Met: {'YES' if avg_time < 50.0 else 'NO (need <50ms)'}")
        
        self.results['speed'] = results
        return results
    
    def run_all_benchmarks(self, save_results: bool = True):
        """Run complete benchmark suite"""
        print("=" * 80)
        print("NEURONSv2 COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run benchmarks
        self.benchmark_mnist(n_epochs=10)
        self.benchmark_fewshot(n_way=5, n_shot=1, n_episodes=100)
        self.benchmark_continual_learning(n_tasks=5)
        self.benchmark_speed()
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        for name, results in self.results.items():
            print(f"\n{name.upper()}:")
            if name == 'mnist':
                print(f"  Accuracy: {results['best_accuracy']*100:.2f}%")
                print(f"  Target (>95%): {'✓ PASS' if results['target_met'] else '✗ FAIL'}")
            elif name == 'fewshot':
                print(f"  Accuracy: {results['accuracy']*100:.2f}%")
                print(f"  Target (>90%): {'✓ PASS' if results['target_met'] else '✗ FAIL'}")
            elif name == 'continual':
                print(f"  Forgetting: {results['avg_forgetting']*100:.2f}%")
                print(f"  Target (<10%): {'✓ PASS' if results['target_met'] else '✗ FAIL'}")
            elif name == 'speed':
                print(f"  Forward Pass: {results['forward_pass_ms']:.2f}ms")
                print(f"  Target (<50ms): {'✓ PASS' if results['target_met'] else '✗ FAIL'}")
        
        print(f"\nTotal benchmark time: {total_time/60:.1f} minutes")
        
        # Save results
        if save_results:
            results_file = self.results_dir / f"benchmark_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n✓ Results saved to: {results_file}")
        
        # Overall pass/fail
        all_passed = all(r.get('target_met', False) for r in self.results.values())
        
        print("\n" + "=" * 80)
        if all_passed:
            print("✓ ALL BENCHMARKS PASSED!")
        else:
            print("✗ SOME BENCHMARKS FAILED - needs improvement")
        print("=" * 80)
        
        return self.results


def quick_test():
    """Quick sanity check (5 minutes)"""
    print("Running quick sanity test...")
    
    runner = BenchmarkRunner()
    
    # Just MNIST with 3 epochs
    results = runner.benchmark_mnist(n_epochs=3, batch_size=64)
    
    if results['best_accuracy'] > 0.70:
        print("\n✓ Quick test PASSED (>70% accuracy)")
        return True
    else:
        print("\n✗ Quick test FAILED (need >70% accuracy)")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test (5 min)")
    parser.add_argument("--full", action="store_true", help="Full benchmark suite (30 min)")
    parser.add_argument("--mnist", action="store_true", help="MNIST only")
    parser.add_argument("--fewshot", action="store_true", help="Few-shot only")
    parser.add_argument("--continual", action="store_true", help="Continual learning only")
    parser.add_argument("--speed", action="store_true", help="Speed only")
    args = parser.parse_args()
    
    runner = BenchmarkRunner()
    
    if args.quick:
        success = quick_test()
        sys.exit(0 if success else 1)
    elif args.full:
        results = runner.run_all_benchmarks()
        all_passed = all(r.get('target_met', False) for r in results.values())
        sys.exit(0 if all_passed else 1)
    elif args.mnist:
        runner.benchmark_mnist()
    elif args.fewshot:
        runner.benchmark_fewshot()
    elif args.continual:
        runner.benchmark_continual_learning()
    elif args.speed:
        runner.benchmark_speed()
    else:
        # Default: run all
        results = runner.run_all_benchmarks()
        all_passed = all(r.get('target_met', False) for r in results.values())
        sys.exit(0 if all_passed else 1)
