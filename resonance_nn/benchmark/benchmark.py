"""
Benchmarking and Evaluation Tools
Verify O(n log n) complexity and performance claims
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from scipy.stats import linregress
from resonance_nn.models.resonance_net import ResonanceNet
from resonance_nn.layers.holographic import HolographicMemory


class ComplexityBenchmark:
    """
    Benchmark computational complexity of Resonance Network
    
    Verifies Theorem 3: O(n log n) complexity claim
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = []
        
    def measure_forward_time(
        self,
        model: torch.nn.Module,
        seq_len: int,
        batch_size: int = 32,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Measure forward pass time
        
        Args:
            model: Model to benchmark
            seq_len: Sequence length
            batch_size: Batch size
            num_runs: Number of runs to average
            
        Returns:
            Timing statistics
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            dummy_input = torch.randn(batch_size, seq_len, model.input_dim).to(self.device)
            _ = model(dummy_input, use_memory=False)
            
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                x = torch.randn(batch_size, seq_len, model.input_dim).to(self.device)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                
                _ = model(x, use_memory=False)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                
                times.append((end - start) * 1000)  # Convert to ms
                
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
        }
    
    def measure_memory(
        self,
        model: torch.nn.Module,
        seq_len: int,
        batch_size: int = 32,
    ) -> float:
        """
        Measure memory usage
        
        Returns:
            Memory in MB
        """
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
        with torch.no_grad():
            x = torch.randn(batch_size, seq_len, model.input_dim).to(self.device)
            _ = model(x, use_memory=False)
            
        if self.device == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            # Rough estimate for CPU
            memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            
        return memory_mb
    
    def run(
        self,
        sequence_lengths: List[int] = [64, 128, 256, 512, 1024],
        input_dim: int = 512,
        num_frequencies: int = 64,
        num_layers: int = 4,
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Run comprehensive benchmark
        
        Args:
            sequence_lengths: List of sequence lengths to test
            input_dim: Input dimension
            num_frequencies: Number of frequencies
            num_layers: Number of layers
            batch_size: Batch size
            
        Returns:
            List of result dictionaries
        """
        print("Running Complexity Benchmark...")
        print(f"Device: {self.device}")
        print(f"Sequence lengths: {sequence_lengths}")
        print()
        
        results = []
        
        for n in sequence_lengths:
            print(f"Testing n={n}...")
            
            # Create model
            model = ResonanceNet(
                input_dim=input_dim,
                num_frequencies=num_frequencies,
                hidden_dim=input_dim,
                num_layers=num_layers,
                holographic_capacity=1000,
            ).to(self.device)
            
            # Measure time
            time_stats = self.measure_forward_time(model, n, batch_size)
            
            # Measure memory
            memory_mb = self.measure_memory(model, n, batch_size)
            
            # Get complexity estimate
            complexity = model.get_complexity_estimate(n)
            
            result = {
                'n': n,
                'time_ms': time_stats['mean'],
                'time_std': time_stats['std'],
                'memory_mb': memory_mb,
                'nlogn': n * np.log2(n),
                'n_squared': n * n,
                'complexity_estimate': complexity['total'],
            }
            
            results.append(result)
            print(f"  Time: {time_stats['mean']:.2f}±{time_stats['std']:.2f} ms")
            print(f"  Memory: {memory_mb:.2f} MB")
            print()
            
        self.results = results
        return results
    
    def analyze_complexity(self) -> Dict[str, float]:
        """
        Analyze complexity and fit to theoretical curves
        
        Returns:
            R² values for different complexity classes
        """
        if not self.results:
            raise ValueError("No results to analyze. Run benchmark first.")
            
        n_values = np.array([r['n'] for r in self.results])
        times = np.array([r['time_ms'] for r in self.results])
        
        # Normalize times to first measurement
        times_normalized = times / times[0]
        
        # Theoretical complexity curves (normalized)
        nlogn_normalized = (n_values * np.log2(n_values)) / (n_values[0] * np.log2(n_values[0]))
        n_squared_normalized = (n_values ** 2) / (n_values[0] ** 2)
        n_normalized = n_values / n_values[0]
        
        # Linear regression to get R² values
        slope_nlogn, intercept_nlogn, r_nlogn, _, _ = linregress(nlogn_normalized, times_normalized)
        slope_n2, intercept_n2, r_n2, _, _ = linregress(n_squared_normalized, times_normalized)
        slope_n, intercept_n, r_n, _, _ = linregress(n_normalized, times_normalized)
        
        return {
            'r_squared_nlogn': r_nlogn ** 2,
            'r_squared_n_squared': r_n2 ** 2,
            'r_squared_linear': r_n ** 2,
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot benchmark results
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.results:
            raise ValueError("No results to plot. Run benchmark first.")
            
        n_values = np.array([r['n'] for r in self.results])
        times = np.array([r['time_ms'] for r in self.results])
        memory = np.array([r['memory_mb'] for r in self.results])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time complexity plot
        ax1 = axes[0]
        ax1.plot(n_values, times, 'o-', label='Measured', linewidth=2, markersize=8)
        
        # Theoretical curves (scaled to fit)
        scale_nlogn = times[-1] / (n_values[-1] * np.log2(n_values[-1]))
        scale_n2 = times[-1] / (n_values[-1] ** 2)
        
        ax1.plot(n_values, scale_nlogn * n_values * np.log2(n_values), 
                '--', label='O(n log n)', alpha=0.7, linewidth=2)
        ax1.plot(n_values, scale_n2 * n_values ** 2, 
                ':', label='O(n²)', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel('Sequence Length (n)', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Computational Complexity', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Memory usage plot
        ax2 = axes[1]
        ax2.plot(n_values, memory, 's-', color='orange', linewidth=2, markersize=8)
        ax2.set_xlabel('Sequence Length (n)', fontsize=12)
        ax2.set_ylabel('Memory (MB)', fontsize=12)
        ax2.set_title('Memory Usage', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def print_table(self):
        """
        Print results as formatted table
        """
        if not self.results:
            raise ValueError("No results to print. Run benchmark first.")
            
        print("\n" + "="*80)
        print("COMPLEXITY VERIFICATION RESULTS")
        print("="*80)
        print(f"{'n':>6} {'Time (ms)':>12} {'nlogn fit':>12} {'n² fit':>12} {'Memory (MB)':>15}")
        print("-"*80)
        
        # Normalize for fit comparison
        times = np.array([r['time_ms'] for r in self.results])
        n_values = np.array([r['n'] for r in self.results])
        
        nlogn_fit = (n_values * np.log2(n_values)) / (n_values[0] * np.log2(n_values[0]))
        n2_fit = (n_values ** 2) / (n_values[0] ** 2)
        
        for i, result in enumerate(self.results):
            print(f"{result['n']:>6} {result['time_ms']:>12.2f} {nlogn_fit[i]:>12.2f} "
                  f"{n2_fit[i]:>12.2f} {result['memory_mb']:>15.2f}")
        
        print("-"*80)
        
        # Print R² values
        analysis = self.analyze_complexity()
        print(f"\nCorrelation with O(n log n): R² = {analysis['r_squared_nlogn']:.4f}")
        print(f"Correlation with O(n²):      R² = {analysis['r_squared_n_squared']:.4f}")
        print(f"Correlation with O(n):       R² = {analysis['r_squared_linear']:.4f}")
        print("="*80 + "\n")


class HolographicMemoryBenchmark:
    """
    Benchmark holographic memory system
    
    Verifies Theorem 2: Information capacity claims
    """
    
    def __init__(self):
        self.results = []
        
    def test_reconstruction_fidelity(
        self,
        pattern_dim: int = 512,
        num_patterns: List[int] = [1, 10, 50, 100, 500],
    ) -> List[Dict]:
        """
        Test reconstruction fidelity with varying number of patterns
        
        Args:
            pattern_dim: Pattern dimension
            num_patterns: List of pattern counts to test
            
        Returns:
            List of results
        """
        print("Testing Holographic Memory Reconstruction Fidelity...")
        print()
        
        results = []
        
        for n_patterns in num_patterns:
            print(f"Testing {n_patterns} patterns...")
            
            # Create memory
            memory = HolographicMemory(
                pattern_dim=pattern_dim,
                hologram_dim=pattern_dim * 2,
                capacity=max(num_patterns),
            )
            
            # Generate and encode patterns
            patterns = [torch.randn(pattern_dim) for _ in range(n_patterns)]
            for pattern in patterns:
                memory.encode(pattern)
                
            # Test reconstruction of each pattern
            fidelities = []
            for pattern in patterns:
                fidelity = memory.get_reconstruction_fidelity(pattern)
                fidelities.append(fidelity)
                
            # Compute statistics
            result = {
                'num_patterns': n_patterns,
                'mean_fidelity': np.mean(fidelities),
                'std_fidelity': np.std(fidelities),
                'min_fidelity': np.min(fidelities),
                'capacity_utilization': memory.get_capacity_utilization(),
                'theoretical_capacity': memory.get_theoretical_capacity(),
            }
            
            results.append(result)
            print(f"  Mean Fidelity: {result['mean_fidelity']:.4f}")
            print(f"  Capacity Utilization: {result['capacity_utilization']:.2%}")
            print()
            
        self.results = results
        return results
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot holographic memory results
        """
        if not self.results:
            raise ValueError("No results to plot.")
            
        num_patterns = [r['num_patterns'] for r in self.results]
        mean_fidelity = [r['mean_fidelity'] for r in self.results]
        std_fidelity = [r['std_fidelity'] for r in self.results]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(num_patterns, mean_fidelity, yerr=std_fidelity, 
                    marker='o', capsize=5, linewidth=2, markersize=8)
        plt.xlabel('Number of Stored Patterns', fontsize=12)
        plt.ylabel('Reconstruction Fidelity', fontsize=12)
        plt.title('Holographic Memory Reconstruction Quality', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


class GradientStabilityTest:
    """
    Test gradient stability during training
    
    Verifies Theorem 1: Stable frequency gradients
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def test_gradient_stability(
        self,
        num_iterations: int = 1000,
        input_dim: int = 512,
    ) -> Dict[str, any]:
        """
        Test gradient stability over training iterations
        
        Returns:
            Stability statistics
        """
        print("Testing Gradient Stability...")
        print(f"Running {num_iterations} iterations...")
        print()
        
        # Create model
        model = ResonanceNet(
            input_dim=input_dim,
            num_frequencies=64,
            num_layers=4,
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Track gradients
        magnitude_grads = []
        phase_grads = []
        total_norms = []
        
        for i in range(num_iterations):
            # Random input
            x = torch.randn(8, 64, input_dim).to(self.device)
            target = torch.randn(8, 64, input_dim).to(self.device)
            
            # Forward and backward
            optimizer.zero_grad()
            output = model(x, use_memory=False)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            
            # Collect gradient statistics
            stats = model.get_gradient_stats()
            if 'layer_0_magnitude_grad_norm' in stats:
                magnitude_grads.append(stats['layer_0_magnitude_grad_norm'])
            if 'layer_0_phase_grad_norm' in stats:
                phase_grads.append(stats['layer_0_phase_grad_norm'])
                
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            total_norms.append(total_norm.item())
            
            optimizer.step()
            
            if (i + 1) % 200 == 0:
                print(f"  Iteration {i+1}: max_grad_norm = {max(total_norms[-200:]):.4f}")
                
        # Analyze stability
        gradient_explosions = sum(1 for g in total_norms if g > 100)
        convergence = loss.item() < 1.0
        
        results = {
            'max_gradient_norm': max(total_norms),
            'mean_gradient_norm': np.mean(total_norms),
            'gradient_explosions': gradient_explosions,
            'convergence_achieved': convergence,
            'final_loss': loss.item(),
            'magnitude_grads': magnitude_grads,
            'phase_grads': phase_grads,
            'total_norms': total_norms,
        }
        
        print()
        print(f"Maximum gradient norm: {results['max_gradient_norm']:.4f}")
        print(f"Gradient explosion events: {results['gradient_explosions']}")
        print(f"Convergence achieved: {results['convergence_achieved']}")
        print(f"Final loss: {results['final_loss']:.6f}")
        print()
        
        return results
