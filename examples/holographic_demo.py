"""
Holographic Memory Demonstration
Reproduces Table 2 from the paper
"""

import torch
import numpy as np
from resonance_nn.layers.holographic import HolographicMemory
from resonance_nn.benchmark import HolographicMemoryBenchmark

def generate_test_patterns(pattern_dim: int, num_patterns: int):
    """Generate different types of test patterns"""
    patterns = []
    
    # Random signal
    patterns.append({
        'name': 'Random Signal',
        'data': torch.randn(pattern_dim)
    })
    
    # Sinusoidal
    t = torch.linspace(0, 4 * np.pi, pattern_dim)
    patterns.append({
        'name': 'Sinusoidal',
        'data': torch.sin(t)
    })
    
    # Mixed frequency
    mixed = torch.sin(t) + 0.5 * torch.sin(3 * t) + 0.25 * torch.sin(7 * t)
    patterns.append({
        'name': 'Mixed Frequency',
        'data': mixed
    })
    
    return patterns


def test_single_pattern(pattern_name: str, pattern_data: torch.Tensor):
    """Test holographic encoding/decoding for a single pattern"""
    pattern_dim = pattern_data.shape[0]
    
    # Create holographic memory
    memory = HolographicMemory(
        pattern_dim=pattern_dim,
        hologram_dim=pattern_dim * 2,
        capacity=1000,
    )
    
    # Encode pattern
    memory.encode(pattern_data.unsqueeze(0))
    
    # Reconstruct
    reconstructed = memory.reconstruct()
    
    # Compute metrics
    reconstruction_error = torch.mean((pattern_data - reconstructed) ** 2).item()
    
    # Mutual information (approximation using correlation)
    correlation = torch.corrcoef(torch.stack([pattern_data, reconstructed]))[0, 1].item()
    mutual_info = -0.5 * np.log(1 - correlation ** 2 + 1e-8)  # Approximation
    
    # Compression ratio
    original_size = pattern_data.numel() * pattern_data.element_size()
    hologram_size = memory.hologram.numel() * memory.hologram.element_size()
    compression = original_size / hologram_size
    
    return {
        'reconstruction_error': reconstruction_error,
        'mutual_info': mutual_info,
        'compression': compression,
    }


def main():
    print("=" * 80)
    print("HOLOGRAPHIC MEMORY DEMONSTRATION")
    print("Reproducing Table 2: Holographic memory performance")
    print("=" * 80)
    print()
    
    pattern_dim = 512
    
    # Generate test patterns
    print("Generating test patterns...")
    patterns = generate_test_patterns(pattern_dim, 3)
    print()
    
    # Test each pattern
    print("Testing holographic encoding and reconstruction...")
    print("-" * 80)
    print(f"{'Test Pattern':<20} {'Recon Error':>15} {'Mutual Info':>15} {'Compression':>15}")
    print("-" * 80)
    
    for pattern in patterns:
        results = test_single_pattern(pattern['name'], pattern['data'])
        print(f"{pattern['name']:<20} {results['reconstruction_error']:>15.3f} "
              f"{results['mutual_info']:>12.2f} bits {results['compression']:>14.1f}x")
    
    print("-" * 80)
    print()
    
    # Test capacity scaling
    print("Testing holographic capacity with multiple patterns...")
    print("-" * 80)
    
    benchmark = HolographicMemoryBenchmark()
    results = benchmark.test_reconstruction_fidelity(
        pattern_dim=pattern_dim,
        num_patterns=[1, 10, 50, 100, 200],
    )
    
    print("\nCapacity Scaling Results:")
    print("-" * 80)
    print(f"{'Num Patterns':<15} {'Mean Fidelity':>20} {'Capacity Used':>20}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['num_patterns']:<15} {result['mean_fidelity']:>20.4f} "
              f"{result['capacity_utilization']:>19.2%}")
    
    print("-" * 80)
    print()
    
    # Plot results
    print("Generating holographic memory plot...")
    benchmark.plot_results('holographic_memory_results.png')
    print("Plot saved as 'holographic_memory_results.png'")
    print()
    
    # Test associative recall
    print("Testing associative recall...")
    print("-" * 80)
    
    memory = HolographicMemory(
        pattern_dim=pattern_dim,
        hologram_dim=pattern_dim * 2,
        capacity=100,
    )
    
    # Store multiple patterns
    stored_patterns = [torch.randn(pattern_dim) for _ in range(10)]
    for pattern in stored_patterns:
        memory.encode(pattern.unsqueeze(0))
    
    # Query with partial pattern
    query = stored_patterns[0] + torch.randn(pattern_dim) * 0.1  # Add noise
    retrieved = memory.reconstruct(query.unsqueeze(0))
    
    # Measure similarity
    similarity = torch.cosine_similarity(stored_patterns[0], retrieved, dim=0).item()
    print(f"Query similarity to original: {similarity:.4f}")
    print(f"Associative recall successful: {similarity > 0.8}")
    print()
    
    # Test information capacity theorem
    print("Verifying Information Capacity Theorem...")
    print("-" * 80)
    
    theoretical_capacity = memory.get_theoretical_capacity()
    print(f"Hologram dimension (A): {memory.hologram_dim}")
    print(f"Wavelength (λ): {memory.wavelength}")
    print(f"SNR: {memory.snr}")
    print(f"Theoretical capacity: {theoretical_capacity:.2f} patterns")
    print(f"Current stored patterns: {memory.num_patterns.item()}")
    print(f"Capacity utilization: {memory.get_capacity_utilization():.2%}")
    print()
    
    print("=" * 80)
    print("Holographic memory demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
