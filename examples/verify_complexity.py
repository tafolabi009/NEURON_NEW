"""
Verify O(n log n) Complexity Claims
Reproduces Table 1 from the paper
"""

import torch
from resonance_nn.benchmark import ComplexityBenchmark

def main():
    print("=" * 80)
    print("RESONANCE NEURAL NETWORKS - COMPLEXITY VERIFICATION")
    print("Reproducing Table 1: Computational complexity validation")
    print("=" * 80)
    print()
    
    # Create benchmark
    benchmark = ComplexityBenchmark(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run benchmark with sequence lengths from the paper
    sequence_lengths = [64, 128, 256, 512, 1024]
    
    results = benchmark.run(
        sequence_lengths=sequence_lengths,
        input_dim=512,
        num_frequencies=64,
        num_layers=4,
        batch_size=32,
    )
    
    # Print table
    benchmark.print_table()
    
    # Analyze complexity
    analysis = benchmark.analyze_complexity()
    
    print("\nCOMPLEXITY ANALYSIS")
    print("-" * 80)
    print(f"Correlation with O(n log n): R² = {analysis['r_squared_nlogn']:.4f}")
    print(f"Correlation with O(n²):      R² = {analysis['r_squared_n_squared']:.4f}")
    print()
    
    if analysis['r_squared_nlogn'] > 0.95:
        print("✓ VERIFIED: Complexity matches O(n log n) prediction")
    else:
        print("⚠ WARNING: Complexity may deviate from O(n log n)")
    print()
    
    # Plot results
    print("Generating complexity plot...")
    benchmark.plot_results('complexity_verification.png')
    print("Plot saved as 'complexity_verification.png'")
    print()
    
    print("=" * 80)
    print("Verification complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
