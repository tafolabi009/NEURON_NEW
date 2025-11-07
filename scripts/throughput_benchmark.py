"""
Throughput Optimization Benchmark
Tests OptimizedResonanceLayer vs standard layer
Target: 1000+ samples/sec (up from 526)
"""

import torch
import time
import numpy as np
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from resonance_nn.layers.resonance import (
    ResonanceLayer,
    WarmupWrapper,
    FusedResonanceStack,
    optimize_resonance_model,
    create_optimized_resonance_layer,
)
from resonance_nn import ResonanceNet


def benchmark_layer(
    layer,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_iterations: int = 100,
    warmup: int = 10,
    device: str = 'cuda',
) -> Dict:
    """Benchmark a single layer"""
    
    layer = layer.to(device)
    layer.eval()
    
    # Generate input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = layer(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = layer(x)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    throughput_samples = batch_size / mean_time
    throughput_tokens = (batch_size * seq_len) / mean_time
    
    return {
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_samples_per_sec': throughput_samples,
        'throughput_tokens_per_sec': throughput_tokens,
    }


def compare_layers():
    """Compare standard vs optimized layers"""
    print("\n" + "="*80)
    print("LAYER COMPARISON: STANDARD VS OPTIMIZED")
    print("="*80)
    
    configs = [
        {'batch_size': 1, 'seq_len': 2048, 'hidden_dim': 768},
        {'batch_size': 8, 'seq_len': 2048, 'hidden_dim': 768},
        {'batch_size': 32, 'seq_len': 2048, 'hidden_dim': 768},
        {'batch_size': 8, 'seq_len': 8192, 'hidden_dim': 768},
    ]
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Config: batch={config['batch_size']}, seq={config['seq_len']}, dim={config['hidden_dim']}")
        print("="*80)
        
        # Standard layer
        print("\n[1] Standard ResonanceLayer")
        standard_layer = ResonanceLayer(
            input_dim=config['hidden_dim'],
            num_frequencies=64,
            optimize=False,
        )
        standard_results = benchmark_layer(
            standard_layer,
            config['batch_size'],
            config['seq_len'],
            config['hidden_dim'],
        )
        print(f"  Time: {standard_results['mean_time_ms']:.2f} ± {standard_results['std_time_ms']:.2f} ms")
        print(f"  Throughput: {standard_results['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Throughput: {standard_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
        
        # Optimized layer
        print("\n[2] Optimized ResonanceLayer")
        optimized_layer = ResonanceLayer(
            input_dim=config['hidden_dim'],
            num_frequencies=64,
            optimize=True,
        )
        optimized_results = benchmark_layer(
            optimized_layer,
            config['batch_size'],
            config['seq_len'],
            config['hidden_dim'],
        )
        print(f"  Time: {optimized_results['mean_time_ms']:.2f} ± {optimized_results['std_time_ms']:.2f} ms")
        print(f"  Throughput: {optimized_results['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Throughput: {optimized_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
        
        # Compiled layer with torch.compile
        print("\n[3] Compiled ResonanceLayer (torch.compile)")
        print("  Note: torch.compile() has issues with complex tensor operations")
        print("  Skipping compiled test for now...")
        # compiled_layer = ResonanceLayer(
        #     input_dim=config['hidden_dim'],
        #     num_frequencies=64,
        #     optimize=True,
        #     use_compile=True,
        # )
        # if hasattr(torch, 'compile'):
        #     compiled_layer = torch.compile(compiled_layer, mode='max-autotune')
        # compiled_results = benchmark_layer(
        #     compiled_layer,
        #     config['batch_size'],
        #     config['seq_len'],
        #     config['hidden_dim'],
        #     warmup=20,  # More warmup for compilation
        # )
        # print(f"  Time: {compiled_results['mean_time_ms']:.2f} ± {compiled_results['std_time_ms']:.2f} ms")
        # print(f"  Throughput: {compiled_results['throughput_samples_per_sec']:.1f} samples/sec")
        # print(f"  Throughput: {compiled_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
        
        # Speedup
        print("\n" + "-"*80)
        print("SPEEDUP ANALYSIS")
        print("-"*80)
        optimized_speedup = optimized_results['throughput_samples_per_sec'] / standard_results['throughput_samples_per_sec']
        # compiled_speedup = compiled_results['throughput_samples_per_sec'] / standard_results['throughput_samples_per_sec']
        
        print(f"Optimized vs Standard: {optimized_speedup:.2f}x")
        # print(f"Compiled vs Standard:  {compiled_speedup:.2f}x")


def benchmark_full_model():
    """Benchmark full model with optimizations"""
    print("\n" + "="*80)
    print("FULL MODEL THROUGHPUT BENCHMARK")
    print("="*80)
    print("Target: 1000+ samples/sec (current: 526)")
    
    batch_size = 32
    seq_len = 2048
    
    # Create standard model
    print("\n[1] Creating standard ResonanceNet...")
    standard_model = ResonanceNet(
        input_dim=768,
        num_frequencies=64,
        hidden_dim=768,
        num_layers=6,
    ).cuda()
    
    standard_params = sum(p.numel() for p in standard_model.parameters())
    print(f"Parameters: {standard_params:,}")
    
    # Benchmark standard model
    print(f"\nBenchmarking standard model (batch={batch_size}, seq={seq_len})...")
    standard_results = benchmark_model(standard_model, batch_size, seq_len)
    
    print(f"\nStandard Model Results:")
    print(f"  Time: {standard_results['mean_time_ms']:.2f} ± {standard_results['std_time_ms']:.2f} ms")
    print(f"  Throughput: {standard_results['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Throughput: {standard_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
    
    # Optimize model
    print("\n[2] Optimizing model with optimize_resonance_model()...")
    print("    (torch.compile() disabled due to complex tensor compatibility)")
    optimized_model = optimize_resonance_model(standard_model, use_compile=False)
    
    # Benchmark optimized model
    print(f"\nBenchmarking optimized model (batch={batch_size}, seq={seq_len})...")
    optimized_results = benchmark_model(optimized_model, batch_size, seq_len, warmup=20)
    
    print(f"\nOptimized Model Results:")
    print(f"  Time: {optimized_results['mean_time_ms']:.2f} ± {optimized_results['std_time_ms']:.2f} ms")
    print(f"  Throughput: {optimized_results['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Throughput: {optimized_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    speedup = optimized_results['throughput_samples_per_sec'] / standard_results['throughput_samples_per_sec']
    print(f"Speedup: {speedup:.2f}x")
    print(f"Previous throughput: 526 samples/sec")
    print(f"Current throughput:  {optimized_results['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"Target:              1000+ samples/sec")
    
    if optimized_results['throughput_samples_per_sec'] >= 1000:
        print("\n✓ TARGET ACHIEVED!")
    else:
        remaining = 1000 - optimized_results['throughput_samples_per_sec']
        print(f"\n✗ Need {remaining:.1f} more samples/sec to reach target")


def benchmark_model(model, batch_size: int, seq_len: int, warmup: int = 10) -> Dict:
    """Benchmark full model"""
    model.eval()
    
    x = torch.randn(batch_size, seq_len, model.input_dim, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model(x)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        'mean_time_ms': mean_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput_samples_per_sec': batch_size / mean_time,
        'throughput_tokens_per_sec': (batch_size * seq_len) / mean_time,
    }


def benchmark_fused_stack():
    """Benchmark FusedResonanceStack"""
    print("\n" + "="*80)
    print("FUSED STACK BENCHMARK")
    print("="*80)
    print("Testing multi-layer kernel fusion")
    
    batch_size = 8
    seq_len = 2048
    hidden_dim = 768
    num_layers = 6
    
    # Individual layers
    print(f"\n[1] Individual Optimized ResonanceLayers (x{num_layers})")
    individual_layers = torch.nn.Sequential(*[
        ResonanceLayer(hidden_dim, 64, optimize=True)
        for _ in range(num_layers)
    ]).cuda()
    
    individual_results = benchmark_layer(
        individual_layers,
        batch_size,
        seq_len,
        hidden_dim,
    )
    print(f"  Throughput: {individual_results['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Fused stack
    print(f"\n[2] FusedResonanceStack (kernel fusion)")
    fused_stack = FusedResonanceStack(
        input_dim=hidden_dim,
        num_frequencies=64,
        num_layers=num_layers,
    ).cuda()
    
    fused_results = benchmark_layer(
        fused_stack,
        batch_size,
        seq_len,
        hidden_dim,
        warmup=20,
    )
    print(f"  Throughput: {fused_results['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Speedup
    speedup = fused_results['throughput_samples_per_sec'] / individual_results['throughput_samples_per_sec']
    print(f"\nFusion speedup: {speedup:.2f}x")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Throughput Optimization Benchmark')
    parser.add_argument('--test', choices=['layers', 'model', 'fused', 'all'], default='all')
    
    args = parser.parse_args()
    
    if args.test in ['layers', 'all']:
        compare_layers()
    
    if args.test in ['fused', 'all']:
        benchmark_fused_stack()
    
    if args.test in ['model', 'all']:
        benchmark_full_model()


if __name__ == '__main__':
    main()
