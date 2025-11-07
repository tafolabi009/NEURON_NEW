"""
Multimodal Benchmark with Warmup Phase
Fixes variance issues by warming up JIT compilation
"""

import torch
import time
import numpy as np
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from resonance_nn.multimodal.fusion import MultiModalResonanceFusion
from resonance_nn.layers.resonance import WarmupWrapper


def benchmark_multimodal_with_warmup(
    model_dim: int = 768,
    num_frequencies: int = 64,
    num_layers: int = 4,
    batch_size: int = 8,
    vision_seq_len: int = 196,  # 14x14 patches
    audio_seq_len: int = 512,
    text_seq_len: int = 256,
    num_iterations: int = 100,
    warmup_iterations: int = 20,
    device: str = 'cuda',
) -> Dict:
    """
    Benchmark multimodal fusion with proper warmup
    
    Target: ±5ms variance (down from ±50.1ms)
    """
    print("\n" + "="*80)
    print("MULTIMODAL BENCHMARK WITH WARMUP")
    print("="*80)
    print(f"Model dim: {model_dim}, Frequencies: {num_frequencies}, Layers: {num_layers}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence lengths - Vision: {vision_seq_len}, Audio: {audio_seq_len}, Text: {text_seq_len}")
    print(f"Warmup iterations: {warmup_iterations}, Benchmark iterations: {num_iterations}")
    
        # Create model
    print("\nCreating MultiModalResonanceFusion...")
    modality_dims = {
        'vision': model_dim,
        'audio': model_dim,
        'text': model_dim,
    }
    model = MultiModalResonanceFusion(
        modality_dims=modality_dims,
        hidden_dim=model_dim,
        num_cross_modal_layers=num_layers,
        num_frequencies=num_frequencies,
    ).to(device)
    
    # Wrap with warmup wrapper
    print("Wrapping with WarmupWrapper...")
    model = WarmupWrapper(model, warmup_iterations=warmup_iterations)
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Generate dummy inputs
    print("\nGenerating dummy inputs...")
    vision_input = torch.randn(batch_size, vision_seq_len, model_dim, device=device)
    audio_input = torch.randn(batch_size, audio_seq_len, model_dim, device=device)
    text_input = torch.randn(batch_size, text_seq_len, model_dim, device=device)
    
    # Package as dictionary
    modality_inputs = {
        'vision': vision_input,
        'audio': audio_input,
        'text': text_input,
    }
    
    # Warmup phase (will be handled by WarmupWrapper automatically)
    print(f"\nWarming up JIT compilation ({warmup_iterations} iterations)...")
    print("This reduces variance by stabilizing CUDA kernels...")
    
    with torch.no_grad():
        _ = model(modality_inputs)
    
    torch.cuda.synchronize()
    print("Warmup complete!")
    
    # Pre-allocate memory to reduce variance
    print("\nPre-allocating output buffer...")
    output_buffer = torch.empty(batch_size, model_dim, device=device)
    
    # Benchmark phase
    print(f"\nRunning benchmark ({num_iterations} iterations)...")
    times = []
    memory_used = []
    
    for i in range(num_iterations):
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        # Synchronize before timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model(modality_inputs)
        
        # Synchronize after computation
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        elapsed = (end_time - start_time) * 1000  # Convert to ms
        times.append(elapsed)
        
        # Memory tracking
        end_mem = torch.cuda.max_memory_allocated()
        memory_used.append((end_mem - start_mem) / 1024**2)  # MB
        
        if (i + 1) % 10 == 0:
            current_mean = np.mean(times)
            current_std = np.std(times)
            print(f"  Iteration {i+1}/{num_iterations}: "
                  f"{current_mean:.2f} ± {current_std:.2f} ms "
                  f"(target: ±5ms)")
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    
    mean_memory = np.mean(memory_used)
    
    # Calculate throughput
    throughput_samples = batch_size / (mean_time / 1000)
    total_tokens = batch_size * (vision_seq_len + audio_seq_len + text_seq_len)
    throughput_tokens = total_tokens / (mean_time / 1000)
    
    # Results
    results = {
        'batch_size': batch_size,
        'model_dim': model_dim,
        'num_frequencies': num_frequencies,
        'num_layers': num_layers,
        'vision_seq_len': vision_seq_len,
        'audio_seq_len': audio_seq_len,
        'text_seq_len': text_seq_len,
        'total_tokens': vision_seq_len + audio_seq_len + text_seq_len,
        'parameters': total_params,
        'warmup_iterations': warmup_iterations,
        'benchmark_iterations': num_iterations,
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'median_time_ms': median_time,
        'mean_memory_mb': mean_memory,
        'throughput_samples_per_sec': throughput_samples,
        'throughput_tokens_per_sec': throughput_tokens,
    }
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Mean time:        {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"Median time:      {median_time:.2f} ms")
    print(f"Min time:         {min_time:.2f} ms")
    print(f"Max time:         {max_time:.2f} ms")
    print(f"Variance:         ±{std_time:.2f} ms {'✓ PASS' if std_time <= 5 else '✗ FAIL (target: ±5ms)'}")
    print(f"\nThroughput:       {throughput_samples:.1f} samples/sec")
    print(f"Throughput:       {throughput_tokens:,.0f} tokens/sec")
    print(f"Memory:           {mean_memory:.1f} MB")
    print(f"Parameters:       {total_params:,}")
    
    # Improvement analysis
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    print("Previous results (without warmup):")
    print("  Variance: ±50.1 ms")
    print("  Throughput: ~334 samples/sec")
    print()
    print("Current results (with warmup):")
    print(f"  Variance: ±{std_time:.2f} ms")
    print(f"  Throughput: {throughput_samples:.1f} samples/sec")
    print()
    if std_time < 50.1:
        improvement = ((50.1 - std_time) / 50.1) * 100
        print(f"Variance improved by {improvement:.1f}%!")
    
    return results


def compare_with_without_warmup():
    """Compare performance with and without warmup"""
    print("\n" + "="*80)
    print("COMPARING WITH/WITHOUT WARMUP")
    print("="*80)
    
    # Test without warmup
    print("\n[1] Testing WITHOUT warmup wrapper...")
    model = MultiModalResonanceFusion(
        modality_dims={'vision': 768, 'audio': 768, 'text': 768},
        hidden_dim=768,
        num_cross_modal_layers=4,
        num_frequencies=64,
    ).cuda()
    model.eval()
    
    vision_input = torch.randn(8, 196, 768, device='cuda')
    audio_input = torch.randn(8, 512, 768, device='cuda')
    text_input = torch.randn(8, 256, 768, device='cuda')
    
    times_no_warmup = []
    for i in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(vision_input, audio_input, text_input)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_no_warmup.append((end - start) * 1000)
    
    std_no_warmup = np.std(times_no_warmup)
    
    # Test with warmup
    print("\n[2] Testing WITH warmup wrapper...")
    model_wrapped = WarmupWrapper(model, warmup_iterations=20)
    
    # Initial forward pass triggers warmup
    with torch.no_grad():
        _ = model_wrapped(vision_input, audio_input, text_input)
    
    times_with_warmup = []
    for i in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_wrapped(vision_input, audio_input, text_input)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_with_warmup.append((end - start) * 1000)
    
    std_with_warmup = np.std(times_with_warmup)
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"Without warmup: {np.mean(times_no_warmup):.2f} ± {std_no_warmup:.2f} ms")
    print(f"With warmup:    {np.mean(times_with_warmup):.2f} ± {std_with_warmup:.2f} ms")
    print()
    print(f"Variance reduction: {((std_no_warmup - std_with_warmup) / std_no_warmup) * 100:.1f}%")


def main():
    """Main benchmark entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multimodal Benchmark with Warmup')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model-dim', type=int, default=768)
    parser.add_argument('--num-frequencies', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmup-iterations', type=int, default=20)
    parser.add_argument('--compare', action='store_true', help='Compare with/without warmup')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_without_warmup()
    else:
        results = benchmark_multimodal_with_warmup(
            model_dim=args.model_dim,
            num_frequencies=args.num_frequencies,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
        )
        
        # Save results
        import json
        with open('multimodal_warmup_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to multimodal_warmup_benchmark.json")


if __name__ == '__main__':
    main()
