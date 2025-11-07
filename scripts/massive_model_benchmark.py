"""
Massive Model Builder and Benchmark
Build and test large-scale Resonance Neural Networks to leverage available compute

Genovo Technologies Research Team
Lead: Oluwatosin Afolabi (afolabi@genovotech.com)
"""

import torch
import torch.nn as nn
import time
import numpy as np
import json
from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from resonance_nn import ResonanceNet
from resonance_nn.layers.resonance import optimize_resonance_model, WarmupWrapper


class MassiveResonanceConfig:
    """Configurations for massive models"""
    
    # Small: ~50M parameters
    SMALL = {
        'name': 'Small-50M',
        'input_dim': 1024,
        'num_frequencies': 128,
        'hidden_dim': 1024,
        'num_layers': 12,
        'expected_params': 50_000_000,
    }
    
    # Medium: ~200M parameters
    MEDIUM = {
        'name': 'Medium-200M',
        'input_dim': 2048,
        'num_frequencies': 256,
        'hidden_dim': 2048,
        'num_layers': 16,
        'expected_params': 200_000_000,
    }
    
    # Large: ~500M parameters
    LARGE = {
        'name': 'Large-500M',
        'input_dim': 3072,
        'num_frequencies': 512,
        'hidden_dim': 3072,
        'num_layers': 20,
        'expected_params': 500_000_000,
    }
    
    # XLarge: ~1B parameters
    XLARGE = {
        'name': 'XLarge-1B',
        'input_dim': 4096,
        'num_frequencies': 768,
        'hidden_dim': 4096,
        'num_layers': 24,
        'expected_params': 1_000_000_000,
    }
    
    # XXLarge: ~3B parameters (if VRAM allows)
    XXLARGE = {
        'name': 'XXLarge-3B',
        'input_dim': 6144,
        'num_frequencies': 1024,
        'hidden_dim': 6144,
        'num_layers': 32,
        'expected_params': 3_000_000_000,
    }
    
    # Ultimate: ~7B parameters (Llama-scale)
    ULTIMATE = {
        'name': 'Ultimate-7B',
        'input_dim': 8192,
        'num_frequencies': 1536,
        'hidden_dim': 8192,
        'num_layers': 40,
        'expected_params': 7_000_000_000,
    }


def build_massive_model(config: Dict, device: str = 'cuda') -> nn.Module:
    """Build a massive resonance model"""
    print(f"\n{'='*80}")
    print(f"BUILDING {config['name']} MODEL")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Input dim:       {config['input_dim']}")
    print(f"  Frequencies:     {config['num_frequencies']}")
    print(f"  Hidden dim:      {config['hidden_dim']}")
    print(f"  Layers:          {config['num_layers']}")
    print(f"  Expected params: {config['expected_params']:,}")
    
    # Build model
    print("\nBuilding model...")
    model = ResonanceNet(
        input_dim=config['input_dim'],
        num_frequencies=config['num_frequencies'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✅ Model built successfully!")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Difference from expected: {abs(total_params - config['expected_params']):,}")
    
    # Move to device
    print(f"\nMoving to {device}...")
    model = model.to(device)
    
    # Check memory
    if device == 'cuda':
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory allocated: {memory_allocated:.2f} GB")
        print(f"  GPU Memory reserved:  {memory_reserved:.2f} GB")
    
    return model


def benchmark_massive_model(
    model: nn.Module,
    config: Dict,
    batch_sizes: List[int] = [1, 4, 8, 16],
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    num_iterations: int = 50,
    device: str = 'cuda',
) -> List[Dict]:
    """Benchmark massive model"""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING {config['name']}")
    print(f"{'='*80}")
    
    model.eval()
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print(f"\nConfig: batch={batch_size}, seq={seq_len}")
            print("-" * 80)
            
            # Check if this will fit in memory
            estimated_memory = (batch_size * seq_len * config['input_dim'] * 4) / 1024**3
            print(f"  Estimated input memory: {estimated_memory:.2f} GB")
            
            if device == 'cuda':
                available_memory = (torch.cuda.get_device_properties(0).total_memory - 
                                  torch.cuda.memory_allocated()) / 1024**3
                print(f"  Available GPU memory:   {available_memory:.2f} GB")
                
                if estimated_memory > available_memory * 0.5:
                    print(f"  ⚠️  Skipping (likely OOM)")
                    continue
            
            try:
                # Generate input
                x = torch.randn(batch_size, seq_len, config['input_dim'], device=device)
                
                # Warmup
                print("  Warming up...")
                for _ in range(5):
                    with torch.no_grad():
                        _ = model(x)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark
                print(f"  Running {num_iterations} iterations...")
                times = []
                
                for _ in range(num_iterations):
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    with torch.no_grad():
                        output = model(x)
                    
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    
                    end = time.perf_counter()
                    times.append(end - start)
                
                # Calculate stats
                mean_time = np.mean(times) * 1000  # ms
                std_time = np.std(times) * 1000
                throughput_samples = batch_size / (mean_time / 1000)
                throughput_tokens = (batch_size * seq_len) / (mean_time / 1000)
                
                # Memory
                if device == 'cuda':
                    memory_used = torch.cuda.max_memory_allocated() / 1024**3
                else:
                    memory_used = 0
                
                result = {
                    'model': config['name'],
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'mean_time_ms': mean_time,
                    'std_time_ms': std_time,
                    'throughput_samples_per_sec': throughput_samples,
                    'throughput_tokens_per_sec': throughput_tokens,
                    'memory_gb': memory_used,
                }
                results.append(result)
                
                print(f"  ✅ Results:")
                print(f"     Time: {mean_time:.2f} ± {std_time:.2f} ms")
                print(f"     Throughput: {throughput_samples:.1f} samples/sec")
                print(f"     Throughput: {throughput_tokens:,.0f} tokens/sec")
                print(f"     Memory: {memory_used:.2f} GB")
                
                # Cleanup
                del x, output
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ❌ OOM: Skipping this configuration")
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                else:
                    raise
    
    return results


def compare_with_llama2(
    resonance_config: Dict,
    batch_size: int = 1,
    seq_len: int = 2048,
    device: str = 'cuda',
):
    """Compare massive Resonance model with Llama 2 7B"""
    print(f"\n{'='*80}")
    print(f"COMPARING {resonance_config['name']} vs LLAMA 2 7B")
    print(f"{'='*80}")
    
    # Benchmark Resonance
    print("\n[1] Benchmarking Resonance Neural Network...")
    resonance_model = build_massive_model(resonance_config, device)
    resonance_results = benchmark_massive_model(
        resonance_model,
        resonance_config,
        batch_sizes=[batch_size],
        seq_lengths=[seq_len],
        num_iterations=100,
    )
    
    del resonance_model
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Benchmark Llama 2
    print("\n[2] Benchmarking Llama 2 7B...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        llama_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.float16,
            device_map='auto',
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        llama_model.eval()
        
        # Generate input
        dummy_text = "This is a test " * (seq_len // 5)
        inputs = tokenizer(
            [dummy_text] * batch_size,
            return_tensors='pt',
            max_length=seq_len,
            truncation=True,
            padding='max_length',
        ).to(device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = llama_model(**inputs)
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = llama_model(**inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        llama_time = np.mean(times) * 1000
        llama_std = np.std(times) * 1000
        llama_throughput = (batch_size * seq_len) / (llama_time / 1000)
        llama_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"\n✅ Llama 2 Results:")
        print(f"   Time: {llama_time:.2f} ± {llama_std:.2f} ms")
        print(f"   Throughput: {llama_throughput:,.0f} tokens/sec")
        print(f"   Memory: {llama_memory:.2f} GB")
        
        # Comparison
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        resonance_time = resonance_results[0]['mean_time_ms']
        resonance_throughput = resonance_results[0]['throughput_tokens_per_sec']
        
        speedup = llama_time / resonance_time
        throughput_ratio = resonance_throughput / llama_throughput
        
        print(f"Resonance vs Llama 2:")
        print(f"  Speed:      {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        print(f"  Throughput: {throughput_ratio:.2f}x {'higher' if throughput_ratio > 1 else 'lower'}")
        print(f"  Memory:     {resonance_results[0]['memory_gb']:.2f} GB vs {llama_memory:.2f} GB")
        
    except Exception as e:
        print(f"Error benchmarking Llama 2: {e}")


def main():
    """Main benchmark suite for massive models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Massive Model Benchmark')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'xlarge', 'xxlarge', 'ultimate'], 
                       default='medium', help='Model size')
    parser.add_argument('--compare-llama', action='store_true', help='Compare with Llama 2 7B')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--seq-lengths', nargs='+', type=int, default=[512, 1024, 2048, 4096])
    
    args = parser.parse_args()
    
    # Select config
    config_map = {
        'small': MassiveResonanceConfig.SMALL,
        'medium': MassiveResonanceConfig.MEDIUM,
        'large': MassiveResonanceConfig.LARGE,
        'xlarge': MassiveResonanceConfig.XLARGE,
        'xxlarge': MassiveResonanceConfig.XXLARGE,
        'ultimate': MassiveResonanceConfig.ULTIMATE,
    }
    config = config_map[args.size]
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"\n{'='*80}")
    print("MASSIVE MODEL BENCHMARK")
    print(f"{'='*80}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Model size: {args.size}")
    
    if args.compare_llama:
        compare_with_llama2(config, batch_size=1, seq_len=2048)
    else:
        # Build and benchmark
        model = build_massive_model(config)
        results = benchmark_massive_model(
            model,
            config,
            batch_sizes=args.batch_sizes,
            seq_lengths=args.seq_lengths,
        )
        
        # Save results
        output_file = f'massive_model_{args.size}_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}")
        
        # Print summary
        print("\nSUMMARY:")
        print("-" * 80)
        for r in results:
            print(f"batch={r['batch_size']:2d} seq={r['seq_len']:4d}: "
                  f"{r['mean_time_ms']:7.2f}ms  "
                  f"{r['throughput_tokens_per_sec']:10,.0f} tok/s  "
                  f"{r['memory_gb']:5.2f} GB")


if __name__ == '__main__':
    main()
