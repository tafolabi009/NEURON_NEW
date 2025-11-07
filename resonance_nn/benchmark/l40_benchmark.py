"""
Extended Benchmarking for L40 GPU
Tests long context, multimodal, and large vocabulary capabilities
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class L40GPUBenchmark:
    """
    Comprehensive benchmarks optimized for NVIDIA L40 GPU
    
    Tests:
    - Long context performance (up to 300K tokens)
    - Multimodal fusion
    - Large vocabulary embedding
    - Memory efficiency
    - Throughput
    """
    
    def __init__(self, device: str = 'cuda', output_dir: str = './benchmark_results'):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check GPU capabilities
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name}")
            print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"Compute Capability: {props.major}.{props.minor}")
        
        self.results = {}
    
    def benchmark_long_context(
        self,
        context_lengths: List[int] = [4096, 16384, 65536, 131072, 262144],
        batch_size: int = 1,
        embed_dim: int = 768,
    ):
        """
        Benchmark long context processing
        """
        print("\n" + "="*80)
        print("LONG CONTEXT BENCHMARK")
        print("="*80)
        
        from resonance_nn.models.long_context import LongContextResonanceNet
        
        results = []
        
        for seq_len in context_lengths:
            print(f"\nTesting context length: {seq_len:,} tokens")
            
            try:
                # Create model
                model = LongContextResonanceNet(
                    input_dim=embed_dim,
                    chunk_size=4096,
                    overlap=512,
                    max_chunks=(seq_len // 4096) + 10,
                ).to(self.device)
                model.eval()
                
                # Create input
                x = torch.randn(batch_size, seq_len, embed_dim).to(self.device)
                
                # Warmup
                with torch.no_grad():
                    _ = model(x, use_memory=False, store_to_memory=False)
                
                # Benchmark
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                start = time.time()
                with torch.no_grad():
                    output = model(x, use_memory=False, store_to_memory=False)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                
                # Memory
                if self.device == 'cuda':
                    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                else:
                    memory_mb = 0
                
                # Tokens per second
                tokens_per_sec = seq_len / elapsed
                
                result = {
                    'seq_len': seq_len,
                    'time_ms': elapsed * 1000,
                    'memory_mb': memory_mb,
                    'tokens_per_sec': tokens_per_sec,
                }
                results.append(result)
                
                print(f"  Time: {elapsed*1000:.1f} ms")
                print(f"  Memory: {memory_mb:.1f} MB")
                print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
                
                # Cleanup
                del model, x, output
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"  FAILED: {e}")
                break
        
        self.results['long_context'] = results
        self._save_results('long_context', results)
        self._plot_long_context(results)
    
    def benchmark_multimodal(
        self,
        batch_size: int = 8,
    ):
        """
        Benchmark multimodal fusion
        """
        print("\n" + "="*80)
        print("MULTIMODAL FUSION BENCHMARK")
        print("="*80)
        
        from resonance_nn.multimodal.fusion import MultiModalResonanceFusion
        
        # Create model
        modality_dims = {
            'text': 768,
            'vision': 768,
            'audio': 512,
        }
        
        model = MultiModalResonanceFusion(
            modality_dims=modality_dims,
            hidden_dim=768,
            num_cross_modal_layers=4,
            num_classes=1000,
        ).to(self.device)
        model.eval()
        
        # Create inputs
        inputs = {
            'text': torch.randn(batch_size, 128, 768).to(self.device),
            'vision': torch.randn(batch_size, 196, 768).to(self.device),
            'audio': torch.randn(batch_size, 200, 512).to(self.device),
        }
        
        # Benchmark
        times = []
        for _ in range(10):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                output = model(inputs)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed * 1000)
        
        result = {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'batch_size': batch_size,
        }
        
        print(f"  Average time: {result['mean_time_ms']:.1f} ± {result['std_time_ms']:.1f} ms")
        print(f"  Samples/sec: {batch_size / (result['mean_time_ms'] / 1000):.1f}")
        
        self.results['multimodal'] = result
    
    def benchmark_vocabulary_scaling(
        self,
        vocab_sizes: List[int] = [50000, 100000, 200000, 500000, 1000000],
        embed_dim: int = 768,
        seq_len: int = 128,
        batch_size: int = 32,
    ):
        """
        Benchmark large vocabulary embedding
        """
        print("\n" + "="*80)
        print("VOCABULARY SCALING BENCHMARK")
        print("="*80)
        
        from resonance_nn.layers.embeddings import HierarchicalVocabularyEmbedding
        
        results = []
        
        for vocab_size in vocab_sizes:
            print(f"\nTesting vocabulary size: {vocab_size:,}")
            
            # Create embedding
            embedding = HierarchicalVocabularyEmbedding(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
            ).to(self.device)
            
            # Create input
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
            
            # Benchmark
            times = []
            for _ in range(10):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    _ = embedding(input_ids)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                times.append(elapsed * 1000)
            
            # Count parameters
            num_params = sum(p.numel() for p in embedding.parameters())
            memory_mb = num_params * 4 / 1024**2  # 4 bytes per float32
            
            result = {
                'vocab_size': vocab_size,
                'mean_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'num_params': num_params,
                'memory_mb': memory_mb,
            }
            results.append(result)
            
            print(f"  Time: {result['mean_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
            print(f"  Parameters: {num_params:,}")
            print(f"  Memory: {memory_mb:.1f} MB")
            
            del embedding, input_ids
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        self.results['vocabulary'] = results
        self._save_results('vocabulary', results)
        self._plot_vocabulary(results)
    
    def benchmark_throughput(
        self,
        model_configs: List[Dict] = None,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        seq_len: int = 512,
    ):
        """
        Benchmark throughput at different batch sizes
        """
        print("\n" + "="*80)
        print("THROUGHPUT BENCHMARK")
        print("="*80)
        
        from resonance_nn.models.resonance_net import ResonanceNet
        
        if model_configs is None:
            model_configs = [{
                'input_dim': 768,
                'num_frequencies': 64,
                'hidden_dim': 768,
                'num_layers': 12,
            }]
        
        results = []
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            model = ResonanceNet(**model_configs[0]).to(self.device)
            model.eval()
            
            x = torch.randn(batch_size, seq_len, model_configs[0]['input_dim']).to(self.device)
            
            # Warmup
            with torch.no_grad():
                _ = model(x, use_memory=False)
            
            # Benchmark
            times = []
            for _ in range(20):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    _ = model(x, use_memory=False)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                times.append(elapsed)
            
            mean_time = np.mean(times)
            throughput = batch_size / mean_time
            
            result = {
                'batch_size': batch_size,
                'mean_time_ms': mean_time * 1000,
                'throughput': throughput,
            }
            results.append(result)
            
            print(f"  Time: {mean_time*1000:.2f} ms")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            
            del model, x
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        self.results['throughput'] = results
        self._plot_throughput(results)
    
    def _save_results(self, name: str, results: List[Dict]):
        """Save results to JSON"""
        import json
        path = self.output_dir / f"{name}_results.json"
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {path}")
    
    def _plot_long_context(self, results: List[Dict]):
        """Plot long context results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        seq_lens = [r['seq_len'] for r in results]
        times = [r['time_ms'] for r in results]
        memory = [r['memory_mb'] for r in results]
        
        # Time plot
        ax1.plot(seq_lens, times, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Sequence Length', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Long Context Processing Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Memory plot
        ax2.plot(seq_lens, memory, 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Sequence Length', fontsize=12)
        ax2.set_ylabel('Memory (MB)', fontsize=12)
        ax2.set_title('Memory Usage', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        path = self.output_dir / 'long_context_benchmark.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {path}")
        plt.close()
    
    def _plot_vocabulary(self, results: List[Dict]):
        """Plot vocabulary scaling"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        vocab_sizes = [r['vocab_size'] for r in results]
        times = [r['mean_time_ms'] for r in results]
        memory = [r['memory_mb'] for r in results]
        
        ax1.plot(vocab_sizes, times, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Vocabulary Size', fontsize=12)
        ax1.set_ylabel('Lookup Time (ms)', fontsize=12)
        ax1.set_title('Vocabulary Embedding Lookup Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        ax2.plot(vocab_sizes, memory, 's-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Vocabulary Size', fontsize=12)
        ax2.set_ylabel('Memory (MB)', fontsize=12)
        ax2.set_title('Embedding Memory Usage', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        path = self.output_dir / 'vocabulary_benchmark.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {path}")
        plt.close()
    
    def _plot_throughput(self, results: List[Dict]):
        """Plot throughput results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        batch_sizes = [r['batch_size'] for r in results]
        throughput = [r['throughput'] for r in results]
        
        ax.plot(batch_sizes, throughput, 'o-', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Throughput (samples/sec)', fontsize=12)
        ax.set_title('Model Throughput vs Batch Size', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / 'throughput_benchmark.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {path}")
        plt.close()
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "="*80)
        print("RUNNING FULL BENCHMARK SUITE FOR L40 GPU")
        print("="*80)
        
        self.benchmark_throughput()
        self.benchmark_vocabulary_scaling()
        self.benchmark_long_context()
        self.benchmark_multimodal()
        
        print("\n" + "="*80)
        print("✓ ALL BENCHMARKS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)
