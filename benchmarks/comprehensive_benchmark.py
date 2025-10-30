"""
Comprehensive Benchmark Suite: NEURONSv2 vs Transformers
Compare speed, memory, and performance on real tasks
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json

from neurons.neuronsv2_unified import (
    NEURONSv2Model,
    NEURONSv2Config,
    create_neuronsv2_small,
    create_neuronsv2_medium,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark"""
    model_name: str
    task: str
    
    # Speed
    tokens_per_second: float
    forward_time_ms: float
    backward_time_ms: float
    
    # Memory
    peak_memory_mb: float
    param_count: int
    
    # Performance
    loss: float
    perplexity: float
    
    # Context
    sequence_length: int
    batch_size: int


class BenchmarkSuite:
    """Complete benchmark suite"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results: List[BenchmarkResult] = []
        
        print(f"Benchmark device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        sequence_lengths: List[int] = [512, 1024, 2048, 4096],
        batch_size: int = 8,
        num_iterations: int = 100,
    ) -> List[BenchmarkResult]:
        """Benchmark a model at different sequence lengths"""
        
        print(f"\n{'='*80}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*80}")
        
        model = model.to(self.device)
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count/1e6:.1f}M")
        
        results = []
        
        for seq_len in sequence_lengths:
            print(f"\nSequence length: {seq_len}")
            
            # Create dummy input
            input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=self.device)
            labels = input_ids.clone()
            
            # Warmup
            print("  Warming up...", end=" ")
            for _ in range(10):
                with torch.no_grad():
                    try:
                        outputs = model(input_ids, labels=labels)
                    except:
                        # Handle models with different interfaces
                        outputs = model(input_ids)
            print("done")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # Forward pass benchmark
            print("  Benchmarking forward...", end=" ")
            forward_times = []
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.time()
                    
                    try:
                        outputs = model(input_ids, labels=labels)
                    except:
                        outputs = model(input_ids)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    forward_times.append(time.time() - start)
            
            avg_forward_time = np.mean(forward_times[10:]) * 1000  # ms
            print(f"done ({avg_forward_time:.2f} ms)")
            
            # Backward pass benchmark
            print("  Benchmarking backward...", end=" ")
            backward_times = []
            
            model.train()
            for _ in range(num_iterations // 2):
                start = time.time()
                
                try:
                    outputs = model(input_ids, labels=labels)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                except:
                    outputs = model(input_ids)
                    loss = outputs.mean()
                
                loss.backward()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                backward_times.append(time.time() - start)
                
                # Zero gradients
                model.zero_grad()
            
            avg_backward_time = np.mean(backward_times[5:]) * 1000  # ms
            print(f"done ({avg_backward_time:.2f} ms)")
            model.eval()
            
            # Memory
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
            else:
                peak_memory = 0.0
            
            # Performance metrics
            with torch.no_grad():
                try:
                    outputs = model(input_ids, labels=labels)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                except:
                    outputs = model(input_ids)
                    loss = torch.nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        labels.view(-1)
                    )
            
            loss_val = loss.item()
            perplexity = np.exp(loss_val)
            
            # Tokens per second
            tokens_per_sec = (batch_size * seq_len) / (avg_forward_time / 1000)
            
            result = BenchmarkResult(
                model_name=model_name,
                task="language_modeling",
                tokens_per_second=tokens_per_sec,
                forward_time_ms=avg_forward_time,
                backward_time_ms=avg_backward_time,
                peak_memory_mb=peak_memory,
                param_count=param_count,
                loss=loss_val,
                perplexity=perplexity,
                sequence_length=seq_len,
                batch_size=batch_size,
            )
            
            results.append(result)
            self.results.append(result)
            
            print(f"  Tokens/sec: {tokens_per_sec:.0f}")
            print(f"  Peak memory: {peak_memory:.1f} MB")
            print(f"  Loss: {loss_val:.4f} | Perplexity: {perplexity:.2f}")
        
        return results
    
    def compare_models(
        self,
        sequence_lengths: List[int] = [512, 1024, 2048, 4096],
        batch_size: int = 8,
    ):
        """Compare NEURONSv2 with baseline transformer"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK: NEURONSv2 vs Transformer")
        print("="*80)
        
        # 1. NEURONSv2
        print("\n[1/2] NEURONSv2")
        neuronsv2 = create_neuronsv2_small()
        neuronsv2_results = self.benchmark_model(
            neuronsv2,
            "NEURONSv2-Small",
            sequence_lengths,
            batch_size,
        )
        
        # 2. Baseline Transformer (GPT-2 style)
        print("\n[2/2] Baseline Transformer")
        transformer = self._create_baseline_transformer()
        transformer_results = self.benchmark_model(
            transformer,
            "Transformer-Small",
            sequence_lengths,
            batch_size,
        )
        
        # Generate comparison report
        self.generate_report(neuronsv2_results, transformer_results)
    
    def _create_baseline_transformer(self) -> nn.Module:
        """Create baseline transformer for comparison"""
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
        class BaselineTransformer(nn.Module):
            def __init__(
                self,
                vocab_size: int = 50257,
                hidden_size: int = 768,
                num_layers: int = 12,
                num_heads: int = 12,
                max_seq_length: int = 2048,
            ):
                super().__init__()
                
                self.token_embedding = nn.Embedding(vocab_size, hidden_size)
                self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
                
                encoder_layer = TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True,
                )
                
                self.transformer = TransformerEncoder(encoder_layer, num_layers)
                self.ln_f = nn.LayerNorm(hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
                
                # Tie weights
                self.lm_head.weight = self.token_embedding.weight
            
            def forward(self, input_ids, labels=None):
                batch_size, seq_len = input_ids.shape
                
                # Embeddings
                token_emb = self.token_embedding(input_ids)
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                pos_emb = self.position_embedding(pos_ids)
                
                x = token_emb + pos_emb
                
                # Causal mask
                mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
                
                # Transformer
                x = self.transformer(x, mask=mask, is_causal=True)
                x = self.ln_f(x)
                
                # LM head
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                
                return {'logits': logits, 'loss': loss}
        
        return BaselineTransformer()
    
    def generate_report(
        self,
        neuronsv2_results: List[BenchmarkResult],
        transformer_results: List[BenchmarkResult],
    ):
        """Generate comparison report"""
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        print("\n{:<20} {:<15} {:<15} {:<15} {:<15}".format(
            "Sequence Length", "Tokens/sec", "Memory (MB)", "Forward (ms)", "Backward (ms)"
        ))
        print("-"*80)
        
        for nr, tr in zip(neuronsv2_results, transformer_results):
            seq_len = nr.sequence_length
            
            print(f"\nSeq Length: {seq_len}")
            
            print(f"  NEURONSv2:    {nr.tokens_per_second:>10.0f}  "
                  f"{nr.peak_memory_mb:>10.1f}      "
                  f"{nr.forward_time_ms:>10.2f}      "
                  f"{nr.backward_time_ms:>10.2f}")
            
            print(f"  Transformer:  {tr.tokens_per_second:>10.0f}  "
                  f"{tr.peak_memory_mb:>10.1f}      "
                  f"{tr.forward_time_ms:>10.2f}      "
                  f"{tr.backward_time_ms:>10.2f}")
            
            # Speedup
            speedup = nr.tokens_per_second / tr.tokens_per_second
            memory_ratio = nr.peak_memory_mb / tr.peak_memory_mb
            
            print(f"  Speedup:      {speedup:>10.2f}x "
                  f"{memory_ratio:>10.2f}x     "
                  f"{tr.forward_time_ms/nr.forward_time_ms:>10.2f}x     "
                  f"{tr.backward_time_ms/nr.backward_time_ms:>10.2f}x")
        
        # Complexity analysis
        print("\n" + "="*80)
        print("COMPLEXITY ANALYSIS")
        print("="*80)
        
        print("\nNEURONSv2:")
        print("  - Spectral processing: O(n log n)")
        print("  - Hierarchical compression: O(log n)")
        print("  - Overall: O(n log n)")
        
        print("\nTransformer:")
        print("  - Self-attention: O(n²)")
        print("  - Overall: O(n²)")
        
        # Extrapolation
        print("\n" + "="*80)
        print("LONG CONTEXT EXTRAPOLATION (200K tokens)")
        print("="*80)
        
        # Use last two points for extrapolation
        if len(neuronsv2_results) >= 2:
            n1, n2 = neuronsv2_results[-2].sequence_length, neuronsv2_results[-1].sequence_length
            t1_n, t2_n = neuronsv2_results[-2].forward_time_ms, neuronsv2_results[-1].forward_time_ms
            t1_t, t2_t = transformer_results[-2].forward_time_ms, transformer_results[-1].forward_time_ms
            
            # Fit to O(n log n) for NEURONSv2
            c_n = t2_n / (n2 * np.log(n2))
            
            # Fit to O(n²) for Transformer
            c_t = t2_t / (n2 ** 2)
            
            # Extrapolate to 200K
            n_long = 200000
            t_neuronsv2_200k = c_n * n_long * np.log(n_long)
            t_transformer_200k = c_t * (n_long ** 2)
            
            print(f"\nEstimated forward pass time (200K tokens):")
            print(f"  NEURONSv2:    {t_neuronsv2_200k/1000:.1f} seconds")
            print(f"  Transformer:  {t_transformer_200k/1000:.1f} seconds "
                  f"({t_transformer_200k/3600000:.1f} hours)")
            print(f"  Speedup:      {t_transformer_200k/t_neuronsv2_200k:.0f}x")
        
        # Save results
        self.save_results()
    
    def save_results(self, path: str = "benchmark_results.json"):
        """Save results to JSON"""
        results_dict = [
            {
                'model_name': r.model_name,
                'task': r.task,
                'tokens_per_second': r.tokens_per_second,
                'forward_time_ms': r.forward_time_ms,
                'backward_time_ms': r.backward_time_ms,
                'peak_memory_mb': r.peak_memory_mb,
                'param_count': r.param_count,
                'loss': r.loss,
                'perplexity': r.perplexity,
                'sequence_length': r.sequence_length,
                'batch_size': r.batch_size,
            }
            for r in self.results
        ]
        
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {path}")
    
    def plot_results(self):
        """Plot benchmark results"""
        if not self.results:
            print("No results to plot")
            return
        
        # Group by model
        models = {}
        for r in self.results:
            if r.model_name not in models:
                models[r.model_name] = []
            models[r.model_name].append(r)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Tokens/sec vs sequence length
        ax = axes[0, 0]
        for model_name, results in models.items():
            seq_lens = [r.sequence_length for r in results]
            tokens_per_sec = [r.tokens_per_second for r in results]
            ax.plot(seq_lens, tokens_per_sec, marker='o', label=model_name)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Tokens/sec')
        ax.set_title('Throughput vs Sequence Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Forward time vs sequence length
        ax = axes[0, 1]
        for model_name, results in models.items():
            seq_lens = [r.sequence_length for r in results]
            forward_times = [r.forward_time_ms for r in results]
            ax.plot(seq_lens, forward_times, marker='o', label=model_name)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Forward Time (ms)')
        ax.set_title('Forward Pass Time vs Sequence Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Memory vs sequence length
        ax = axes[1, 0]
        for model_name, results in models.items():
            seq_lens = [r.sequence_length for r in results]
            memory = [r.peak_memory_mb for r in results]
            ax.plot(seq_lens, memory, marker='o', label=model_name)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage vs Sequence Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Speedup
        ax = axes[1, 1]
        if len(models) == 2:
            model_names = list(models.keys())
            results1 = models[model_names[0]]
            results2 = models[model_names[1]]
            
            seq_lens = [r.sequence_length for r in results1]
            speedups = [r1.tokens_per_second / r2.tokens_per_second 
                       for r1, r2 in zip(results1, results2)]
            
            ax.plot(seq_lens, speedups, marker='o', color='green')
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Speedup')
            ax.set_title(f'Speedup: {model_names[0]} vs {model_names[1]}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        print("\nPlots saved to benchmark_results.png")


if __name__ == "__main__":
    print("="*80)
    print("NEURONSv2 Comprehensive Benchmark Suite")
    print("="*80)
    
    # Run benchmarks
    suite = BenchmarkSuite()
    
    # Test sequence lengths
    sequence_lengths = [512, 1024, 2048, 4096]
    
    # Compare models
    suite.compare_models(
        sequence_lengths=sequence_lengths,
        batch_size=8,
    )
    
    # Plot results
    # suite.plot_results()  # Uncomment if matplotlib available
    
    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)
