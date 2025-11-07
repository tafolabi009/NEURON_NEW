"""
Comparative Benchmark: Resonance Net vs Llama 2 vs GPT-4 (via API)

Tests on same hardware for fair comparison
"""

import torch
import time
import numpy as np
import json
from typing import Dict, List, Optional
import psutil
import GPUtil


class ComparativeBenchmark:
    """
    Compare Resonance Networks against SOTA models
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = {}
        
    def benchmark_resonance_net(
        self,
        model,
        batch_size: int = 8,
        seq_len: int = 2048,
        num_iterations: int = 100,
    ) -> Dict:
        """Benchmark Resonance Network"""
        print("\n" + "="*80)
        print("BENCHMARKING RESONANCE NEURAL NETWORK")
        print("="*80)
        
        model = model.to(self.device)
        model.eval()
        
        # Generate dummy input
        x = torch.randn(batch_size, seq_len, model.input_dim, device=self.device)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"Running {num_iterations} iterations...")
        times = []
        memory_used = []
        
        for i in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            if torch.cuda.is_available():
                end_mem = torch.cuda.max_memory_allocated()
                memory_used.append((end_mem - start_mem) / 1024**2)  # MB
        
        results = {
            'model': 'Resonance Neural Network',
            'batch_size': batch_size,
            'seq_len': seq_len,
            'avg_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'throughput_samples_per_sec': batch_size / np.mean(times),
            'throughput_tokens_per_sec': (batch_size * seq_len) / np.mean(times),
            'avg_memory_mb': np.mean(memory_used) if memory_used else 0,
            'parameters': sum(p.numel() for p in model.parameters()),
        }
        
        print(f"\nResults:")
        print(f"  Average time: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Throughput: {results['throughput_tokens_per_sec']:.0f} tokens/sec")
        print(f"  Memory: {results['avg_memory_mb']:.1f} MB")
        print(f"  Parameters: {results['parameters']:,}")
        
        return results
    
    def benchmark_llama2(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        batch_size: int = 8,
        seq_len: int = 2048,
        num_iterations: int = 100,
    ) -> Dict:
        """
        Benchmark Llama 2 model
        Requires transformers library and model access
        """
        print("\n" + "="*80)
        print(f"BENCHMARKING LLAMA 2: {model_name}")
        print("="*80)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import login
            import os
            
            # Login with token from environment
            hf_token = os.environ.get('HF_TOKEN', None)
            if hf_token:
                try:
                    login(token=hf_token)
                except:
                    pass  # Already logged in
            
            print("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                token=hf_token,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model.eval()
            
            # Generate dummy input
            dummy_text = "This is a test " * (seq_len // 5)
            inputs = tokenizer(
                [dummy_text] * batch_size,
                return_tensors='pt',
                max_length=seq_len,
                truncation=True,
                padding='max_length',
            ).to(self.device)
            
            # Warmup
            print("Warming up...")
            for _ in range(10):
                with torch.no_grad():
                    _ = model(**inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            print(f"Running {num_iterations} iterations...")
            times = []
            memory_used = []
            
            for i in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    start_mem = torch.cuda.memory_allocated()
                
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    output = model(**inputs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                
                if torch.cuda.is_available():
                    end_mem = torch.cuda.max_memory_allocated()
                    memory_used.append((end_mem - start_mem) / 1024**2)
            
            results = {
                'model': f'Llama 2 ({model_name})',
                'batch_size': batch_size,
                'seq_len': seq_len,
                'avg_time_ms': np.mean(times) * 1000,
                'std_time_ms': np.std(times) * 1000,
                'throughput_samples_per_sec': batch_size / np.mean(times),
                'throughput_tokens_per_sec': (batch_size * seq_len) / np.mean(times),
                'avg_memory_mb': np.mean(memory_used) if memory_used else 0,
                'parameters': sum(p.numel() for p in model.parameters()),
            }
            
            print(f"\nResults:")
            print(f"  Average time: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
            print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
            print(f"  Throughput: {results['throughput_tokens_per_sec']:.0f} tokens/sec")
            print(f"  Memory: {results['avg_memory_mb']:.1f} MB")
            print(f"  Parameters: {results['parameters']:,}")
            
            return results
            
        except Exception as e:
            print(f"Error benchmarking Llama 2: {e}")
            print("Make sure you have transformers installed and model access")
            return None
    
    def benchmark_gpt4_api(
        self,
        api_key: str,
        batch_size: int = 8,
        seq_len: int = 2048,
        num_iterations: int = 100,
    ) -> Dict:
        """
        Benchmark GPT-4 via API
        Note: This measures API latency, not pure inference time
        """
        print("\n" + "="*80)
        print("BENCHMARKING GPT-4 (API)")
        print("="*80)
        print("Note: Results include network latency")
        
        try:
            import openai
            openai.api_key = api_key
            
            # Generate test prompt
            test_prompt = "This is a test prompt. " * (seq_len // 10)
            
            print(f"Running {num_iterations} API calls...")
            times = []
            
            for i in range(num_iterations):
                start_time = time.perf_counter()
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=100,
                )
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_iterations}")
            
            results = {
                'model': 'GPT-4 (API)',
                'batch_size': 1,  # API doesn't support batching
                'seq_len': seq_len,
                'avg_time_ms': np.mean(times) * 1000,
                'std_time_ms': np.std(times) * 1000,
                'throughput_samples_per_sec': 1 / np.mean(times),
                'throughput_tokens_per_sec': seq_len / np.mean(times),
                'avg_memory_mb': 0,  # N/A for API
                'parameters': 1_760_000_000_000,  # Estimated GPT-4 size
                'note': 'Includes network latency',
            }
            
            print(f"\nResults:")
            print(f"  Average time: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
            print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
            print(f"  (Includes network latency)")
            
            return results
            
        except Exception as e:
            print(f"Error benchmarking GPT-4: {e}")
            print("Make sure you have openai installed and valid API key")
            return None
    
    def run_comparative_analysis(
        self,
        resonance_model,
        llama_model_name: Optional[str] = None,
        gpt4_api_key: Optional[str] = None,
        batch_sizes: List[int] = [1, 8, 32],
        seq_lengths: List[int] = [512, 2048, 8192],
    ):
        """Run comprehensive comparison"""
        print("\n" + "="*80)
        print("COMPARATIVE BENCHMARK ANALYSIS")
        print("="*80)
        
        all_results = []
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"\n\nConfiguration: batch_size={batch_size}, seq_len={seq_len}")
                print("-" * 80)
                
                # Benchmark Resonance Net
                try:
                    results = self.benchmark_resonance_net(
                        resonance_model,
                        batch_size=batch_size,
                        seq_len=seq_len,
                    )
                    all_results.append(results)
                except Exception as e:
                    print(f"Error with Resonance Net: {e}")
                
                # Benchmark Llama 2 (if provided)
                if llama_model_name:
                    try:
                        results = self.benchmark_llama2(
                            llama_model_name,
                            batch_size=batch_size,
                            seq_len=seq_len,
                        )
                        if results:
                            all_results.append(results)
                    except Exception as e:
                        print(f"Error with Llama 2: {e}")
                
                # Benchmark GPT-4 (if API key provided)
                # Note: Only run once since API doesn't batch
                if gpt4_api_key and batch_size == 1:
                    try:
                        results = self.benchmark_gpt4_api(
                            gpt4_api_key,
                            seq_len=seq_len,
                        )
                        if results:
                            all_results.append(results)
                    except Exception as e:
                        print(f"Error with GPT-4: {e}")
        
        # Save results
        output_file = 'comparative_benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}")
        
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: List[Dict]):
        """Print comparison summary"""
        print("\n\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Group by configuration
        configs = {}
        for r in results:
            key = (r['batch_size'], r['seq_len'])
            if key not in configs:
                configs[key] = []
            configs[key].append(r)
        
        for config, models in configs.items():
            batch_size, seq_len = config
            print(f"\nConfiguration: batch_size={batch_size}, seq_len={seq_len}")
            print("-" * 80)
            
            # Sort by throughput
            models_sorted = sorted(models, key=lambda x: x['throughput_tokens_per_sec'], reverse=True)
            
            print(f"{'Model':<40} {'Time (ms)':<15} {'Throughput (tok/s)':<20} {'Memory (MB)':<15}")
            print("-" * 80)
            
            for m in models_sorted:
                print(f"{m['model']:<40} "
                      f"{m['avg_time_ms']:>8.2f} ± {m['std_time_ms']:<4.2f} "
                      f"{m['throughput_tokens_per_sec']:>18,.0f} "
                      f"{m['avg_memory_mb']:>14,.1f}")


def main():
    """Main benchmark script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comparative Benchmark')
    parser.add_argument('--llama-model', type=str, help='Llama 2 model name')
    parser.add_argument('--gpt4-api-key', type=str, help='GPT-4 API key')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 32])
    parser.add_argument('--seq-lengths', nargs='+', type=int, default=[512, 2048, 8192])
    
    args = parser.parse_args()
    
    # Create Resonance model
    from resonance_nn import ResonanceNet
    
    print("Creating Resonance Neural Network...")
    resonance_model = ResonanceNet(
        input_dim=768,
        num_frequencies=64,
        hidden_dim=768,
        num_layers=6,
    )
    
    # Run comparison
    benchmark = ComparativeBenchmark()
    results = benchmark.run_comparative_analysis(
        resonance_model=resonance_model,
        llama_model_name=args.llama_model,
        gpt4_api_key=args.gpt4_api_key,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
    )


if __name__ == '__main__':
    main()
