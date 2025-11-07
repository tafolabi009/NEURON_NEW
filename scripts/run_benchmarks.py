#!/usr/bin/env python3
"""
Run benchmarks on L40 GPU

Usage:
    python scripts/run_benchmarks.py --all
    python scripts/run_benchmarks.py --long-context
    python scripts/run_benchmarks.py --multimodal
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from resonance_nn.benchmark.l40_benchmark import L40GPUBenchmark


def parse_args():
    parser = argparse.ArgumentParser(description='Run L40 GPU Benchmarks')
    
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    parser.add_argument('--long-context', action='store_true',
                       help='Run long context benchmark')
    parser.add_argument('--multimodal', action='store_true',
                       help='Run multimodal benchmark')
    parser.add_argument('--vocabulary', action='store_true',
                       help='Run vocabulary scaling benchmark')
    parser.add_argument('--throughput', action='store_true',
                       help='Run throughput benchmark')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create benchmark
    benchmark = L40GPUBenchmark(output_dir=args.output_dir)
    
    # Run benchmarks
    if args.all:
        benchmark.run_all_benchmarks()
    else:
        if args.throughput:
            benchmark.benchmark_throughput()
        if args.vocabulary:
            benchmark.benchmark_vocabulary_scaling()
        if args.long_context:
            benchmark.benchmark_long_context()
        if args.multimodal:
            benchmark.benchmark_multimodal()
        
        if not any([args.throughput, args.vocabulary, args.long_context, args.multimodal]):
            print("No benchmark selected. Use --all or specify individual benchmarks.")
            print("Run with --help for usage information.")


if __name__ == '__main__':
    main()
