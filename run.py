"""
Main runner script for Resonance Neural Networks
Run all demos and benchmarks
"""

import argparse
import sys


def run_quickstart():
    """Run quick start example"""
    print("\n" + "="*80)
    print("RUNNING QUICK START")
    print("="*80 + "\n")
    from examples.quickstart import main
    main()


def run_complexity_verification():
    """Run complexity verification"""
    print("\n" + "="*80)
    print("RUNNING COMPLEXITY VERIFICATION")
    print("="*80 + "\n")
    from examples.verify_complexity import main
    main()


def run_holographic_demo():
    """Run holographic memory demo"""
    print("\n" + "="*80)
    print("RUNNING HOLOGRAPHIC MEMORY DEMO")
    print("="*80 + "\n")
    from examples.holographic_demo import main
    main()


def run_gradient_stability():
    """Run gradient stability verification"""
    print("\n" + "="*80)
    print("RUNNING GRADIENT STABILITY TEST")
    print("="*80 + "\n")
    from examples.gradient_stability import main
    main()


def run_sequence_modeling():
    """Run sequence modeling example"""
    print("\n" + "="*80)
    print("RUNNING SEQUENCE MODELING")
    print("="*80 + "\n")
    from examples.sequence_modeling import main
    main()


def run_all():
    """Run all examples"""
    print("\n" + "="*80)
    print("RUNNING ALL EXAMPLES AND BENCHMARKS")
    print("="*80 + "\n")
    
    try:
        run_quickstart()
        run_complexity_verification()
        run_holographic_demo()
        run_gradient_stability()
        # Skip sequence modeling in "all" mode as it takes longer
        print("\nNote: Skipping sequence modeling (use --sequence to run)")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Resonance Neural Networks - Examples and Benchmarks"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quickstart', 'complexity', 'holographic', 'gradient', 'sequence', 'all'],
        default='all',
        help='Which example/benchmark to run'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("RESONANCE NEURAL NETWORKS")
    print("Frequency-Domain Information Processing with Holographic Memory")
    print("="*80)
    
    if args.mode == 'quickstart':
        run_quickstart()
    elif args.mode == 'complexity':
        run_complexity_verification()
    elif args.mode == 'holographic':
        run_holographic_demo()
    elif args.mode == 'gradient':
        run_gradient_stability()
    elif args.mode == 'sequence':
        run_sequence_modeling()
    elif args.mode == 'all':
        run_all()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
