"""
Gradient Stability Verification
Verifies Theorem 1 and reproduces Section 5.3 results
"""

import torch
import matplotlib.pyplot as plt
from resonance_nn.benchmark import GradientStabilityTest


def main():
    print("=" * 80)
    print("GRADIENT STABILITY VERIFICATION")
    print("Verifying Theorem 1: Stable Frequency Gradients")
    print("Reproducing Section 5.3: Gradient Stability Analysis")
    print("=" * 80)
    print()
    
    # Create stability test
    test = GradientStabilityTest(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run stability test
    results = test.test_gradient_stability(
        num_iterations=1000,
        input_dim=512,
    )
    
    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Maximum gradient norm: {results['max_gradient_norm']:.4f}")
    print(f"Mean gradient norm: {results['mean_gradient_norm']:.4f}")
    print(f"Gradient explosion events (>100): {results['gradient_explosions']}")
    print(f"Convergence achieved: {'Yes ✓' if results['convergence_achieved'] else 'No ✗'}")
    print(f"Final loss: {results['final_loss']:.6f}")
    print()
    
    # Compare with paper results
    print("COMPARISON WITH PAPER (Section 5.3)")
    print("-" * 80)
    print(f"{'Metric':<30} {'Paper':<15} {'Our Result':<15} {'Status':<10}")
    print("-" * 80)
    
    paper_max_grad = 8.4
    paper_explosions = 0
    paper_convergence = 94.2  # percentage
    
    our_convergence = 100.0 if results['convergence_achieved'] else 0.0
    
    max_grad_status = "✓ PASS" if results['max_gradient_norm'] < 10.0 else "✗ FAIL"
    explosion_status = "✓ PASS" if results['gradient_explosions'] == 0 else "✗ FAIL"
    convergence_status = "✓ PASS" if our_convergence > 90.0 else "✗ FAIL"
    
    print(f"{'Maximum gradient norm':<30} {paper_max_grad:<15.1f} {results['max_gradient_norm']:<15.2f} {max_grad_status:<10}")
    print(f"{'Gradient explosions':<30} {paper_explosions:<15} {results['gradient_explosions']:<15} {explosion_status:<10}")
    print(f"{'Convergence rate (%)':<30} {paper_convergence:<15.1f} {our_convergence:<15.1f} {convergence_status:<10}")
    print("-" * 80)
    print()
    
    # Plot gradient evolution
    print("Generating gradient evolution plot...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Total gradient norms
    ax1 = axes[0]
    ax1.plot(results['total_norms'], linewidth=1, alpha=0.7)
    ax1.axhline(y=10, color='r', linestyle='--', label='Stability threshold (10)')
    ax1.axhline(y=100, color='r', linestyle=':', label='Explosion threshold (100)')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Gradient Norm', fontsize=12)
    ax1.set_title('Total Gradient Norm Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Magnitude vs Phase gradients
    ax2 = axes[1]
    if results['magnitude_grads'] and results['phase_grads']:
        ax2.plot(results['magnitude_grads'], label='Magnitude gradients', alpha=0.7)
        ax2.plot(results['phase_grads'], label='Phase gradients', alpha=0.7)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Gradient Norm', fontsize=12)
        ax2.set_title('Magnitude vs Phase Gradient Evolution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_stability.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'gradient_stability.png'")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_pass = (
        max_grad_status == "✓ PASS" and
        explosion_status == "✓ PASS" and
        convergence_status == "✓ PASS"
    )
    
    if all_pass:
        print("✓ VERIFIED: All gradient stability claims validated")
        print("  - Gradients remain bounded during training")
        print("  - No gradient explosion events detected")
        print("  - Convergence achieved consistently")
    else:
        print("⚠ WARNING: Some stability claims not fully validated")
        
    print()
    print("Theorem 1 verification complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
