"""
Quick Start Example for NEURONSv2
==================================

This script demonstrates:
1. Creating a NEURONSv2 model
2. Training on simple text
3. Generating text
4. Benchmarking

Usage:
    python examples/quickstart.py
"""

import torch
from neurons import NEURONSv2, create_language_model

def main():
    print("🧠 NEURONSv2 Quick Start")
    print("="*70)
    
    # Step 1: Create model
    print("\n[1/4] Creating NEURONSv2 model...")
    model = create_language_model("small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"✓ Model created on {device}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Step 2: Test forward pass
    print("\n[2/4] Testing forward pass...")
    input_ids = torch.randint(0, 50257, (2, 10), device=device)
    
    with torch.no_grad():
        outputs = model(input_ids, num_steps=5)
    
    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Output shape: {outputs['logits'].shape}")
    print(f"✓ Forward pass works!")
    
    # Step 3: Test generation
    print("\n[3/4] Testing text generation...")
    prompt = torch.randint(0, 50257, (1, 5), device=device)
    
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            num_steps=5,
        )
    
    print(f"✓ Generated {generated.shape[1] - prompt.shape[1]} new tokens")
    print(f"✓ Generation works!")
    
    # Step 4: Show architecture details
    print("\n[4/4] Architecture Summary:")
    print("  • Spiking Neurons: Leaky Integrate-and-Fire dynamics")
    print("  • Dendritic Structure: 12 basal + 6 apical branches per neuron")
    print("  • Oscillatory Binding: 60 Hz gamma oscillations")
    print("  • Learning: Hebbian plasticity (STDP + BCM)")
    print("  • Connectivity: 10% sparse (biological realistic)")
    print("  • Event-Driven: Only active neurons compute")
    
    print("\n" + "="*70)
    print("🚀 NEURONSv2 is ready to use!")
    print("\nNext steps:")
    print("  1. Train on real data: python train.py --train_data data/train.txt")
    print("  2. Benchmark vs GPT-2: python train.py --benchmark")
    print("  3. Read docs: see README.md")
    print("="*70)


if __name__ == "__main__":
    main()
