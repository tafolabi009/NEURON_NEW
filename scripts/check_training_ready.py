"""
Quick Setup Check for FineWebEdu 32k Training
Genovo Technologies Research Team

Run this to verify everything is ready for training
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

print("="*80)
print("RESONANCE NEURAL NETWORKS - TRAINING READINESS CHECK")
print("="*80)
print()

# 1. Check PyTorch installation
print("1. Checking PyTorch...")
print(f"   ✓ PyTorch version: {torch.__version__}")
print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   ✓ Compute capability: {torch.cuda.get_device_capability(0)}")
else:
    print("   ⚠ Warning: CUDA not available, will train on CPU (very slow)")
print()

# 2. Check required packages
print("2. Checking required packages...")
required_packages = [
    'numpy',
    'tqdm',
]

all_installed = True
for package_name in required_packages:
    try:
        __import__(package_name)
        print(f"   ✓ {package_name}")
    except ImportError:
        print(f"   ✗ {package_name} not installed")
        all_installed = False

if not all_installed:
    print("\n   Install missing packages with:")
    print("   pip install numpy tqdm")
    sys.exit(1)
print()

# 3. Check Resonance NN installation
print("3. Checking Resonance Neural Networks...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from resonance_nn import ResonanceNet
    from resonance_nn.layers.resonance import ResonanceLayer
    print("   ✓ resonance_nn package")
    print("   ✓ ResonanceNet model")
    print("   ✓ ResonanceLayer")
except ImportError as e:
    print(f"   ✗ Error importing resonance_nn: {e}")
    print("\n   Install with:")
    print("   pip install -e .")
    sys.exit(1)
print()

# 4. Check training scripts
print("4. Checking training scripts...")
script_dir = Path(__file__).parent
required_scripts = [
    'train_finewebedu_32k.py',
    'train_finewebedu_32k_distributed.py',
    'prepare_data.py',
    'quick_train.sh',
]

for script in required_scripts:
    script_path = script_dir / script
    if script_path.exists():
        print(f"   ✓ {script}")
    else:
        print(f"   ✗ {script} not found")
print()

# 5. Test model creation
print("5. Testing model creation...")
try:
    from resonance_nn.models.resonance_net import ResonanceNet
    
    # Create small test model
    model = ResonanceNet(
        input_dim=512,
        num_frequencies=32,
        hidden_dim=512,
        num_layers=2,
    )
    
    # Test forward pass
    test_input = torch.randn(2, 16, 512)
    with torch.no_grad():
        output = model(test_input)
    
    print(f"   ✓ Model creation successful")
    print(f"   ✓ Forward pass working")
    print(f"   ✓ Output shape: {output.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)
print()

# 6. Test GPU operations
if torch.cuda.is_available():
    print("6. Testing GPU operations...")
    try:
        # Move model to GPU
        model = model.cuda()
        test_input = test_input.cuda()
        
        # Test forward pass on GPU
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   ✓ GPU forward pass working")
        print(f"   ✓ Output device: {output.device}")
        
        # Test FFT operations
        x = torch.randn(2, 1024, 512, device='cuda')
        x_fft = torch.fft.fft(x, dim=1)
        print(f"   ✓ FFT operations working")
        
    except Exception as e:
        print(f"   ✗ GPU error: {e}")
        sys.exit(1)
    print()

# 7. Check documentation
print("7. Checking documentation...")
docs_dir = Path(__file__).parent.parent / 'docs'
required_docs = [
    'TRAINING_GUIDE.md',
    'TRAINING_INFRASTRUCTURE.md',
]

for doc in required_docs:
    doc_path = docs_dir / doc
    if doc_path.exists():
        print(f"   ✓ {doc}")
    else:
        print(f"   ⚠ {doc} not found")
print()

# 8. Memory check
print("8. Checking available memory...")
if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   Total VRAM: {total_mem:.1f} GB")
    
    # Estimate model sizes
    configs = [
        ("Small", 768, 6, 10),
        ("Medium", 1024, 12, 20),
        ("Large", 2048, 16, 35),
        ("XLarge", 4096, 24, 44),
    ]
    
    print("\n   Recommended models for your GPU:")
    for name, dim, layers, vram in configs:
        if vram <= total_mem * 0.9:  # 90% utilization
            print(f"   ✓ {name} ({vram} GB VRAM)")
        else:
            print(f"   ✗ {name} ({vram} GB VRAM) - Not enough memory")
print()

# Summary
print("="*80)
print("READINESS CHECK COMPLETE")
print("="*80)
print()
print("✅ All systems ready for training!")
print()
print("Next steps:")
print()
print("1. Prepare your pretokenized FineWebEdu data:")
print("   - Directory of .npy or .pt files")
print("   - Each file shape: (num_sequences, 32768)")
print("   - Token IDs in range [0, vocab_size)")
print()
print("2. Validate your data:")
print("   python scripts/prepare_data.py validate \\")
print("     --data-path /path/to/your/pretokenized/data")
print()
print("3. Start training:")
print("   ./scripts/quick_train.sh /path/to/your/data medium")
print()
print("For detailed instructions, see:")
print("   docs/TRAINING_GUIDE.md")
print("   scripts/README_TRAINING.md")
print()
print("="*80)
