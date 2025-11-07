#!/bin/bash
# Quick setup script for L40 GPU

echo "=========================================="
echo "Resonance Neural Networks V2.0 - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Check CUDA availability
echo ""
echo "Checking CUDA..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"

# Check GPU
echo ""
echo "GPU Information:"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run examples:     python examples/v2_quickstart.py"
echo "  2. Run benchmarks:   python scripts/run_benchmarks.py --all"
echo "  3. Start training:   python scripts/train_l40.py --model language"
echo ""
echo "Documentation:"
echo "  - V2_FEATURES.md              (Feature guide)"
echo "  - IMPLEMENTATION_SUMMARY.md   (Complete summary)"
echo "  - ARCHITECTURE.md             (Architecture details)"
echo ""
