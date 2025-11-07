#!/usr/bin/env python3
"""
Validation Script - Verify all V2 features are working
Run this to ensure everything is properly installed
"""

import sys
import importlib

def check_import(module_path, description):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_path)
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description}")
        print(f"   Error: {e}")
        return False

def main():
    print("="*80)
    print("RESONANCE NEURAL NETWORKS V2.0 - VALIDATION")
    print("="*80)
    print()
    
    all_passed = True
    
    # Core modules
    print("Core Modules:")
    all_passed &= check_import("resonance_nn.layers.resonance", "Resonance layers")
    all_passed &= check_import("resonance_nn.layers.holographic", "Holographic memory")
    all_passed &= check_import("resonance_nn.layers.embeddings", "Large vocabulary embeddings")
    all_passed &= check_import("resonance_nn.models.resonance_net", "Base models")
    print()
    
    # Long context
    print("Long Context (260K-300K tokens):")
    all_passed &= check_import("resonance_nn.models.long_context", "Long context models")
    print()
    
    # Multimodal
    print("Multimodal Capabilities:")
    all_passed &= check_import("resonance_nn.multimodal.vision", "Vision processing (NO CNN)")
    all_passed &= check_import("resonance_nn.multimodal.audio", "Audio processing")
    all_passed &= check_import("resonance_nn.multimodal.fusion", "Cross-modal fusion")
    print()
    
    # Specialized models
    print("Specialized Models:")
    all_passed &= check_import("resonance_nn.models.specialized.language_model", "Language model")
    all_passed &= check_import("resonance_nn.models.specialized.code_model", "Code model")
    all_passed &= check_import("resonance_nn.models.specialized.vision_model", "Vision model")
    all_passed &= check_import("resonance_nn.models.specialized.audio_model", "Audio model")
    print()
    
    # Export utilities
    print("Export & Deployment:")
    all_passed &= check_import("resonance_nn.export", "Export utilities (ONNX, TorchScript)")
    print()
    
    # Benchmarking
    print("Benchmarking:")
    all_passed &= check_import("resonance_nn.benchmark.benchmark", "Original benchmarks")
    all_passed &= check_import("resonance_nn.benchmark.l40_benchmark", "L40 GPU benchmarks")
    print()
    
    # Training
    print("Training:")
    all_passed &= check_import("resonance_nn.training.trainer", "Training utilities")
    print()
    
    # Test basic functionality
    print("Functional Tests:")
    try:
        import torch
        from resonance_nn.models.long_context import LongContextResonanceNet
        model = LongContextResonanceNet(input_dim=64, chunk_size=256, max_chunks=4)
        x = torch.randn(1, 512, 64)
        output = model(x, use_memory=False)
        assert output.shape == x.shape
        print("✅ Long context model forward pass")
    except Exception as e:
        print(f"❌ Long context model forward pass: {e}")
        all_passed = False
    
    try:
        from resonance_nn.layers.embeddings import HierarchicalVocabularyEmbedding
        embedding = HierarchicalVocabularyEmbedding(vocab_size=10000, embed_dim=64)
        ids = torch.randint(0, 10000, (2, 32))
        emb = embedding(ids)
        assert emb.shape == (2, 32, 64)
        print("✅ Large vocabulary embedding")
    except Exception as e:
        print(f"❌ Large vocabulary embedding: {e}")
        all_passed = False
    
    try:
        from resonance_nn.multimodal.fusion import MultiModalResonanceFusion
        model = MultiModalResonanceFusion(
            modality_dims={'text': 64, 'vision': 64},
            hidden_dim=64,
            num_classes=10,
        )
        inputs = {
            'text': torch.randn(2, 32, 64),
            'vision': torch.randn(2, 49, 64),
        }
        logits = model(inputs)
        assert logits.shape == (2, 10)
        print("✅ Multimodal fusion")
    except Exception as e:
        print(f"❌ Multimodal fusion: {e}")
        all_passed = False
    
    print()
    print("="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print()
        print("Next steps:")
        print("  1. Run examples:     python examples/v2_quickstart.py")
        print("  2. Run benchmarks:   python scripts/run_benchmarks.py --all")
        print("  3. Start training:   python scripts/train_l40.py --model language")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print()
        print("Please check the errors above and ensure:")
        print("  - All dependencies are installed: pip install -e .")
        print("  - PyTorch is available: pip install torch")
        return 1

if __name__ == '__main__':
    sys.exit(main())
