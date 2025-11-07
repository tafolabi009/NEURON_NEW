"""
Resonance Neural Networks V2.0 - Quick Start Examples
Demonstrates new features: long context, multimodal, large vocab
"""

import torch
import torch.nn as nn


def example_1_long_context():
    """Example 1: Ultra-long context processing (256K tokens)"""
    print("="*80)
    print("EXAMPLE 1: Long Context Processing (256K tokens)")
    print("="*80)
    
    from resonance_nn.models.long_context import LongContextResonanceNet
    
    # Create model
    model = LongContextResonanceNet(
        input_dim=768,
        chunk_size=4096,
        overlap=512,
        max_chunks=70,  # ~280K tokens
    )
    
    # Create ultra-long sequence
    seq_len = 65536  # 64K tokens (use smaller for demo)
    batch_size = 1
    x = torch.randn(batch_size, seq_len, 768)
    
    print(f"Input shape: {x.shape}")
    print(f"Processing {seq_len:,} tokens with O(n log n) complexity...")
    
    # Process
    output = model(x, use_memory=True, store_to_memory=True)
    
    print(f"Output shape: {output.shape}")
    print(f"✓ Successfully processed {seq_len:,} tokens!")
    
    # Memory estimate
    estimate = model.get_memory_usage_estimate(seq_len)
    print(f"\nMemory estimate for {seq_len:,} tokens:")
    print(f"  Total memory: {estimate['total_memory_mb']:.1f} MB")
    print(f"  Efficiency: {estimate['memory_efficiency']}")
    print()


def example_2_large_vocabulary():
    """Example 2: Large vocabulary embeddings (1M tokens)"""
    print("="*80)
    print("EXAMPLE 2: Large Vocabulary (1M tokens)")
    print("="*80)
    
    from resonance_nn.layers.embeddings import HierarchicalVocabularyEmbedding
    
    # Create embeddings for 1M vocabulary
    vocab_size = 1000000
    embed_dim = 768
    
    print(f"Creating embedding for {vocab_size:,} tokens...")
    
    embedding = HierarchicalVocabularyEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )
    
    # Look up tokens
    input_ids = torch.randint(0, vocab_size, (4, 128))
    embeddings = embedding(input_ids)
    
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"✓ Successfully embedded 1M vocabulary!")
    print()


def example_3_multimodal_fusion():
    """Example 3: Multimodal fusion (text + vision + audio)"""
    print("="*80)
    print("EXAMPLE 3: Multimodal Fusion")
    print("="*80)
    
    from resonance_nn.multimodal.fusion import MultiModalResonanceFusion
    
    # Create fusion model
    model = MultiModalResonanceFusion(
        modality_dims={
            'text': 768,
            'vision': 768,
            'audio': 512,
        },
        hidden_dim=768,
        num_cross_modal_layers=4,
        num_classes=1000,
    )
    
    # Create multimodal inputs
    batch_size = 8
    inputs = {
        'text': torch.randn(batch_size, 128, 768),    # Text tokens
        'vision': torch.randn(batch_size, 196, 768),   # Image patches
        'audio': torch.randn(batch_size, 200, 512),    # Audio frames
    }
    
    print("Input modalities:")
    for name, tensor in inputs.items():
        print(f"  {name}: {tensor.shape}")
    
    # Fuse modalities
    logits = model(inputs)
    
    print(f"\nFused output shape: {logits.shape}")
    print("✓ Successfully fused text, vision, and audio!")
    
    # Handle missing modality
    print("\nTesting with missing audio...")
    partial_inputs = {
        'text': inputs['text'],
        'vision': inputs['vision'],
    }
    
    logits_partial = model.forward_with_missing_modalities(
        partial_inputs,
        available_modalities=['text', 'vision'],
    )
    
    print(f"Output with partial inputs: {logits_partial.shape}")
    print("✓ Handled missing modality gracefully!")
    print()


def example_4_vision_no_cnn():
    """Example 4: Vision processing WITHOUT CNN"""
    print("="*80)
    print("EXAMPLE 4: Vision Processing (NO CNN, pure frequency)")
    print("="*80)
    
    from resonance_nn.multimodal.vision import ResonanceVisionEncoder
    
    # Create vision model (NO CNN!)
    model = ResonanceVisionEncoder(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_layers=12,
        num_classes=1000,
    )
    
    # Process images
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input images: {images.shape}")
    print("Processing with 2D FFT and frequency resonance...")
    print("(NO convolutions, NO pooling)")
    
    logits = model(images)
    
    print(f"\nOutput logits: {logits.shape}")
    print("✓ Pure frequency-domain vision processing!")
    
    # Extract hierarchical features
    features = model.extract_hierarchical_features(images)
    print(f"\nExtracted {len(features)} feature levels")
    for i, feat in enumerate(features):
        print(f"  Level {i+1}: {feat.shape}")
    print()


def example_5_language_model():
    """Example 5: Language model with 256K context"""
    print("="*80)
    print("EXAMPLE 5: Language Model (256K context)")
    print("="*80)
    
    from resonance_nn.models.specialized import ResonanceLanguageModel
    
    # Create language model
    model = ResonanceLanguageModel(
        vocab_size=50000,
        embed_dim=768,
        max_seq_length=262144,  # 256K!
        use_long_context=True,
    )
    
    print(f"Model initialized with 256K max context")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Process text
    batch_size = 1
    seq_len = 8192  # 8K tokens (use smaller for demo)
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    
    print(f"\nInput tokens: {input_ids.shape}")
    
    # Get logits
    logits = model(input_ids)
    print(f"Output logits: {logits.shape}")
    
    # Generate text
    prompt = torch.randint(0, 50000, (1, 20))
    print(f"\nGenerating from prompt of length {prompt.shape[1]}...")
    
    generated = model.generate(
        input_ids=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
    )
    
    print(f"Generated {generated.shape[1]} tokens total")
    print("✓ Language model with ultra-long context!")
    print()


def example_6_model_export():
    """Example 6: Export model for deployment"""
    print("="*80)
    print("EXAMPLE 6: Model Export for Integration")
    print("="*80)
    
    from resonance_nn.models.resonance_net import ResonanceNet
    from resonance_nn.export import ModelExporter, ModelPackager
    import tempfile
    import os
    
    # Create a simple model
    model = ResonanceNet(
        input_dim=512,
        num_frequencies=64,
        hidden_dim=512,
        num_layers=4,
    )
    
    config = {
        'input_dim': 512,
        'num_frequencies': 64,
        'hidden_dim': 512,
        'num_layers': 4,
    }
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Exporting to: {temp_dir}")
    
    # Export
    exporter = ModelExporter(model, config)
    
    # PyTorch format
    pt_path = os.path.join(temp_dir, 'model.pt')
    exporter.export_pytorch(pt_path)
    
    # TorchScript
    example_input = torch.randn(1, 128, 512)
    ts_path = os.path.join(temp_dir, 'model_scripted.pt')
    exporter.export_torchscript(ts_path, example_input, use_trace=True)
    
    print("\n✓ Model exported successfully!")
    print(f"  PyTorch: {os.path.basename(pt_path)}")
    print(f"  TorchScript: {os.path.basename(ts_path)}")
    print(f"  Config: model_config.json")
    print("\nReady for integration into applications!")
    print()


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("RESONANCE NEURAL NETWORKS V2.0 - QUICK START")
    print("="*80)
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    try:
        example_1_long_context()
    except Exception as e:
        print(f"Example 1 failed: {e}\n")
    
    try:
        example_2_large_vocabulary()
    except Exception as e:
        print(f"Example 2 failed: {e}\n")
    
    try:
        example_3_multimodal_fusion()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")
    
    try:
        example_4_vision_no_cnn()
    except Exception as e:
        print(f"Example 4 failed: {e}\n")
    
    try:
        example_5_language_model()
    except Exception as e:
        print(f"Example 5 failed: {e}\n")
    
    try:
        example_6_model_export()
    except Exception as e:
        print(f"Example 6 failed: {e}\n")
    
    print("="*80)
    print("✓ ALL EXAMPLES COMPLETED!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run benchmarks: python scripts/run_benchmarks.py --all")
    print("  2. Train a model: python scripts/train_l40.py --model language")
    print("  3. Read V2_FEATURES.md for detailed documentation")
    print()


if __name__ == '__main__':
    main()
