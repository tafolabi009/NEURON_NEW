# Massive Model Testing Guide

**Genovo Technologies Research Team**  
**Lead Researcher:** Oluwatosin Afolabi (afolabi@genovotech.com)  
**Date:** November 7, 2025  
**HuggingFace Token:** Configured ✅

---

## 🚀 Available Model Sizes

We've prepared 6 model configurations to leverage your compute power:

### 1. **Small - 50M Parameters**
```bash
python3 scripts/massive_model_benchmark.py --size small
```
**Config:**
- Input dim: 1024
- Frequencies: 128
- Hidden dim: 1024
- Layers: 12
- Est. VRAM: ~2-3 GB

**Use case:** Fast prototyping, edge deployment

---

### 2. **Medium - 200M Parameters** 
```bash
python3 scripts/massive_model_benchmark.py --size medium
```
**Config:**
- Input dim: 2048
- Frequencies: 256
- Hidden dim: 2048
- Layers: 16
- Est. VRAM: ~8-10 GB

**Use case:** Production models, research

---

### 3. **Large - 500M Parameters**
```bash
python3 scripts/massive_model_benchmark.py --size large
```
**Config:**
- Input dim: 3072
- Frequencies: 512
- Hidden dim: 3072
- Layers: 20
- Est. VRAM: ~15-20 GB

**Use case:** High-performance NLP, vision

---

### 4. **XLarge - 1B Parameters**
```bash
python3 scripts/massive_model_benchmark.py --size xlarge
```
**Config:**
- Input dim: 4096
- Frequencies: 768
- Hidden dim: 4096
- Layers: 24
- Est. VRAM: ~25-30 GB

**Use case:** State-of-the-art performance

---

### 5. **XXLarge - 3B Parameters**
```bash
python3 scripts/massive_model_benchmark.py --size xxlarge
```
**Config:**
- Input dim: 6144
- Frequencies: 1024
- Hidden dim: 6144
- Layers: 32
- Est. VRAM: ~35-40 GB

**Use case:** Maximum scale on L40S (44GB VRAM)

---

### 6. **Ultimate - 7B Parameters** (Llama-scale)
```bash
python3 scripts/massive_model_benchmark.py --size ultimate
```
**Config:**
- Input dim: 8192
- Frequencies: 1536
- Hidden dim: 8192
- Layers: 40
- Est. VRAM: ~42-44 GB (uses full L40S capacity)

**Use case:** Direct Llama 2 7B comparison

---

## 🧪 Test Commands

### Basic Benchmark
```bash
# Test medium model with default configs
python3 scripts/massive_model_benchmark.py --size medium

# Test large model with custom batch/sequence
python3 scripts/massive_model_benchmark.py --size large \
  --batch-sizes 1 2 4 8 \
  --seq-lengths 512 1024 2048 4096
```

### Compare with Llama 2 7B
```bash
# Direct comparison (requires same VRAM)
python3 scripts/massive_model_benchmark.py --size ultimate --compare-llama

# Or compare different size
python3 scripts/massive_model_benchmark.py --size large --compare-llama
```

### Run Full Suite
```bash
# Test all sizes (takes ~1-2 hours)
for size in small medium large xlarge; do
  echo "Testing $size..."
  python3 scripts/massive_model_benchmark.py --size $size
done
```

---

## 📊 Expected Performance

Based on O(n log n) complexity and current optimizations:

| Model Size | Parameters | Seq=2048 | Throughput | VRAM |
|------------|------------|----------|------------|------|
| Small | 50M | ~15ms | 800K tok/s | ~3 GB |
| Medium | 200M | ~30ms | 650K tok/s | ~10 GB |
| Large | 500M | ~50ms | 500K tok/s | ~20 GB |
| XLarge | 1B | ~80ms | 350K tok/s | ~30 GB |
| XXLarge | 3B | ~150ms | 200K tok/s | ~40 GB |
| **Ultimate** | **7B** | **~250ms** | **150K tok/s** | **~44 GB** |

**Llama 2 7B** (for comparison):
- Seq=2048: ~180ms (O(n²) attention)
- Throughput: ~90K tok/s
- VRAM: ~28 GB (FP16)

**Expected advantage:** 1.5-2x faster than Llama 2 at similar scale

---

## 🎯 Benchmarking Strategies

### Strategy 1: **Memory Profiling**
```bash
# Find optimal batch size for your VRAM
python3 scripts/massive_model_benchmark.py --size large \
  --batch-sizes 1 2 4 8 16 32 \
  --seq-lengths 2048
```

### Strategy 2: **Throughput Optimization**
```bash
# Find peak throughput config
python3 scripts/massive_model_benchmark.py --size xlarge \
  --batch-sizes 32 64 128 \
  --seq-lengths 512
```

### Strategy 3: **Latency Testing**
```bash
# Single-sample inference speed
python3 scripts/massive_model_benchmark.py --size ultimate \
  --batch-sizes 1 \
  --seq-lengths 512 1024 2048 4096 8192
```

### Strategy 4: **Long Context**
```bash
# Test ultra-long sequences
python3 scripts/massive_model_benchmark.py --size medium \
  --batch-sizes 1 2 4 \
  --seq-lengths 8192 16384 32768 65536
```

---

## 🔬 Advanced Testing

### Multi-GPU Setup
If you have multiple GPUs, modify the script:

```python
# In massive_model_benchmark.py
model = nn.DataParallel(model)  # Automatic multi-GPU
# Or
model = model.to('cuda:0')  # Specific GPU
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(x)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Checkpointing (for training large models)
```python
from torch.utils.checkpoint import checkpoint

# In forward pass
def forward_with_checkpointing(self, x):
    return checkpoint(self.layer, x)
```

---

## 📈 Scaling Analysis

### Expected Complexity Scaling

| Seq Length | Small (50M) | Medium (200M) | Large (500M) | Ultimate (7B) |
|------------|-------------|---------------|--------------|---------------|
| 512 | 5ms | 10ms | 15ms | 50ms |
| 1024 | 8ms | 16ms | 25ms | 80ms |
| 2048 | 15ms | 30ms | 50ms | 150ms |
| 4096 | 28ms | 55ms | 95ms | 280ms |
| 8192 | 52ms | 105ms | 180ms | 520ms |

*All times approximate for batch=1 on L40S GPU*

### Memory Scaling

```
Memory = (parameters × 4 bytes) + (activations × batch_size × seq_len × hidden_dim × 4)

For 7B model at batch=1, seq=2048:
  Parameters: 7B × 4 = 28 GB
  Activations: 2048 × 8192 × 4 = 64 MB
  Total: ~28 GB (FP32) or ~14 GB (FP16)
```

---

## 🚨 Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `--batch-sizes 1`
2. Use gradient checkpointing (training only)
3. Enable FP16: Already enabled via `torch.amp.autocast`
4. Reduce sequence length: `--seq-lengths 512 1024`

### Slow Performance

**Solutions:**
1. Enable optimizations: Models use `optimize=True` by default
2. Use larger batch sizes for better GPU utilization
3. Check if CPU is bottleneck: Use `nvidia-smi` to monitor GPU usage
4. Disable unnecessary logging/checkpoints

### Model Loading Errors

**Solutions:**
1. Ensure HuggingFace token is set: `python3 scripts/hf_login.py`
2. Check internet connection for model downloads
3. Clear cache: `rm -rf ~/.cache/huggingface/`

---

## 📊 Results Analysis

After running benchmarks, analyze results:

```python
import json

# Load results
with open('massive_model_xlarge_results.json') as f:
    results = json.load(f)

# Find peak throughput
peak = max(results, key=lambda x: x['throughput_tokens_per_sec'])
print(f"Peak: {peak['throughput_tokens_per_sec']:,.0f} tok/s at batch={peak['batch_size']}, seq={peak['seq_len']}")

# Find best latency
best_latency = min(results, key=lambda x: x['mean_time_ms'])
print(f"Best latency: {best_latency['mean_time_ms']:.2f}ms at batch={best_latency['batch_size']}, seq={best_latency['seq_len']}")
```

---

## 🎯 Recommended Test Sequence

For comprehensive testing on L40S GPU (44 GB VRAM):

```bash
# 1. Verify setup
python3 scripts/hf_login.py

# 2. Quick test with small model
python3 scripts/massive_model_benchmark.py --size small

# 3. Test medium model (optimal for most tasks)
python3 scripts/massive_model_benchmark.py --size medium \
  --batch-sizes 1 4 8 16 \
  --seq-lengths 512 2048 8192

# 4. Test large model (production scale)
python3 scripts/massive_model_benchmark.py --size large \
  --batch-sizes 1 4 8 \
  --seq-lengths 512 2048 4096

# 5. Test XLarge (1B params - research scale)
python3 scripts/massive_model_benchmark.py --size xlarge \
  --batch-sizes 1 2 4 \
  --seq-lengths 512 2048

# 6. Test Ultimate vs Llama 2 (if enough VRAM)
python3 scripts/massive_model_benchmark.py --size ultimate --compare-llama

# 7. Run comparative benchmark with Llama 2 access
python3 scripts/comparative_benchmark.py --llama-model meta-llama/Llama-2-7b-hf
```

---

## 💡 Pro Tips

1. **Start Small:** Test with `--size small` first to verify setup
2. **Monitor VRAM:** Use `nvidia-smi -l 1` in another terminal
3. **Save Results:** All benchmarks save JSON files automatically
4. **Compare Configs:** Use `--compare-llama` to validate against SOTA
5. **Batch Size:** Larger batches = better throughput, smaller = better latency
6. **Sequence Length:** O(n log n) means doubling seq_len doesn't double time

---

## 📞 Support

**Issues or questions?**
- Email: afolabi@genovotech.com
- Check: `PERFORMANCE_RESULTS.md` for baseline results
- Review: `OPTIMIZATION_GUIDE.md` for optimization tips

---

## 🎉 Next Steps

After running benchmarks:

1. ✅ Document your results
2. ✅ Identify optimal configuration for your use case
3. ✅ Compare with Llama 2 7B (now with HF token)
4. ✅ Train on real datasets (future work)
5. ✅ Deploy to production

**You now have the tools to build and test models up to 7B parameters with optimized O(n log n) complexity!**

---

**Status:** ✅ Ready to leverage your L40S GPU (44 GB VRAM)  
**HuggingFace:** ✅ Token configured for Llama 2 access  
**Models:** 6 sizes from 50M to 7B parameters available  
**Benchmarks:** Comprehensive suite ready to run
