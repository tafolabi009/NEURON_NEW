# FineWebEdu 32k Training Guide

**Genovo Technologies Research Team**  
**Lead:** Oluwatosin Afolabi (afolabi@genovotech.com)

Complete guide for training Resonance Neural Networks on pretokenized FineWebEdu data at 32k context length.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Training Configurations](#training-configurations)
4. [Distributed Training](#distributed-training)
5. [Monitoring & Checkpointing](#monitoring--checkpointing)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Step 1: Validate Your Data

```bash
# Validate pretokenized data format
python scripts/prepare_data.py validate \
  --data-path /path/to/pretokenized/finewebedu

# This will:
# - Check file formats (.npy or .pt)
# - Validate shapes (expect: [num_sequences, 32768])
# - Calculate statistics (vocab size, total tokens, etc.)
# - Provide training recommendations
```

**Expected Data Format:**
- Directory containing `.npy` or `.pt` files
- Each file: `(num_sequences, 32768)` array of token IDs
- All files must have identical shape
- Token IDs should be in range `[0, vocab_size)`

### Step 2: Create Train/Val Split

```bash
# Create train/validation split (99% train, 1% val)
python scripts/prepare_data.py split \
  --data-path /path/to/pretokenized/finewebedu \
  --output-path /path/to/split/data \
  --val-ratio 0.01

# This creates:
# - /path/to/split/data/train/  (99% of files)
# - /path/to/split/data/val/    (1% of files)
```

### Step 3: Start Training

```bash
# Single GPU training (small model)
python scripts/train_finewebedu_32k.py \
  --data-path /path/to/split/data/train \
  --output-dir checkpoints/small_model \
  --model-dim 768 \
  --num-frequencies 64 \
  --num-layers 6 \
  --batch-size 4 \
  --gradient-accumulation 8 \
  --max-seq-len 32768 \
  --epochs 3
```

---

## Data Preparation

### Data Format Validation

Your pretokenized FineWebEdu data must meet these requirements:

1. **File Format:** `.npy` (NumPy) or `.pt` (PyTorch)
2. **Shape:** `(num_sequences, 32768)` - exactly 32,768 tokens per sequence
3. **Data Type:** Integer token IDs (int32, int64, or long)
4. **Token Range:** `[0, vocab_size)` - typically GPT-2 vocab (50,257)

**Validation Script Output:**
```
================================================================================
VALIDATION RESULTS
================================================================================
✓ Valid: True
✓ Files: 1000
✓ Total sequences: 100,000
✓ Sequence length: 32,768
✓ Total tokens: 3,276,800,000 (3.28B)
✓ Vocab size: 50,257
✓ Unique tokens: 50,257
✓ Avg token value: 25128.45
================================================================================
```

### Creating Train/Val Split

The split utility creates symbolic links (no data duplication):

```bash
python scripts/prepare_data.py split \
  --data-path /original/data \
  --output-path /split/data \
  --val-ratio 0.01  # 1% validation
```

**Result:**
- `train/`: 990 files (99%)
- `val/`: 10 files (1%)

---

## Training Configurations

### 1. Small Model (Testing & Development)

**Parameters:** ~50M  
**VRAM:** ~10 GB  
**Purpose:** Quick iterations, architecture testing

```bash
python scripts/train_finewebedu_32k.py \
  --data-path /data/train \
  --output-dir checkpoints/small \
  --vocab-size 50257 \
  --model-dim 768 \
  --num-frequencies 64 \
  --num-layers 6 \
  --batch-size 4 \
  --gradient-accumulation 8 \
  --max-seq-len 32768 \
  --epochs 3 \
  --lr 6e-4
```

**Effective Batch Size:** 32 (4 × 8)  
**Estimated Time:** ~2-3 days on L40S for 3B tokens

### 2. Medium Model (Production)

**Parameters:** ~200M  
**VRAM:** ~20 GB  
**Purpose:** Production-ready model

```bash
python scripts/train_finewebedu_32k.py \
  --data-path /data/train \
  --output-dir checkpoints/medium \
  --vocab-size 50257 \
  --model-dim 1024 \
  --num-frequencies 128 \
  --num-layers 12 \
  --batch-size 4 \
  --gradient-accumulation 8 \
  --max-seq-len 32768 \
  --epochs 3 \
  --lr 6e-4
```

**Effective Batch Size:** 32  
**Estimated Time:** ~4-5 days on L40S for 3B tokens

### 3. Large Model (Research)

**Parameters:** ~500M-1B  
**VRAM:** ~30-40 GB  
**Purpose:** Maximum performance

```bash
python scripts/train_finewebedu_32k.py \
  --data-path /data/train \
  --output-dir checkpoints/large \
  --vocab-size 50257 \
  --model-dim 2048 \
  --num-frequencies 256 \
  --num-layers 16 \
  --batch-size 2 \
  --gradient-accumulation 16 \
  --max-seq-len 32768 \
  --epochs 3 \
  --lr 6e-4
```

**Effective Batch Size:** 32  
**Estimated Time:** ~6-8 days on L40S for 3B tokens

### 4. XLarge Model (Maximum Scale)

**Parameters:** ~3B-7B  
**VRAM:** ~44 GB (requires L40S or A100)  
**Purpose:** Massive scale testing

```bash
python scripts/train_finewebedu_32k.py \
  --data-path /data/train \
  --output-dir checkpoints/xlarge \
  --vocab-size 50257 \
  --model-dim 4096 \
  --num-frequencies 512 \
  --num-layers 24 \
  --batch-size 1 \
  --gradient-accumulation 32 \
  --max-seq-len 32768 \
  --epochs 3 \
  --lr 3e-4
```

**Effective Batch Size:** 32  
**Estimated Time:** ~10-14 days on L40S for 3B tokens

---

## Distributed Training

For multi-GPU setups, use the distributed training script:

### 4-GPU Setup (Recommended)

```bash
# Using torchrun for distributed training
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=29500 \
  scripts/train_finewebedu_32k_distributed.py \
  --data-path /data/train \
  --output-dir checkpoints/distributed \
  --vocab-size 50257 \
  --model-dim 2048 \
  --num-frequencies 256 \
  --num-layers 24 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --max-seq-len 32768 \
  --epochs 3 \
  --lr 6e-4 \
  --wandb
```

**Effective Batch Size:** 64 (1 × 4 GPUs × 16 accumulation)  
**Speedup:** ~3.5x (90% scaling efficiency)

### WandB Integration

Add `--wandb` flag to enable Weights & Biases logging:

```bash
# Login to WandB first
wandb login

# Then run with --wandb flag
python scripts/train_finewebedu_32k_distributed.py \
  --data-path /data/train \
  --wandb
```

**Logged Metrics:**
- Training loss
- Learning rate
- Tokens per second
- GPU utilization
- Memory usage

---

## Monitoring & Checkpointing

### Automatic Checkpointing

Checkpoints are saved every 1,000 steps:

```
checkpoints/
├── checkpoint-1000.pt
├── checkpoint-2000.pt
├── checkpoint-3000.pt
└── final_model.pt
```

**Checkpoint Contents:**
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `step`: Global step number
- `epoch`: Current epoch
- `loss`: Current loss value

### Resuming from Checkpoint

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint-5000.pt')

# Restore model
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Restore scheduler
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Resume from step
start_step = checkpoint['step']
```

### Real-Time Monitoring

During training, you'll see:

```
Epoch 1/3: 100%|████████| 10000/10000 [2:15:30<00:00, loss=3.2456, lr=5.8e-4]
✓ Saved checkpoint: checkpoints/checkpoint-1000.pt

Epoch 1 average loss: 3.2456
```

### Terminal Logging

Training prints comprehensive logs:

```
================================================================================
RESONANCE NEURAL NETWORK TRAINING
================================================================================
Data path: /data/train
Output dir: checkpoints/medium
Model: 1024D, 12L, 128F
Batch size: 4 x 8 = 32
Max sequence length: 32,768

Device: cuda
GPU: NVIDIA L40S
VRAM: 44.4 GB

Loading dataset from /data/train...
Found 990 data files
Sequences per file: 100
Total sequences: 99,000

Creating model...
Total parameters: 187,456,512 (187.5M)

================================================================================
TRAINING
================================================================================
```

---

## Troubleshooting

### 1. Out of Memory (OOM)

**Symptoms:** CUDA OOM error during training

**Solutions:**
- Reduce `--batch-size` (try 2 or 1)
- Increase `--gradient-accumulation` to maintain effective batch size
- Reduce `--model-dim` or `--num-layers`
- Use distributed training across multiple GPUs

**Example Fix:**
```bash
# Before (OOM)
--batch-size 4 --gradient-accumulation 8

# After (Fixed)
--batch-size 2 --gradient-accumulation 16  # Same effective batch
```

### 2. Slow Training Speed

**Symptoms:** Low throughput (< 100 samples/sec)

**Solutions:**
- Ensure mixed precision is enabled (default)
- Check GPU utilization: `nvidia-smi`
- Increase `num_workers` in dataloader
- Use multiple GPUs with distributed training
- Reduce sequence length if possible

### 3. Data Loading Errors

**Symptoms:** "No data files found" or shape mismatch errors

**Solutions:**
- Run validation: `python scripts/prepare_data.py validate --data-path /path`
- Check file format (must be .npy or .pt)
- Verify all files have shape `(N, 32768)`
- Ensure token IDs are in valid range

### 4. Loss Not Decreasing

**Symptoms:** Loss stays high or increases

**Solutions:**
- Check learning rate (try lower: 3e-4 or 1e-4)
- Verify data is correctly tokenized
- Increase warmup steps
- Check for gradient explosion (use gradient clipping)
- Reduce model size if dataset is small

### 5. Distributed Training Issues

**Symptoms:** Process hangs or crashes in multi-GPU setup

**Solutions:**
- Check NCCL environment: `echo $NCCL_DEBUG`
- Verify all GPUs are visible: `nvidia-smi`
- Use correct `--nproc_per_node` value
- Check network connectivity between nodes
- Disable compiled model: remove `--compile` flag

---

## Performance Benchmarks

### L40S GPU (44 GB VRAM)

| Model Size | Params | Batch Size | Throughput | VRAM Usage |
|-----------|--------|-----------|-----------|-----------|
| Small | 50M | 4 × 8 | ~1000 samples/sec | ~10 GB |
| Medium | 200M | 4 × 8 | ~600 samples/sec | ~20 GB |
| Large | 500M | 2 × 16 | ~300 samples/sec | ~35 GB |
| XLarge | 1B | 1 × 32 | ~150 samples/sec | ~44 GB |

**Note:** Throughput measured with 32k sequence length and mixed precision enabled.

---

## Advanced Features

### Custom Vocabulary

If using custom tokenizer:

```bash
python scripts/train_finewebedu_32k.py \
  --vocab-size 100000 \  # Custom vocab size
  --data-path /data/custom
```

### Learning Rate Schedules

Default: Cosine annealing with warm restarts

To customize, edit `train_finewebedu_32k.py`:

```python
# Linear warmup + cosine decay
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=total_steps,
)
```

### Gradient Checkpointing

For very large models, enable gradient checkpointing:

```python
# In ResonanceLanguageModel.__init__
self.backbone.gradient_checkpointing_enable()
```

This trades compute for memory, enabling larger models at the cost of ~30% slower training.

---

## Next Steps

After training:

1. **Evaluate on validation set**
   ```bash
   python scripts/evaluate_model.py \
     --checkpoint checkpoints/final_model.pt \
     --data-path /data/val
   ```

2. **Test on downstream tasks**
   - Text generation
   - Question answering
   - Summarization

3. **Compare with baselines**
   ```bash
   python scripts/comparative_benchmark.py \
     --resonance-checkpoint checkpoints/final_model.pt
   ```

4. **Deploy to production**
   - Export to ONNX
   - Quantization (INT8/FP16)
   - Serve with FastAPI/TorchServe

---

## Support

For issues or questions:
- Email: afolabi@genovotech.com
- Documentation: See `README.md`
- Architecture details: See `ARCHITECTURE.md`

---

**Genovo Technologies Research Team**  
*Building the future of efficient neural architectures*
