# Training Scripts for FineWebEdu 32k

Complete training infrastructure for Resonance Neural Networks on pretokenized FineWebEdu data at 32k context length.

---

## Quick Start

### Simplest Path (Recommended)

```bash
# 1. Validate your data
./scripts/quick_train.sh /path/to/your/pretokenized/data medium

# That's it! This will:
# - Validate data format
# - Create train/val split
# - Train a 200M parameter model
# - Save checkpoints every 1000 steps
```

---

## Available Scripts

### 1. **quick_train.sh** - One-Command Training

The easiest way to start training:

```bash
# Small model (50M params, ~10GB VRAM)
./scripts/quick_train.sh /path/to/data small

# Medium model (200M params, ~20GB VRAM) - RECOMMENDED
./scripts/quick_train.sh /path/to/data medium

# Large model (500M params, ~35GB VRAM)
./scripts/quick_train.sh /path/to/data large

# XLarge model (1B params, ~44GB VRAM)
./scripts/quick_train.sh /path/to/data xlarge
```

**Features:**
- Automatic data validation
- Auto train/val split creation
- Pre-configured model sizes
- Checkpointing enabled
- Progress monitoring

---

### 2. **prepare_data.py** - Data Preparation

Validate and prepare your pretokenized data:

```bash
# Validate data format
python scripts/prepare_data.py validate \
  --data-path /path/to/pretokenized/data

# Create train/val split
python scripts/prepare_data.py split \
  --data-path /path/to/pretokenized/data \
  --output-path /path/to/output \
  --val-ratio 0.01  # 1% validation
```

**Output:**
- Data statistics (num files, sequences, tokens)
- Training time estimates
- Recommended configurations
- `validation_results.json` with metadata

---

### 3. **train_finewebedu_32k.py** - Single GPU Training

Full-featured training script for single GPU:

```bash
python scripts/train_finewebedu_32k.py \
  --data-path /path/to/train \
  --output-dir checkpoints/my_model \
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

**Options:**
- `--data-path`: Path to training data
- `--output-dir`: Where to save checkpoints (default: `checkpoints`)
- `--vocab-size`: Vocabulary size (default: 50257 - GPT-2)
- `--model-dim`: Model dimension (768, 1024, 2048, 4096)
- `--num-frequencies`: Number of resonance frequencies
- `--num-layers`: Number of layers
- `--batch-size`: Batch size per GPU
- `--gradient-accumulation`: Gradient accumulation steps
- `--max-seq-len`: Max sequence length (default: 32768)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--no-amp`: Disable mixed precision training

**Features:**
- Mixed precision (AMP) by default
- Automatic checkpointing (every 1000 steps)
- Progress bars with loss/LR
- Cosine annealing LR schedule
- Gradient clipping
- Cached data loading

---

### 4. **train_finewebedu_32k_distributed.py** - Multi-GPU Training

Distributed training for scaling across multiple GPUs:

```bash
# 4-GPU training
torchrun \
  --nproc_per_node=4 \
  scripts/train_finewebedu_32k_distributed.py \
  --data-path /path/to/train \
  --output-dir checkpoints/distributed \
  --model-dim 2048 \
  --num-frequencies 256 \
  --num-layers 24 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --wandb
```

**Additional Options:**
- `--wandb`: Enable Weights & Biases logging

**Features:**
- PyTorch DDP (DistributedDataParallel)
- Synchronized batch normalization
- Rank 0 logging
- Distributed sampler
- WandB integration (optional)

**Scaling Efficiency:**
- 2 GPUs: ~1.9x speedup
- 4 GPUs: ~3.5x speedup
- 8 GPUs: ~6.5x speedup

---

## Data Format Requirements

Your pretokenized FineWebEdu data must be:

### File Format
- Extension: `.npy` (NumPy) or `.pt` (PyTorch)
- Organized in a directory

### Shape Requirements
```python
# Each file must have shape:
(num_sequences, 32768)

# Example:
data.shape = (100, 32768)  # 100 sequences of 32k tokens each
```

### Token Requirements
- Data type: `int32`, `int64`, or `torch.long`
- Token IDs in range `[0, vocab_size)`
- Typically GPT-2 vocabulary: 50,257 tokens

### Directory Structure
```
pretokenized_data/
├── file_0000.npy  # (100, 32768)
├── file_0001.npy  # (100, 32768)
├── file_0002.npy  # (100, 32768)
└── ...
```

---

## Model Size Configurations

### Small (Testing)
**Parameters:** ~50M  
**VRAM:** ~10 GB

```bash
--model-dim 768
--num-frequencies 64
--num-layers 6
--batch-size 4
--gradient-accumulation 8
```

### Medium (Production)
**Parameters:** ~200M  
**VRAM:** ~20 GB  
**RECOMMENDED for most users**

```bash
--model-dim 1024
--num-frequencies 128
--num-layers 12
--batch-size 4
--gradient-accumulation 8
```

### Large (Research)
**Parameters:** ~500M  
**VRAM:** ~35 GB

```bash
--model-dim 2048
--num-frequencies 256
--num-layers 16
--batch-size 2
--gradient-accumulation 16
```

### XLarge (Maximum)
**Parameters:** ~1B-3B  
**VRAM:** ~44 GB (requires L40S or A100)

```bash
--model-dim 4096
--num-frequencies 512
--num-layers 24
--batch-size 1
--gradient-accumulation 32
```

---

## Training Time Estimates

Based on L40S GPU (44GB VRAM) with 3 billion tokens:

| Model | Batch Size | Throughput | Time (1 Epoch) |
|-------|-----------|------------|----------------|
| Small | 32 | ~1000 samples/sec | ~2 days |
| Medium | 32 | ~600 samples/sec | ~3 days |
| Large | 32 | ~300 samples/sec | ~5 days |
| XLarge | 32 | ~150 samples/sec | ~10 days |

*Note: With 4 GPUs, training time reduces by ~3.5x*

---

## Monitoring Training

### Terminal Output

During training:
```
Epoch 1/3: 100%|███████████| 10000/10000 [2:15:30<00:00, loss=3.2456, lr=5.8e-4]
✓ Saved checkpoint: checkpoints/checkpoint-1000.pt

Epoch 1 average loss: 3.2456
```

### Checkpoints

Saved every 1000 steps:
```
checkpoints/
├── checkpoint-1000.pt   # Step 1000
├── checkpoint-2000.pt   # Step 2000
├── checkpoint-3000.pt   # Step 3000
└── final_model.pt       # Final model
```

### WandB Dashboard (Distributed)

Enable with `--wandb` flag:
- Real-time loss curves
- Learning rate schedule
- GPU utilization
- Memory usage
- Tokens per second

---

## Troubleshooting

### Out of Memory

**Problem:** CUDA OOM error

**Solution:** Reduce batch size or model size
```bash
# Before (OOM)
--batch-size 4

# After (Fixed)
--batch-size 2 --gradient-accumulation 16  # Same effective batch
```

### Slow Training

**Problem:** Low throughput (< 100 samples/sec)

**Solutions:**
1. Check GPU utilization: `nvidia-smi`
2. Use multiple GPUs
3. Ensure data is on fast storage (NVMe SSD)
4. Increase `num_workers` in dataloader

### Data Format Errors

**Problem:** "No data files found" or shape mismatch

**Solution:** Run data validation
```bash
python scripts/prepare_data.py validate --data-path /path/to/data
```

### Loss Not Decreasing

**Problem:** Loss stays high

**Solutions:**
1. Lower learning rate: `--lr 3e-4` or `--lr 1e-4`
2. Check data quality
3. Increase warmup steps
4. Use gradient clipping (already enabled)

---

## Example Workflows

### Workflow 1: Quick Test

```bash
# Test with small model on subset of data
./scripts/quick_train.sh /path/to/data small
```

### Workflow 2: Production Training

```bash
# 1. Validate data
python scripts/prepare_data.py validate --data-path /path/to/data

# 2. Create split
python scripts/prepare_data.py split \
  --data-path /path/to/data \
  --output-path /path/to/split

# 3. Train medium model
./scripts/quick_train.sh /path/to/split medium
```

### Workflow 3: Distributed Large-Scale

```bash
# 1. Prepare data
python scripts/prepare_data.py split \
  --data-path /path/to/data \
  --output-path /path/to/split

# 2. Train on 4 GPUs with WandB
torchrun --nproc_per_node=4 \
  scripts/train_finewebedu_32k_distributed.py \
  --data-path /path/to/split/train \
  --model-dim 2048 \
  --num-layers 24 \
  --wandb
```

---

## Next Steps After Training

1. **Evaluate model:**
   ```bash
   python scripts/evaluate_model.py \
     --checkpoint checkpoints/final_model.pt \
     --data-path /path/to/val
   ```

2. **Compare with baselines:**
   ```bash
   python scripts/comparative_benchmark.py \
     --resonance-checkpoint checkpoints/final_model.pt
   ```

3. **Test generation:**
   ```bash
   python examples/text_generation.py \
     --checkpoint checkpoints/final_model.pt \
     --prompt "Once upon a time"
   ```

---

## Support

For questions or issues:
- **Documentation:** See `docs/TRAINING_GUIDE.md`
- **Email:** afolabi@genovotech.com
- **Architecture:** See `ARCHITECTURE.md`

---

**Genovo Technologies Research Team**  
*Building the future of efficient neural architectures*
