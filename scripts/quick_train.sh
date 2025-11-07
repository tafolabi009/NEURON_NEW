#!/bin/bash

# Quick Start Training Script for FineWebEdu 32k
# Genovo Technologies Research Team
# 
# Usage:
#   ./quick_train.sh /path/to/pretokenized/data [small|medium|large|xlarge]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ "$#" -lt 1 ]; then
    echo -e "${RED}Error: Missing data path${NC}"
    echo "Usage: $0 /path/to/data [small|medium|large|xlarge]"
    exit 1
fi

DATA_PATH="$1"
MODEL_SIZE="${2:-medium}"  # Default to medium

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}Error: Data path does not exist: $DATA_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FineWebEdu 32k Training - Quick Start${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Data path: $DATA_PATH"
echo "Model size: $MODEL_SIZE"
echo ""

# Step 1: Validate data
echo -e "${YELLOW}Step 1: Validating data...${NC}"
python scripts/prepare_data.py validate --data-path "$DATA_PATH"

if [ $? -ne 0 ]; then
    echo -e "${RED}Data validation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Data validation passed${NC}"
echo ""

# Step 2: Create train/val split (if not already split)
if [ ! -d "$DATA_PATH/train" ]; then
    echo -e "${YELLOW}Step 2: Creating train/val split...${NC}"
    
    SPLIT_PATH="${DATA_PATH}_split"
    python scripts/prepare_data.py split \
        --data-path "$DATA_PATH" \
        --output-path "$SPLIT_PATH" \
        --val-ratio 0.01
    
    TRAIN_PATH="$SPLIT_PATH/train"
    VAL_PATH="$SPLIT_PATH/val"
    
    echo -e "${GREEN}✓ Split created at $SPLIT_PATH${NC}"
else
    echo -e "${YELLOW}Step 2: Using existing train/val split${NC}"
    TRAIN_PATH="$DATA_PATH/train"
    VAL_PATH="$DATA_PATH/val"
fi

echo ""

# Step 3: Configure model based on size
echo -e "${YELLOW}Step 3: Configuring $MODEL_SIZE model...${NC}"

case $MODEL_SIZE in
    small)
        MODEL_DIM=768
        NUM_FREQ=64
        NUM_LAYERS=6
        BATCH_SIZE=4
        GRAD_ACCUM=8
        LR=6e-4
        ;;
    medium)
        MODEL_DIM=1024
        NUM_FREQ=128
        NUM_LAYERS=12
        BATCH_SIZE=4
        GRAD_ACCUM=8
        LR=6e-4
        ;;
    large)
        MODEL_DIM=2048
        NUM_FREQ=256
        NUM_LAYERS=16
        BATCH_SIZE=2
        GRAD_ACCUM=16
        LR=6e-4
        ;;
    xlarge)
        MODEL_DIM=4096
        NUM_FREQ=512
        NUM_LAYERS=24
        BATCH_SIZE=1
        GRAD_ACCUM=32
        LR=3e-4
        ;;
    *)
        echo -e "${RED}Error: Invalid model size. Choose from: small, medium, large, xlarge${NC}"
        exit 1
        ;;
esac

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))

echo "Configuration:"
echo "  Model dim: $MODEL_DIM"
echo "  Frequencies: $NUM_FREQ"
echo "  Layers: $NUM_LAYERS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Effective batch: $EFFECTIVE_BATCH"
echo "  Learning rate: $LR"
echo ""

# Step 4: Start training
OUTPUT_DIR="checkpoints/${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"

echo -e "${YELLOW}Step 4: Starting training...${NC}"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo -e "${GREEN}Training in progress...${NC}"
echo ""

python scripts/train_finewebedu_32k.py \
    --data-path "$TRAIN_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --vocab-size 50257 \
    --model-dim $MODEL_DIM \
    --num-frequencies $NUM_FREQ \
    --num-layers $NUM_LAYERS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation $GRAD_ACCUM \
    --max-seq-len 32768 \
    --epochs 3 \
    --lr $LR

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Training Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate: python scripts/evaluate_model.py --checkpoint $OUTPUT_DIR/final_model.pt"
    echo "  2. Compare: python scripts/comparative_benchmark.py --resonance-checkpoint $OUTPUT_DIR/final_model.pt"
    echo ""
else
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi
