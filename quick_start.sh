#!/bin/bash
# Quick start script for fine-tuning Qwen 2.5 0.5B on financial news

set -e  # Exit on error

echo "============================================================"
echo "🚀 Quick Start: Financial News Fine-tuning"
echo "============================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "data/sample_financial_news_data_updated.json" ]; then
    echo -e "${RED}❌ Error: data/sample_financial_news_data_updated.json not found!${NC}"
    echo "   Please run this script from the IP-LLM directory"
    exit 1
fi

# Step 1: Prepare dataset
echo -e "${GREEN}📊 Step 1: Preparing dataset...${NC}"
python prepare_dataset.py \
    --input data/sample_financial_news_data_updated.json \
    --output-dir data \
    --test-size 0.2 \
    --seed 42

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Dataset preparation failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Dataset prepared successfully!${NC}"
echo ""

# Step 2: Check if we should proceed with training
echo -e "${YELLOW}⚠️  Next step: Training${NC}"
echo ""
echo "To start training, run:"
echo "  python finetune/finetuning.py"
echo ""
echo "Or if using SLURM:"
echo "  srun --account=msccsit2024 --gpus-per-node=1 --partition=normal \\"
echo "       --gres=gpu:1 --time=02:00:00 --cpus-per-task=4 --mem=16G \\"
echo "       --pty /bin/bash"
echo "  python finetune/finetuning.py"
echo ""
echo -e "${GREEN}✅ Quick start complete!${NC}"

