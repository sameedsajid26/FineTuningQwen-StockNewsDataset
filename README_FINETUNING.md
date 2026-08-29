# Financial News Fine-tuning Guide

Complete end-to-end guide for fine-tuning Qwen 2.5 7B Instruct on financial news data.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Understanding the Process](#understanding-the-process)
6. [Training on H800 GPU](#training-on-h800-gpu)
7. [Troubleshooting](#troubleshooting)
8. [Using the Fine-tuned Model](#using-the-fine-tuned-model)

---

## 🎯 Overview

This project fine-tunes the **Qwen 2.5 7B Instruct** model to extract structured information from financial news articles. The model learns to identify:
- Company name and ticker
- Cause of stock movement
- Effect (stock price change)
- Event category (earnings_miss, deal_announced, etc.)
- Impact direction (positive/negative)

**Key Features:**
- ✅ Balanced train/eval split by event category
- ✅ LoRA for efficient fine-tuning (only ~1% of parameters trained)
- ✅ Comprehensive logging and debugging
- ✅ Automatic GPU detection and optimization
- ✅ Evaluation during training

---

## 📦 Prerequisites

### Software Requirements

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements

- **GPU**: NVIDIA H800 (or any CUDA-compatible GPU with 8GB+ VRAM)
- **RAM**: 16GB+ recommended
- **Storage**: 5GB+ free space

---

## 📁 Project Structure

```
IP-LLM/
├── data/
│   ├── sample_financial_news_data_updated.json  # Raw training data (JSON format)
│   ├── train.json                   # Training set (created by prepare_dataset.py)
│   └── eval.json                    # Evaluation set (created by prepare_dataset.py)
├── prepare_dataset.py                # Data preparation script
├── finetune/
│   └── finetuning.py                # Main fine-tuning script
├── qwen2.5-7b-finetuned/         # Fine-tuned model (created after training)
├── logs/                            # Training logs (TensorBoard)
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Step-by-Step Guide

### Step 1: Prepare Your Environment

```bash
# Activate your environment (if using conda/venv)
# conda activate your_env
# or
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare the Dataset

This step:
- Loads data from `data/sample_financial_news_data_updated.json`
- Validates data format
- Creates balanced train/eval split (ensures all event categories in both sets)
- Saves processed data to `data/` directory

```bash
python prepare_dataset.py \
    --input data/sample_financial_news_data_updated.json \
    --output-dir data \
    --test-size 0.2 \
    --seed 42
```

**What you'll see:**
```
============================================================
🚀 Financial News Dataset Preparation
============================================================

📂 Step 1: Loading data from data/sample_financial_news_data_updated.json...
✅ Loaded 200+ examples

🔍 Step 2: Validating data...
  Total examples: 159
  Valid examples: 159
  Invalid examples: 0

⚖️  Step 3: Creating balanced train/eval split...

📊 Event Category Distribution:
  analyst_action: 15 examples
  corporate_governance: 12 examples
  deal_announced: 25 examples
  earnings_beat: 20 examples
  earnings_miss: 18 examples
  ...

✅ Split Summary:
  Training examples: 127
  Evaluation examples: 32
```

### Step 3: Verify Data (Optional but Recommended)

```bash
# Check the data files
python -c "
import json
with open('data/train.json') as f:
    train = json.load(f)
with open('data/eval.json') as f:
    eval_data = json.load(f)
print(f'Training: {len(train)} examples')
print(f'Evaluation: {len(eval_data)} examples')
print(f'\\nFirst training example:')
print(json.dumps(train[0], indent=2))
"
```

### Step 4: Start Training on H800 GPU

#### Option A: Using SLURM (Recommended for H800)

```bash
# Request GPU node
srun --account=msccsit2024 \
     --gpus-per-node=1 \
     --partition=normal \
     --gres=gpu:1 \
     --time=02:00:00 \
     --cpus-per-task=4 \
     --mem=16G \
     --pty /bin/bash

# Once on the node, navigate to your project
cd /path/to/IP-LLM

# Run training
python finetune/finetuning.py
```

#### Option B: Direct Training (if you have direct GPU access)

```bash
python finetune/finetuning.py
```

**Training Output:**
```
======================================================================
🚀 Qwen 2.5 7B Instruct Fine-tuning
======================================================================

STEP 1: Device Setup
======================================================================
✅ Using CUDA device: NVIDIA H800
   GPU Memory: 80.00 GB

STEP 2: Model Loading
======================================================================
📥 Loading model: Qwen/Qwen2.5-7B-Instruct
✅ Tokenizer loaded
✅ Model loaded
   Parameters: 7.62B
   Trainable: ~5M (~0.07%)

STEP 3: Data Loading
======================================================================
✅ Loaded 127 training examples
✅ Loaded 32 evaluation examples

...

STEP 8: Starting Training
======================================================================
🚀 Training started! Check logs/ directory for TensorBoard logs.
```

### Step 5: Monitor Training

**Option 1: TensorBoard (Recommended)**
```bash
# In a separate terminal
tensorboard --logdir=./logs --port=6006

# Then open http://localhost:6006 in your browser
```

**Option 2: Check Log Files**
```bash
# View latest training logs
tail -f logs/training.log  # if logging to file
```

**Option 3: Watch GPU Usage**
```bash
# In another terminal (if on the same node)
watch -n 1 nvidia-smi
```

### Step 6: Verify Training Completed

After training, you should see:
```
✅ Training Complete!
Final training loss: 0.1234

✅ Model saved to: ./qwen2.5-7b-finetuned
✅ Tokenizer saved to: ./qwen2.5-7b-finetuned
```

Check the output directory:
```bash
ls -lh qwen2.5-7b-finetuned/
# Should contain:
# - adapter_config.json
# - adapter_model.bin (or adapter_model.safetensors)
# - tokenizer files
```

---

## 🧠 Understanding the Process

### 1. Data Preparation (`prepare_dataset.py`)

**What it does:**
- Reads JSONL file line by line
- Extracts `event_category` from each example's assistant response
- Groups examples by category
- Splits each category proportionally (80% train, 20% eval)
- Ensures every category appears in both train and eval sets

**Why balanced split?**
- Prevents overfitting to common categories
- Ensures model sees all event types during training
- Provides fair evaluation across all categories

### 2. Fine-tuning (`finetuning.py`)

**Architecture:**
```
Base Model (Qwen 2.5 7B)
    ↓
LoRA Adapters (only these are trained)
    ↓
Fine-tuned Model
```

**LoRA (Low-Rank Adaptation):**
- Only trains ~0.07% of model parameters (~5M of 7.6B)
- Much faster and uses less memory
- Can be merged back into base model if needed

**Training Process:**
1. Load base model and tokenizer
2. Apply LoRA adapters to attention layers
3. Tokenize training data using chat template
4. Train on financial news examples
5. Evaluate periodically on held-out set
6. Save adapter weights

### 3. Data Flow

```
JSONL File
    ↓
prepare_dataset.py
    ↓
train.json + eval.json
    ↓
finetuning.py (tokenization)
    ↓
Tokenized Datasets
    ↓
SFTTrainer (training loop)
    ↓
LoRA Adapters
```

---

## 🖥️ Training on H800 GPU

### SLURM Script Example

Create `train.sh`:
```bash
#!/bin/bash
#SBATCH --account=msccsit2024
#SBATCH --job-name=qwen_finetune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Load modules (adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate environment
# source activate your_env

# Run training
cd /path/to/IP-LLM
python finetune/finetuning.py
```

Submit job:
```bash
sbatch train.sh
```

Check status:
```bash
squeue -u $USER
```

### Resource Recommendations

| Resource | Recommended | Minimum |
|----------|-------------|---------|
| GPU Memory | 16GB+ | 8GB |
| System RAM | 16GB | 8GB |
| CPU Cores | 4+ | 2 |
| Training Time | 1-2 hours | 3-4 hours |

---

## 🔧 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
- Reduce `per_device_train_batch_size` in `finetuning.py` (try 2 or 1)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `max_length` (try 256 instead of 512)

### Issue: "File not found: data/train.json"

**Solution:**
```bash
# Make sure you ran prepare_dataset.py first
python prepare_dataset.py --input financial_news_training.jsonl --output-dir data
```

### Issue: "Invalid JSON in assistant response"

**Solution:**
- Check your JSONL file format
- Each line should be valid JSON
- Assistant message should contain valid JSON

### Issue: Training loss not decreasing

**Possible causes:**
- Learning rate too high/low (try 1e-5 or 2e-5)
- Not enough training steps (increase epochs)
- Data quality issues (check your examples)

### Issue: Model not saving

**Solution:**
- Check disk space: `df -h`
- Check write permissions: `ls -ld qwen2.5-7b-finetuned/`
- Ensure training completes (not interrupted)

---

## 📊 Monitoring Training

### Key Metrics to Watch

1. **Training Loss**: Should decrease over time
2. **Evaluation Loss**: Should track training loss (if diverging, may be overfitting)
3. **Learning Rate**: Should follow warmup then decay
4. **GPU Utilization**: Should be 80-100% during training

### TensorBoard Commands

```bash
# Start TensorBoard
tensorboard --logdir=./logs

# View specific run
tensorboard --logdir=./logs --port=6006
```

### Checkpoints

Training saves checkpoints at:
- `qwen2.5-7b-finetuned/checkpoint-100/`
- `qwen2.5-7b-finetuned/checkpoint-200/`
- etc.

To resume from checkpoint:
```python
# Modify finetuning.py to add:
# training_args.resume_from_checkpoint = "qwen2.5-7b-finetuned/checkpoint-200"
```

---

## 🎯 Using the Fine-tuned Model

### Load and Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned adapters
model = PeftModel.from_pretrained(base_model, "./qwen2.5-7b-finetuned")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./qwen2.5-7b-finetuned")

# Prepare input
messages = [
    {"role": "system", "content": "You are a financial news analyst. Extract structured information from news articles."},
    {"role": "user", "content": "Analyze this financial news:\n\nTesla shares rose 5% after reporting strong earnings."}
]

# Generate
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Merge LoRA Adapters (Optional)

To create a standalone model without needing base + adapters:

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "./qwen2.5-7b-finetuned")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./qwen2.5-7b-finetuned-merged")
```

---

## 📝 Summary

**Quick Start:**
```bash
# 1. Prepare data
python prepare_dataset.py

# 2. Train (on GPU node)
srun --gpus-per-node=1 --pty /bin/bash
python finetune/finetuning.py

# 3. Use model
python test_model.py  # (create this to test)
```

**Expected Results:**
- Training loss: ~0.1-0.3 (depends on data)
- Evaluation loss: Similar to training loss
- Training time: 1-2 hours on H800
- Model size: ~20MB (just adapters) or ~15GB (merged)

---

## 🆘 Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review training logs in `logs/` directory
3. Verify data format with `prepare_dataset.py` output
4. Check GPU memory usage: `nvidia-smi`

---

**Happy Fine-tuning! 🚀**

