# Quick Start Guide

## 🚀 Fast Track (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python prepare_dataset.py
```
This loads `data/sample_financial_news_data_updated.json`, converts it to the messages format, and creates balanced train/eval splits in `data/` directory.

### 3. Train on H800 GPU
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

# Run training
python finetune/finetuning.py
```

## 📋 What Each Script Does

### `prepare_dataset.py`
- ✅ Loads `financial_news_training.jsonl`
- ✅ Validates data format
- ✅ Creates balanced train/eval split by event category
- ✅ Saves to `data/train.json` and `data/eval.json`

### `finetune/finetuning.py`
- ✅ Loads Qwen 2.5 0.5B Instruct model
- ✅ Applies LoRA adapters (efficient fine-tuning)
- ✅ Trains on financial news data
- ✅ Evaluates during training
- ✅ Saves model to `qwen2.5-0.5b-finetuned/`

### `test_model.py`
- ✅ Loads fine-tuned model
- ✅ Tests on sample financial news
- ✅ Shows extracted structured information

## 🎯 Expected Output

**After `prepare_dataset.py`:**
```
✅ Loaded 159 examples
✅ Split Summary:
  Training examples: 127
  Evaluation examples: 32
```

**After `finetuning.py`:**
```
✅ Training Complete!
Final training loss: 0.1234
✅ Model saved to: ./qwen2.5-0.5b-finetuned
```

## 📊 Training Time

- **H800 GPU**: ~1-2 hours
- **A100 GPU**: ~1-2 hours  
- **V100 GPU**: ~2-3 hours
- **CPU**: Not recommended (10+ hours)

## 🔍 Verify Everything Works

```bash
# Check data files exist
ls -lh data/train.json data/eval.json

# Check model was created
ls -lh qwen2.5-0.5b-finetuned/

# Test the model
python test_model.py
```

## 📚 Full Documentation

See `README_FINETUNING.md` for complete details.

## 🆘 Common Issues

**"CUDA out of memory"**
- Reduce batch size in `finetuning.py`: `per_device_train_batch_size = 2`

**"File not found: data/train.json"**
- Run `python prepare_dataset.py` first

**Training loss not decreasing**
- Check learning rate (try 1e-5)
- Verify data quality

---

**That's it! Happy fine-tuning! 🎉**

