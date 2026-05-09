#!/usr/bin/env python3
"""
Fine-tuning Script for Qwen 2.5 0.5B Instruct Model

This script fine-tunes the Qwen 2.5 0.5B Instruct model on financial news data
using LoRA (Low-Rank Adaptation) for efficient training.

Key features:
- LoRA for parameter-efficient fine-tuning
- Automatic device detection (CUDA/MPS/CPU)
- Comprehensive logging and debugging
- Evaluation during training
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def setup_device():
    """
    Detect and setup the best available device.
    
    Returns:
        device string and device object
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_obj = torch.device("cuda")
        print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        device_obj = torch.device("mps")
        print("✅ Using Apple Silicon (MPS)")
    else:
        device = "cpu"
        device_obj = torch.device("cpu")
        print("⚠️  Using CPU (training will be slow)")
    
    return device, device_obj


def load_model_and_tokenizer(model_name: str, device: str):
    """
    Load model and tokenizer with proper configuration.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device string
        
    Returns:
        model and tokenizer
    """
    print(f"\n📥 Loading model: {model_name}")
    print("   This may take a few minutes on first run...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    
    print(f"✅ Tokenizer loaded")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Pad token: {tokenizer.pad_token}")
    
    # Load model
    print(f"\n📥 Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Move to device if not using device_map
    if device != "cuda" or "device_map" not in str(model):
        model = model.to(device)
    
    print(f"✅ Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    return model, tokenizer


def load_training_data(train_path: str, eval_path: str):
    """
    Load and validate training data.
    
    Args:
        train_path: Path to training JSON file
        eval_path: Path to evaluation JSON file
        
    Returns:
        train_dataset and eval_dataset
    """
    print(f"\n📂 Loading training data...")
    
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")
    
    # Load JSON files
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    print(f"✅ Loaded {len(train_data)} training examples")
    print(f"✅ Loaded {len(eval_data)} evaluation examples")
    
    # Validate data format
    if len(train_data) == 0:
        raise ValueError("Training data is empty!")
    if len(eval_data) == 0:
        raise ValueError("Evaluation data is empty!")
    
    # Check structure
    if "messages" not in train_data[0]:
        raise ValueError("Training data must have 'messages' field")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    return train_dataset, eval_dataset


def preprocess_function(examples, tokenizer, max_length: int = 512):
    """
    Preprocess examples using chat template.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    # Handle both single examples and batches
    if "messages" in examples:
        # Single example
        messages = examples["messages"]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        tokenized = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    else:
        # Batch processing
        prompts = []
        for messages in examples["messages"]:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            prompts.append(prompt)
        
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized


def setup_lora(model, r: int = 16, lora_alpha: int = 64, lora_dropout: float = 0.1):
    """
    Setup LoRA configuration and apply to model.
    
    Args:
        model: Base model
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        
    Returns:
        PEFT model
    """
    print(f"\n🔧 Setting up LoRA...")
    print(f"   Rank (r): {r}")
    print(f"   Alpha: {lora_alpha}")
    print(f"   Dropout: {lora_dropout}")
    
    # Determine target modules based on model architecture
    # For Qwen models, these are typically the attention projection layers
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Try to find available modules
    model_modules = set([name.split('.')[0] for name, _ in model.named_modules()])
    available_targets = [m for m in target_modules if any(m in mod for mod in model_modules)]
    
    if not available_targets:
        # Fallback: use common module names
        available_targets = ["q_proj", "v_proj"]
        print(f"⚠️  Using default target modules: {available_targets}")
    else:
        available_targets = available_targets[:2]  # Use first 2 found
        print(f"✅ Using target modules: {available_targets}")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=available_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA applied")
    print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
    print(f"   Total parameters: {total_params / 1e6:.2f}M")
    
    return model


def main():
    """Main training function."""
    print("=" * 70)
    print("🚀 Qwen 2.5 7B Instruct Fine-tuning")
    print("=" * 70)
    
    # Configuration
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    train_path = "data_ready_20260419_v2/train.json"
    eval_path = "data_ready_20260419_v2/eval.json"
    output_dir = "./qwen2.5-7b-finetuned-3may26"
    logging_dir = "./logs"
    
    # Training hyperparameters
    max_length = 512
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 16
    learning_rate = 5e-5
    num_train_epochs = 15
    warmup_steps = 50
    logging_steps = 10
    eval_steps = 50
    save_steps = 100
    
    # LoRA hyperparameters
    lora_r = 16
    lora_alpha = 64
    lora_dropout = 0.1
    
    # Step 1: Setup device
    print("\n" + "=" * 70)
    print("STEP 1: Device Setup")
    print("=" * 70)
    device, device_obj = setup_device()
    
    # Step 2: Load model and tokenizer
    print("\n" + "=" * 70)
    print("STEP 2: Model Loading")
    print("=" * 70)
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Step 3: Load data
    print("\n" + "=" * 70)
    print("STEP 3: Data Loading")
    print("=" * 70)
    train_dataset, eval_dataset = load_training_data(train_path, eval_path)
    
    # Step 4: Preprocess data
    print("\n" + "=" * 70)
    print("STEP 4: Data Preprocessing")
    print("=" * 70)
    print("Tokenizing datasets (this may take a moment)...")
    
    def tokenize_function(examples):
        return preprocess_function(examples, tokenizer, max_length=max_length)
    
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )
    
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing evaluation data"
    )
    
    print(f"✅ Tokenization complete")
    print(f"   Training examples: {len(tokenized_train)}")
    print(f"   Evaluation examples: {len(tokenized_eval)}")
    
    # Step 5: Setup LoRA
    print("\n" + "=" * 70)
    print("STEP 5: LoRA Setup")
    print("=" * 70)
    model = setup_lora(model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    # Step 6: Training arguments
    print("\n" + "=" * 70)
    print("STEP 6: Training Configuration")
    print("=" * 70)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        
        # Training parameters
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        
        # Logging and saving
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        
        # Other settings
        fp16=device == "cuda",  # Use FP16 on CUDA
        bf16=False,
        dataloader_pin_memory=True if device == "cuda" else False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    print(f"✅ Training arguments configured:")
    print(f"   Batch size: {per_device_train_batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_train_epochs}")
    print(f"   Max length: {max_length}")
    print(f"   Output directory: {output_dir}")
    
    # Step 7: Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Step 8: Create trainer
    print("\n" + "=" * 70)
    print("STEP 7: Trainer Setup")
    print("=" * 70)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator
    )
    
    print("✅ Trainer created")
    
    # Step 9: Training
    print("\n" + "=" * 70)
    print("STEP 8: Starting Training")
    print("=" * 70)
    print("🚀 Training started! Check logs/ directory for TensorBoard logs.")
    print("   You can monitor training with: tensorboard --logdir=./logs")
    print()
    
    try:
        train_result = trainer.train()
        
        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        print(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Step 10: Save model
        print("\n" + "=" * 70)
        print("STEP 9: Saving Model")
        print("=" * 70)
        
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model saved to: {output_dir}")
        print(f"✅ Tokenizer saved to: {output_dir}")
        
        # Final evaluation
        print("\n" + "=" * 70)
        print("STEP 10: Final Evaluation")
        print("=" * 70)
        
        eval_results = trainer.evaluate()
        print(f"✅ Final evaluation results:")
        for key, value in eval_results.items():
            print(f"   {key}: {value:.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💾 Saving checkpoint...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Checkpoint saved to: {output_dir}")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 70)
    print("🎉 Fine-tuning Complete!")
    print("=" * 70)
    print(f"\n📁 Model saved to: {output_dir}")
    print(f"📊 Logs saved to: {logging_dir}")
    print(f"\n💡 To use the model:")
    print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"   from peft import PeftModel")
    print(f"   ")
    print(f"   base_model = AutoModelForCausalLM.from_pretrained('{model_name}')")
    print(f"   model = PeftModel.from_pretrained(base_model, '{output_dir}')")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")


if __name__ == "__main__":
    main()
