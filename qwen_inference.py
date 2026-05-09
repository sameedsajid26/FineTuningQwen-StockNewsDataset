#!/usr/bin/env python3
"""
Inference utilities for the fine-tuned Qwen 2.5 7B model.

This module:
- Loads the base Qwen/Qwen2.5-7B-Instruct model with LoRA adapters (qwen2.5-7b-finetuned).
- Uses the SAME prompt format as training / evaluation.
- Provides a simple `analyze_financial_news` function that returns a parsed dict:
  {
      "company": str or None,
      "ticker": str or None,
      "cause": str or None,
      "effect": str or None,
      "event_category": str or None,
      "impact_direction": str or None,
      "raw_text": original model text (for debugging)
  }
"""

import json
import os
import re
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# Default adapter path for the fine-tuned LoRA weights
ADAPTER_PATH = "./qwen2.5-7b-finetuned"


# ---------------------------------------------------------------------------
# Lazy / cached model loading
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None


def _load_finetuned_model(
    base_model_name: str = BASE_MODEL_NAME,
    adapter_path: str = ADAPTER_PATH,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the fine-tuned model with LoRA adapters.

    This mirrors the logic from `test_finetuned_model.load_finetuned_model`
    but is simplified for single-text inference.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"📥 Loading base model: {base_model_name} on device={device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device != "cuda":
        base_model = base_model.to(device)

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")

    print(f"📥 Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("📥 Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Global accessor for model/tokenizer, loaded once per process.

    Streamlit should wrap this with `st.cache_resource` in the UI layer.
    """
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        _MODEL, _TOKENIZER = _load_finetuned_model()
    return _MODEL, _TOKENIZER


# ---------------------------------------------------------------------------
# Prompt construction (MATCH TRAINING FORMAT)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a financial news analyst specializing in cause and effect of stock price movement. "
    "Extract structured information from news articles.\n"
    "Output a JSON object with these exact fields:\n"
    "- company: The company name\n"
    "- ticker: The stock ticker symbol\n"
    "- cause: What caused the stock movement\n"
    "- effect: The resulting stock price movement\n"
    "- event_category: One of [earnings_beat, earnings_miss, guidance_change, "
    "deal_announced, analyst_action, regulatory_action, product_event, "
    "corporate_governance, macro_sentiment, no_stock_movement]\n"
    "- impact_direction: One of [positive, negative, neutral]"
)


def build_messages(news_text: str) -> Any:
    """
    Build chat-style messages for the Qwen chat template.
    """
    user_prompt = (
        "Extract structured information from this financial news article.\n\n"
        "Financial news:\n\n"
        f"{news_text}\n\n"
        "Respond with only the JSON object, no additional text."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

JSON_FIELD_NAMES = [
    "company",
    "ticker",
    "cause",
    "effect",
    "event_category",
    "impact_direction",
]


def _extract_json_text(raw: str) -> str:
    """
    Extract the JSON object text from the model's raw string output.

    Handles cases like:
    - Plain JSON
    - Wrapped in Markdown code fences
    - Extra text before/after JSON
    """
    text = raw.strip()

    # If wrapped in ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?(.*)```", text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Try to find the first '{' and last '}' to isolate JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def parse_prediction(raw: str) -> Dict[str, Any]:
    """
    Parse the model raw output into a structured dict with the expected fields.

    Returns a dict with all 6 fields (possibly None) plus `raw_text`.
    """
    cleaned = _extract_json_text(raw)

    data: Dict[str, Any]
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    result: Dict[str, Any] = {name: None for name in JSON_FIELD_NAMES}
    for name in JSON_FIELD_NAMES:
        if isinstance(data.get(name), str) and data[name].strip():
            result[name] = data[name].strip()
        else:
            # also try lowercase keys just in case
            lower_keys = {k.lower(): v for k, v in data.items()}
            if name in lower_keys and isinstance(lower_keys[name], str):
                val = lower_keys[name].strip()
                result[name] = val or None

    result["raw_text"] = raw
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_raw(news_text: str, max_new_tokens: int = 256) -> str:
    """
    Run the fine-tuned model on the given news text and return the raw output string.
    """
    if not news_text or not news_text.strip():
        raise ValueError("news_text must be a non-empty string")

    model, tokenizer = get_model_and_tokenizer()

    messages = build_messages(news_text)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Strip the prompt tokens from the output
    generated_ids = [
        output_ids[len(input_ids) :]  # noqa: E203  (black formatting)
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def analyze_financial_news(news_text: str) -> Dict[str, Any]:
    """
    High-level helper: analyze a single financial news article and return
    structured fields plus the raw model output.
    """
    raw = generate_raw(news_text)
    parsed = parse_prediction(raw)
    parsed["raw_text"] = raw
    return parsed


__all__ = [
    "analyze_financial_news",
    "generate_raw",
    "parse_prediction",
    "get_model_and_tokenizer",
]


