#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timezone

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make repo-root imports work when script is run as: python scripts/demo_live_cli.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SYSTEM_PROMPT = """You are a financial news analyst specializing in cause and effect of stock price movement. Extract structured information from news articles.
Output a JSON object with these exact fields:
- company: The company name
- ticker: The stock ticker symbol
- cause: What caused the stock movement
- effect: The resulting stock price movement
- event_category: One of [earnings_beat, earnings_miss, guidance_change, deal_announced, analyst_action, regulatory_action, product_event, corporate_governance, macro_sentiment, no_stock_movement]
- impact_direction: One of [positive, negative, neutral]"""

FIELDS = ["company", "ticker", "cause", "effect", "event_category", "impact_direction"]
VALID_CATEGORIES = {
    "earnings_beat",
    "earnings_miss",
    "guidance_change",
    "deal_announced",
    "analyst_action",
    "regulatory_action",
    "product_event",
    "corporate_governance",
    "macro_sentiment",
    "no_stock_movement",
}
VALID_DIRECTIONS = {"positive", "negative", "neutral"}


def extract_json_from_response(response: str):
    response = re.sub(r"```json\s*", "", response)
    response = re.sub(r"```\s*", "", response).strip()
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None
    payload = match.group(0)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def load_baseline(base_model: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_finetuned(base_model: str, adapter: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, adapter)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_model(model, tokenizer, news_text: str):
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
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed = extract_json_from_response(raw)
    if parsed is None:
        parsed = {}
    parsed["raw_text"] = raw
    return parsed


def quality_flags(pred):
    missing = [f for f in FIELDS if not pred.get(f)]
    cat = str(pred.get("event_category") or "").strip().lower()
    direction = str(pred.get("impact_direction") or "").strip().lower()
    return {
        "schema_valid": len(missing) == 0,
        "missing_fields": missing,
        "category_valid": cat in VALID_CATEGORIES,
        "impact_valid": direction in VALID_DIRECTIONS,
    }


def print_block(title: str, payload: dict):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    clean = {k: payload.get(k) for k in FIELDS}
    print(json.dumps(clean, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Live CLI demo: baseline vs finetuned on one news snippet")
    parser.add_argument("--news", type=str, default="", help="News text. If empty, will prompt interactively.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="./qwen2.5-7b-finetuned-3may26")
    parser.add_argument("--skip-baseline", action="store_true", help="Run only finetuned inference.")
    parser.add_argument(
        "--save",
        type=str,
        default="runs/live_demo/live_demo_runs.jsonl",
        help="JSONL path to append run results; set empty string to disable.",
    )
    args = parser.parse_args()

    news = args.news.strip()
    if not news:
        print("Paste news text, then press Ctrl-D (Linux/macOS) or Ctrl-Z then Enter (Windows):")
        news = "\n".join(line.rstrip("\n") for line in __import__("sys").stdin).strip()
    if not news:
        raise ValueError("No news text provided.")

    print(f"\nDevice: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("\nInput News:")
    print(textwrap.shorten(news, width=260, placeholder=" ..."))

    ft_model, ft_tokenizer = load_finetuned(args.base_model, args.adapter)
    start = time.time()
    finetuned = run_model(ft_model, ft_tokenizer, news)
    ft_latency = time.time() - start
    ft_flags = quality_flags(finetuned)

    baseline = None
    bl_latency = None
    bl_flags = None
    if not args.skip_baseline:
        bl_model, bl_tokenizer = load_baseline(args.base_model)
        start = time.time()
        baseline = run_model(bl_model, bl_tokenizer, news)
        bl_latency = time.time() - start
        bl_flags = quality_flags(baseline)

    print_block("FINETUNED OUTPUT", finetuned)
    print("Quality:", json.dumps(ft_flags, ensure_ascii=False))
    print(f"Latency: {ft_latency:.2f}s")

    if baseline is not None:
        print_block("BASELINE OUTPUT", baseline)
        print("Quality:", json.dumps(bl_flags, ensure_ascii=False))
        print(f"Latency: {bl_latency:.2f}s")
        if ft_latency > 0:
            print(f"\nSpeed ratio (baseline/finetuned): {bl_latency / ft_latency:.2f}x")

    if args.save.strip():
        save_path = args.save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "news": news,
            "base_model": args.base_model,
            "adapter": args.adapter,
            "finetuned": {
                "output": {k: finetuned.get(k) for k in FIELDS},
                "quality": ft_flags,
                "latency_s": round(ft_latency, 4),
            },
            "baseline": None
            if baseline is None
            else {
                "output": {k: baseline.get(k) for k in FIELDS},
                "quality": bl_flags,
                "latency_s": round(bl_latency, 4),
            },
        }
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved run record: {save_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
