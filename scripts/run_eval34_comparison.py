#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are a financial news analyst specializing in cause and effect of stock price movement. Extract structured information from news articles.
Output a JSON object with these exact fields:
- company: The company name
- ticker: The stock ticker symbol
- cause: What caused the stock movement
- effect: The resulting stock price movement
- event_category: One of [earnings_beat, earnings_miss, guidance_change, deal_announced, analyst_action, regulatory_action, product_event, corporate_governance, macro_sentiment, no_stock_movement]
- impact_direction: One of [positive, negative, neutral]"""


def normalize(v):
    if v is None:
        return ""
    return str(v).strip().lower()


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


def load_eval_samples(eval_path: str):
    data = json.load(open(eval_path, "r", encoding="utf-8"))
    samples = []
    for item in data:
        messages = item.get("messages", [])
        user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        assistant_msg = next((m.get("content", "{}") for m in messages if m.get("role") == "assistant"), "{}")

        news = user_msg.split("Financial news:")[-1]
        if "Respond with only" in news:
            news = news.split("Respond with only")[0]
        news = news.strip()

        try:
            gt = json.loads(assistant_msg)
        except json.JSONDecodeError:
            gt = {}

        samples.append({"text": news, "ground_truth": gt})
    return samples


def generate_predictions(model, tokenizer, samples):
    out = []
    for idx, s in enumerate(samples, 1):
        user_prompt = f"""Extract structured information from this financial news article.

Financial news:

{s['text']}

Respond with only the JSON object, no additional text."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = [o[len(i):] for i, o in zip(inputs.input_ids, outputs)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        parsed = extract_json_from_response(response)
        out.append(
            {
                "test_case": idx,
                "input": s["text"],
                "output": parsed if parsed is not None else "JSON_PARSE_ERROR",
                "status": "success" if parsed is not None else "parse_error",
            }
        )
        if idx % 5 == 0:
            print(f"processed {idx}/{len(samples)}")
    return out


def calc_exact_metrics(samples, preds):
    total = min(len(samples), len(preds))
    fields = ["company", "ticker", "event_category", "impact_direction"]
    matches = {f: 0 for f in fields}
    category_mismatches = Counter()
    parse_errors = 0

    for i in range(total):
        gt = samples[i]["ground_truth"]
        pred = preds[i].get("output", {})
        if not isinstance(pred, dict):
            parse_errors += 1
            continue

        for field in fields:
            if normalize(gt.get(field)) == normalize(pred.get(field)):
                matches[field] += 1

        gt_cat = normalize(gt.get("event_category"))
        pred_cat = normalize(pred.get("event_category"))
        if gt_cat != pred_cat:
            category_mismatches[(gt_cat, pred_cat)] += 1

    metrics = {
        "total_compared": total,
        "parse_errors": parse_errors,
        "exact_match": {
            f: {
                "matches": matches[f],
                "total": total,
                "accuracy": (matches[f] / total if total else 0.0),
            }
            for f in fields
        },
        "top_category_mismatches": [
            {"ground_truth": k[0], "predicted": k[1], "count": v}
            for k, v in category_mismatches.most_common(10)
        ],
    }
    return metrics


def write_markdown(out_path, baseline_metrics, finetuned_metrics):
    lines = []
    lines.append("# Eval34 Baseline vs Finetuned")
    lines.append("")
    lines.append("## Exact Match Accuracy")
    for f in ["company", "ticker", "event_category", "impact_direction"]:
        b = baseline_metrics["exact_match"][f]["accuracy"] * 100
        t = finetuned_metrics["exact_match"][f]["accuracy"] * 100
        d = t - b
        lines.append(f"- `{f}`: baseline `{b:.1f}%` -> finetuned `{t:.1f}%` ({d:+.1f} pp)")
    lines.append("")
    lines.append("## Parse Errors")
    lines.append(
        f"- baseline: {baseline_metrics['parse_errors']}/{baseline_metrics['total_compared']}, "
        f"finetuned: {finetuned_metrics['parse_errors']}/{finetuned_metrics['total_compared']}"
    )
    lines.append("")
    lines.append("## Top Finetuned Category Mismatches")
    if not finetuned_metrics["top_category_mismatches"]:
        lines.append("- none")
    else:
        for m in finetuned_metrics["top_category_mismatches"][:8]:
            lines.append(
                f"- `{m['ground_truth']} -> {m['predicted']}`: {m['count']}"
            )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="data_ready_20260419_v2/eval.json")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="./qwen2.5-7b-finetuned-3may26")
    parser.add_argument("--outdir", default="runs/2026-05-03_eval34_comparison")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    samples = load_eval_samples(args.eval)
    print("samples:", len(samples))

    # Baseline
    print("\n[1/2] Running baseline predictions...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    baseline_preds = generate_predictions(base_model, base_tokenizer, samples)
    baseline_path = os.path.join(args.outdir, "baseline_results_qwen7b_eval34.json")
    json.dump(baseline_preds, open(baseline_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    baseline_metrics = calc_exact_metrics(samples, baseline_preds)
    print("baseline saved:", baseline_path)

    del base_model
    torch.cuda.empty_cache()

    # Finetuned
    print("\n[2/2] Running finetuned predictions...")
    ft_base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    ft_model = PeftModel.from_pretrained(ft_base, args.adapter)
    ft_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token
    finetuned_preds = generate_predictions(ft_model, ft_tokenizer, samples)
    finetuned_path = os.path.join(args.outdir, "finetuned_results_qwen7b_eval34.json")
    json.dump(finetuned_preds, open(finetuned_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    finetuned_metrics = calc_exact_metrics(samples, finetuned_preds)
    print("finetuned saved:", finetuned_path)

    summary_json = os.path.join(args.outdir, "comparison_eval34_exact.json")
    json.dump(
        {"baseline": baseline_metrics, "finetuned": finetuned_metrics},
        open(summary_json, "w", encoding="utf-8"),
        indent=2,
    )
    summary_md = os.path.join(args.outdir, "comparison_eval34_summary.md")
    write_markdown(summary_md, baseline_metrics, finetuned_metrics)
    print("comparison saved:", summary_json)
    print("summary saved:", summary_md)


if __name__ == "__main__":
    main()
