<div align="center">

# 📈 Fine-Tuning Qwen 2.5 7B for Financial Cause → Effect Extraction

**Teaching a small open LLM to read financial news like an analyst — turning messy headlines into clean, structured, machine-readable signals.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-FFD21E)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-6E56CF)](https://github.com/huggingface/peft)
[![Base Model](https://img.shields.io/badge/Base-Qwen2.5--7B--Instruct-2563EB)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

</div>

---

## 🎯 What this project does

General-purpose LLMs can summarize a news headline, but they struggle with the two things a financial pipeline actually needs: **domain-specific causal nuance** and **strict, consistent output structure**. This project fine-tunes **Qwen 2.5 7B Instruct** using **LoRA** to extract the *cause-and-effect* behind a stock movement from a raw news snippet, and emit it as a fixed JSON schema.

```text
INPUT  ►  "Nio ADRs slump 6.7% after its sovereign-wealth backer
           was sued for allegedly inflating revenue…"

OUTPUT ►  {
             "company":          "Nio Inc.",
             "ticker":           "NIO",
             "cause":            "Backer sued for allegedly inflating revenue",
             "effect":           "ADRs fell 6.7%",
             "event_category":   "regulatory_action",
             "impact_direction": "negative"
           }
```

### The 6-field schema

| Field | Description |
|---|---|
| `company` | Primary entity mentioned in the text |
| `ticker` | Stock symbol (e.g. `AAPL`, `MSFT`) |
| `cause` | The specific event/action driving the movement *(free-form)* |
| `effect` | The resulting market movement or outcome *(free-form)* |
| `event_category` | One of 10 fixed categories *(see below)* |
| `impact_direction` | `positive` · `negative` · `neutral` |

<details>
<summary><b>The 10 event categories</b></summary>

`earnings_beat` · `earnings_miss` · `guidance_change` · `deal_announced` · `analyst_action` · `regulatory_action` · `product_event` · `corporate_governance` · `macro_sentiment` · `no_stock_movement`

</details>

---

## 📊 Results at a glance

The fine-tuned model was compared against the **Qwen 2.5 7B base** on a held-out test set, using a hybrid evaluation: **exact-match** for categorical fields, **semantic similarity** (`all-MiniLM-L6-v2` cosine) for free-form text, and an **LLM-as-a-judge** (GPT-4o) for reasoning quality.

| Metric | Baseline | Fine-Tuned | Δ |
|---|:---:|:---:|:---:|
| **Cause accuracy** *(GPT-4o judged)* | 0.693 | **0.852** | 🟢 **+15.8%** |
| Cause similarity *(embedding cosine)* | 0.653 | **0.758** | 🟢 +10.5% |
| Effect similarity *(embedding cosine)* | 0.772 | **0.805** | 🟢 +3.3% |
| Ticker match | 58.1% | **61.3%** | 🟢 +3.2% |
| Event-category match | 64.5% | 64.5% | ⚪ 0.0% |
| Impact-direction match | 83.9% | 83.9% | ⚪ 0.0% |

> **The honest takeaway:** fine-tuning delivered a clear win on the *hard* part — extracting the **specific cause** and enforcing schema adherence (no more hallucinated, out-of-schema labels). But on **simple classification** (sentiment, event category) it changed *nothing* — a modern instruction-tuned base model already handles those well. That "where it helped vs. where it didn't" split is the most useful finding in the whole project: **fine-tune for nuance and structure, not for tasks the base model already nails.**

---

## 🧠 Approach: LoRA (Parameter-Efficient Fine-Tuning)

A full fine-tune of a 7B model needs ~30 GB of VRAM and risks *catastrophic forgetting*. Instead, **LoRA** freezes the base weights and trains two small low-rank matrices injected into the attention layers — so only a tiny fraction of parameters are updated, the base model keeps its general ability, and training fits comfortably on a single GPU.

```text
   Base weights (frozen)  ❄️        Low-rank adapters (trained)  🔥
   ┌──────────────────┐            ┌──────────────┐
   │        W         │     +      │    A × B      │   →   Specialist model
   │   7.6B params    │            │   ~5M params  │       (no forgetting)
   └──────────────────┘            └──────────────┘
```

### Training configuration

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| LoRA rank (`r`) | 16 |
| LoRA `alpha` | 64 |
| LoRA `dropout` | 0.1 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable params | **~5M (~0.07% of 7.6B)** |
| Epochs | 15 |
| Learning rate | 5e-5 (cosine, 10% warmup) |
| Effective batch size | 16 |
| Precision | FP16 |
| Trainer | `SFTTrainer` (TRL) |
| Hardware | 1× NVIDIA H800 (80 GB), HKUST SuperPOD |

---

## 🗂️ Data pipeline

The dataset was **self-curated** — quality mattered far more than volume for a task this nuanced.

1. **Source** — short news snippets from Bloomberg, Seeking Alpha, and Yahoo Finance covering earnings, deals, governance, macro events, and analyst actions.
2. **Annotate** — a human-in-the-loop workflow: hand-labeled seed examples, then Gemini 2.5 few-shot drafting, then a **manual review of every sample** (cause/effect wording and event category were frequently corrected).
3. **Format** — each record wrapped into a `system → user → assistant` chat format for instruction tuning.
4. **Split** — **stratified by `event_category`** so every event type appears in both train and eval (~130 curated examples → 127 train / 32 eval), plus a separate held-out test set for the baseline-vs-fine-tuned comparison.
5. **Validate** — malformed JSON samples discarded to guarantee schema integrity.

---

## 📁 Repository structure

```
FineTuningQwen-StockNewsDataset/
├── prepare_dataset.py              # Stratified train/eval split by event_category
├── finetune/
│   └── finetuning.py               # LoRA fine-tuning of Qwen 2.5 7B (SFTTrainer)
├── qwen_inference.py               # Load base + adapters → analyze_financial_news()
├── scripts/
│   ├── run_testset_comparison.py   # Baseline vs fine-tuned on the test set
│   ├── run_eval34_comparison.py    # Per-example evaluation comparison
│   └── demo_live_cli.py            # Interactive CLI demo on live/new headlines
├── quick_start.sh                  # One-shot data-prep helper
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

### 1. Install

```bash
git clone https://github.com/sameedsajid26/FineTuningQwen-StockNewsDataset.git
cd FineTuningQwen-StockNewsDataset
pip install -r requirements.txt
```

### 2. Prepare the dataset

```bash
python prepare_dataset.py \
    --input data/sample_financial_news_data_updated.json \
    --output-dir data \
    --test-size 0.2 \
    --seed 42
```

### 3. Fine-tune

```bash
python finetune/finetuning.py
```

<details>
<summary><b>Running on a SLURM cluster (e.g. HKUST SuperPOD)</b></summary>

```bash
srun --gpus-per-node=1 --partition=normal --gres=gpu:1 \
     --time=02:00:00 --cpus-per-task=4 --mem=16G --pty /bin/bash
python finetune/finetuning.py
```

Monitor with TensorBoard: `tensorboard --logdir=./logs`

</details>

### 4. Run inference

```python
from qwen_inference import analyze_financial_news

result = analyze_financial_news(
    "Tesla shares rose 5% after reporting record quarterly deliveries."
)
print(result)
# {'company': 'Tesla', 'ticker': 'TSLA', 'cause': 'record quarterly deliveries',
#  'effect': 'shares rose 5%', 'event_category': 'earnings_beat',
#  'impact_direction': 'positive'}
```

### 5. Reproduce the comparison

```bash
python scripts/run_testset_comparison.py      # baseline vs fine-tuned metrics
python scripts/demo_live_cli.py               # try it on your own headlines
```

---

## 💡 Key learnings

- **The data is the project.** ~130 clean, human-reviewed examples moved the needle more than any modeling change. Careful annotation beat raw volume.
- **Fine-tuning is a scalpel, not a hammer.** It shines on domain nuance, strict structure, privacy, and cheap high-volume inference — and adds nothing to classification a base model already does well. Knowing *which* is the actual engineering judgment.
- **Always benchmark a baseline first.** Without the zero-shot baseline, there's no honest way to claim the fine-tuning did anything.
- **Evaluation needs to match the field.** Exact-match works for tickers and categories; free-form cause/effect needs semantic similarity + an LLM judge, or you unfairly penalize correct-but-reworded answers.

---

## 🛠️ Tech stack

`PyTorch` · `🤗 Transformers` · `PEFT (LoRA)` · `TRL (SFTTrainer)` · `Accelerate` · `sentence-transformers` · `scikit-learn` · `TensorBoard`

---

## 📝 Note on model size

The pipeline was prototyped on the lightweight **Qwen 2.5 0.5B** for fast iteration, then scaled to **Qwen 2.5 7B Instruct** for the final run (see `finetune/finetuning.py`). The code is identical either way — only the base checkpoint changes — so you can swap in a smaller model to experiment on modest hardware.

---

## 🙏 Acknowledgements

Independent Project at **HKUST** (MSc Computer Science). Compute provided by the **HKUST SuperPOD**. Built on [Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and the Hugging Face ecosystem.

## 📄 License

Released under the MIT License — see `LICENSE` for details.

---

<div align="center">
<sub>Built by <b>Sameed Sajid</b> · fine-tuning, data curation, and evaluation.</sub>
</div>
