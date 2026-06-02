"""
04_evaluate_standard.py

Standard fine-tuning evaluation: perplexity and ROUGE-L on the held-out eval set.
This is what most people run and call done.

The article's point: these metrics will look good even when reasoning transfer
has barely happened. Run this first, then run 05_reasoning_probe.py and compare.

Usage:
    python scripts/04_evaluate_standard.py --config configs/train_config.yaml
"""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def compute_perplexity(
    model,
    tokenizer,
    examples: list,
    device: str,
    max_examples: int = 200,
) -> float:
    """
    Compute average perplexity on a sample of eval examples.
    Perplexity = exp(average cross-entropy loss over all tokens).
    Lower is better. A well-fine-tuned model on its own corpus
    typically drops from ~40 to ~10-15.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    sample = examples[:max_examples]

    with torch.no_grad():
        for ex in tqdm(sample, desc="Computing perplexity"):
            input_ids = torch.tensor([ex["input_ids"]], dtype=torch.long).to(device)
            labels = torch.tensor([ex["labels"]], dtype=torch.long).to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # outputs.loss is mean loss per token in this batch
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def compute_rouge(
    model,
    tokenizer,
    examples: list,
    device: str,
    max_examples: int = 100,
    prompt_tokens: int = 64,
    continuation_tokens: int = 128,
) -> dict:
    """
    ROUGE-L between model-generated continuations and actual continuations
    from the eval set. We split each example into a prompt (first
    prompt_tokens tokens) and a reference (next continuation_tokens tokens),
    then ask the model to generate from the prompt and score against reference.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    model.eval()

    precision_scores = []
    recall_scores = []
    f1_scores = []

    sample = examples[:max_examples]

    with torch.no_grad():
        for ex in tqdm(sample, desc="Computing ROUGE-L"):
            token_ids = ex["input_ids"]

            if len(token_ids) < prompt_tokens + 10:
                continue  # Too short to split meaningfully

            prompt_ids = token_ids[:prompt_tokens]
            reference_ids = token_ids[prompt_tokens : prompt_tokens + continuation_tokens]

            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)

            generated = model.generate(
                input_ids=prompt_tensor,
                max_new_tokens=continuation_tokens,
                do_sample=False,  # Greedy for deterministic eval
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode only the newly generated portion
            generated_ids = generated[0][prompt_tokens:].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)

            if not generated_text.strip() or not reference_text.strip():
                continue

            scores = scorer.score(reference_text, generated_text)
            precision_scores.append(scores["rougeL"].precision)
            recall_scores.append(scores["rougeL"].recall)
            f1_scores.append(scores["rougeL"].fmeasure)

    if not f1_scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--max_perplexity_examples", type=int, default=200)
    parser.add_argument("--max_rouge_examples", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = cfg["probe"]["checkpoint"]
    eval_file = cfg["data"]["eval_file"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Eval file: {eval_file}\n")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    print("Loading eval data...")
    examples = load_jsonl(eval_file)
    print(f"Eval examples: {len(examples):,}\n")

    # ── Perplexity ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("PERPLEXITY")
    print("=" * 60)
    ppl = compute_perplexity(model, tokenizer, examples, device, args.max_perplexity_examples)
    print(f"\n  Perplexity: {ppl:.2f}")
    print(f"  Interpretation: lower = model is less surprised by your writing")
    print(f"  Baseline (untrained Mistral 7B on this corpus): ~40-50")
    print(f"  Target after fine-tuning: <15\n")

    # ── ROUGE-L ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("ROUGE-L")
    print("=" * 60)
    rouge = compute_rouge(model, tokenizer, examples, device, args.max_rouge_examples)
    print(f"\n  ROUGE-L Precision: {rouge['precision']:.4f}")
    print(f"  ROUGE-L Recall:    {rouge['recall']:.4f}")
    print(f"  ROUGE-L F1:        {rouge['f1']:.4f}")
    print(f"  Interpretation: higher = model completions share more tokens with your actual writing")
    print(f"  Baseline: ~0.35-0.40  |  Target: >0.55\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
  Metric          Score       What it measures
  ─────────────── ─────────── ──────────────────────────────────────
  Perplexity      {ppl:<11.2f} Token prediction accuracy
  ROUGE-L F1      {rouge['f1']:<11.4f} Lexical overlap with your style

  What these metrics DON'T measure:
  → Whether the model reasons the way you reason
  → Whether it applies your judgment to novel situations
  → Whether its opinions are contextual or just memorized

  Run 05_reasoning_probe.py to measure what matters.
""")


if __name__ == "__main__":
    main()
