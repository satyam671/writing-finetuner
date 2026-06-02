"""
05_reasoning_probe.py

The reasoning probe. This is the evaluation that standard metrics miss.

Workflow:
  1. Loads probe prompts from configs/train_config.yaml
  2. For each prompt, asks YOU to write your answer first (blind)
  3. Then generates the fine-tuned model's response
  4. Presents both side by side
  5. Asks you to score the model on three axes:
       - Structural similarity (0-100)
       - Reasoning depth      (0-100)
       - Opinion consistency  (0-100)
  6. Saves all results to probe_results/results.json for later analysis

The discipline: write your answer before you see the model's output.
Anchoring kills the experiment. The script enforces this.

Usage:
    python scripts/05_reasoning_probe.py --config configs/train_config.yaml
"""

import argparse
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── Generation ───────────────────────────────────────────────────────────────

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    device: str,
) -> str:
    """Generate a response from the fine-tuned model for a probe prompt."""

    # Format as Mistral instruct chat template
    chat = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ─── Display helpers ──────────────────────────────────────────────────────────

def divider(char: str = "─", width: int = 70) -> str:
    return char * width


def wrap(text: str, width: int = 68, indent: str = "  ") -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if line.strip() == "":
            wrapped.append("")
        else:
            wrapped.extend(
                textwrap.wrap(line, width=width, initial_indent=indent, subsequent_indent=indent)
            )
    return "\n".join(wrapped)


def get_score(label: str) -> int:
    """Prompt user for a score 0-100, validate, and return."""
    while True:
        raw = input(f"  {label} (0-100): ").strip()
        try:
            score = int(raw)
            if 0 <= score <= 100:
                return score
            else:
                print("  Please enter a number between 0 and 100.")
        except ValueError:
            print("  Please enter a whole number.")


def get_multiline_input(prompt_text: str) -> str:
    """
    Collect multiline input from the user.
    They type their answer and hit Enter twice (blank line) to finish.
    """
    print(prompt_text)
    print("  (Type your answer. Press Enter twice when done.)\n")
    lines = []
    while True:
        try:
            line = input("  > ")
        except EOFError:
            break
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--skip_model", action="store_true",
                        help="Skip model loading — just collect your answers for later comparison")
    args = parser.parse_args()

    cfg = load_config(args.config)
    probe_cfg = cfg["probe"]
    prompts = probe_cfg["prompts"]

    results_dir = Path(probe_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = None, None

    if not args.skip_model:
        checkpoint = probe_cfg["checkpoint"]
        print(f"Loading fine-tuned model from: {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()
        print("Model loaded.\n")

    results = []

    print(divider("═"))
    print("  REASONING PROBE")
    print(f"  {len(prompts)} prompts | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(divider("═"))
    print("""
  HOW THIS WORKS:
  For each prompt, you write your answer first — before seeing the model's.
  Then the model generates its response.
  Then you score the model on three axes.

  The point is not the number. It's the act of comparing.
  Write honestly. Score honestly.
""")

    input("  Press ENTER to begin the probe.\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{divider('═')}")
        print(f"  PROMPT {i} of {len(prompts)}")
        print(divider())
        print(wrap(prompt))
        print(divider())

        # ── Step 1: Get user's answer (blind) ─────────────────────────────
        your_response = get_multiline_input("\n  YOUR ANSWER (write before seeing the model's):")

        # ── Step 2: Generate model response ───────────────────────────────
        model_response = ""
        if model is not None and tokenizer is not None:
            print(f"\n  Generating model response...")
            model_response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=probe_cfg["max_new_tokens"],
                temperature=probe_cfg["temperature"],
                do_sample=probe_cfg["do_sample"],
                device=device,
            )
        else:
            model_response = "[Model not loaded — run without --skip_model to get model output]"

        # ── Step 3: Side-by-side display ──────────────────────────────────
        print(f"\n{divider()}")
        print("  YOUR RESPONSE:")
        print(divider("·"))
        print(wrap(your_response))

        print(f"\n{divider()}")
        print("  MODEL RESPONSE:")
        print(divider("·"))
        print(wrap(model_response))
        print(divider())

        # ── Step 4: Scoring ───────────────────────────────────────────────
        print("""
  SCORE THE MODEL (compare against your response, not against perfection):

  Structural similarity: Does it organize the argument the way you would?
  Reasoning depth:       Does it surface the trade-offs you'd surface,
                         or stop one layer too soon?
  Opinion consistency:   Does it take the position you'd take, or hedge
                         where you'd commit?
""")
        structural = get_score("Structural similarity")
        depth      = get_score("Reasoning depth     ")
        opinion    = get_score("Opinion consistency ")

        composite = round((structural + depth + opinion) / 3)

        print(f"\n  Composite score: {composite}/100")

        if composite >= 70:
            verdict = "Strong transfer — the model reasoned like you on this prompt."
        elif composite >= 45:
            verdict = "Partial transfer — surface match, reasoning shallower."
        else:
            verdict = "Surface only — sounds like you, doesn't think like you."

        print(f"  Verdict: {verdict}\n")

        results.append({
            "prompt_index": i,
            "prompt": prompt,
            "your_response": your_response,
            "model_response": model_response,
            "scores": {
                "structural_similarity": structural,
                "reasoning_depth": depth,
                "opinion_consistency": opinion,
                "composite": composite,
            },
            "verdict": verdict,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{divider('═')}")
    print("  PROBE COMPLETE — SUMMARY")
    print(divider("═"))

    struct_avg  = sum(r["scores"]["structural_similarity"] for r in results) / len(results)
    depth_avg   = sum(r["scores"]["reasoning_depth"]       for r in results) / len(results)
    opinion_avg = sum(r["scores"]["opinion_consistency"]   for r in results) / len(results)
    composite   = sum(r["scores"]["composite"]             for r in results) / len(results)

    print(f"""
  Prompts evaluated:      {len(results)}

  Structural similarity:  {struct_avg:.1f} / 100
  Reasoning depth:        {depth_avg:.1f} / 100
  Opinion consistency:    {opinion_avg:.1f} / 100
  ─────────────────────────────────
  Composite (avg):        {composite:.1f} / 100

  Interpretation:
    > 70  — meaningful reasoning transfer. Model thinks like you.
    45-70 — surface transfer dominates. Sounds right, thinner than you.
    < 45  — surface only. Editing required before any output is publishable.

  Run 04_evaluate_standard.py and compare those scores to these.
  That gap is the finding.
""")

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"probe_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "checkpoint": probe_cfg.get("checkpoint", "unknown"),
        "num_prompts": len(results),
        "summary_scores": {
            "structural_similarity_avg": round(struct_avg, 2),
            "reasoning_depth_avg": round(depth_avg, 2),
            "opinion_consistency_avg": round(opinion_avg, 2),
            "composite_avg": round(composite, 2),
        },
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also write a clean latest symlink-style copy for easy reference
    latest_path = results_dir / "results_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Results saved → {out_path}")
    print(f"  Latest copy   → {latest_path}\n")


if __name__ == "__main__":
    main()
