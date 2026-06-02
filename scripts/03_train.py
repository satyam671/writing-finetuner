"""
03_train.py

LoRA fine-tuning for Mistral 7B Instruct v0.2 on a personal writing corpus.
All hyperparameters are read from configs/train_config.yaml.

Usage:
    python scripts/03_train.py --config configs/train_config.yaml

What this does:
    1. Loads Mistral 7B in bfloat16
    2. Attaches a LoRA adapter (q_proj, v_proj only — as described in the article)
    3. Trains on your chunked corpus using HuggingFace Trainer
    4. Saves the best checkpoint to checkpoints/final/
    5. Merges LoRA weights into the base model for inference
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# ─── Callbacks ────────────────────────────────────────────────────────────────

class PerplexityCallback(TrainerCallback):
    """Log perplexity alongside loss at each eval step."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"])
            metrics["eval_perplexity"] = round(ppl, 2)
            step = state.global_step
            print(f"\n  Step {step:>5} | eval_loss: {metrics['eval_loss']:.4f} | perplexity: {ppl:.2f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    output_dir = Path(train_cfg["output_dir"])
    final_dir = output_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print(f"Loading tokenizer: {model_cfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        padding_side="right",  # Required for causal LM training
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ── Load base model ─────────────────────────────────────────────────────
    print(f"Loading base model: {model_cfg['base_model']}")
    dtype = torch.bfloat16 if model_cfg["torch_dtype"] == "bfloat16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=dtype,
        device_map=model_cfg["device_map"],
    )
    model.config.use_cache = False  # Required when using gradient checkpointing

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    # ── Attach LoRA adapter ─────────────────────────────────────────────────
    print("Attaching LoRA adapter...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load datasets ───────────────────────────────────────────────────────
    print("Loading datasets...")
    train_dataset = load_jsonl(data_cfg["train_file"])
    eval_dataset = load_jsonl(data_cfg["eval_file"])

    print(f"  Train: {len(train_dataset):,} chunks")
    print(f"  Eval:  {len(eval_dataset):,} chunks")

    # Data collator handles padding within batch
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # ── Training arguments ──────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        seed=train_cfg["seed"],
        report_to="tensorboard",
        logging_dir=str(output_dir / "logs"),
        remove_unused_columns=False,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[PerplexityCallback()],
    )

    print("\nStarting training...")
    print(f"  Effective batch size: {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  Epochs: {train_cfg['num_train_epochs']}")
    print(f"  Output: {output_dir}/\n")

    trainer.train()

    # ── Save final adapter ──────────────────────────────────────────────────
    print("\nSaving best checkpoint...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ── Merge LoRA weights into base for inference ──────────────────────────
    print("Merging LoRA weights into base model...")
    merged_dir = output_dir / "merged"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=dtype,
        device_map=model_cfg["device_map"],
    )
    peft_model = PeftModel.from_pretrained(base_model, str(final_dir))
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    print(f"\nDone.")
    print(f"  LoRA adapter:   {final_dir}/")
    print(f"  Merged model:   {merged_dir}/")
    print(f"\nUse the merged model for inference in 05_reasoning_probe.py")


if __name__ == "__main__":
    main()
