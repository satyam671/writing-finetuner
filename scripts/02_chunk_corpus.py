"""
02_chunk_corpus.py

Chunk cleaned text files into fixed-token windows for training.
Uses the actual Mistral tokenizer so chunk sizes are exact, not approximate.
Writes train/eval JSONL splits ready for the training script.

Usage:
    python scripts/02_chunk_corpus.py \
        --input_dir data/cleaned \
        --output_dir data/chunks \
        --config configs/train_config.yaml
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def chunk_token_ids(
    token_ids: List[int],
    chunk_size: int,
    overlap: float
) -> List[List[int]]:
    """
    Slide a window over token_ids with the specified overlap.
    overlap=0.10 means each chunk shares 10% of tokens with the next.
    """
    step = max(1, int(chunk_size * (1 - overlap)))
    chunks = []

    for start in range(0, len(token_ids), step):
        chunk = token_ids[start : start + chunk_size]
        # Only keep chunks that are at least half full
        if len(chunk) >= chunk_size // 2:
            chunks.append(chunk)

    return chunks


def build_training_example(token_ids: List[int]) -> Dict:
    """
    Format a chunk as a causal LM training example.
    input_ids and labels are the same — standard causal LM setup.
    """
    return {
        "input_ids": token_ids,
        "labels": token_ids,
    }


def write_jsonl(examples: List[Dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chunk cleaned corpus into training examples.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["base_model"]
    chunk_size = config["data"]["max_seq_length"]
    overlap = config["data"]["chunk_overlap"]
    split_ratio = config["data"]["train_eval_split"]
    seed = config["training"]["seed"]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}. Run 01_prepare_corpus.py first.")
        return

    print(f"Tokenizing and chunking {len(txt_files)} files...")
    all_chunks = []

    for filepath in tqdm(txt_files, desc="Chunking"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize without adding special tokens at the boundaries —
        # we want raw token sequences to slide the window over
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) < chunk_size // 2:
            continue  # File too short even for one partial chunk

        chunks = chunk_token_ids(token_ids, chunk_size, overlap)
        examples = [build_training_example(c) for c in chunks]
        all_chunks.extend(examples)

    print(f"\nTotal chunks: {len(all_chunks):,}")

    # Shuffle before split so train/eval are drawn from all files
    random.seed(seed)
    random.shuffle(all_chunks)

    split_idx = int(len(all_chunks) * split_ratio)
    train_examples = all_chunks[:split_idx]
    eval_examples = all_chunks[split_idx:]

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    write_jsonl(train_examples, train_path)
    write_jsonl(eval_examples, eval_path)

    print(f"\nSplit:")
    print(f"  Train: {len(train_examples):,} chunks → {train_path}")
    print(f"  Eval:  {len(eval_examples):,} chunks  → {eval_path}")
    print(f"\nEach chunk: {chunk_size} tokens, {int(overlap * 100)}% overlap")


if __name__ == "__main__":
    main()
