# writing-finetuner

Fine-tune Mistral 7B on your own writing corpus. Evaluate whether the model actually learned your reasoning — or just your surface patterns.

Built as the companion repo for: **"I Trained a Model on My Own Writing for Six Months. It Learned the Wrong Things."** (**Medium**)
---

## What this repo does

Most fine-tuning tutorials stop at "the loss dropped, ship it." This one doesn't. It includes:

1. **Corpus preparation** — clean and chunk your writing from Medium HTML exports, plain text, or Markdown
2. **LoRA fine-tuning** — Mistral 7B Instruct v0.2 with reproducible hyperparameters
3. **Standard evaluation** — perplexity and ROUGE-L on a held-out set
4. **Reasoning probe** — the part most people skip. Compares your model's outputs to your own on novel situations it hasn't seen
5. **Depth scoring** — a structured comparison across structural similarity, reasoning depth, and opinion consistency

The thesis of the companion article: standard metrics will tell you your fine-tune is working. The reasoning probe will tell you the truth.

---

## Hardware requirements

| Config | Min VRAM | Notes |
|---|---|---|
| LoRA (bf16) | 24 GB | RTX 3090 / A10G / A100 40GB |
| LoRA (fp16) | 20 GB | Some headroom loss |
| Gradient checkpointing ON | ~16 GB | Slower, but fits |

Training time on RTX 3090: ~6 hours per epoch on a 400K-word corpus.

---

## Project structure

```
writing-finetuner/
├── configs/
│   └── train_config.yaml          # All hyperparameters in one place
├── data/
│   ├── raw/                       # Drop your raw exports here
│   ├── cleaned/                   # Output of 01_prepare_corpus.py
│   └── chunks/                    # Output of 02_chunk_corpus.py
├── scripts/
│   ├── 01_prepare_corpus.py       # Clean HTML/text → plain text
│   ├── 02_chunk_corpus.py         # Chunk into 512-token windows
│   ├── 03_train.py                # LoRA fine-tuning
│   ├── 04_evaluate_standard.py    # Perplexity + ROUGE-L
│   └── 05_reasoning_probe.py      # The actual test
├── probe_results/                 # Probe outputs saved here
├── checkpoints/                   # Model checkpoints
├── notebooks/
│   └── explore_results.ipynb      # Visual exploration of probe results
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/satyam671/writing-finetuner.git
cd writing-finetuner
pip install -r requirements.txt
```

### 2. Prepare your corpus

Drop your writing into `data/raw/`. Supported formats:
- Medium HTML exports (`.html`)
- Plain text files (`.txt`)
- Markdown files (`.md`)

```bash
python scripts/01_prepare_corpus.py --input_dir data/raw --output_dir data/cleaned
python scripts/02_chunk_corpus.py --input_dir data/cleaned --output_dir data/chunks --chunk_size 512 --overlap 0.10
```

### 3. Configure training

Edit `configs/train_config.yaml` — or leave defaults for the setup described in the article.

### 4. Train

```bash
python scripts/03_train.py --config configs/train_config.yaml
```

### 5. Evaluate

```bash
# Standard metrics
python scripts/04_evaluate_standard.py --config configs/train_config.yaml

# Reasoning probe — the important one
python scripts/05_reasoning_probe.py --config configs/train_config.yaml
```

The probe will prompt you to write your own answers before it shows the model's outputs. Don't skip this step.

---

## Expected training output

```
Epoch 1/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  2000/2000
  train_loss: 2.143 → 1.624  |  eval_loss: 2.198 → 1.703
  
Epoch 2/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  2000/2000
  train_loss: 1.624 → 1.341  |  eval_loss: 1.703 → 1.412

Epoch 3/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  2000/2000
  train_loss: 1.341 → 1.203  |  eval_loss: 1.412 → 1.289
  perplexity: 42.3 → 12.1
  ROUGE-L:    0.38 → 0.61

Checkpoint saved → checkpoints/final/
Training time: 18h 42m
```

---

## The reasoning probe — what to expect

After running `05_reasoning_probe.py`, you'll see output like this:

```
============================================================
PROMPT 1/5
------------------------------------------------------------
An ML model that performed well in staging is now producing
degraded outputs in production. Walk through your diagnostic
process.
------------------------------------------------------------
[Wrote your answer? Press ENTER to see model output]

MODEL OUTPUT:
When a model degrades in production, there are several things
worth investigating. First, check the data pipeline for issues.
Distribution shift is a common cause of production issues...

------------------------------------------------------------
Score this response on reasoning depth (0-100): _
```

The depth score you assign is yours to calibrate. The point isn't a precise number — it's the act of comparing, response by response, whether the model reasons the way you do or just sounds the way you do.

---

## License

MIT. Use it, fork it, run your own experiment.

---

## Citation

If you use this repo in your own writing or research:

```
Sahu, S. (2026). writing-finetuner: Fine-tune and evaluate reasoning
transfer in LLMs trained on personal writing corpora.
GitHub. https://github.com/satyam671/writing-finetuner
```
