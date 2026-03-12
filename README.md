# GPU_GPT

**GPU_GPT** is a focused PyTorch codebase for building and understanding the two core functions of a GPT-style language model system:

- **Pre-training**
- **Inference**

The goal of this repository is not to be a giant framework.
It is to be a clear, hackable foundation for learning how modern LLM systems are built, especially from the perspective of **GPU-aware engineering**.

---

## Why this repo exists

If you want to understand large language models seriously, you need more than isolated snippets.
You need a source tree that shows how the main pieces connect:

- how text becomes tokens
- how tokens flow through a transformer
- how pre-training computes loss and updates weights
- how checkpoints are saved and loaded
- how inference generates text autoregressively
- how device handling and future parallelism fit into the architecture

This repo is designed to make those connections explicit.

---

## Core workflows

### 1) Pre-training

The pre-training pipeline covers:

- loading a training corpus
- building a tokenizer
- creating next-token prediction batches
- defining a GPT-style decoder-only transformer
- running optimization with AdamW
- saving checkpoints and tokenizer artifacts

Entry point:

```bash
python main.py pretrain --train-path your_text.txt
```

### 2) Inference

The inference pipeline covers:

- loading a trained checkpoint
- loading tokenizer artifacts
- encoding a prompt
- autoregressive token generation
- decoding generated output back into text

Entry point:

```bash
python main.py inference \
  --checkpoint-path artifacts/model.pt \
  --tokenizer-path artifacts/tokenizer.json \
  --prompt "Hello GPU world"
```

---

## Repository structure

```text
GPU_GPT/
├── main.py                    # Main CLI entrypoint
├── config.py                  # Config dataclasses for training/inference/model
├── data.py                    # Text loading, character tokenizer, batch sampling
├── model.py                   # GPT-style decoder-only transformer model
├── checkpoint.py              # Save/load checkpoints
├── parallelism.py             # Device selection, seed setup, mixed precision helpers
├── pretrain.py                # Pre-training workflow
├── inference.py               # Inference workflow
│
├── data_parallel_gpt2.py
├── sequence_parallelism_gpt2_example.py
├── tensor_parallel_gpt2_lesson2.py
├── tensor_parallelism_example.py
├── tensor_sequence_parallel_gpt2.py
├── neural_network_example.py  # Earlier/auxiliary learning examples
└── README.md
```

---

## Design philosophy

### Clear first, scalable next

This codebase is intentionally built so you can understand it end-to-end.
That means:

- small number of files
- explicit flow from input text to generated text
- minimal hidden magic
- architecture that can later grow into multi-GPU training and optimized inference

### Built for learning advanced GPU programming

Even though the current pre-training and inference path is intentionally compact, the repo is organized so it can evolve toward more advanced topics like:

- mixed precision training
- distributed data parallelism
- tensor parallelism
- sequence parallelism
- checkpoint sharding
- KV-cache optimization
- memory-aware inference paths

This is meant to be a **learning scaffold** for serious LLM systems work.

---

## Quick start

### 1) Prepare a text corpus

Create a UTF-8 text file, for example:

```text
hello gpu world
hello transformer
this is a tiny training corpus
```

Save it as `train.txt`.

### 2) Run pre-training

```bash
python main.py pretrain \
  --train-path train.txt \
  --out-dir artifacts \
  --max-steps 200 \
  --batch-size 16 \
  --block-size 128 \
  --n-embd 128 \
  --n-head 4 \
  --n-layer 4
```

This will produce artifacts such as:

- `artifacts/model.pt`
- `artifacts/tokenizer.json`

### 3) Run inference

```bash
python main.py inference \
  --checkpoint-path artifacts/model.pt \
  --tokenizer-path artifacts/tokenizer.json \
  --prompt "hello"
```

---

## CLI reference

### Pre-training

```bash
python main.py pretrain \
  --train-path <text-file> \
  --out-dir artifacts \
  --batch-size 16 \
  --block-size 128 \
  --max-steps 200 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --eval-interval 20 \
  --save-every 100 \
  --n-embd 128 \
  --n-head 4 \
  --n-layer 4 \
  --dropout 0.1 \
  --seed 42
```

### Inference

```bash
python main.py inference \
  --checkpoint-path <checkpoint> \
  --tokenizer-path <tokenizer> \
  --prompt "your prompt" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 20
```

---

## What the current implementation includes

- character-level tokenizer for end-to-end simplicity
- GPT-style decoder-only architecture
- causal self-attention
- next-token prediction objective
- AdamW optimizer
- checkpoint save/load support
- autoregressive generation with temperature and top-k sampling
- basic device and mixed-precision helpers

---

## What should come next

The strongest next upgrades for this repo are:

1. **Validation split + eval loop**
2. **Resume training from checkpoint**
3. **Top-p / nucleus sampling**
4. **Batch inference**
5. **DDP pre-training**
6. **Tensor parallel inference/training paths**
7. **Profiling and memory instrumentation**
8. **More production-grade tokenizer/data pipeline**

---

## Important note

This repository is currently a **learning-oriented foundation**, not a full production training stack.
It is meant to make the mechanics of GPT pre-training and inference understandable, modifiable, and extensible.

If your goal is to master advanced GPU programming for LLMs, this repo should be treated as the core source tree you evolve step by step toward production-grade systems.

---

## Vision

The long-term vision of **GPU_GPT** is simple:

> Build a codebase that teaches how real LLM systems work by making pre-training and inference explicit, testable, and progressively more GPU-efficient.

That means moving from:

- toy examples
- to correct workflows
- to scalable distributed systems
- to production-level GPU engineering

---

## License

MIT
