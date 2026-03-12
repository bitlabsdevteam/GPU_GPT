#!/usr/bin/env python3
from __future__ import annotations

import argparse

from config import InferenceConfig, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU_GPT entrypoint for the two core workflows: pre-training and inference."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("pretrain", help="Run pre-training on a text corpus")
    train.add_argument("--train-path", required=True, help="Path to a UTF-8 text file for training")
    train.add_argument("--out-dir", default="artifacts", help="Directory to store checkpoints/tokenizer")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--block-size", type=int, default=128)
    train.add_argument("--max-steps", type=int, default=200)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--eval-interval", type=int, default=20)
    train.add_argument("--save-every", type=int, default=100)
    train.add_argument("--n-embd", type=int, default=128)
    train.add_argument("--n-head", type=int, default=4)
    train.add_argument("--n-layer", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)

    infer = subparsers.add_parser("inference", help="Run autoregressive inference from a saved checkpoint")
    infer.add_argument("--checkpoint-path", required=True)
    infer.add_argument("--tokenizer-path", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--max-new-tokens", type=int, default=80)
    infer.add_argument("--temperature", type=float, default=0.8)
    infer.add_argument("--top-k", type=int, default=20)
    infer.add_argument("--device", default=None, help="Optional override, e.g. cpu or cuda")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pretrain":
        from pretrain import run_pretraining

        cfg = TrainingConfig(
            train_path=args.train_path,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_steps=args.max_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval,
            save_every=args.save_every,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
            seed=args.seed,
        )
        run_pretraining(cfg)
        return 0

    if args.command == "inference":
        from inference import run_inference

        cfg = InferenceConfig(
            checkpoint_path=args.checkpoint_path,
            tokenizer_path=args.tokenizer_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        run_inference(cfg)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
