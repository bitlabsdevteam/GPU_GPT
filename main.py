#!/usr/bin/env python3
"""Simple launcher for the GPU_GPT example scripts.

Usage:
  python main.py --list
  python main.py neural_network_example
  python main.py data_parallel_gpt2
  python main.py tensor_parallel_gpt2_lesson2 --nproc 2
  python main.py --all
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "neural_network_example": {
        "path": ROOT / "neural_network_example.py",
        "mode": "python",
        "description": "Simple NumPy neural network example.",
    },
    "data_parallel_gpt2": {
        "path": ROOT / "data_parallel_gpt2.py",
        "mode": "python",
        "description": "Hugging Face GPT-2 with DataParallel when multiple GPUs are available.",
    },
    "tensor_sequence_parallel_gpt2": {
        "path": ROOT / "tensor_sequence_parallel_gpt2.py",
        "mode": "python",
        "description": "Single-process tensor+sequence parallel GPT-2 structure demo.",
    },
    "tensor_parallelism_example": {
        "path": ROOT / "tensor_parallelism_example.py",
        "mode": "torchrun",
        "description": "Distributed DDP training toy example.",
    },
    "sequence_parallelism_gpt2_example": {
        "path": ROOT / "sequence_parallelism_gpt2_example.py",
        "mode": "torchrun",
        "description": "Distributed sequence-parallel GPT-2 example.",
    },
    "tensor_parallel_gpt2_lesson2": {
        "path": ROOT / "tensor_parallel_gpt2_lesson2.py",
        "mode": "torchrun",
        "description": "PyTorch native tensor parallel lesson.",
    },
}


def list_scripts() -> None:
    print("Available scripts:\n")
    for name, meta in SCRIPTS.items():
        print(f"- {name:32} [{meta['mode']}]  {meta['description']}")


def run_python_script(path: Path) -> int:
    runpy.run_path(str(path), run_name="__main__")
    return 0


def run_torchrun_script(path: Path, nproc: int) -> int:
    torchrun = shutil.which("torchrun")
    if not torchrun:
        raise RuntimeError("torchrun is not installed or not on PATH")

    cmd = [torchrun, f"--nproc_per_node={nproc}", str(path)]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT), env=os.environ.copy())


def run_one(name: str, nproc: int) -> int:
    if name not in SCRIPTS:
        raise KeyError(f"Unknown script: {name}")

    meta = SCRIPTS[name]
    path = meta["path"]
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"\n=== Running {name} ===")
    if meta["mode"] == "python":
        return run_python_script(path)
    if meta["mode"] == "torchrun":
        return run_torchrun_script(path, nproc=nproc)
    raise ValueError(f"Unsupported mode: {meta['mode']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU_GPT example scripts easily.")
    parser.add_argument("script", nargs="?", help="Script name to run (use --list to see options)")
    parser.add_argument("--list", action="store_true", help="List available scripts")
    parser.add_argument("--all", action="store_true", help="Run all scripts in a sensible order")
    parser.add_argument("--nproc", type=int, default=2, help="Processes/GPUs to use for torchrun-based scripts")
    args = parser.parse_args()

    if args.list:
        list_scripts()
        return 0

    if args.all:
        order = [
            "neural_network_example",
            "tensor_sequence_parallel_gpt2",
            "data_parallel_gpt2",
            "tensor_parallelism_example",
            "sequence_parallelism_gpt2_example",
            "tensor_parallel_gpt2_lesson2",
        ]
        for name in order:
            code = run_one(name, nproc=args.nproc)
            if code != 0:
                return code
        return 0

    if not args.script:
        parser.print_help()
        print("\nTip: python main.py --list")
        return 1

    return run_one(args.script, nproc=args.nproc)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
