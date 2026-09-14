#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from model import STUDENT
from prepare_openmath import main as prepare

ADAPTER = "outputs/sft"


def main() -> None:
    if not Path("data/gsm8k/train.jsonl").exists():
        prepare()
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", STUDENT,
        "--data", "data/gsm8k",
        "--train",
        "--batch-size", "1",
        "--iters", "800",
        "--learning-rate", "1e-5",
        "--adapter-path", ADAPTER,
        "--max-seq-length", "2048",
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print("saved", ADAPTER)


if __name__ == "__main__":
    main()
