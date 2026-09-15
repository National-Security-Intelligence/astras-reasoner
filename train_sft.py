#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from prepare_curriculum import main as prepare


def main() -> None:
    if not Path("data/gsm8k/train.jsonl").exists():
        prepare()
    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", "lora_config.yaml"]
    print(" ".join(cmd))
    print("LR: warmup 30 steps 2e-7->2e-5, cosine decay to 2e-7 over 300 iters")
    subprocess.check_call(cmd)
    print("saved outputs/sft")


if __name__ == "__main__":
    main()
