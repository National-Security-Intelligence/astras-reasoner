#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from model import SYSTEM

OUT = Path("data/gsm8k")
N = 20_000


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    raw = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    rows: list[dict] = []
    for ex in raw:
        if "gsm8k" not in (ex.get("problem_source") or "").lower():
            continue
        q = (ex.get("problem") or "").strip()
        sol = (ex.get("generated_solution") or "").strip()
        ans = (ex.get("expected_answer") or "").strip()
        if not (q and sol and ans):
            continue
        if "\\boxed" not in sol:
            sol = sol + f"\n\\boxed{{{ans}}}"
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": SYSTEM + "\n\n" + q},
                    {"role": "assistant", "content": sol},
                ]
            }
        )
        if len(rows) >= N:
            break
    if len(rows) < 100:
        raise SystemExit(f"only {len(rows)} rows")
    n_val = min(200, max(1, len(rows) // 20))
    dump(OUT / "train.jsonl", rows[:-n_val])
    dump(OUT / "valid.jsonl", rows[-n_val:])
    print(f"wrote {len(rows) - n_val} train {n_val} valid (OpenMathInstruct-2, CC-BY-4.0 NVIDIA)")


if __name__ == "__main__":
    main()
