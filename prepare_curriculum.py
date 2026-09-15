#!/usr/bin/env python3
"""Build GSM8K *train-only* curriculum jsonl. Never touches the test split."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset

from model import SYSTEM

ROOT = Path("data/curriculum")
TRAIN_DIR = Path("data/gsm8k")
HASH = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
N_VAL = 50
N1 = 300
N2 = 500


def qid(question: str) -> str:
    return hashlib.sha256(question.strip().encode()).hexdigest()[:16]


def difficulty(answer: str) -> tuple[int, int]:
    calc = answer.count("<<")
    lines = len([ln for ln in answer.splitlines() if ln.strip() and "####" not in ln])
    steps = calc if calc else max(1, lines)
    return (steps, len(answer))


def gold_num(answer: str) -> str:
    found = HASH.findall(answer.replace(",", ""))
    return found[-1] if found else ""


def record(question: str, answer: str) -> dict:
    ans = answer.strip()
    num = gold_num(ans)
    if num and "\\boxed" not in ans:
        ans = ans + f"\n\\boxed{{{num}}}"
    return {
        "id": qid(question),
        "n_steps": difficulty(answer)[0],
        "messages": [
            {"role": "user", "content": SYSTEM + "\n\n" + question.strip()},
            {"role": "assistant", "content": ans},
        ],
    }


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mlx_row(row: dict) -> dict:
    return {"messages": row["messages"]}


def main() -> None:
    raw = load_dataset("openai/gsm8k", "main", split="train")
    scored = [record(x["question"], x["answer"]) for x in raw]
    scored.sort(key=lambda r: (r["n_steps"], len(r["messages"][1]["content"])))

    stride = max(1, len(scored) // N_VAL)
    val_idx = sorted({min(i * stride, len(scored) - 1) for i in range(N_VAL)})[:N_VAL]
    val_set = set(val_idx)
    val = [scored[i] for i in val_idx]
    pool = [r for i, r in enumerate(scored) if i not in val_set]
    stage1 = pool[:N1]
    stage2 = pool[:N2]

    dump(ROOT / "val.jsonl", val)
    dump(ROOT / "stage1_300.jsonl", stage1)
    dump(ROOT / "stage2_500.jsonl", stage2)
    dump(TRAIN_DIR / "train.jsonl", [mlx_row(r) for r in stage1])
    dump(TRAIN_DIR / "valid.jsonl", [mlx_row(r) for r in val])

    meta = {
        "source": "openai/gsm8k train only",
        "test_split": False,
        "val": len(val),
        "stage1": len(stage1),
        "stage2": len(stage2),
        "stage1_steps": [r["n_steps"] for r in stage1[:5]] + ["..."] + [stage1[-1]["n_steps"]],
        "stage2_max_steps": stage2[-1]["n_steps"],
        "ids_stage1": [r["id"] for r in stage1],
        "ids_val": [r["id"] for r in val],
    }
    (ROOT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"train-only curriculum  val={len(val)}  "
        f"stage1={len(stage1)} steps {stage1[0]['n_steps']}-{stage1[-1]['n_steps']}  "
        f"stage2={len(stage2)} steps {stage2[0]['n_steps']}-{stage2[-1]['n_steps']}"
    )
    print(f"wrote {TRAIN_DIR} (stage1 + val) and {ROOT}")
    print("test split was not loaded.")


if __name__ == "__main__":
    main()
