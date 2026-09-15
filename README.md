# Astras

Portable math reasoner on Apple MLX. **Scripts only** — no weights in git.

- Student: [Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning) (Microsoft, **MIT**)
- MLX: `mlx-community/Phi-4-mini-reasoning-4bit`
- Eval: GSM8K test, `--sample 200 --seed 0`
- Train data: GSM8K **train split only** (curriculum). Test never enters jsonl.

## Eval

```bash
uv sync
uv run ruff check .
uv run python eval_gsm8k.py --sample 200 --seed 0
```

Measured (no adapter): **131/200 = 65.5%** with the boxed prompt. See RESULTS.md.

## Curriculum SFT (after eval finishes)

```bash
uv run python prepare_curriculum.py   # 300 easy + 50 train-val; stage2 500 ready
uv run python train_sft.py            # LoRA 2e-5 cosine, 300 iters, not 0.005
uv run python eval_gsm8k.py --sample 200 --seed 0 --adapter outputs/sft
```

`data/` is gitignored. Do not commit jsonl or adapters.
