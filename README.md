# Astras

Portable math reasoner on Apple MLX. **uv** + **Ruff**. Weights are not in git.

```bash
uv sync
uv run ruff check .
uv run python eval_gsm8k.py --sample 200 --seed 0
```

Student: Phi-4-mini-reasoning (MIT, ~2.2 GB 4-bit).

See NOTICE.md.
