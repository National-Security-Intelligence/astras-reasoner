# Astras

Scripts only. Weights are not in git. Student: Phi-4-mini-reasoning (MIT).

```bash
uv sync
uv run python eval_gsm8k.py --sample 200 --seed 0
```

Prep **train-only** curriculum (300 easy, 500 grow, 50 val from train; never test):

```bash
uv run python prepare_curriculum.py
```

Writes `data/curriculum/` and copies stage1 into `data/gsm8k/` for LoRA. Then `uv run python train_sft.py` (2e-5 cosine, not 0.005).
