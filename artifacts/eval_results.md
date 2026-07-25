# Eval - Base vs QLoRA Fine-tuned (held-out test set)

Model: Qwen/Qwen2.5-1.5B-Instruct - test rows: 368 - greedy, max_new_tokens=64.

| Metric | Base (zero-shot) | Fine-tuned | Delta (ft - base) |
|---|---|---|---|
| schema_validity | 0.1386 | 1.0000 | +0.8614 |
| accuracy | 0.0272 | 0.6060 | +0.5788 |
| macro_f1 | 0.0413 | 0.5986 | +0.5573 |

Directional improvement (accuracy & macro_f1): **YES**.
