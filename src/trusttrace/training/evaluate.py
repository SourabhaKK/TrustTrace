"""Evaluate base vs QLoRA fine-tuned model on the held-out test set.

The comparison is the deliverable (not the fine-tuned number alone). For each test row we
generate JSON (greedy, max_new_tokens=64 — deliberately generous so the base model's more
verbose zero-shot output isn't truncated) and compute, per model:

- schema_validity: fraction whose output parses into the strict ClassificationOutput schema
- accuracy:        category correct (unparseable / invalid category = wrong)
- macro_f1:        macro-F1 over the 7 categories

Metrics separate "did it emit valid schema" from "did it get the category right", so the
fine-tune's gains on schema adherence vs classification are both visible.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from trusttrace.schema import Category, ClassificationOutput
from trusttrace.training.dataset import load_split
from trusttrace.training.prompt import render_for_inference
from trusttrace.training.train import ADAPTER_DIR, MODEL_ID

log = logging.getLogger(__name__)

_CATEGORY_VALUES = {c.value for c in Category}
_LABELS = [c.value for c in Category]
_JSON_RE = re.compile(r"\{[^{}]*\}")
RESULTS_PATH = Path("artifacts/eval_results.md")


def _load_base(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16
    )
    model.eval()
    return model, tok


def extract_prediction(text: str) -> tuple[bool, str | None]:
    """(schema_valid, predicted_category). schema_valid is strict; category is lenient."""
    m = _JSON_RE.search(text)
    if not m:
        return False, None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return False, None
    schema_valid = True
    try:
        ClassificationOutput.model_validate(obj)
    except Exception:  # noqa: BLE001
        schema_valid = False
    cat = obj.get("category") if isinstance(obj, dict) else None
    return schema_valid, (cat if cat in _CATEGORY_VALUES else None)


@torch.no_grad()
def _generate(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    outputs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        new = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
        log.info("generated %d/%d", min(i + batch_size, len(prompts)), len(prompts))
    return outputs


def _metrics(texts: list[str], gold: list[str]) -> dict[str, float]:
    valids, preds = zip(*(extract_prediction(t) for t in texts))
    n = len(gold)
    y_pred = [p if p is not None else "INVALID" for p in preds]
    return {
        "schema_validity": sum(valids) / n,
        "accuracy": sum(p == g for p, g in zip(y_pred, gold)) / n,
        "macro_f1": f1_score(gold, y_pred, labels=_LABELS, average="macro", zero_division=0),
    }


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    if not ADAPTER_DIR.exists():
        raise SystemExit(f"Adapter not found at {ADAPTER_DIR} — run training first.")

    df = load_split("test")
    if args.limit:
        df = df.head(args.limit)
    gold = df["category"].tolist()

    base, tok = _load_base(args.model_id)
    prompts = [render_for_inference(tok, t, max_length=args.max_len) for t in df["text"].tolist()]

    log.info("=== base model ===")
    base_texts = _generate(base, tok, prompts, args.batch_size, args.max_new_tokens)
    base_metrics = _metrics(base_texts, gold)

    log.info("=== fine-tuned model ===")
    from peft import PeftModel

    ft = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    ft.eval()
    ft_texts = _generate(ft, tok, prompts, args.batch_size, args.max_new_tokens)
    ft_metrics = _metrics(ft_texts, gold)

    _report(base_metrics, ft_metrics, n=len(gold))


def _report(base: dict, ft: dict, n: int) -> None:
    rows = [
        ("schema_validity", base["schema_validity"], ft["schema_validity"]),
        ("accuracy", base["accuracy"], ft["accuracy"]),
        ("macro_f1", base["macro_f1"], ft["macro_f1"]),
    ]
    lines = [
        "# Eval - Base vs QLoRA Fine-tuned (held-out test set)",
        "",
        f"Model: {MODEL_ID} - test rows: {n} - greedy, max_new_tokens=64.",
        "",
        "| Metric | Base (zero-shot) | Fine-tuned | Delta (ft - base) |",
        "|---|---|---|---|",
    ]
    for name, b, f in rows:
        lines.append(f"| {name} | {b:.4f} | {f:.4f} | {f - b:+.4f} |")
    improved = ft["accuracy"] >= base["accuracy"] and ft["macro_f1"] >= base["macro_f1"]
    lines += ["", f"Directional improvement (accuracy & macro_f1): **{'YES' if improved else 'NO'}**."]
    text = "\n".join(lines)

    # Persist FIRST so results survive even if console printing fails.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(text + "\n", encoding="utf-8")
    (RESULTS_PATH.parent / "eval_metrics.json").write_text(
        json.dumps({"base": base, "fine_tuned": ft, "n": n}, indent=2), encoding="utf-8"
    )
    try:
        print("\n" + text + "\n")
    except UnicodeEncodeError:
        print("\n" + text.encode("ascii", "replace").decode() + "\n")
    print(f"written -> {RESULTS_PATH}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="evaluate only first N rows (debug)")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
