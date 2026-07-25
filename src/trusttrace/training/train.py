"""QLoRA fine-tune of Qwen2.5-1.5B-Instruct into the content-policy classifier.

Locked config (PRD §11.3): nf4 4-bit + bf16 compute + double-quant; LoRA r=16 alpha=32
dropout=0.05 on q/k/v/o_proj; lr=2e-4, 3 epochs, paged_adamw_8bit, cosine + warmup, gradient
checkpointing, max_seq_len=512. Completion-only loss (see training/prompt.py).

Usage:
    python -m trusttrace.training.train --smoke     # tiny run to prove the loop + VRAM
    python -m trusttrace.training.train             # full 3-epoch fine-tune -> artifacts/adapter
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path("artifacts/adapter")
CHECKPOINT_DIR = Path("models/qlora")  # git-ignored working dir


def _load_model_and_tokenizer(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16
    )
    model.config.use_cache = False
    return model, tokenizer


def _apply_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return model


def train(args: argparse.Namespace) -> None:
    from transformers import Trainer, TrainingArguments

    from trusttrace.training.dataset import CausalCollator, build_dataset, load_split

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — QLoRA training requires an NVIDIA GPU.")
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = _load_model_and_tokenizer(args.model_id)
    model = _apply_lora(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("trainable params: %d (%.3f%% of %d)", trainable, 100 * trainable / total, total)

    train_df = load_split("train")
    val_df = load_split("val")
    train_ds = build_dataset(train_df, tokenizer, args.max_len, limit=64 if args.smoke else None)
    eval_ds = build_dataset(val_df, tokenizer, args.max_len, limit=16 if args.smoke else None)

    common = dict(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if args.smoke else 10,
        report_to="none" if args.smoke else ["mlflow"],
    )
    if args.smoke:
        targs = TrainingArguments(max_steps=8, eval_strategy="no", save_strategy="no", **common)
    else:
        targs = TrainingArguments(
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="no",
            **common,
        )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalCollator(pad_token_id=tokenizer.pad_token_id),
    )

    try:
        result = trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        raise SystemExit(f"OOM during training (peak {peak:.2f} GiB). Reduce batch/seq or model.") from exc

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n=== {'SMOKE' if args.smoke else 'FULL'} run done ===")
    print(f"train_loss={result.training_loss:.4f}  steps={result.global_step}  peak_VRAM={peak:.2f} GiB")

    if not args.smoke:
        metrics = trainer.evaluate()
        print(f"final eval_loss={metrics.get('eval_loss'):.4f}")
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ADAPTER_DIR))
        tokenizer.save_pretrained(str(ADAPTER_DIR))
        print(f"adapter saved -> {ADAPTER_DIR}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "trusttrace-qlora")
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny run: 64 rows, 8 steps, no save")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=512)
    train(p.parse_args())


if __name__ == "__main__":
    main()
