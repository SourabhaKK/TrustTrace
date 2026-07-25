"""Dual-model canary classifier.

Loads the 4-bit base model ONCE and attaches the QLoRA adapter as a ``PeftModel``. Each
request is classified by both paths from the same weights:
- fine-tuned path: adapter enabled (default)
- base path: inside ``model.disable_adapter()``

Because the two paths mutate the SAME model's adapter state, and FastAPI runs sync handlers
in a threadpool (requests overlap), ``classify()`` holds a ``threading.Lock`` across the
entire base+fine-tuned inference so one request cannot flip the adapter mid-way through
another. GPU inference is serialized anyway, so the lock costs little.

Scope (CLAUDE.md): only parsed structured fields leave this module — raw model text is never
returned or logged.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from trusttrace.parse import ParseResult, parse_classification
from trusttrace.training.prompt import render_for_inference
from trusttrace.training.train import ADAPTER_DIR, MODEL_ID


@dataclass
class DualResult:
    """Both paths' parsed outputs for one request."""

    finetuned: ParseResult
    base: ParseResult


class DualClassifier:
    def __init__(self, model, tokenizer, device: str = "cuda", max_len: int = 512, max_new_tokens: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_id: str = MODEL_ID, adapter_dir=ADAPTER_DIR, **kwargs) -> "DualClassifier":
        """Load base (4-bit nf4) + LoRA adapter as one PeftModel on the GPU."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model.eval()
        return cls(model, tokenizer, device="cuda", **kwargs)

    @torch.no_grad()
    def _generate_text(self, prompt: str) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        gen = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new = gen[0, enc["input_ids"].shape[1] :]
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def _infer_both(self, prompt: str) -> tuple[str, str]:
        """Generate both paths under the lock (no adapter-state races across requests)."""
        with self._lock:
            ft_text = self._generate_text(prompt)          # adapter enabled -> fine-tuned
            with self.model.disable_adapter():             # base path
                base_text = self._generate_text(prompt)
        return ft_text, base_text

    def classify(self, text: str) -> DualResult:
        """Classify one input through both paths; returns parsed structured results."""
        prompt = render_for_inference(self.tokenizer, text, self.max_len)
        ft_text, base_text = self._infer_both(prompt)
        return DualResult(
            finetuned=parse_classification(ft_text),
            base=parse_classification(base_text),
        )


_singleton: DualClassifier | None = None


def get_classifier() -> DualClassifier:
    """Lazily load and cache the dual classifier (call site: FastAPI dependency)."""
    global _singleton
    if _singleton is None:
        _singleton = DualClassifier.load()
    return _singleton
