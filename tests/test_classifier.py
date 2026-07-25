"""Concurrency test for DualClassifier's adapter-state lock (no GPU needed).

A fake model records a 'violation' whenever its adapter-enabled flag changes DURING a
generate() call — i.e. another thread flipped the shared adapter mid-generation. With the
lock, generates are serialized so no violation can occur; without it, races appear. This
proves the lock does its job and that the test actually bites.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time

os.environ["OTEL_DISABLE_OTLP"] = "1"  # spans go nowhere; this test is about the lock

import torch  # noqa: E402

from trusttrace.serving.classifier import DualClassifier  # noqa: E402

_FAKE_JSON = '{"category": "none", "severity": "none", "confidence": 1.0}'


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, prompt, return_tensors=None, add_special_tokens=None):
        return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}

    def decode(self, tokens, skip_special_tokens=True):
        return _FAKE_JSON


class _FakeModel:
    def __init__(self):
        self._enabled = True
        self.violations: list[tuple[bool, bool]] = []

    @contextlib.contextmanager
    def disable_adapter(self):
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = True

    def generate(self, input_ids=None, **kwargs):
        start = self._enabled
        time.sleep(0.002)  # widen the race window (releases the GIL)
        if self._enabled != start:  # another thread flipped the adapter mid-generation
            self.violations.append((start, self._enabled))
        return torch.cat([input_ids, torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)], dim=1)


def _hammer(clf: DualClassifier, iters: int) -> None:
    for _ in range(iters):
        clf._infer_both("dummy prompt")


def _run(clf: DualClassifier, threads: int = 8, iters: int = 40) -> None:
    workers = [threading.Thread(target=_hammer, args=(clf, iters)) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()


def test_lock_prevents_adapter_race():
    model = _FakeModel()
    clf = DualClassifier(model, _FakeTokenizer(), device="cpu", max_new_tokens=1)
    _run(clf)
    assert model.violations == [], f"adapter-state race under lock: {model.violations[:3]}"


def test_race_is_detectable_without_lock():
    # Neutralize the lock -> the same load must surface races, proving the guard is real.
    model = _FakeModel()
    clf = DualClassifier(model, _FakeTokenizer(), device="cpu", max_new_tokens=1)
    clf._lock = contextlib.nullcontext()
    _run(clf)
    assert model.violations, "expected adapter-state races once the lock is removed"
