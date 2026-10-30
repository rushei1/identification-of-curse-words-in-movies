"""Binary classification metrics (token level: profane = positive)."""

from __future__ import annotations


def confusion(gold: list, pred: list) -> dict:
    assert len(gold) == len(pred)
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for g, p in zip(gold, pred):
        c[("t" if g == p else "f") + ("p" if p else "n")] += 1
    return c


def _div(a, b):
    return a / b if b else None  # None = undefined (reported as n/a, never as 0)


def scores(c: dict) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    return {
        **c,
        "n": tp + fp + fn + tn,
        "accuracy": _div(tp + tn, tp + fp + fn + tn),
        "precision": p,
        "recall": r,
        "f1": _div(2 * p * r, p + r) if p is not None and r is not None else None,
        "specificity": _div(tn, tn + fp),
        "false_positive_rate": _div(fp, fp + tn),
        "false_negative_rate": _div(fn, fn + tp),
    }
