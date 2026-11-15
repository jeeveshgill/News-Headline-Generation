"""ROUGE-1 / ROUGE-2 / ROUGE-L implemented from scratch so results are reproducible and explainable."""
import re
from collections import Counter

import numpy as np


def _tok(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def _prf(overlap, n_hyp, n_ref):
    p = overlap / n_hyp if n_hyp else 0.0
    r = overlap / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def rouge_n(ref, hyp, n=1):
    g = lambda w: Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    rg, hg = g(_tok(ref)), g(_tok(hyp))
    return _prf(sum((rg & hg).values()), sum(hg.values()), sum(rg.values()))


def rouge_l(ref, hyp):
    a, b = _tok(ref), _tok(hyp)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return _prf(dp[-1][-1], len(b), len(a))


def evaluate(refs, hyps) -> dict:
    r1 = [rouge_n(r, h, 1) for r, h in zip(refs, hyps)]
    r2 = [rouge_n(r, h, 2) for r, h in zip(refs, hyps)]
    rl = [rouge_l(r, h) for r, h in zip(refs, hyps)]
    m = lambda xs, k: round(float(np.mean([x[k] for x in xs])), 4)
    return {"ROUGE-1 P": m(r1, 0), "ROUGE-1 R": m(r1, 1), "ROUGE-1 F1": m(r1, 2),
            "ROUGE-2 F1": m(r2, 2), "ROUGE-L F1": m(rl, 2),
            "Avg length (words)": round(float(np.mean([len(h.split()) for h in hyps])), 2)}
