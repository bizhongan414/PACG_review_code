"""Scalar PACG reference; framework integration is specified in TRAINING.md."""
import math
from statistics import mean


def pacg_credit(base, delta, direct_spans, residual, valid_mask, *, valid=True,
                margin=0.05, alpha=10.0, floor=0.30, min_residual=8):
    """No factual-support labels are accepted. Returns (credit, span_gates).

    Invalid scoring/alignment must set valid=False. Each direct span is a list
    of response-token indices. Overlap uses the stricter (minimum) gate.
    """
    n = len(base)
    if len(delta) != n or len(valid_mask) != n:
        raise ValueError("Token-array lengths differ")
    if not (margin >= 0 and alpha >= 0 and 0 < floor <= 1 and min_residual >= 1):
        raise ValueError("Invalid gate parameters")
    gates = [1.0] * n
    residual = set(residual)
    spans = [set(s) for s in direct_spans]
    indices = residual.union(*spans)
    if any(type(t) is not int or not 0 <= t < n for t in indices):
        return list(base), gates
    residual = {t for t in residual if valid_mask[t]}
    spans = [{t for t in s if valid_mask[t]} for s in spans]
    if residual.intersection(set().union(*spans)):
        raise ValueError("Residual control must exclude routed claims")
    if (not valid or len(residual) < min_residual or
            any(not math.isfinite(delta[t]) for t in indices if valid_mask[t])):
        return list(base), gates
    control = mean(delta[t] for t in residual)
    for span in spans:
        if not span:
            continue
        corrected = mean(delta[t] for t in span) - control
        gate = max(floor, math.exp(-alpha * max(0.0, corrected + margin)))
        for t in span:
            gates[t] = min(gates[t], gate)
    credit = [a * gates[t] if valid_mask[t] and a > 0 else a
              for t, a in enumerate(base)]
    return credit, gates


def clipped_token_mean_loss(ratios, advantages, loss_mask, low=0.20, high=0.28):
    """DAPO-form scalar surrogate, not an effective gradient coefficient.

    Flatten only tokens actually admitted to the actor update. Parent GRPO
    normalization and other regularizers must remain in the parent trainer.
    """
    if not len(ratios) == len(advantages) == len(loss_mask):
        raise ValueError("Token-array lengths differ")
    if not 0 <= low < 1 or high < 0:
        raise ValueError("Invalid clipping thresholds")
    terms = []
    for r, a, valid in zip(ratios, advantages, loss_mask):
        if valid:
            if not math.isfinite(r) or r <= 0 or not math.isfinite(a):
                raise ValueError("Invalid policy ratio or advantage")
            clipped = min(1 + high, max(1 - low, r))
            terms.append(-min(r*a, clipped*a))
    if not terms:
        raise ValueError("No actor-loss tokens")
    return mean(terms)
