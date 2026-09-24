"""Exact, small-input numerical references for the paper's diagnostics."""
import math
from collections import defaultdict
from statistics import mean


def log_softmax(logits):
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("Expected finite full-vocabulary logits")
    maximum = max(logits)
    shifted = [x - maximum for x in logits]
    normalizer = math.log(sum(math.exp(x) for x in shifted))
    return [x - normalizer for x in shifted]


def token_scores(original_logits, counterfactual_logits, targets):
    if not len(original_logits) == len(counterfactual_logits) == len(targets):
        raise ValueError("Response-token lengths differ")
    efs, delta = [], []
    for original, cf, target in zip(original_logits, counterfactual_logits, targets):
        if len(original) != len(cf) or not 0 <= target < len(original):
            raise ValueError("Vocabulary/target mismatch")
        p, q = log_softmax(original), log_softmax(cf)
        efs.append(max(0.0, sum(math.exp(a) * (a-b) for a, b in zip(p, q))))
        delta.append(q[target] - p[target])
    return efs, delta


def claim_metrics(efs, delta, span, residual, min_residual=8):
    span, residual = sorted(set(span)), sorted(set(residual))
    if len(efs) != len(delta) or set(span) & set(residual):
        raise ValueError("Inconsistent arrays or overlapping residual control")
    if any(t < 0 or t >= len(delta) for t in span + residual):
        raise ValueError("Token index out of range")
    if not span or len(residual) < min_residual:
        return None
    if any(not math.isfinite(delta[t]) or not math.isfinite(efs[t])
           for t in span + residual):
        return None
    raw, control = mean(delta[t] for t in span), mean(delta[t] for t in residual)
    return {"efs": mean(efs[t] for t in span), "raw_persistence": raw,
            "residual_shift": control, "corrected_persistence": raw-control}


def reference_summary(rows, metric="corrected_persistence", unsupported_only=False):
    """One row per valid direct claim, containing question_id and rollout_id."""
    rollouts = defaultdict(list)
    for row in rows:
        if unsupported_only and row.get("evidence_status") not in {"unsupported", "contradicted"}:
            continue
        value = row.get(metric)
        if value is not None and math.isfinite(value):
            rollouts[(row["question_id"], row["rollout_id"])].append(value)
    questions = defaultdict(list)
    for (question, _), values in rollouts.items():
        questions[question].append(mean(values))
    values = [mean(v) for v in questions.values()]
    return {"mean": mean(values) if values else None,
            "n_questions": len(values), "n_rollouts": len(rollouts)}


def benchmark_accuracy(rows, expected_responses=8):
    """Rows: question_id, rollout_id, correct (binary verified answer score)."""
    questions, seen = defaultdict(list), set()
    for row in rows:
        key = (row["question_id"], row["rollout_id"])
        if key in seen or row["correct"] not in (0, 1):
            raise ValueError("Duplicate response or nonbinary answer score")
        seen.add(key)
        questions[key[0]].append(row["correct"])
    if not questions or any(len(v) != expected_responses for v in questions.values()):
        raise ValueError("Incomplete evaluation; do not silently drop responses")
    return 100 * mean(mean(v) for v in questions.values())


def nine_benchmark_mean(scores):
    if len(scores) != 9 or any(not math.isfinite(x) or not 0 <= x <= 100
                              for x in scores.values()):
        raise ValueError("Expected nine distinct benchmark accuracies in percent")
    return mean(scores.values())
