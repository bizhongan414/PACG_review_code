"""Full-distribution EFS and signed residual-corrected persistence; no model calls."""
import numpy as np


def token_metrics(original_logits, counterfactual_logits, targets):
    """Inputs [T,V], [T,V], [T]. KL is orig || cf, not a sampled-token estimator."""
    orig = np.asarray(original_logits, dtype=np.float64)
    cf = np.asarray(counterfactual_logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if orig.shape != cf.shape or orig.ndim != 2 or targets.shape != orig.shape[:1]:
        raise ValueError("Expected matched [T,V] logits and [T] targets")
    if not np.isfinite(orig).all() or not np.isfinite(cf).all():
        raise ValueError("Nonfinite logits")
    orig = orig - orig.max(-1, keepdims=True)
    cf = cf - cf.max(-1, keepdims=True)
    orig -= np.log(np.exp(orig).sum(-1, keepdims=True))
    cf -= np.log(np.exp(cf).sum(-1, keepdims=True))
    efs = (np.exp(orig) * (orig - cf)).sum(-1)
    delta = cf[np.arange(len(targets)), targets] - orig[np.arange(len(targets)), targets]
    return efs, delta


def claim_metrics(efs, delta, spans, offsets, min_residual=8):
    """Exclude direct and indirect visual spans from the residual reference."""
    efs, delta = np.asarray(efs), np.asarray(delta)
    if len(offsets) != len(delta) or efs.shape != delta.shape:
        raise ValueError("Token/offset shape mismatch")
    residual = np.array([b > a for a, b in offsets], dtype=bool)
    direct = []
    for span in spans:
        a, b = span["char_start"], span["char_end"]
        tokens = [i for i, (x, y) in enumerate(offsets) if y > x and x < b and y > a]
        visual = (span.get("span_group") in {"direct_vef", "derived_vef"}
                  or span.get("visual_dependency") in {"direct", "indirect"})
        if visual:
            residual[tokens] = False
        if span.get("span_group") == "direct_vef" and tokens:
            direct.append((span, tokens))
    count = int(residual.sum())
    calibration = float(delta[residual].mean()) if count >= min_residual else None
    result = []
    for span, tokens in direct:
        result.append({**span, "token_indices": tokens, "efs": float(efs[tokens].mean()),
                       "raw_persistence": float(delta[tokens].mean()),
                       "corrected_persistence": float(delta[tokens].mean() - calibration)
                       if calibration is not None else None})
    return {"residual_tokens": count, "residual_shift": calibration, "claims": result}


def question_macro(rows, metric, unsupported_only=False):
    """Tokens -> spans -> response -> question; never treat spans as IID samples."""
    questions = {}
    for row in rows:
        values = [claim[metric] for claim in row["claims"]
                  if claim.get(metric) is not None and (not unsupported_only or
                      claim.get("evidence_status") in {"unsupported", "contradicted"} or
                      claim.get("analysis_category") == "hallucinated_or_unsupported_direct_vef")]
        if values:
            key = (str(row.get("benchmark", "")), str(row["question_id"]))
            questions.setdefault(key, []).append(float(np.mean(values)))
    return {key: float(np.mean(values)) for key, values in questions.items()}


def paired_bootstrap(base, current, seed=42, draws=5000):
    """Paired question-clustered difference and percentile CI; question-macro weighting."""
    if set(base) != set(current) or not base:
        raise ValueError("Paired comparison requires identical nonempty question populations")
    differences = np.array([current[k] - base[k] for k in sorted(base)])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(draws)])
    return {"change": float(differences.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "questions": len(differences), "draws": draws}

