"""Anonymous, provider-independent routing reference. No network side effects."""
import json


PROMPT = """Identify minimal contiguous assertions of information purportedly
read directly from the supplied image(s): counts, attributes, OCR, chart values,
spatial relations, and geometric properties. Directness is an evidential role,
not factual correctness. Route false visual observations as direct claims too.
Exclude question-only facts, hypotheses, option restatements, calculations,
general knowledge, final answers, and meta-reasoning. Split observations from
their downstream deductions. Also mark derived/indirect visual spans for
residual exclusion. Return JSON with a spans list. Each item must contain exact
text, start and end character offsets (half-open), span_group (direct_vef,
derived_vef, non_vef_control, other), and visual_dependency (direct, indirect,
none). Offsets refer to the unchanged response. Do not assess the final answer.
For offline auditing only, optionally add evidence_status (supported,
contradicted, unsupported, unknown); this field never determines routing.
Return an empty spans list if no spans qualify; do not fabricate observations.
"""


def align_span(response, offsets, span):
    """Reject invalid/ambiguous spans instead of guessing a substring match."""
    a, b = span.get("start"), span.get("end")
    if (type(a) is not int or type(b) is not int or
            not 0 <= a < b <= len(response) or response[a:b] != span.get("text")):
        raise ValueError("Span must exactly match its character interval")
    for lo, hi in offsets:
        if not 0 <= lo <= hi <= len(response):
            raise ValueError("Invalid response-token offsets")
    indices = [i for i, (lo, hi) in enumerate(offsets)
               if lo < hi and lo < b and hi > a]
    if not indices:
        raise ValueError("Span has no response tokens")
    covered = set()
    for i in indices:
        covered.update(range(max(a, offsets[i][0]), min(b, offsets[i][1])))
    if any(not response[j].isspace() and j not in covered for j in range(a, b)):
        raise ValueError("Incomplete character coverage")
    return indices


def annotate(router, question, response, images):
    """router(request) -> JSON string/dict; images are caller-owned inputs."""
    request = {"instruction": PROMPT, "question": question,
               "trace": response, "images": images}
    result = router(request)
    if isinstance(result, str):
        result = json.loads(result)
    if not isinstance(result, dict) or not isinstance(result.get("spans"), list):
        raise ValueError("Expected a JSON object with a spans list")
    if any(not isinstance(s, dict) for s in result["spans"]):
        raise ValueError("Every span must be an object")
    return result["spans"]


def route(response, offsets, spans, valid_mask):
    """Return direct token sets, residual indices, and routing validity.

    Any alignment failure invalidates this rollout conservatively. Support
    labels are deliberately ignored. All annotated visual dependencies are
    excluded from the residual control, not only direct claims.
    """
    if len(offsets) != len(valid_mask):
        raise ValueError("Offset/mask lengths differ")
    direct, excluded = [], set()
    try:
        for s in spans:
            indices = [t for t in align_span(response, offsets, s) if valid_mask[t]]
            if not indices:
                raise ValueError("No valid span tokens")
            if s.get("span_group") == "direct_vef":
                direct.append(indices)
            if (s.get("span_group") in {"direct_vef", "derived_vef"} or
                    s.get("visual_dependency") in {"direct", "indirect"} or
                    s.get("is_vef") is True):
                excluded.update(indices)
    except (ValueError, TypeError, AttributeError):
        return [], [], False
    residual = [t for t, valid in enumerate(valid_mask) if valid and t not in excluded]
    return direct, residual, True
