"""Question-macro mechanism estimates and matched question-clustered confidence intervals."""
import argparse
import json

import numpy as np

from .metrics import paired_bootstrap, question_macro


def read(filename):
    with open(filename) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [str(row["sample_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Empty results or duplicate sample IDs")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--base")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    rows = read(args.input)
    baseline = read(args.base) if args.base else None
    if baseline:
        def identities(data):
            return {str(row["sample_id"]): (row["reference_sha256"], row["response_token_ids"],
                    row["protocol"], row["intervention_seed"], row["intervention_valid"]) for row in data}
        if identities(rows) != identities(baseline):
            raise ValueError("Fixed references, tokenization or intervention protocol differ")
    output = {"responses": len(rows), "valid_responses": sum(row["intervention_valid"] for row in rows),
              "aggregation": "span -> response -> question (question macro)"}
    for label, metric, unsup in (("direct_efs", "efs", False),
                                 ("overall_persistence", "corrected_persistence", False),
                                 ("unsupported_persistence", "corrected_persistence", True)):
        values = question_macro(rows, metric, unsup)
        output[label] = {"mean": float(np.mean(list(values.values()))) if values else None,
                         "questions": len(values)}
        if baseline:
            base = question_macro(baseline, metric, unsup)
            output[label]["paired"] = paired_bootstrap(base, values, args.seed, args.bootstrap)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

