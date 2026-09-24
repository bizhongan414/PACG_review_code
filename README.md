# Anonymous review code: claim-level credit assignment

This is a selective, independently written reference implementation of the
paper's core computations, not the complete distributed training repository.
It contains no experimental data, checkpoints, recorded model outputs, service
credentials, author identifiers, or machine-specific paths. The examples are
synthetic and must not be interpreted as experimental results.

## Contents

- `annotation.py`: provider-independent multimodal annotation request, strict
  character-span/token alignment, direct-claim routing and residual selection.
- `metrics.py`: full-distribution EFS, signed/corrected persistence, offline
  unsupported-claim aggregation, and nine-benchmark accuracy summaries.
- `rl.py`: conservative positive-only PACG gates and a scalar reference for
  the asymmetric clipped token-mean objective.
- `TRAINING.md`: integration pseudocode and the boundary of this release.
- `test_core.py`: synthetic checks of the mathematical and routing invariants.

Run from this directory with Python 3.10 or later (standard library only):

```sh
python3 -m unittest -v test_core
```

## Conventions

All token arrays contain **response tokens only**, with a separate valid-token
mask excluding padding and non-text positions. Offsets are half-open character
intervals in the exact original response, not in a normalized or retokenized
version. The caller must map offsets to the actual generated token IDs using the
same tokenizer/template as training. Failed or ambiguous alignment is rejected.

`token_scores` accepts logits already aligned to the predicted response token:
in a causal model, logits at position `j-1` predict the token at position `j`.
Hold response IDs and every prefix fixed between the two image conditions.
Pass the complete vocabulary logits; top-k approximations are not exact EFS.
The pure-Python implementation prioritizes inspectability, not throughput.

`claim_metrics` averages tokens within a span. `reference_summary` averages
eligible spans within rollout, rollouts within question, then questions equally.
Question IDs must include dataset identity to avoid collisions. Undefined
subgroups are omitted and their valid question counts are returned; they are
never silently converted to zero. Unsupported includes `unsupported` and
`contradicted`, not `unknown`. These labels are used only offline and are not
accepted by the gate function.

For task accuracy, supply independently verified binary answer scores to
`benchmark_accuracy`, average responses within question, and supply exactly
nine benchmark means to `nine_benchmark_mean`. It does not implement each
benchmark's answer extraction or verifier. Do not compute pass@8 by taking the
maximum response score. Reported tables are not embedded as computed results.

## Annotation boundary

`annotate` requires an injected frozen-router callable and original image inputs.
It performs no network access itself. Its compact prompt is a review-oriented
restatement of the routing criteria, **not a claim of byte-for-byte identity
with the deployed annotation prompt**. The manuscript appendix documents the
experimental prompt. Gold answers and source-model identifiers are not sent.
The response and images supplied by the caller may themselves contain sensitive
information: use an approved local or remote router and appropriate data access.

## RL boundary

Pass the parent's actual token credit into PACG, including VPPO shaping where
applicable. Do not replace VPPO credit by a broadcast sequence advantage.
Invalid routing/scoring or fewer than eight residual tokens gives unit gates.
The numerical loss is not an autograd trainer: the training integration must
detach gates, preserve the parent's loss normalization, masks, clipping,
sampling, regularization, and optimizer. EFS is diagnostic, not a gate input.

No complete training framework, deployment launcher, dataset, annotation service,
checkpoint replay, online effective-update-mass reconstruction, bootstrap
pipeline, or experiment reproduction claim is included. Those components are
outside this selective release; the supplied tests verify formulas only.
