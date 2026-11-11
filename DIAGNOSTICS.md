# Fixed-rollout SPD diagnostic

Run all comparisons on the **same Base responses, claims, image intervention,
questions and tokenizer**. Do not annotate fresh checkpoint-dependent responses
when measuring changes on a fixed reference.

## Input and reference generation

Local evaluation input is one JSON object per line:

```json
{"sample_id":"q1","question_id":"q1","benchmark":"MMK12","question":"<image>\nHow many circles are visible?","image":"images/q1.png","answer":"3"}
```

Image paths are relative to the input JSONL or absolute. Question IDs must be
stable, with a benchmark name if IDs can repeat across benchmarks. For multiple
responses, sample IDs must be unique. No private dataset paths are built in.

```bash
# Generate one frozen population from the base model, not each trained checkpoint.
python -m pacg.evaluate --model /models/Qwen2.5-VL-7B-Instruct \
  --input /data/crossbench_questions.jsonl --output base_responses.jsonl \
  --n 8 --temperature 1.0
```

This writes the generated response, source question, image, exact-match accuracy
and generation settings. The printed summary averages responses within question,
questions within benchmark, then benchmarks equally. With nine benchmark names,
the final mean is the unweighted nine-benchmark average. Verifier behavior depends
on the provided answer representation; check dataset-specific extraction before
treating it as the official score of a non-mathematical benchmark.

## Span annotation

```bash
python -m pacg.annotate --input base_responses.jsonl --output fixed_reference.jsonl \
  --url http://annotator-host:8000/v1/chat/completions --model annotator
```

The exported prompt is the Chinese prompt from the selected production helper,
with directness, evidence status, derived claims and residual controls separated.
Its full rendered text is in `prompts/visual_claim_annotation.txt`. Annotation
temperature defaults to 0 for offline analysis, versus 0.8 in the inherited
online training client; explicitly choose the intended configuration. The offline
client refuses exhausted failures instead of treating them as an empty claim set.

Each annotated span includes `text`, `span_group`, `evidence_status`,
`analysis_category` and exact `char_start`/`char_end`. Automatic offline alignment
accepts unique exact substrings. If the same span text occurs more than once,
provide verified offsets rather than silently choosing the wrong occurrence.
Do not use factual evidence labels for training eligibility or gating.

## Fixed-response teacher forcing

```bash
python -m pacg.diagnose --model /models/Qwen2.5-VL-7B-Instruct \
  --reference fixed_reference.jsonl --output base_scores.jsonl
python -m pacg.diagnose --model /models/merged-dapo-checkpoint \
  --reference fixed_reference.jsonl --output dapo_scores.jsonl
python -m pacg.diagnose --model /models/merged-dapo-pacg-checkpoint \
  --reference fixed_reference.jsonl --output pacg_scores.jsonl
```

This implementation currently supports Qwen2.5-VL and one image per reference.
It teacher-forces the same response with fixed prefixes and targets; it never
regenerates under the corrupted image. EFS is the token-average full-vocabulary
KL(original || corrupted). Raw persistence is the mean target-token log-probability
change (corrupted minus original); corrected persistence subtracts the residual
token mean. Direct and derived visual spans are excluded from the residual set;
at least eight residual text tokens are required. Evidence-status labels only
select analysis subgroups, including contradicted + unsupported direct claims.

The training kernel retains its original numerical guard, clamping token
log-probability differences to [-20,20]; offline diagnostics report unclipped
differences. This distinction matters for extreme probability ratios.

Random masking uses 14×14 patches, probability 0.5 and a sample-specific seed
derived from SHA256(seed, sample_id). Thus every checkpoint sees the same mask.
The processed image, prompt IDs and visual grid remain fixed in geometry.
`--operator coarse_ink` supplies the documented 245-threshold/9×9 expansion
robustness variant. Use a new output file for each intervention and compare only
matching protocols. Images outside the valid changed-area range [0.005,0.95]
are excluded from mechanism aggregates, with invalidity reported in the output.

References are not truncated. If a sample is too long, explicitly revise the
reference protocol or increase `--max-length` within the model's capacity. Each
output stores target IDs, a reference/image hash, corruption seed and protocol.
The response text is tokenized once per checkpoint and required to round-trip;
the paired reporter checks that all compared target IDs are identical. If a
reference already supplies `response_token_ids`, these must match exactly.

Scoring materializes the checkpoint's vocabulary logits and computes KL in
32-token chunks on CPU. It uses more memory/time than target-only likelihood
scoring; target-token low-variance KL is not substituted for full EFS.

## Estimates and uncertainty

```bash
python -m pacg.report --input dapo_scores.jsonl --base base_scores.jsonl
python -m pacg.report --input pacg_scores.jsonl --base dapo_scores.jsonl
```

The report gives direct EFS, overall corrected persistence and unsupported
corrected persistence. It averages tokens within span, spans within response,
responses within question, then questions equally. Paired changes and percentile
95% CIs use question-clustered bootstrap (5000 draws by default). Benchmark names
are part of the question identity. Spans are never treated as independent samples.
This default is question-macro pooling, not a claim that all historical tables
used the same benchmark-macro convention.

The paired reporter refuses differing reference hashes, targets, intervention
protocols or question populations. It does not silently intersect populations.
To reproduce checkpoint/phase trajectories, score every checkpoint on the same
reference, record the resulting means, and plot EFS against negative unsupported
persistence. No numeric endpoints or confidence intervals are invented.

