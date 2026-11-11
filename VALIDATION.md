# Export validation

The curated release passed 25 CPU tests covering:

- GRPO/DAPO/VPPO launch configurations with PACG enabled and disabled, merged
  against the actual exported veRL dataclass schema and post-init logic.
- The actual training gate function: residual calibration, overlapping spans,
  non-routed tokens, common likelihood drift and invalid-intervention fallback.
- The actual actor's positive-only credit assignment, preserving zero/negative
  parent advantages.
- Full-vocabulary KL and signed target likelihood shifts, residual exclusion of
  derived visual claims, question aggregation and paired bootstrap populations.
- Frozen response/prefix/target alignment through the diagnostic scoring path
  using a small model/processor double, without pretending to run Qwen weights.
- Exact span alignment and annotation helper loading, service payload validation,
  retries (including HTTP 403), request/batch budgets and annotation concurrency.

Both shell entry points pass `bash -n`. All exported Python files parse. All four
offline CLI `--help` entry points load without private experiment modules. Package
metadata parses, and the source scan found no bundled credentials, private
endpoints or personal filesystem paths in release code.

Checks were run in an isolated CPU environment; they do **not** establish GPU
memory feasibility, CUDA wheel compatibility, FSDP/vLLM integration, annotation
model quality, or reproduction of published experimental values. No real GPU
training or external annotation request was executed during this export.

On the intended CUDA node, install the working training environment, deploy the
annotator separately, and run `MAX_STEPS=2` before a full run. The training launcher
checks local datasets, CUDA GPU count and a real multimodal annotation request
before starting the trainer. Record the resolved configuration and environment
versions alongside each run.

