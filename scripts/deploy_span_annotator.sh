#!/usr/bin/env bash
# Run on separate annotation GPUs; no downloads, OSS access, or cluster submission.
set -euo pipefail
: "${ANNOTATOR_MODEL_PATH:?Set ANNOTATOR_MODEL_PATH to a local multimodal model directory}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
args=(--host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
  --model "$ANNOTATOR_MODEL_PATH" --served-model-name "${ANNOTATOR_MODEL:-annotator}"
  --tensor-parallel-size "${TP_SIZE:-8}" --trust-remote-code
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --max-num-seqs "${MAX_NUM_SEQS:-32}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --enable-prefix-caching)
[[ "${ENFORCE_EAGER:-0}" != 1 ]] || args+=(--enforce-eager)
[[ -z "${ANNOTATOR_API_KEY:-}" ]] || args+=(--api-key "$ANNOTATOR_API_KEY")
exec python -m vllm.entrypoints.openai.api_server "${args[@]}" "$@"

