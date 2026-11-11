#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
parent="${PARENT:-dapo}"
case "$parent" in
  grpo) online=false; loss=seq; clip_high=0.20; visual=false ;;
  dapo) online=true; loss=token; clip_high=0.28; visual=false ;;
  vppo) online=true; loss=token; clip_high=0.28; visual=true ;;
  *) echo 'PARENT must be grpo, dapo, or vppo' >&2; exit 2 ;;
esac
pacg="${USE_PACG:-true}"
case "$pacg" in true|false) ;; *) echo 'USE_PACG must be true or false' >&2; exit 2 ;; esac
run_name="${RUN_NAME:-${parent}_pacg_7b_seed${SEED:-42}}"
output="${OUTPUT_DIR:-$PWD/outputs/$run_name}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$output/tensorboard}"
args=(config=examples/configs/pacg_7b.yaml
  "data.train_files=${TRAIN_DATA:-}" "data.val_files=${VAL_DATA:-}"
  "worker.actor.model.model_path=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
  "data.seed=${SEED:-42}" "worker.rollout.seed=${SEED:-42}"
  "data.rollout_batch_size=${ROLLOUT_BATCH_SIZE:-384}"
  "data.val_batch_size=${VAL_BATCH_SIZE:-512}"
  "worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE:-128}"
  "worker.actor.micro_batch_size_per_device_for_update=${UPDATE_MICRO_BATCH:-2}"
  "worker.actor.micro_batch_size_per_device_for_experience=${EXPERIENCE_MICRO_BATCH:-4}"
  "algorithm.use_pa_grpo=$pacg" "algorithm.online_filtering=$online"
  "worker.actor.loss_avg_mode=$loss" "worker.actor.clip_ratio_high=$clip_high"
  "algorithm.use_vppo_on_perception=$visual" "algorithm.use_advantage_shaping=$visual"
  algorithm.top_p_perception_tokens=0.4 algorithm.advantage_scaling_min=0.9
  "algorithm.entropy_penalty_coef=${ENTROPY_COEF:-0.10}"
  "algorithm.disable_kl=${DISABLE_KL:-true}" "algorithm.use_kl_loss=${USE_KL_LOSS:-false}"
  "algorithm.kl_coef=${KL_COEF:-0}"
  "algorithm.pa_margin=${PACG_MARGIN:-0.05}" "algorithm.pa_alpha=${PACG_ALPHA:-10}"
  "algorithm.pa_gate_floor=${PACG_FLOOR:-0.30}"
  "algorithm.pa_annotator_url=${ANNOTATOR_URL:-http://127.0.0.1:8000/v1/chat/completions}"
  "algorithm.pa_annotator_model=${ANNOTATOR_MODEL:-annotator}"
  "algorithm.pa_annotator_concurrency=${ANNOTATOR_CONCURRENCY:-16}"
  "algorithm.pa_annotation_batch_budget=${ANNOTATION_BATCH_BUDGET:-3600}"
  "algorithm.pa_annotator_temperature=${ANNOTATOR_TEMPERATURE:-0.8}"
  "algorithm.pa_annotation_log_dir=$output/annotations"
  "algorithm.pa_gate_log_dir=$output/gates"
  "worker.reward.reward_function_kwargs.format_weight=${FORMAT_WEIGHT:-0.0}"
  "trainer.n_gpus_per_node=${N_GPUS:-8}" "trainer.max_steps=${MAX_STEPS:-200}"
  "trainer.experiment_name=$run_name" "trainer.save_checkpoint_path=$output"
  "trainer.rollout_data_dir=$output/rollouts" "trainer.val_data_dir=$output/validation"
  "trainer.debug_event_log_path=$output/debug_events.jsonl"
  "trainer.save_freq=${SAVE_FREQ:-20}" "trainer.val_freq=${VAL_FREQ:-20}"
)
if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf '%q ' python -m verl.trainer.main "${args[@]}" "$@"; printf '\n'
  exit 0
fi
: "${TRAIN_DATA:?Set TRAIN_DATA to a local training JSONL/parquet file or directory}"
: "${VAL_DATA:?Set VAL_DATA to a local validation JSONL/parquet file or directory}"
[[ -e "$TRAIN_DATA" && -e "$VAL_DATA" ]] || { echo 'Local dataset does not exist' >&2; exit 2; }
python -m pacg.preflight "${args[@]}" "$@"
mkdir -p "$output" "$TENSORBOARD_DIR"
python -m verl.trainer.main "${args[@]}" "$@" 2>&1 | tee "$output/train.log"

