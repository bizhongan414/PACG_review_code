# PACG: training and sensitivity–persistence diagnostics

This repository contains executable PACG training code on GRPO, DAPO and VPPO,
and a standalone fixed-rollout diagnostic for Sensitivity–Persistence Decoupling
(SPD). The training backend is the veRL/EasyR1-derived runtime used in VPPO_RL;
this is a curated runtime export, not the earlier illustrative review-code package.

The release excludes OSS download/upload, cluster submission, private service
addresses, API credentials, experiment logs, datasets, and unrelated experiment
launchers. All data/model paths and annotation endpoints are supplied by the user.
The original `VPPO_RL` tree is not modified by this export.

## Included code

| Purpose | Entry point |
| --- | --- |
| Qwen2.5-VL-7B training, any parent with/without PACG | `scripts/train_qwen25vl7b.sh` |
| PACG routing, residual calibration and span gates | `verl/trainer/pa_grpo.py` |
| Positive-only advantage gating and VPPO credit shaping | `verl/workers/actor/dp_actor.py` |
| PPO loss and group-relative advantages | `verl/trainer/core_algos.py` |
| Annotation model deployment | `scripts/deploy_span_annotator.sh` |
| Actual annotation prompt, parser, image serialization | `pacg/annotation.py`, `prompts/visual_claim_annotation.txt` |
| Fixed-response annotation | `python -m pacg.annotate` |
| Full-distribution EFS / corrected persistence | `python -m pacg.diagnose` |
| Question-level estimates and paired bootstrap changes | `python -m pacg.report` |
| Local verifiable-answer evaluation/reference generation | `python -m pacg.evaluate` |
| FSDP checkpoint conversion for inference/diagnosis | `scripts/model_merger.py` |

## Environment

Use Linux, Python 3.10–3.12, CUDA GPUs, and a compatible PyTorch/vLLM/FlashAttention
installation. Prefer the CUDA environment in which the original VPPO_RL training
already works; vLLM and PyTorch CUDA wheels must be mutually compatible. This
export does not introduce a newly validated CUDA/container version lock.

```bash
pip install -e .
# If not already present in the working CUDA environment:
pip install vllm 'flash-attn>=2.4.3' liger-kernel --no-build-isolation
```

`transformers` is constrained to the backend's existing supported range
4.54–4.57.0. `omegaconf` is pinned to 2.3.0. Install PyTorch using the wheel
matching the CUDA environment before installing training dependencies. Run from
the repository root. Do not install upstream `verl` over this vendored package.

CPU verification:

```bash
python -m unittest discover -s tests -v
bash -n scripts/train_qwen25vl7b.sh
bash -n scripts/deploy_span_annotator.sh
PARENT=dapo bash scripts/train_qwen25vl7b.sh --dry-run
```

CPU checks cover actual exported gate/actor code, annotation retries and failures,
all six launcher configurations, residual controls, full-KL directionality and
question aggregation. They do not substitute for a GPU/FSDP/vLLM training test.

## 1. Deploy a visual-claim annotator

Run on **separate GPUs or a separate host** from the training allocation. Supply a
local multimodal instruct model that supports OpenAI-style image inputs and JSON
output (for example the Qwen multimodal model used in your own training setup).
This script does not download model weights or contact model registries.

```bash
ANNOTATOR_MODEL_PATH=/models/your-multimodal-annotator \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP_SIZE=8 \
bash scripts/deploy_span_annotator.sh
```

The served name defaults to `annotator`, port 8000, context 32768, max concurrent
sequences 32. Override `TP_SIZE`, `PORT`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS` for your
hardware and model. The model must support the `enable_thinking=false` chat
template option used by the client. Optional authentication uses the
`ANNOTATOR_API_KEY` environment variable on both server and client.

The client ignores proxy environment variables by default. Training starts with
a real image+prompt smoke request before starting Ray workers. A single service
serving 3072 rollout annotations may require substantial time: tune concurrency
and `ANNOTATION_BATCH_BUDGET` to measured throughput. Individual failed samples
retain unit gates; the training aborts when the configured failure-rate or batch
time budget is exceeded. Authentication/configuration failures are not silently
treated as successful routing.

## 2. Train Qwen2.5-VL-7B

Prepare local JSONL/parquet files or directories containing them. Training rows:

```json
{"problem":"<image>\nHow many circles are visible?","answer":"3","images":["/data/images/example.png"]}
```

`problem` must contain the corresponding `<image>` placeholder. Other dataset
column names can be set with `data.prompt_key`, `data.answer_key`, `data.image_key`.
Use absolute image paths, embedded image objects supported by the dataset loader,
or `data.image_dir=/data/images`. Set the validation data separately. There is
no automatic OSS or evaluation-dataset download.

```bash
export MODEL_PATH=/models/Qwen2.5-VL-7B-Instruct
export TRAIN_DATA=/data/ViRL39K/train.parquet
export VAL_DATA=/data/MMK12/validation.parquet
export ANNOTATOR_URL=http://annotator-host:8000/v1/chat/completions
export ANNOTATOR_MODEL=annotator

# DAPO + PACG (default); launch within an already allocated CUDA node.
PARENT=dapo N_GPUS=8 bash scripts/train_qwen25vl7b.sh

# Other compositions; set different output/run names for separate runs.
PARENT=vppo RUN_NAME=vppo_pacg_seed42 bash scripts/train_qwen25vl7b.sh
PARENT=grpo RUN_NAME=grpo_pacg_seed42 bash scripts/train_qwen25vl7b.sh

# Matched parent only: same data and shared settings, no annotation service needed.
PARENT=dapo USE_PACG=false RUN_NAME=dapo_seed42 bash scripts/train_qwen25vl7b.sh
```

The public launcher defaults are explicit in `examples/configs/pacg_7b.yaml`:
margin 0.05, attenuation 10, floor 0.30, entropy coefficient 0.10, no reference
model or KL loss (KL coefficient 0), rollout group 8, batch 384, actor batch 128, response length
2048, learning rate 1e-6, seed 42, maximum 200 steps. GRPO uses sequence averaging,
symmetric clipping and no online filtering; DAPO uses token averaging, asymmetric
clipping and online filtering; VPPO additionally enables the inherited visual
selector and advantage shaping. These named configurations describe the exported
implementation, rather than asserting exact equality to every historical run.
By default the outcome reward is answer correctness only. `FORMAT_WEIGHT=0.1`
enables a 0.9 accuracy + 0.1 format reward; use the same setting for paired runs.
The inherited entropy penalty is the sampled-token negative log-probability
surrogate implemented by the backend, not full-vocabulary Shannon entropy.

Other overrides: `ENTROPY_COEF`, `PACG_MARGIN`, `PACG_ALPHA`, `PACG_FLOOR`, `SEED`,
`MAX_STEPS`, `ROLLOUT_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`, `UPDATE_MICRO_BATCH`,
`EXPERIENCE_MICRO_BATCH`, `SAVE_FREQ`, `VAL_FREQ`, `OUTPUT_DIR`,
`ANNOTATOR_CONCURRENCY`, and any trailing veRL dotlist argument. Reference KL is
disabled by default. To enable it explicitly, set all three:
`DISABLE_KL=false USE_KL_LOSS=true KL_COEF=0.01`.

Try two updates before committing to a full run:

```bash
MAX_STEPS=2 RUN_NAME=dapo_pacg_smoke bash scripts/train_qwen25vl7b.sh
```

The current online PACG router conservatively gates **single-image** rollouts;
multi-image/video trajectories keep parent credit. This inherited limitation is
explicit; the vLLM image limit does not imply multi-image claim routing support.
Factual `supported/unsupported` labels do not enter the training gate.

## 3. Checkpoints, logs and inference

Local output is `outputs/<RUN_NAME>/`: training log, TensorBoard events,
annotations, gates, rollouts, validation outputs and checkpoints. Checkpoints
are saved every 20 updates and at completion (latest three retained). No upload
is performed. Do not reuse a run directory unless resuming intentionally.
Resume with a trailing `trainer.load_checkpoint_path=/path/global_step_100`.

```bash
tensorboard --logdir outputs
python scripts/model_merger.py --local_dir outputs/dapo_pacg_7b_seed42/global_step_200/actor
```

The merger writes weights alongside metadata into `actor/huggingface/`. Use this
merged model directory for the standalone diagnostic/evaluation. A checkpoint's
`huggingface` directory contains **metadata only until the shards are merged**.
The merger only performs local conversion; its original upload functionality
has been removed. PACG has no inference-time routing or scoring components.

## 4. Accuracy evaluation and fixed Base reference

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for the complete annotation → fixed scoring
→ paired comparison workflow. `pacg.evaluate` uses local JSONL questions and
rule-based math answer verification; it reports mean accuracy, not pass@8.
It is intended for verifiable instances. It does not reproduce LLM-as-a-judge
benchmarks or provide unavailable raw benchmark datasets.

## Scope and provenance

Original Apache-2.0 notices are preserved. Shared backend utilities for other
architectures and disabled extensions remain where they are dependencies of the
working runtime. No SRPO training implementation or unrelated experiment
submission scripts are included. The runtime supports local checkpointing and
TensorBoard; neither requires a cloud account.

The standalone SPD pipeline implements the paper's equations and an explicit
fixed-reference protocol. Published table numbers are not hardcoded, and the
historical frozen responses, span annotations and exact image masks are not
bundled. Matching a published measurement requires those exact artifacts and
its aggregation/filtering protocol. EFS/persistence are responsiveness measures,
not factuality certificates; lower persistence is not proof of a false claim.

