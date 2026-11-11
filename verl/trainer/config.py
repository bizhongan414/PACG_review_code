# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        if self.image_dir is not None:
            if os.path.exists(self.image_dir):  # ray job uses absolute path
                self.image_dir = os.path.abspath(self.image_dir)
            else:
                print(f"Image directory {self.image_dir} not found.")
                self.image_dir = None

        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                print(f"Format prompt file {self.format_prompt} not found.")
                self.format_prompt = None


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""

    use_vppo_on_entropy: bool = False
    top_p_entropy_tokens: float = 0.2
    """use vppo based on entropy"""

    use_vppo_on_perception: bool = False
    top_p_perception_tokens: float = 0.2
    """use vppo based on perception"""

    use_entropy_penalty: bool = False
    entropy_penalty_coef: float = 0.06
    """use entropy penalty for training"""

    use_advantage_shaping: bool = False
    advantage_scaling_min: float = 0.8

    # Structured Role-Aware Policy Optimization, arXiv:2605.07274.

    # CFPO: maximize factual/counterfactual policy divergence.
    use_kl_cmve: bool = False
    kl_cmve_penalty: str = "low_var_kl"
    kl_cmve_coef: float = 2e-2
    use_cmve_entropy_loss: bool = False
    cmve_entropy_loss_coef: float = 3e-2
    use_ori_entropy_loss: bool = False
    ori_entropy_loss_coef: float = 3e-2

    # Counterfactual Persistence-Aware GRPO.  Support/evidence labels are not
    # used by the gate; only direct-VEF membership and signed retraction are.
    use_pa_grpo: bool = False
    # ``direct`` routes only extractor-identified direct visual claims.
    # ``all_tokens`` is a no-router ablation that skips the extractor and
    # applies the same persistence test to every valid response token.
    pa_routing_mode: str = "direct"
    # Hierarchical PA keeps the original span gate and adds a conservative
    # prompt-group-relative rollout weight.  The final direct-VEF multiplier
    # remains capped at one, so HPA only withholds positive credit.
    use_hpa_grpo: bool = False
    hpa_beta_rollout: float = 5.0
    hpa_rollout_lambda: float = 0.5
    hpa_rollout_weight_floor: float = 0.8
    hpa_rollout_weight_ceiling: float = 1.2
    hpa_final_gate_floor: float = 0.30
    hpa_final_gate_ceiling: float = 1.0
    # ``random_patch_blackening`` matches VPPO's 14x14 grid-patch
    # intervention. ``coarse_ink`` is retained as an ablation; ``cfpo`` reuses
    # the latent counterfactual forward and requires CFPO to be enabled.
    pa_counterfactual_provider: str = "random_patch_blackening"
    pa_patch_size: int = 14
    pa_black_prob: float = 0.5
    pa_gate_mode: str = "persistence"
    pa_gate_seed: int = 42
    pa_efs_threshold: float = 0.10
    pa_margin: float = 0.05
    pa_alpha: float = 10.0
    pa_gate_floor: float = 0.30
    pa_delta_clip: float = 20.0
    pa_residual_min_tokens: int = 8
    pa_vef_confidence_min: float = 0.0
    pa_min_token_match_ratio: float = 0.95
    pa_foreground_threshold: int = 245
    pa_ink_dilation: int = 3
    pa_min_changed_ratio: float = 0.005
    pa_max_changed_ratio: float = 0.95
    pa_annotator_script: str = "pacg/annotation.py"
    pa_annotator_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    pa_annotator_model: str = "auto"
    pa_annotator_api_key_env: str = "ANNOTATOR_API_KEY"
    pa_annotator_timeout: int = 120
    pa_annotator_connect_timeout: float = 10.0
    pa_annotator_request_budget: float = 240.0
    pa_annotation_batch_budget: float = 900.0
    pa_annotation_heartbeat: float = 30.0
    # Comma-separated equivalent replicas; also accepts PA_ANNOTATOR_URLS.
    pa_annotator_urls: str = ""
    pa_annotation_streaming: bool = True
    pa_rollout_chunk_size: int = 128
    pa_annotator_max_retries: int = 2
    pa_annotator_retry_backoff: float = 2.0
    pa_annotator_concurrency: int = 128
    pa_annotation_progress_every: int = 32
    pa_annotator_temperature: float = 0.8
    pa_annotator_top_k: int = 50
    pa_annotator_top_p: float = 0.95
    pa_annotator_max_trace_chars: int = 20000
    pa_annotator_trust_env: bool = False
    pa_max_annotation_failure_rate: float = 0.20
    pa_annotation_log_dir: Optional[str] = None
    pa_counterfactual_log_dir: Optional[str] = None
    pa_counterfactual_log_limit_per_step: int = 64
    pa_gate_log_dir: Optional[str] = None

    def post_init(self):
        if self.use_hpa_grpo or self.use_kl_cmve or self.pa_gate_mode != "persistence" or self.pa_routing_mode != "direct":
            raise ValueError("This release supports GRPO/DAPO/VPPO with optional direct-claim PACG only")
        if self.use_kl_cmve and self.kl_cmve_coef <= 0:
            raise ValueError("kl_cmve_coef must be positive when CFPO is enabled")
        if not self.use_pa_grpo:
            if self.use_hpa_grpo:
                raise ValueError("HPA-GRPO requires use_pa_grpo=true")
            return
        if self.pa_gate_mode not in {"persistence", "constant", "shuffled", "efs"}:
            raise ValueError("Unknown pa_gate_mode")
        if self.pa_gate_mode != "persistence" and (self.pa_routing_mode != "direct" or self.use_hpa_grpo):
            raise ValueError("Gate ablations require direct routing without HPA")
        if self.pa_efs_threshold < 0:
            raise ValueError("pa_efs_threshold must be nonnegative")
        if self.pa_routing_mode not in {"direct", "all_tokens"}:
            raise ValueError("pa_routing_mode must be direct or all_tokens")
        if self.pa_routing_mode == "all_tokens" and self.use_hpa_grpo:
            raise ValueError("pa_routing_mode=all_tokens does not support HPA-GRPO")
        if self.pa_counterfactual_provider not in {
            "random_patch", "random_patch_blackening", "coarse_ink", "cfpo"
        }:
            raise ValueError(
                "pa_counterfactual_provider must be random_patch, random_patch_blackening, "
                "coarse_ink, or cfpo"
            )
        if self.pa_counterfactual_provider == "cfpo" and not self.use_kl_cmve:
            raise ValueError("pa_counterfactual_provider=cfpo requires use_kl_cmve=true")
        if self.pa_patch_size < 1:
            raise ValueError("pa_patch_size must be positive")
        if not 0.0 <= self.pa_black_prob <= 1.0:
            raise ValueError("pa_black_prob must be in [0, 1]")
        if not 0.0 < self.pa_gate_floor <= 1.0:
            raise ValueError("pa_gate_floor must be in (0, 1]")
        if self.pa_alpha < 0.0 or self.pa_margin < 0.0:
            raise ValueError("pa_alpha and pa_margin must be non-negative")
        if self.pa_residual_min_tokens < 1:
            raise ValueError("pa_residual_min_tokens must be positive")
        if self.pa_annotator_max_retries < 0:
            raise ValueError("pa_annotator_max_retries must be non-negative")
        for name in ("pa_annotator_timeout", "pa_annotator_connect_timeout",
                     "pa_annotator_request_budget", "pa_annotation_batch_budget",
                     "pa_annotation_heartbeat"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pa_annotator_concurrency < 1:
            raise ValueError("pa_annotator_concurrency must be positive")
        if self.pa_rollout_chunk_size < 1:
            raise ValueError("pa_rollout_chunk_size must be positive")
        if self.pa_annotation_progress_every < 1:
            raise ValueError("pa_annotation_progress_every must be positive")
        if self.pa_counterfactual_log_limit_per_step < 0:
            raise ValueError("pa_counterfactual_log_limit_per_step must be non-negative")
        if not 0.0 <= self.pa_max_annotation_failure_rate <= 1.0:
            raise ValueError("pa_max_annotation_failure_rate must be in [0, 1]")
        if self.use_hpa_grpo:
            if self.hpa_beta_rollout < 0 or self.hpa_rollout_lambda < 0:
                raise ValueError("HPA beta and lambda must be non-negative")
            if not 0 < self.hpa_rollout_weight_floor <= self.hpa_rollout_weight_ceiling:
                raise ValueError("invalid HPA rollout weight bounds")
            if not 0 < self.hpa_final_gate_floor <= self.hpa_final_gate_ceiling <= 1:
                raise ValueError("HPA final gate bounds must satisfy 0 < floor <= ceiling <= 1")


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    val_data_dir: Optional[str] = None
    """directory to dump full validation rollouts as jsonl, one file per validation step"""
    rollout_data_dir: Optional[str] = None
    """directory to dump full training rollouts as jsonl, one file per training step"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""
    debug_event_log_path: Optional[str] = None
    """append durable JSONL stage events for post-mortem debugging"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # ray job uses absolute path
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):  # ray job uses absolute path
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None
        if self.debug_event_log_path is not None:
            self.debug_event_log_path = os.path.abspath(self.debug_event_log_path)


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

        # Use vppo based on entropy
        self.worker.actor.use_vppo_on_entropy = self.algorithm.use_vppo_on_entropy
        self.worker.actor.top_p_entropy_tokens = self.algorithm.top_p_entropy_tokens

        # Use vppo based on perception
        self.worker.actor.use_vppo_on_perception = self.algorithm.use_vppo_on_perception
        self.worker.actor.top_p_perception_tokens = self.algorithm.top_p_perception_tokens

        # Use entropy penalty for training
        self.worker.actor.use_entropy_penalty = self.algorithm.use_entropy_penalty
        self.worker.actor.entropy_penalty_coef = self.algorithm.entropy_penalty_coef

        self.worker.actor.use_advantage_shaping = self.algorithm.use_advantage_shaping
        self.worker.actor.advantage_scaling_min = self.algorithm.advantage_scaling_min
        self.worker.actor.use_pa_grpo = self.algorithm.use_pa_grpo

        self.worker.actor.use_kl_cmve = self.algorithm.use_kl_cmve
        self.worker.actor.kl_cmve_penalty = self.algorithm.kl_cmve_penalty
        self.worker.actor.kl_cmve_coef = self.algorithm.kl_cmve_coef
        self.worker.actor.use_cmve_entropy_loss = self.algorithm.use_cmve_entropy_loss
        self.worker.actor.cmve_entropy_loss_coef = self.algorithm.cmve_entropy_loss_coef
        self.worker.actor.use_ori_entropy_loss = self.algorithm.use_ori_entropy_loss
        self.worker.actor.ori_entropy_loss_coef = self.algorithm.ori_entropy_loss_coef
        if self.algorithm.use_kl_cmve and not self.worker.actor.model.apply_cmve:
            raise ValueError("CFPO requires worker.actor.model.apply_cmve=true")

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
