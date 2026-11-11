"""Validate local training inputs and annotation service before allocating Ray workers."""
import json
import os
import sys
from pathlib import Path

import requests
from omegaconf import OmegaConf


def main():
    from verl.trainer.config import PPOConfig
    cli = OmegaConf.from_cli(sys.argv[1:])
    filename = cli.pop("config")
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.load(filename), cli))
    config.deep_post_init()
    for name in (config.data.train_files, config.data.val_files):
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"Local dataset does not exist: {path}")
        if path.is_dir() and not any(p.suffix in {".parquet", ".json", ".jsonl"} for p in path.iterdir()):
            raise ValueError(f"No supported data files in {path}")
    import torch
    if torch.cuda.device_count() < config.trainer.n_gpus_per_node:
        raise RuntimeError("Not enough visible CUDA GPUs for trainer.n_gpus_per_node")
    if config.algorithm.use_pa_grpo:
        from .annotation import build_prompt, image_to_data_url, parse_json_content
        from PIL import Image
        import tempfile
        session = requests.Session()
        session.trust_env = config.algorithm.pa_annotator_trust_env
        headers = {}
        key = os.getenv(config.algorithm.pa_annotator_api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "smoke.png"
            Image.new("RGB", (112, 112), "white").save(image)
            payload = {"model": config.algorithm.pa_annotator_model, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": [
                           {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                           {"type": "text", "text": build_prompt(
                               {"sample_id": "smoke", "question": "Describe the image."},
                               "The image has a white background.", 20000)}]}]}
            response = session.post(config.algorithm.pa_annotator_url, headers=headers, json=payload,
                                    timeout=(10, config.algorithm.pa_annotator_timeout))
            if not response.ok:
                raise RuntimeError(f"Annotation smoke request returned HTTP {response.status_code}")
            parsed = parse_json_content(response.json()["choices"][0]["message"]["content"])
            if not isinstance(parsed.get("spans"), list):
                raise ValueError("Annotation response must contain a spans list")
    print(json.dumps({"preflight": "passed", "GPUs": config.trainer.n_gpus_per_node,
                      "PACG": config.algorithm.use_pa_grpo}))


if __name__ == "__main__":
    main()

