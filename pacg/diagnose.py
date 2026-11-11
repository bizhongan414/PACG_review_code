"""Teacher-force a frozen response under original/degraded images; never regenerate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .annotate import align_exact
from .metrics import claim_metrics
from .reference import digest, messages, read_rows


def intervene(image, seed, patch_size=14, black_prob=0.5, operator="random_patch"):
    if patch_size < 1 or not 0 <= black_prob <= 1:
        raise ValueError("Invalid patch intervention parameters")
    if operator == "coarse_ink":
        from PIL import ImageFilter
        mask = image.convert("L").point(lambda value: 255 if value < 245 else 0)
        mask = mask.filter(ImageFilter.MaxFilter(9))
        result = image.copy()
        result.paste((0, 0, 0), mask=mask)
        return result
    array = np.array(image).copy()
    rng = np.random.RandomState(seed)
    for y in range(0, array.shape[0], patch_size):
        for x in range(0, array.shape[1], patch_size):
            if rng.rand() < black_prob:
                array[y:y + patch_size, x:x + patch_size] = 0
    return Image.fromarray(array)


def score_row(model, processor, row, args):
    import torch
    from verl.utils.dataset import process_image
    tokenizer = processor.tokenizer
    trace = row["response"]
    spans = align_exact(trace, row["spans"])
    encoded = tokenizer(trace, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if not ids or tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != trace:
        raise ValueError("Response must round-trip through the fixed reference tokenizer")
    if row.get("response_token_ids") is not None and row["response_token_ids"] != ids:
        raise ValueError("Reference token IDs differ; do not compare checkpoints with different tokenization")
    image = process_image(row["image"], args.min_pixels, args.max_pixels)
    seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{row["sample_id"]}'.encode()).digest()[:4], "big")
    cf = intervene(image, seed, args.patch_size, args.black_prob, args.operator)
    changed = float(np.any(np.asarray(image) != np.asarray(cf), axis=-1).mean())
    prompt = processor.apply_chat_template(messages(row["question"]), tokenize=False, add_generation_prompt=True)
    inputs = [processor(images=[im], text=[prompt], add_special_tokens=False, return_tensors="pt")
              for im in (image, cf)]
    if not torch.equal(inputs[0]["input_ids"], inputs[1]["input_ids"]) or not torch.equal(
            inputs[0]["image_grid_thw"], inputs[1]["image_grid_thw"]):
        raise ValueError("Intervention changed prompt IDs or visual geometry")
    prompt_length = inputs[0]["input_ids"].shape[1]
    if prompt_length + len(ids) > args.max_length:
        raise ValueError("Reference exceeds --max-length; refuse to silently truncate a claim")
    target = torch.tensor(ids, dtype=torch.long)
    logits = []
    with torch.inference_mode():
        for data in inputs:
            data["input_ids"] = torch.cat([data["input_ids"], target[None]], dim=1)
            data["attention_mask"] = torch.ones_like(data["input_ids"])
            data = {key: value.to(model.device) for key, value in data.items()}
            output = model(**data, use_cache=False)
            # Position immediately before each response target predicts that target.
            logits.append(output.logits[0, prompt_length - 1:prompt_length + len(ids) - 1].cpu())
            del output, data
        efs, delta = [], []
        for start in range(0, len(ids), args.chunk_tokens):
            orig = logits[0][start:start + args.chunk_tokens].float().log_softmax(-1)
            corrupt = logits[1][start:start + args.chunk_tokens].float().log_softmax(-1)
            tokens = target[start:start + args.chunk_tokens, None]
            efs.extend((orig.exp() * (orig - corrupt)).sum(-1).tolist())
            delta.extend((corrupt.gather(-1, tokens) - orig.gather(-1, tokens)).squeeze(-1).tolist())
    result = claim_metrics(efs, delta, spans, offsets, args.min_residual)
    # Invalid corruption is reported, never included in retractability aggregates.
    valid = args.min_changed_ratio <= changed <= args.max_changed_ratio
    if not valid:
        for claim in result["claims"]:
            claim["corrected_persistence"] = None
            claim["efs"] = None
    return {**result, "sample_id": row["sample_id"], "question_id": row["question_id"],
            "benchmark": row.get("benchmark", ""), "reference_sha256": digest(row),
            "response_token_ids": ids, "intervention_valid": valid, "changed_ratio": changed,
            "intervention_seed": seed, "checkpoint": args.model}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model directory/ID, not an FSDP shard directory")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--operator", choices=["random_patch", "coarse_ink"], default="random_patch")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--black-prob", type=float, default=0.5)
    parser.add_argument("--min-changed-ratio", type=float, default=0.005)
    parser.add_argument("--max-changed-ratio", type=float, default=0.95)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--min-residual", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    args = parser.parse_args()
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if args.chunk_tokens < 1:
        parser.error("--chunk-tokens must be positive")
    rows = read_rows(args.reference)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    if not processor.tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    with open(args.output, "x") as handle:
        for row in rows:
            result = score_row(model, processor, row, args)
            result["protocol"] = {key: value for key, value in vars(args).items()
                                  if key not in {"model", "output", "reference"}}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()

