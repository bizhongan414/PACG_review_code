"""Routing and persistence-aware credit gates for PA-GRPO.

In the default mode, the annotator runs on the trainer driver as filtered
rollout microbatches become available and never sends factual-support labels into the policy
update.  The ``all_tokens`` ablation bypasses the annotator entirely. Failed,
unaligned, or invalid rollouts receive an all-one multiplier (base-optimizer
fallback).
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import requests
import torch
from PIL import Image, ImageChops, ImageFilter, ImageOps

from ..protocol import DataProto
from ..utils.dataset import process_image
from ..workers.perc_utils import random_patch_blackening


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
DIRECT_GROUP = "direct_vef"
VISUAL_GROUPS = {"direct_vef", "derived_vef"}
VISUAL_DEPENDENCIES = {"direct", "indirect"}
IMAGE_COUNTERFACTUAL_PROVIDERS = {
    "random_patch",
    "random_patch_blackening",
    "coarse_ink",
}
PA_ANNOTATION_TENSOR_KEYS = (
    "pa_direct_token_mask",
    "pa_visual_token_mask",
    "pa_response_text_token_mask",
    "pa_intervention_valid",
    "pa_changed_area_ratio",
)
PA_ANNOTATION_NON_TENSOR_KEYS = (
    "pa_direct_spans",
    "pa_counterfactual_multi_modal_data",
)


class AnnotationServiceError(RuntimeError):
    """Non-recoverable service configuration error; do not train ungated."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_annotation_helpers(script_path: str):
    path = Path(script_path)
    if not path.is_absolute():
        path = _repo_root() / path
    spec = importlib.util.spec_from_file_location("pa_qwen3_span_annotator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load PA span annotator helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The requested smoke script delegates prompt/parser construction here.
    helpers = module.load_helpers()
    return helpers, path


def _alpha_composite_white(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    _alpha_composite_white(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_hash(image: Image.Image) -> str:
    buffer = io.BytesIO()
    _alpha_composite_white(image).save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def make_coarse_ink_counterfactual(
    image: Image.Image,
    *,
    threshold: int,
    dilation: int,
) -> Image.Image:
    """Blacken coarse foreground/ink pixels, matching Phase-1 targeted ink."""
    rgb = _alpha_composite_white(image)
    gray = ImageOps.grayscale(rgb)
    mask = gray.point(lambda value: 255 if value < threshold else 0).convert("L")
    if dilation > 1:
        size = dilation if dilation % 2 == 1 else dilation + 1
        mask = mask.filter(ImageFilter.MaxFilter(size=size))
    black = Image.new("RGB", rgb.size, (0, 0, 0))
    return Image.composite(black, rgb, mask)


def changed_pixel_ratio(original: Image.Image, counterfactual: Image.Image) -> float:
    difference = ImageChops.difference(
        _alpha_composite_white(original), counterfactual.convert("RGB")
    )
    changed = difference.convert("L").point(lambda value: 1 if value else 0)
    changed_count = sum(changed.histogram()[1:])
    width, height = original.size
    return changed_count / max(width * height, 1)


def _strip_thinking(content: str) -> str:
    content = content.strip()
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    return content


def _validate_annotation_payload(parsed: Any) -> dict[str, Any]:
    """Validate the annotator's required JSON structure.

    Structural errors are deliberately raised from ``_request_once`` so the
    existing request retry loop treats them exactly like transport/model
    failures.  An exhausted retry budget is then counted in the rollout-level
    annotation failure rate instead of being silently interpreted as an empty
    span list.
    """
    if not isinstance(parsed, dict):
        raise ValueError(
            "annotator output must be a top-level JSON object, "
            f"got {type(parsed).__name__}"
        )
    if "spans" not in parsed:
        raise ValueError('annotator JSON object must explicitly contain "spans"')
    spans = parsed["spans"]
    if not isinstance(spans, list):
        raise ValueError(
            'annotator JSON field "spans" must be a list, '
            f"got {type(spans).__name__}"
        )
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            raise ValueError(
                f'annotator JSON field "spans[{index}]" must be an object, '
                f"got {type(span).__name__}"
            )
    return parsed


def _span_is_visual(span: Mapping[str, Any]) -> bool:
    return bool(
        span.get("is_vef")
        or span.get("span_group") in VISUAL_GROUPS
        or span.get("visual_dependency") in VISUAL_DEPENDENCIES
    )


def _find_exact_spans(trace: str, spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    cursor = 0
    for span_id, raw in enumerate(spans):
        # Keep alignment best-effort even if a cached or externally supplied
        # annotation contains one malformed span.
        if not isinstance(raw, Mapping):
            continue
        try:
            span = dict(raw)
            raw_text = span.get("text")
            if not isinstance(raw_text, str):
                continue
            text = raw_text
        except (TypeError, ValueError):
            continue
        if not text.strip():
            continue
        start = trace.find(text, cursor)
        method = "cursor_exact"
        if start < 0:
            start = trace.find(text)
            method = "global_exact"
        if start < 0:
            continue
        end = start + len(text)
        cursor = max(cursor, end)
        span.update(
            span_id=span_id,
            char_start=start,
            char_end=end,
            alignment_method=method,
        )
        aligned.append(span)
    return aligned


def _tokenize_trace(tokenizer, trace: str) -> tuple[list[int], list[tuple[int, int]]]:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("PA-GRPO requires a fast tokenizer for strict span alignment")
    encoded = tokenizer(trace, add_special_tokens=False, return_offsets_mapping=True)
    return (
        [int(token_id) for token_id in encoded["input_ids"]],
        [(int(start), int(end)) for start, end in encoded["offset_mapping"]],
    )


def _trace_to_response_map(
    trace_ids: Sequence[int], response_ids: Sequence[int]
) -> tuple[dict[int, int], float]:
    trace_ids = list(map(int, trace_ids))
    response_ids = list(map(int, response_ids))
    if trace_ids == response_ids:
        return {index: index for index in range(len(trace_ids))}, 1.0
    matcher = SequenceMatcher(a=trace_ids, b=response_ids, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    return mapping, len(mapping) / max(len(trace_ids), 1)


def _align_spans_to_response_tokens(
    *,
    tokenizer,
    trace: str,
    response_ids: Sequence[int],
    spans: Sequence[Mapping[str, Any]],
    minimum_match_ratio: float,
) -> tuple[list[dict[str, Any]], list[int], float]:
    trace_ids, offsets = _tokenize_trace(tokenizer, trace)
    mapping, match_ratio = _trace_to_response_map(trace_ids, response_ids)
    if match_ratio < minimum_match_ratio:
        raise ValueError(
            f"trace/response token match {match_ratio:.4f} < {minimum_match_ratio:.4f}"
        )

    aligned: list[dict[str, Any]] = []
    for raw in spans:
        if not isinstance(raw, Mapping):
            continue
        try:
            start, end = int(raw["char_start"]), int(raw["char_end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if start < 0 or end <= start or end > len(trace):
            continue
        trace_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        mapped = [mapping[index] for index in trace_indices if index in mapping]
        if not trace_indices or len(mapped) != len(trace_indices):
            continue
        mapped = sorted(mapped)
        if mapped != list(range(mapped[0], mapped[-1] + 1)):
            continue
        try:
            span = dict(raw)
        except (TypeError, ValueError):
            continue
        span["token_indices"] = mapped
        span["token_start"] = mapped[0]
        span["token_end"] = mapped[-1] + 1
        aligned.append(span)

    response_text_indices = sorted(set(mapping.values()))
    return aligned, response_text_indices, match_ratio


class OnlineVEFAnnotator:
    """Concurrent OpenAI-compatible Qwen VEF annotation client."""

    def __init__(self, config: Any):
        self.config = config
        self.helpers, self.script_path = _load_annotation_helpers(config.pa_annotator_script)
        self._thread_local = threading.local()
        urls = getattr(config, "pa_annotator_urls", "") or os.environ.get("PA_ANNOTATOR_URLS", "")
        self.urls = [url.strip() for url in urls.split(",") if url.strip()] or [config.pa_annotator_url]
        self._lock = threading.Lock()
        self._next_endpoint = 0
        self.stop = threading.Event()
        self.events: list[dict[str, Any]] = []

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = bool(self.config.pa_annotator_trust_env)
            self._thread_local.session = session
        return session

    def _request_once(self, *, prompt: str, image: Image.Image, endpoint: str,
                      remaining: float) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.pa_annotator_api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.config.pa_annotator_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": self.config.pa_annotator_temperature,
            "top_k": self.config.pa_annotator_top_k,
            "top_p": self.config.pa_annotator_top_p,
            "n": 1,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self._session().post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=(min(getattr(self.config, "pa_annotator_connect_timeout", 10.0), remaining),
                     min(self.config.pa_annotator_timeout, remaining)),
        )
        if response.status_code in RETRYABLE_STATUS:
            raise requests.HTTPError(
                f"retryable HTTP {response.status_code}",
                response=response,
            )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        content = _strip_thinking(str(choice["message"].get("content") or ""))
        if not content:
            raise ValueError("annotator returned empty content")
        parsed = _validate_annotation_payload(
            self.helpers.parse_json_content(content)
        )
        return {
            "parsed": parsed,
            "usage": body.get("usage"),
            "finish_reason": choice.get("finish_reason"),
        }

    def annotate_one(
        self,
        *,
        rollout_id: str,
        question: str,
        trace: str,
        image: Image.Image,
        queued_at: float | None = None,
    ) -> dict[str, Any]:
        row = {"sample_id": rollout_id, "question": question}
        prompt = self.helpers.build_prompt(row, trace, self.config.pa_annotator_max_trace_chars)
        last_error: Exception | None = None
        started_at = time.perf_counter()
        attempt = 0
        retries_used = 0
        forbidden_retries_used = 0
        request_budget = getattr(self.config, "pa_annotator_request_budget", 240.0)
        while True:
            remaining = request_budget - (time.perf_counter() - started_at)
            if self.stop.is_set() or remaining <= 0:
                break
            attempt += 1
            with self._lock:
                endpoint_index = self._next_endpoint % len(self.urls)
                self._next_endpoint += 1
            attempt_start = time.perf_counter()
            event = {"rollout_id": rollout_id, "attempt": attempt,
                     "endpoint_index": endpoint_index, "http_status": 0,
                     "error_type": "", "success": False,
                     "queue_wait_s": max(0.0, started_at - queued_at) if queued_at is not None else 0.0}
            try:
                result = self._request_once(prompt=prompt, image=image,
                                            endpoint=self.urls[endpoint_index], remaining=remaining)
                result["attempts"] = attempt
                result["latency_seconds"] = time.perf_counter() - started_at
                event["success"] = True
                return result
            except Exception as error:
                # Never include response bodies, URLs, or credentials in diagnostics.
                status = getattr(getattr(error, "response", None), "status_code", 0) or 0
                event.update(http_status=status, error_type=type(error).__name__)
                last_error = f"{type(error).__name__} (HTTP {status})"
                if status in {401, 404}:
                    self.stop.set()
                    raise AnnotationServiceError(
                        f"Annotation replica {endpoint_index} returned HTTP {status}; check credentials/model/endpoint"
                    ) from None
                retryable = (status == 403 or status in RETRYABLE_STATUS if status else
                             isinstance(error, (requests.Timeout, requests.ConnectionError,
                                                ValueError, KeyError, IndexError)))
                if not retryable:
                    break
                # A transient gateway 403 gets three additional attempts. If
                # exhausted, fail only this sample; collect() records the error
                # and the existing unit-gate fallback/failure-rate guard applies.
                if status == 403:
                    if forbidden_retries_used >= 3:
                        break
                    forbidden_retries_used += 1
                else:
                    if retries_used >= self.config.pa_annotator_max_retries:
                        break
                    retries_used += 1
                if retryable:
                    delay = min(self.config.pa_annotator_retry_backoff ** (attempt - 1), 8.0)
                    retry_after = getattr(getattr(error, "response", None), "headers", {}).get("Retry-After")
                    try:
                        delay = max(delay, float(retry_after)) if retry_after else delay
                    except (TypeError, ValueError):
                        pass
                    remaining = request_budget - (time.perf_counter() - started_at)
                    if remaining <= delay:
                        break
                    self.stop.wait(min(delay + random.uniform(0, 0.5), remaining))
            finally:
                event["elapsed_s"] = time.perf_counter() - attempt_start
                with self._lock:
                    self.events.append(event)
        raise RuntimeError(
            f"span annotation stopped after {time.perf_counter() - started_at:.1f}s: "
            f"{last_error or 'cancelled or request budget exhausted'}"
        ) from None


class PAAnnotationPipeline:
    """Controller-only futures in generation order; never sent to GPU workers.

    Submit only retained rollouts, before subsequent generation microbatches.
    One shared executor caps concurrency across all microbatches and replicas.
    """

    def __init__(self, config, tokenizer, global_step):
        self.config, self.tokenizer, self.global_step = config, tokenizer, global_step
        self.annotator = OnlineVEFAnnotator(config)
        self.executor = ThreadPoolExecutor(max_workers=config.pa_annotator_concurrency)
        self.jobs = []
        self.fingerprints = []
        self.started = None
        self.closed = False

    def enqueue(self, batch):
        if self.started is None:
            self.started = time.monotonic()
        for index in range(len(batch)):
            ids = batch.batch["responses"][index][batch.batch["response_mask"][index].bool()].tolist()
            trace = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            uid = str(batch.non_tensor_batch["uid"][index])
            self.fingerprints.append((uid, hashlib.sha256(trace.encode()).hexdigest()))
            image, _ = _first_original_image(batch.non_tensor_batch["multi_modal_data"][index])
            if image is None or not trace:
                self.jobs.append(None)
                continue
            self.jobs.append(self.executor.submit(
                self.annotator.annotate_one,
                rollout_id=f"step{self.global_step}:{uid}:{len(self.jobs)}",
                question=str(batch.non_tensor_batch["question"][index]), trace=trace, image=image,
                queued_at=time.perf_counter(),
            ))

    def collect(self, traces, uids):
        expected = [(str(uid), hashlib.sha256(trace.encode()).hexdigest()) for uid, trace in zip(uids, traces)]
        if expected != self.fingerprints:
            raise RuntimeError("PA streaming annotation order/content mismatch; refusing incorrect gates")
        results = [None] * len(self.jobs)
        errors = {i: "missing image or empty response" for i, f in enumerate(self.jobs) if f is None}
        pending = {f: i for i, f in enumerate(self.jobs) if f is not None}
        attempted = len(pending)
        heartbeat = getattr(self.config, "pa_annotation_heartbeat", 30.0)
        budget = getattr(self.config, "pa_annotation_batch_budget", 900.0)
        # Budget limits the collection/wait stage, not preceding useful GPU work.
        deadline = time.monotonic() + budget
        while pending:
            # Consume completed requests first, even if GPU generation took longer
            # than the annotation budget. Completed work is still usable.
            done = {f for f in pending if f.done()}
            remaining = deadline - time.monotonic()
            if not done:
                if remaining <= 0:
                    raise TimeoutError(f"PA annotation batch budget {budget}s exceeded; "
                                       f"pending={len(pending)}/{attempted}. Training aborted, not silently ungated.")
                done, _ = wait(pending, timeout=min(heartbeat, remaining), return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    results[index] = future.result()
                except AnnotationServiceError:
                    raise
                except Exception as error:
                    errors[index] = str(error)
            now = time.monotonic()
            if not pending or now - getattr(self, "_last_progress", self.started) >= heartbeat:
                self._last_progress = now
                with self.annotator._lock:
                    events = list(self.annotator.events)
                print(f"[PA annotation] step={self.global_step} completed={attempted-len(pending)}/{attempted} "
                      f"pending={len(pending)} failed={len(errors)} elapsed_s={now-self.started:.1f} "
                      f"retries={sum(e['attempt'] > 1 for e in events)} "
                      f"http429={sum(e['http_status'] == 429 for e in events)}", flush=True)
        return results, errors, attempted

    def diagnostics(self):
        with self.annotator._lock:
            events = list(self.annotator.events)
        latencies = [e["elapsed_s"] for e in events]
        queues = [e["queue_wait_s"] for e in events if e["attempt"] == 1]
        return {
            "pa/request_attempt_count": float(len(events)),
            "pa/request_retry_count": float(sum(e["attempt"] > 1 for e in events)),
            "pa/request_http429_count": float(sum(e["http_status"] == 429 for e in events)),
            "pa/request_http403_count": float(sum(e["http_status"] == 403 for e in events)),
            "pa/request_http5xx_count": float(sum(e["http_status"] >= 500 for e in events)),
            "pa/request_timeout_count": float(sum("Timeout" in e["error_type"] for e in events)),
            "pa/request_invalid_payload_count": float(sum(e["error_type"] in {"ValueError", "KeyError", "IndexError", "JSONDecodeError"} for e in events)),
            "pa/request_queue_s_p95": float(np.quantile(queues, .95)) if queues else 0.0,
            "pa/request_attempt_s_p95": float(np.quantile(latencies, .95)) if latencies else 0.0,
            "pa/request_attempt_s_max": float(max(latencies, default=0.0)),
            "pa/annotation_endpoint_count": float(len(self.annotator.urls)),
            "pa/annotation_pipeline_s": time.monotonic() - self.started if self.started else 0.0,
        }

    def close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        if abort:
            self.annotator.stop.set()
        self.executor.shutdown(wait=not abort, cancel_futures=abort)
        diagnostics = self.diagnostics()
        print(f"[PA annotation diagnostics] step={self.global_step} aborted={abort} "
              f"{json.dumps(diagnostics)}", flush=True)
        if self.config.pa_annotation_log_dir:
            log_dir = Path(self.config.pa_annotation_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            with self.annotator._lock:
                events = list(self.annotator.events)
            with (log_dir / f"step_{self.global_step}_requests.jsonl").open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")
            with (log_dir / f"step_{self.global_step}_request_summary.json").open("w", encoding="utf-8") as handle:
                json.dump({"aborted": abort, "unfinished": sum(f is not None and not f.done() for f in self.jobs),
                           "metrics": diagnostics}, handle, indent=2)


def _first_original_image(multi_modal_data: Any) -> tuple[Image.Image | None, int]:
    if not isinstance(multi_modal_data, Mapping):
        return None, 0
    images = multi_modal_data.get("images") or []
    if not images:
        return None, 0
    return process_image(images[0], min_pixels=None, max_pixels=None), len(images)


def _prepare_all_token_pa_batch(
    *,
    batch: DataProto,
    config: Any,
    global_step: int,
    images: list[Image.Image | None],
    image_counts: list[int],
    multi_modal: Sequence[Any],
    uids: Sequence[Any],
) -> dict[str, float]:
    """Build counterfactuals without invoking the direct-claim extractor."""
    response_mask = batch.batch["response_mask"].bool()
    batch_size, _ = response_mask.shape
    direct_mask = response_mask.clone()
    visual_mask = torch.zeros_like(direct_mask)
    text_token_mask = response_mask.clone()
    intervention_valid = torch.zeros(batch_size, dtype=torch.bool)
    changed_ratios = torch.zeros(batch_size, dtype=torch.float32)
    counterfactual_data: list[dict[str, Any]] = []
    span_records: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    records: list[dict[str, Any]] = []
    saved_counterfactuals = 0

    for index in range(batch_size):
        original_mm = multi_modal[index]
        fallback_mm = dict(original_mm) if isinstance(original_mm, Mapping) else {}
        image = images[index]
        cf_mm = dict(fallback_mm)
        ratio = 0.0
        valid = False
        cf_image = image
        if image is not None and config.pa_counterfactual_provider in IMAGE_COUNTERFACTUAL_PROVIDERS:
            if config.pa_counterfactual_provider in {"random_patch", "random_patch_blackening"}:
                cf_image = random_patch_blackening(
                    image,
                    patch_size=config.pa_patch_size,
                    black_prob=config.pa_black_prob,
                )
            else:
                cf_image = make_coarse_ink_counterfactual(
                    image,
                    threshold=config.pa_foreground_threshold,
                    dilation=config.pa_ink_dilation,
                )
            ratio = changed_pixel_ratio(image, cf_image)
            valid = (
                image_counts[index] == 1
                and bool(response_mask[index].any())
                and config.pa_min_changed_ratio <= ratio <= config.pa_max_changed_ratio
            )
            if image_counts[index] == 1:
                cf_mm["images"] = [cf_image]
        elif image is not None and config.pa_counterfactual_provider == "cfpo":
            valid = image_counts[index] == 1 and bool(response_mask[index].any())

        changed_ratios[index] = ratio
        intervention_valid[index] = valid
        counterfactual_data.append(cf_mm)
        record = {
            "step": global_step,
            "rollout_index": index,
            "uid": str(uids[index]),
            "status": "ok" if valid else "invalid_intervention",
            "routing_mode": "all_tokens",
            "extractor_invoked": False,
            "image_count": image_counts[index],
            "response_token_count": int(response_mask[index].sum().item()),
            "changed_area_ratio": ratio,
            "intervention_valid": valid,
            "counterfactual_provider": config.pa_counterfactual_provider,
        }
        if (
            config.pa_counterfactual_log_dir
            and valid
            and cf_image is not None
            and config.pa_counterfactual_provider in IMAGE_COUNTERFACTUAL_PROVIDERS
            and saved_counterfactuals < config.pa_counterfactual_log_limit_per_step
        ):
            cf_dir = Path(config.pa_counterfactual_log_dir) / f"step_{global_step}"
            cf_dir.mkdir(parents=True, exist_ok=True)
            safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(uids[index]))[:80]
            cf_path = cf_dir / f"{index:05d}_{safe_uid}.png"
            cf_image.save(cf_path)
            record["counterfactual_image_path"] = str(cf_path)
            saved_counterfactuals += 1
        records.append(record)

    batch.batch["pa_direct_token_mask"] = direct_mask
    batch.batch["pa_visual_token_mask"] = visual_mask
    batch.batch["pa_response_text_token_mask"] = text_token_mask
    batch.batch["pa_intervention_valid"] = intervention_valid
    batch.batch["pa_changed_area_ratio"] = changed_ratios
    span_record_array = np.empty(batch_size, dtype=object)
    span_record_array[:] = span_records
    counterfactual_array = np.empty(batch_size, dtype=object)
    counterfactual_array[:] = counterfactual_data
    batch.non_tensor_batch["pa_direct_spans"] = span_record_array
    batch.non_tensor_batch["pa_counterfactual_multi_modal_data"] = counterfactual_array

    if config.pa_annotation_log_dir:
        log_dir = Path(config.pa_annotation_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        output = log_dir / f"step_{global_step}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return {
        "pa/routing_mode_all_tokens": 1.0,
        "pa/extractor_invoked": 0.0,
        "pa/annotation_attempted": 0.0,
        "pa/annotation_failed": 0.0,
        "pa/annotation_failure_rate": 0.0,
        "pa/annotation_success_rate": 1.0,
        "pa/direct_span_count": 0.0,
        "pa/no_direct_rollout_rate": 0.0,
        "pa/rollout_with_direct_rate": float(direct_mask.any(dim=1).float().mean().item()),
        "pa/intervention_valid_rate": float(intervention_valid.float().mean().item()),
        "pa/changed_area_ratio_mean": float(changed_ratios.mean().item()),
        "pa/counterfactual_images_saved": float(saved_counterfactuals),
    }


def prepare_pa_grpo_batch(
    *,
    batch: DataProto,
    tokenizer,
    config: Any,
    global_step: int,
    annotation_pipeline: PAAnnotationPipeline | None = None,
) -> dict[str, float]:
    """Annotate every rollout and attach masks/counterfactuals to ``batch``."""
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"].bool()
    batch_size, response_length = responses.shape
    questions = batch.non_tensor_batch.get("question", np.array([""] * batch_size, dtype=object))
    multi_modal = batch.non_tensor_batch.get(
        "multi_modal_data", np.array([None] * batch_size, dtype=object)
    )
    uids = batch.non_tensor_batch.get("uid", np.array([""] * batch_size, dtype=object))

    traces: list[str] = []
    valid_response_ids: list[list[int]] = []
    images: list[Image.Image | None] = []
    image_counts: list[int] = []
    for index in range(batch_size):
        token_ids = responses[index][response_mask[index]].tolist()
        valid_response_ids.append([int(value) for value in token_ids])
        traces.append(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        image, image_count = _first_original_image(multi_modal[index])
        images.append(image)
        image_counts.append(image_count)

    if config.pa_routing_mode == "all_tokens":
        return _prepare_all_token_pa_batch(
            batch=batch,
            config=config,
            global_step=global_step,
            images=images,
            image_counts=image_counts,
            multi_modal=multi_modal,
            uids=uids,
        )

    pipeline = annotation_pipeline or PAAnnotationPipeline(config, tokenizer, global_step)
    try:
        if annotation_pipeline is None:
            pipeline.enqueue(batch)
        results, errors, attempted = pipeline.collect(traces, uids)
    except BaseException:
        pipeline.close(abort=True)
        raise
    else:
        pipeline.close()
    request_metrics = pipeline.diagnostics()
    failure_rate = len(errors) / max(batch_size, 1)
    if failure_rate > config.pa_max_annotation_failure_rate:
        examples = "; ".join(f"{idx}: {message[:160]}" for idx, message in list(errors.items())[:3])
        raise RuntimeError(
            f"PA annotation failure rate {failure_rate:.3f} exceeds "
            f"{config.pa_max_annotation_failure_rate:.3f}. Examples: {examples}"
        )

    direct_mask = torch.zeros((batch_size, response_length), dtype=torch.bool)
    visual_mask = torch.zeros_like(direct_mask)
    text_token_mask = torch.zeros_like(direct_mask)
    intervention_valid = torch.zeros(batch_size, dtype=torch.bool)
    changed_ratios = torch.zeros(batch_size, dtype=torch.float32)
    span_records: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    counterfactual_data: list[dict[str, Any]] = []
    annotation_records: list[dict[str, Any]] = []
    aligned_direct_count = 0
    aligned_rollouts = 0
    no_direct_rollouts = 0
    annotation_attempt_counts: list[int] = []
    annotation_latencies: list[float] = []
    saved_counterfactuals = 0

    for index in range(batch_size):
        original_mm = multi_modal[index]
        fallback_mm = dict(original_mm) if isinstance(original_mm, Mapping) else {}
        image = images[index]
        result = results[index]
        record: dict[str, Any] = {
            "step": global_step,
            "rollout_index": index,
            "uid": str(uids[index]),
            "status": "error" if result is None else "ok",
            "error": errors.get(index),
            "question": str(questions[index]),
            "trace": traces[index],
            "image_count": image_counts[index],
            "annotator_model": config.pa_annotator_model,
        }
        if result is None or image is None:
            counterfactual_data.append(fallback_mm)
            annotation_records.append(record)
            continue

        exact_spans: list[dict[str, Any]] = []
        try:
            exact_spans = _find_exact_spans(
                traces[index], result["parsed"]["spans"]
            )
            aligned, response_text_indices, match_ratio = _align_spans_to_response_tokens(
                tokenizer=tokenizer,
                trace=traces[index],
                response_ids=valid_response_ids[index],
                spans=exact_spans,
                minimum_match_ratio=config.pa_min_token_match_ratio,
            )
        except Exception as error:
            record.update(
                status="alignment_error",
                error=str(error),
                spans=exact_spans,
            )
            counterfactual_data.append(fallback_mm)
            annotation_records.append(record)
            continue
        aligned_rollouts += 1
        annotation_attempt_counts.append(int(result["attempts"]))
        annotation_latencies.append(float(result["latency_seconds"]))

        for token_index in response_text_indices:
            if token_index < response_length:
                text_token_mask[index, token_index] = True

        direct_spans = []
        for span in aligned:
            if not isinstance(span, Mapping):
                continue
            try:
                indices = [
                    int(token)
                    for token in span.get("token_indices", [])
                    if 0 <= int(token) < response_length
                ]
            except (TypeError, ValueError, OverflowError):
                continue
            if not indices:
                continue
            if _span_is_visual(span):
                visual_mask[index, indices] = True
            try:
                confidence = float(span.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                # A malformed confidence invalidates only this span, not the
                # remaining aligned spans or the full training batch.
                continue
            if (
                span.get("span_group") == DIRECT_GROUP
                and confidence >= config.pa_vef_confidence_min
            ):
                direct_mask[index, indices] = True
                direct_spans.append(span)
                aligned_direct_count += 1

        # The requested annotator accepts one image.  Multi-image/video rollouts
        # are annotated for diagnostics but conservatively fall back to GRPO.
        if config.pa_counterfactual_provider in IMAGE_COUNTERFACTUAL_PROVIDERS:
            if config.pa_counterfactual_provider in {"random_patch", "random_patch_blackening"}:
                # Intentionally call the same implementation and defaults as
                # VPPO, while exposing them in AlgorithmConfig for ablation.
                cf_image = random_patch_blackening(
                    image,
                    patch_size=config.pa_patch_size,
                    black_prob=config.pa_black_prob,
                )
            else:
                cf_image = make_coarse_ink_counterfactual(
                    image,
                    threshold=config.pa_foreground_threshold,
                    dilation=config.pa_ink_dilation,
                )
            ratio = changed_pixel_ratio(image, cf_image)
            technically_valid = (
                image_counts[index] == 1
                and bool(direct_spans)
                and config.pa_min_changed_ratio <= ratio <= config.pa_max_changed_ratio
            )
        else:
            # CFPO performs a latent, attention-guided intervention inside the
            # actor. No second image is attached to this batch.
            cf_image = image
            ratio = 0.0
            technically_valid = image_counts[index] == 1 and bool(direct_spans)
        changed_ratios[index] = ratio
        if not direct_spans:
            no_direct_rollouts += 1
        intervention_valid[index] = technically_valid
        cf_mm = dict(fallback_mm)
        if (
            image_counts[index] == 1
            and config.pa_counterfactual_provider in IMAGE_COUNTERFACTUAL_PROVIDERS
        ):
            cf_mm["images"] = [cf_image]
        counterfactual_data.append(cf_mm)
        span_records[index] = direct_spans
        record.update(
            spans=aligned,
            direct_spans=direct_spans,
            token_match_ratio=match_ratio,
            response_text_token_count=len(response_text_indices),
            changed_area_ratio=ratio,
            intervention_valid=technically_valid,
            counterfactual_provider=config.pa_counterfactual_provider,
            counterfactual_patch_size=(
                config.pa_patch_size
                if config.pa_counterfactual_provider in {"random_patch", "random_patch_blackening"}
                else None
            ),
            counterfactual_black_prob=(
                config.pa_black_prob
                if config.pa_counterfactual_provider in {"random_patch", "random_patch_blackening"}
                else None
            ),
            original_image_sha256=_image_hash(image),
            counterfactual_image_sha256=_image_hash(cf_image),
            attempts=result["attempts"],
            latency_seconds=result["latency_seconds"],
            usage=result.get("usage"),
            finish_reason=result.get("finish_reason"),
        )
        if (
            config.pa_counterfactual_log_dir
            and technically_valid
            and config.pa_counterfactual_provider in IMAGE_COUNTERFACTUAL_PROVIDERS
            and saved_counterfactuals < config.pa_counterfactual_log_limit_per_step
        ):
            cf_dir = Path(config.pa_counterfactual_log_dir) / f"step_{global_step}"
            cf_dir.mkdir(parents=True, exist_ok=True)
            safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(uids[index]))[:80]
            cf_path = cf_dir / f"{index:05d}_{safe_uid}.png"
            cf_image.save(cf_path)
            record["counterfactual_image_path"] = str(cf_path)
            saved_counterfactuals += 1
        annotation_records.append(record)

    batch.batch["pa_direct_token_mask"] = direct_mask
    batch.batch["pa_visual_token_mask"] = visual_mask
    batch.batch["pa_response_text_token_mask"] = text_token_mask
    batch.batch["pa_intervention_valid"] = intervention_valid
    batch.batch["pa_changed_area_ratio"] = changed_ratios
    span_record_array = np.empty(batch_size, dtype=object)
    span_record_array[:] = span_records
    counterfactual_array = np.empty(batch_size, dtype=object)
    counterfactual_array[:] = counterfactual_data
    batch.non_tensor_batch["pa_direct_spans"] = span_record_array
    batch.non_tensor_batch["pa_counterfactual_multi_modal_data"] = counterfactual_array

    if config.pa_annotation_log_dir:
        log_dir = Path(config.pa_annotation_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        output = log_dir / f"step_{global_step}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for record in annotation_records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return {
        **request_metrics,
        "pa/routing_mode_all_tokens": 0.0,
        "pa/extractor_invoked": 1.0,
        "pa/annotation_attempted": float(attempted),
        "pa/annotation_failed": float(len(errors)),
        "pa/annotation_failure_rate": failure_rate,
        "pa/annotation_success_rate": float((batch_size - len(errors)) / max(batch_size, 1)),
        "pa/annotation_attempts_mean": float(np.mean(annotation_attempt_counts)) if annotation_attempt_counts else 0.0,
        "pa/annotation_attempts_max": float(max(annotation_attempt_counts)) if annotation_attempt_counts else 0.0,
        "pa/annotation_latency_s_mean": float(np.mean(annotation_latencies)) if annotation_latencies else 0.0,
        "pa/annotation_latency_s_p95": float(np.quantile(annotation_latencies, 0.95)) if annotation_latencies else 0.0,
        "pa/alignment_valid_rollout_rate": float(aligned_rollouts / max(batch_size, 1)),
        "pa/direct_span_count": float(aligned_direct_count),
        "pa/no_direct_rollout_rate": float(no_direct_rollouts / max(batch_size, 1)),
        "pa/rollout_with_direct_rate": float((direct_mask.any(dim=1)).float().mean().item()),
        "pa/intervention_valid_rate": float(intervention_valid.float().mean().item()),
        "pa/changed_area_ratio_mean": float(changed_ratios.mean().item()),
        "pa/counterfactual_images_saved": float(saved_counterfactuals),
    }


def merge_pa_annotation_batch(
    *,
    target: DataProto,
    annotated: DataProto,
    reorder_indices: torch.Tensor,
) -> None:
    """Merge async PA outputs into the length-balanced training batch.

    Annotation starts from generation order while GPU scoring uses the
    length-balanced order. Reordering the completed annotation-only batch with
    the exact same permutation preserves rollout/span alignment.
    """
    annotated.reorder(reorder_indices)
    for key in PA_ANNOTATION_TENSOR_KEYS:
        if key not in annotated.batch:
            raise KeyError(f"async PA annotation did not produce tensor key: {key}")
        target.batch[key] = annotated.batch[key]
    for key in PA_ANNOTATION_NON_TENSOR_KEYS:
        if key not in annotated.non_tensor_batch:
            raise KeyError(f"async PA annotation did not produce non-tensor key: {key}")
        target.non_tensor_batch[key] = annotated.non_tensor_batch[key]


def _compute_all_token_multipliers(
    batch: DataProto, config: Any, *, global_step: int | None
) -> dict[str, float]:
    """Vectorized no-router ablation: every response token is gate-eligible."""
    old_logp = batch.batch["old_log_probs"].float()
    cf_logp = batch.batch["pa_cf_log_probs"].float()
    response_mask = batch.batch["response_mask"].bool()
    valid = batch.batch["pa_intervention_valid"].bool()
    delta = torch.clamp(cf_logp - old_logp, -config.pa_delta_clip, config.pa_delta_clip)
    residual_count = response_mask.sum(dim=-1)
    residual_delta = (delta * response_mask).sum(dim=-1) / residual_count.clamp(min=1)
    corrected = delta - residual_delta.unsqueeze(-1)
    violation = torch.relu(corrected + config.pa_margin)
    gate = torch.exp(-config.pa_alpha * violation).clamp(
        min=config.pa_gate_floor, max=1.0
    )
    eligible_rollout = valid & (residual_count >= config.pa_residual_min_tokens)
    eligible = response_mask & eligible_rollout.unsqueeze(-1)
    multiplier = torch.where(eligible, gate, torch.ones_like(gate))

    if not torch.all((multiplier >= config.pa_gate_floor) & (multiplier <= 1.0)):
        raise AssertionError("all-token PA multiplier left configured [floor, 1] bounds")
    batch.batch["pa_token_multiplier"] = multiplier
    batch.batch["pa_residual_token_mask"] = response_mask
    batch.batch["pa_residual_delta"] = residual_delta

    advantages = batch.batch["advantages"]
    positive = (advantages > 0) & response_mask
    positive_before = positive.float().sum()
    retention = (multiplier * positive).sum() / positive_before.clamp(min=1)
    gated_positive_rate = ((multiplier < 1.0) & positive).sum() / positive.sum().clamp(min=1)
    floor_positive_rate = (
        torch.isclose(
            multiplier,
            torch.tensor(config.pa_gate_floor, device=multiplier.device),
        )
        & positive
    ).sum() / positive.sum().clamp(min=1)

    corrected_values = corrected[eligible].detach().cpu().numpy()
    gate_values = multiplier[eligible].detach().cpu().numpy()
    per_rollout_retention = [
        float(multiplier[index][positive[index]].mean().item())
        for index in range(multiplier.shape[0])
        if positive[index].any()
    ]

    if config.pa_gate_log_dir and global_step is not None:
        gate_dir = Path(config.pa_gate_log_dir)
        gate_dir.mkdir(parents=True, exist_ok=True)
        output = gate_dir / f"step_{global_step}.jsonl"
        uids = batch.non_tensor_batch.get(
            "uid", np.array([""] * multiplier.shape[0], dtype=object)
        )
        with output.open("w", encoding="utf-8") as handle:
            for index in range(multiplier.shape[0]):
                token_mask = eligible[index]
                token_gates = multiplier[index][token_mask]
                handle.write(
                    json.dumps(
                        {
                            "step": global_step,
                            "uid": str(uids[index]),
                            "batch_index": index,
                            "routing_mode": "all_tokens",
                            "extractor_invoked": False,
                            "eligible_token_count": int(token_mask.sum().item()),
                            "gate_mean": float(token_gates.mean().item())
                            if token_gates.numel()
                            else 1.0,
                            "residual_delta": float(residual_delta[index].item()),
                            "intervention_valid": bool(valid[index].item()),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def stat(values: np.ndarray, fn, fallback: float) -> float:
        return float(fn(values)) if values.size else fallback

    eligible_count = int(eligible.sum().item())
    gated_count = int(((multiplier < 1.0) & eligible).sum().item())
    return {
        "pa/routing_mode_all_tokens": 1.0,
        "pa/extractor_invoked": 0.0,
        "pa/residual_control_coverage": float(eligible_rollout.float().mean().item()),
        "pa/residual_token_count_mean": float(residual_count.float().mean().item()),
        # Existing span-named metrics are retained for dashboard compatibility;
        # in this ablation their unit is an eligible response token.
        "pa/eligible_span_count": float(eligible_count),
        "pa/gated_span_count": float(gated_count),
        "pa/gated_span_rate": float(gated_count / max(eligible_count, 1)),
        "pa/corrected_delta_mean": stat(corrected_values, np.mean, 0.0),
        "pa/corrected_delta_p10": stat(corrected_values, lambda x: np.quantile(x, 0.10), 0.0),
        "pa/corrected_delta_median": stat(corrected_values, np.median, 0.0),
        "pa/corrected_delta_p90": stat(corrected_values, lambda x: np.quantile(x, 0.90), 0.0),
        "pa/corrected_positive_rate": stat(corrected_values, lambda x: np.mean(x > 0.0), 0.0),
        "pa/margin_violation_rate": stat(
            corrected_values, lambda x: np.mean(x > -config.pa_margin), 0.0
        ),
        "pa/gate_mean": stat(gate_values, np.mean, 1.0),
        "pa/gate_p10": stat(gate_values, lambda x: np.quantile(x, 0.10), 1.0),
        "pa/gate_median": stat(gate_values, np.median, 1.0),
        "pa/gate_p90": stat(gate_values, lambda x: np.quantile(x, 0.90), 1.0),
        "pa/direct_response_token_fraction": 1.0,
        "pa/positive_credit_retention": float(retention.item()),
        "pa/positive_advantage_mass_retention": float(
            (multiplier * batch.batch["advantages"].float().clamp_min(0) * positive_direct).sum()
            / (batch.batch["advantages"].float().clamp_min(0) * positive_direct).sum().clamp(min=1e-12)
        ),
        "pa/positive_credit_retention_sample_macro": float(np.mean(per_rollout_retention))
        if per_rollout_retention
        else 1.0,
        "pa/downweighted_positive_direct_token_rate": float(gated_positive_rate.item()),
        "pa/floor_positive_direct_token_rate": float(floor_positive_rate.item()),
        "hpa/enabled": 0.0,
        "hpa/eligible_rollout_count": 0.0,
        "hpa/comparable_group_count": 0.0,
        "hpa/positive_group_count": 0.0,
        "hpa/noncomparable_positive_group_rate": 0.0,
        "hpa/rollout_burden_mean": 0.0,
        "hpa/rollout_burden_p90": 0.0,
        "hpa/rollout_quality_mean": 1.0,
        "hpa/rollout_weight_mean": 1.0,
        "hpa/rollout_weight_min": 1.0,
        "hpa/rollout_weight_max": 1.0,
        "hpa/quality_span_count_corr": 0.0,
        "hpa/quality_response_length_corr": 0.0,
    }


def compute_pa_token_multipliers(
    batch: DataProto, config: Any, *, global_step: int | None = None
) -> dict[str, float]:
    """Compute signed, residual-corrected span gates on the trainer driver."""
    if config.pa_routing_mode == "all_tokens":
        return _compute_all_token_multipliers(batch, config, global_step=global_step)

    old_logp = batch.batch["old_log_probs"].float()
    cf_logp = batch.batch["pa_cf_log_probs"].float()
    response_mask = batch.batch["response_mask"].bool()
    visual_mask = batch.batch["pa_visual_token_mask"].bool()
    text_mask = batch.batch["pa_response_text_token_mask"].bool()
    valid = batch.batch["pa_intervention_valid"].bool()
    delta = torch.clamp(cf_logp - old_logp, -config.pa_delta_clip, config.pa_delta_clip)
    residual_mask = response_mask & text_mask & ~visual_mask
    residual_count = residual_mask.sum(dim=-1)
    residual_sum = (delta * residual_mask).sum(dim=-1)
    residual_delta = residual_sum / residual_count.clamp(min=1)

    multiplier = torch.ones_like(delta, dtype=torch.float32)
    corrected_values: list[float] = []
    gate_values: list[float] = []
    gated_spans = 0
    eligible_spans = 0
    gate_records: list[dict[str, Any]] = []
    gate_work: list[dict[str, Any]] = []
    span_records = batch.non_tensor_batch["pa_direct_spans"]
    uids = batch.non_tensor_batch.get("uid", np.array([""] * len(span_records), dtype=object))
    rollout_violations: list[list[torch.Tensor]] = [[] for _ in range(len(span_records))]

    for batch_index, spans in enumerate(span_records):
        if not valid[batch_index] or residual_count[batch_index] < config.pa_residual_min_tokens:
            continue
        for span in spans:
            indices = [
                int(index)
                for index in span.get("token_indices", [])
                if 0 <= int(index) < delta.shape[1] and response_mask[batch_index, int(index)]
            ]
            if not indices:
                continue
            eligible_spans += 1
            raw_span_delta = delta[batch_index, indices].mean()
            corrected_delta = raw_span_delta - residual_delta[batch_index]
            violation = torch.relu(corrected_delta + config.pa_margin)
            gate = torch.exp(-config.pa_alpha * violation).clamp(
                min=config.pa_gate_floor, max=1.0
            )
            rollout_violations[batch_index].append(violation)
            corrected_values.append(float(corrected_delta.item()))
            gate_values.append(float(gate.item()))
            gate_work.append(
                dict(
                    batch_index=batch_index,
                    indices=indices,
                    span=span,
                    raw_span_delta=raw_span_delta,
                    corrected_delta=corrected_delta,
                    violation=violation,
                    span_gate=gate,
                )
            )

    gate_mode = "persistence"

    rollout_burden = torch.full(
        (len(span_records),), float("nan"), dtype=torch.float32, device=multiplier.device
    )
    for batch_index, violations in enumerate(rollout_violations):
        if violations:
            rollout_burden[batch_index] = torch.stack(violations).mean()

    rollout_quality = torch.full_like(rollout_burden, float("nan"))
    finite_burden = torch.isfinite(rollout_burden)
    rollout_quality[finite_burden] = torch.exp(
        -config.hpa_beta_rollout * rollout_burden[finite_burden]
    )
    rollout_weight = torch.ones_like(rollout_burden)
    comparable_groups = 0
    eligible_rollouts = 0
    positive_group_count = 0
    if config.use_hpa_grpo:
        advantages = batch.batch["advantages"].float()
        sequence_advantage = (advantages * response_mask).sum(dim=-1) / response_mask.sum(
            dim=-1
        ).clamp(min=1)
        for uid in dict.fromkeys(str(value) for value in uids):
            group_indices = [
                index for index, value in enumerate(uids) if str(value) == uid
            ]
            if any(sequence_advantage[index].item() > 0 for index in group_indices):
                positive_group_count += 1
            eligible_indices = [
                index
                for index in group_indices
                if sequence_advantage[index].item() > 0 and finite_burden[index].item()
            ]
            eligible_rollouts += len(eligible_indices)
            if len(eligible_indices) < 2:
                continue
            comparable_groups += 1
            index_tensor = torch.tensor(
                eligible_indices, dtype=torch.long, device=multiplier.device
            )
            quality = rollout_quality[index_tensor]
            raw_weight = (
                1.0 + config.hpa_rollout_lambda * (quality - quality.mean())
            ).clamp(
                min=config.hpa_rollout_weight_floor,
                max=config.hpa_rollout_weight_ceiling,
            )
            rollout_weight[index_tensor] = raw_weight / raw_weight.mean().clamp(min=1e-8)

    final_gate_values: list[float] = []
    for work in gate_work:
        batch_index = work["batch_index"]
        span_gate = work["span_gate"]
        final_gate = span_gate
        if config.use_hpa_grpo:
            final_gate = (span_gate * rollout_weight[batch_index]).clamp(
                min=config.hpa_final_gate_floor,
                max=config.hpa_final_gate_ceiling,
            )
        index_tensor = torch.tensor(
            work["indices"], dtype=torch.long, device=multiplier.device
        )
        multiplier[batch_index, index_tensor] = torch.minimum(
            multiplier[batch_index, index_tensor], final_gate
        )
        final_gate_values.append(float(final_gate.item()))
        if final_gate.item() < 1.0 - 1e-8:
            gated_spans += 1
        span = work["span"]
        indices = work["indices"]
        gate_records.append(
            {
                    "step": global_step,
                    "uid": str(uids[batch_index]),
                    "batch_index": batch_index,
                    "span_id": int(span.get("span_id", -1)),
                    "span_text": span.get("text"),
                    "token_start": int(min(indices)),
                    "token_end": int(max(indices) + 1),
                    "raw_span_delta": float(work["raw_span_delta"].item()),
                    "residual_delta": float(residual_delta[batch_index].item()),
                    "corrected_delta": float(work["corrected_delta"].item()),
                    "violation": float(work["violation"].item()),
                    "gate_mode": gate_mode,
                    "persistence_gate": float(work.get("persistence_gate", span_gate).item()),
                    "efs_k3": work.get("efs_k3"),
                    "span_gate": float(span_gate.item()),
                    "rollout_burden": float(rollout_burden[batch_index].item()),
                    "rollout_quality": float(rollout_quality[batch_index].item()),
                    "rollout_weight": float(rollout_weight[batch_index].item()),
                    "gate": float(final_gate.item()),
                    "residual_token_count": int(residual_count[batch_index].item()),
                    "intervention_valid": True,
                    "hierarchical": bool(config.use_hpa_grpo),
                }
            )

    # No direct token may be changed by accidental indexing; invalid rollouts
    # and insufficient-control rollouts must remain exactly standard GRPO.
    direct_mask = batch.batch["pa_direct_token_mask"].bool()
    multiplier[~direct_mask] = 1.0
    ineligible = (~valid) | (residual_count < config.pa_residual_min_tokens)
    multiplier[ineligible] = 1.0
    configured_floor = (
        config.hpa_final_gate_floor if config.use_hpa_grpo else config.pa_gate_floor
    )
    if not torch.all((multiplier >= configured_floor) & (multiplier <= 1.0)):
        raise AssertionError("PA multiplier left configured [floor, 1] bounds")
    if not torch.all(multiplier[~direct_mask] == 1.0):
        raise AssertionError("PA changed a non-direct-VEF token")

    batch.batch["pa_token_multiplier"] = multiplier
    batch.batch["pa_residual_token_mask"] = residual_mask
    batch.batch["pa_residual_delta"] = residual_delta

    positive_token_mask = (batch.batch["advantages"] > 0) & response_mask
    positive_direct = positive_token_mask & direct_mask
    positive_credit_before = positive_direct.float().sum()
    positive_credit_after = (multiplier * positive_direct).sum()
    retention = positive_credit_after / positive_credit_before.clamp(min=1)
    gated_token_rate = ((multiplier < 1.0) & positive_direct).sum() / positive_direct.sum().clamp(min=1)
    floor_token_rate = (
        (torch.isclose(multiplier, torch.tensor(configured_floor, device=multiplier.device)) & positive_direct).sum()
        / positive_direct.sum().clamp(min=1)
    )
    direct_token_fraction = (direct_mask & response_mask).sum() / response_mask.sum().clamp(min=1)
    per_rollout_retention: list[float] = []
    for batch_index in range(multiplier.shape[0]):
        mask = positive_direct[batch_index]
        if mask.any():
            per_rollout_retention.append(float(multiplier[batch_index][mask].mean().item()))

    finite_indices = torch.where(finite_burden)[0].tolist()
    burden_values = [float(rollout_burden[index].item()) for index in finite_indices]
    quality_values = [float(rollout_quality[index].item()) for index in finite_indices]
    weight_values = [
        float(rollout_weight[index].item())
        for index in finite_indices
        if rollout_weight[index].item() != 1.0 or config.use_hpa_grpo
    ]
    span_counts = [float(len(rollout_violations[index])) for index in finite_indices]
    response_lengths = [
        float(response_mask[index].sum().item()) for index in finite_indices
    ]

    def safe_corr(left: list[float], right: list[float]) -> float:
        if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    if config.pa_gate_log_dir and global_step is not None:
        gate_dir = Path(config.pa_gate_log_dir)
        gate_dir.mkdir(parents=True, exist_ok=True)
        output = gate_dir / f"step_{global_step}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for record in gate_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "pa/residual_control_coverage": float(
            (valid & (residual_count >= config.pa_residual_min_tokens)).float().mean().item()
        ),
        "pa/residual_token_count_mean": float(residual_count.float().mean().item()),
        "pa/eligible_span_count": float(eligible_spans),
        "pa/gated_span_count": float(gated_spans),
        "pa/gated_span_rate": float(gated_spans / max(eligible_spans, 1)),
        "pa/corrected_delta_mean": float(np.mean(corrected_values)) if corrected_values else 0.0,
        "pa/corrected_delta_p10": float(np.quantile(corrected_values, 0.10)) if corrected_values else 0.0,
        "pa/corrected_delta_median": float(np.median(corrected_values)) if corrected_values else 0.0,
        "pa/corrected_delta_p90": float(np.quantile(corrected_values, 0.90)) if corrected_values else 0.0,
        "pa/corrected_positive_rate": float(np.mean(np.asarray(corrected_values) > 0.0)) if corrected_values else 0.0,
        "pa/margin_violation_rate": float(np.mean(np.asarray(corrected_values) > -config.pa_margin)) if corrected_values else 0.0,
        "pa/gate_mean": float(np.mean(final_gate_values)) if final_gate_values else 1.0,
        "pa/gate_p10": float(np.quantile(final_gate_values, 0.10)) if final_gate_values else 1.0,
        "pa/gate_median": float(np.median(final_gate_values)) if final_gate_values else 1.0,
        "pa/gate_p90": float(np.quantile(final_gate_values, 0.90)) if final_gate_values else 1.0,
        "pa/direct_response_token_fraction": float(direct_token_fraction.item()),
        "pa/positive_credit_retention": float(retention.item()),
        "pa/positive_advantage_mass_retention": float(
            (multiplier * batch.batch["advantages"].float().clamp_min(0) * positive_direct).sum()
            / (batch.batch["advantages"].float().clamp_min(0) * positive_direct).sum().clamp(min=1e-12)
        ),
        "pa/positive_credit_retention_sample_macro": float(np.mean(per_rollout_retention)) if per_rollout_retention else 1.0,
        "pa/downweighted_positive_direct_token_rate": float(gated_token_rate.item()),
        "pa/floor_positive_direct_token_rate": float(floor_token_rate.item()),
        "hpa/enabled": float(bool(config.use_hpa_grpo)),
        "hpa/eligible_rollout_count": float(eligible_rollouts),
        "hpa/comparable_group_count": float(comparable_groups),
        "hpa/positive_group_count": float(positive_group_count),
        "hpa/noncomparable_positive_group_rate": float(
            (positive_group_count - comparable_groups) / max(positive_group_count, 1)
        ),
        "hpa/rollout_burden_mean": float(np.mean(burden_values)) if burden_values else 0.0,
        "hpa/rollout_burden_p90": float(np.quantile(burden_values, 0.90)) if burden_values else 0.0,
        "hpa/rollout_quality_mean": float(np.mean(quality_values)) if quality_values else 1.0,
        "hpa/rollout_weight_mean": float(np.mean(weight_values)) if weight_values else 1.0,
        "hpa/rollout_weight_min": float(min(weight_values)) if weight_values else 1.0,
        "hpa/rollout_weight_max": float(max(weight_values)) if weight_values else 1.0,
        "hpa/quality_span_count_corr": safe_corr(quality_values, span_counts),
        "hpa/quality_response_length_corr": safe_corr(quality_values, response_lengths),
    }
