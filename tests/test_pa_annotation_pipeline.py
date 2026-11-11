"""CPU-only tests: load production definitions without importing Ray/CUDA.

Run: python3 -m unittest discover -s tests -p test_pa_annotation_pipeline.py -v
"""
from __future__ import annotations

import ast
from collections import defaultdict
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import Mock, patch

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "verl/trainer/pa_grpo.py"
tree = ast.parse(SOURCE.read_text())
names = {"OnlineVEFAnnotator", "PAAnnotationPipeline", "AnnotationServiceError", "_validate_annotation_payload"}
definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions], type_ignores=[])
ns = dict(globals(), RETRYABLE_STATUS={408, 409, 425, 429, 500, 502, 503, 504},
          _load_annotation_helpers=lambda _: (SimpleNamespace(build_prompt=lambda *args: "prompt", parse_json_content=json.loads), "stub"),
          _image_data_url=lambda image: "data:image/png;base64,stub", _strip_thinking=lambda text: text,
          _first_original_image=lambda data: (data, 1))
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
Client, Pipeline = ns["OnlineVEFAnnotator"], ns["PAAnnotationPipeline"]


def config(**overrides):
    values = dict(pa_annotator_script="stub", pa_annotator_url="http://replica-one/v1/chat/completions",
                  pa_annotator_urls="", pa_annotator_api_key_env="TEST_UNUSED_PA_KEY", pa_annotator_model="same-model",
                  pa_annotator_temperature=.8, pa_annotator_top_k=50, pa_annotator_top_p=.95,
                  pa_annotator_trust_env=False, pa_annotator_timeout=.05, pa_annotator_connect_timeout=.02,
                  pa_annotator_max_trace_chars=20000, pa_annotator_max_retries=2, pa_annotator_retry_backoff=0,
                  pa_annotator_request_budget=.2, pa_annotator_concurrency=2, pa_annotation_heartbeat=.01,
                  pa_annotation_batch_budget=.1, pa_annotation_log_dir=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class Mask:
    def __init__(self, value):
        self.value = np.asarray(value)
    def __getitem__(self, index):
        return Mask(self.value[index])
    def bool(self):
        return self.value.astype(bool)


class Batch:
    def __init__(self, ids):
        self.batch = {"responses": np.asarray([[i] for i in ids]), "response_mask": Mask([[1] for _ in ids])}
        self.non_tensor_batch = {"uid": [str(i) for i in ids], "question": ["q"] * len(ids), "multi_modal_data": ["image"] * len(ids)}
    def __len__(self):
        return len(self.non_tensor_batch["uid"])


TOKENIZER = SimpleNamespace(decode=lambda ids, **kwargs: str(ids[0]))


class Proto:
    """Small array-backed DataProto double for executing the real batch builder."""
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}
    def __len__(self):
        return len(next(iter({**self.batch, **self.non_tensor_batch}.values())))
    def __getitem__(self, idx):
        return Proto({k: v[idx] for k, v in self.batch.items()}, {k: v[idx] for k, v in self.non_tensor_batch.items()}, self.meta_info)
    @classmethod
    def from_single_dict(cls, data, meta_info):
        tensors = {k: v for k, v in data.items() if k in {"input_ids", "attention_mask", "position_ids"}}
        return cls(tensors, {k: v for k, v in data.items() if k not in tensors}, meta_info)
    def pop(self, batch_keys=(), non_tensor_batch_keys=(), meta_info_keys=()):
        return Proto({k: self.batch.pop(k) for k in batch_keys},
                     {k: self.non_tensor_batch.pop(k) for k in non_tensor_batch_keys},
                     {k: self.meta_info.pop(k) for k in meta_info_keys})
    def repeat(self, repeat_times, interleave):
        assert interleave
        return Proto({k: np.repeat(v, repeat_times, axis=0) for k, v in self.batch.items()},
                     {k: np.repeat(v, repeat_times, axis=0) for k, v in self.non_tensor_batch.items()}, dict(self.meta_info))
    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        self.meta_info.update(other.meta_info)
        return self
    @classmethod
    def concat(cls, batches):
        return cls({k: np.concatenate([b.batch[k] for b in batches]) for k in batches[0].batch},
                   {k: np.concatenate([b.non_tensor_batch[k] for b in batches]) for k in batches[0].non_tensor_batch}, dict(batches[0].meta_info))


trainer_tree = ast.parse((ROOT / "verl/trainer/ray_trainer.py").read_text())
batch_method = next(node for node in ast.walk(trainer_tree) if isinstance(node, ast.FunctionDef) and node.name == "_make_batch_data")
batch_ns = dict(globals(), DataProto=Proto, ray=SimpleNamespace(get=lambda value: value),
                reduce_metrics=lambda values: {k: float(np.mean(v)) for k, v in values.items()})
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), batch_method], type_ignores=[])), "ray_trainer.py", "exec"), batch_ns)


def batch_trainer(filtering=False, max_tries=2):
    def source(start):
        ids = np.arange(start, start + 4)
        return {"input_ids": ids[:, None], "attention_mask": np.ones((4, 1)), "position_ids": ids[:, None],
                "raw_prompt_ids": ids, "multi_modal_data": np.array(["image"] * 4), "question": np.array(["q"] * 4)}
    calls = []
    def generate(batch):
        ids = batch.non_tensor_batch["raw_prompt_ids"]
        calls.append(ids.tolist())
        return Proto({"responses": np.repeat(ids, 2)[:, None], "response_mask": np.ones((len(ids) * 2, 1))},
                     {"multi_modal_data": np.repeat(batch.non_tensor_batch["multi_modal_data"], 2)})
    def reward(batch):
        scores = [0 if int(row[0]) < 2 else (i % 2) for i, row in enumerate(batch.batch["responses"])]
        return np.array(scores)[:, None], {"accuracy": scores}
    trainer = SimpleNamespace(
        config=SimpleNamespace(data=SimpleNamespace(min_pixels=1, max_pixels=1, video_fps=1, rollout_batch_size=4),
                               worker=SimpleNamespace(rollout=SimpleNamespace(n=2)),
                               algorithm=SimpleNamespace(pa_rollout_chunk_size=1, adv_estimator="grpo", online_filtering=filtering,
                                                         filter_key="accuracy", filter_low=0, filter_high=1),
                               trainer=SimpleNamespace(max_try_make_batch=max_tries)),
        actor_rollout_ref_wg=SimpleNamespace(world_size=2, generate_sequences=generate),
        reward_fn=SimpleNamespace(compute_reward=SimpleNamespace(remote=reward)),
        train_dataloader=[source(0), source(4)], _maybe_dump_train_rollouts=lambda batch: None)
    trainer.data_iterator = iter(trainer.train_dataloader)
    return trainer, calls


class AnnotationTests(unittest.TestCase):
    def test_real_batch_builder_streams_in_order_and_aligns_world_size(self):
        trainer, calls = batch_trainer(max_tries=1)
        submitted = []
        def callback(batch):
            submitted.extend(batch.batch["responses"][:, 0].tolist())
            self.assertEqual(len(calls), len(submitted) // 4)
        result = batch_ns["_make_batch_data"](trainer, {}, callback)
        self.assertEqual(calls, [[0, 1], [2, 3]])
        self.assertEqual(submitted, result.batch["responses"][:, 0].tolist())

    def test_real_batch_builder_filters_before_annotation(self):
        trainer, calls = batch_trainer(filtering=True)
        submitted = []
        result = batch_ns["_make_batch_data"](trainer, {}, lambda b: submitted.extend(b.batch["responses"][:, 0].tolist()))
        self.assertEqual(submitted, [2, 2, 3, 3, 4, 4, 5, 5])
        self.assertEqual(submitted, result.batch["responses"][:, 0].tolist())
        self.assertEqual(calls, [[0, 1], [2, 3], [4, 5]])

    def test_real_batch_builder_without_callback_is_unchanged(self):
        trainer, calls = batch_trainer()
        result = batch_ns["_make_batch_data"](trainer, {})
        self.assertEqual(calls, [[0, 1, 2, 3]])
        self.assertEqual(len(result), 8)

    def test_replica_retry_diagnostics(self):
        client = Client(config(pa_annotator_urls="http://a,http://b", pa_annotator_request_budget=2))
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "0"
        client._request_once = Mock(side_effect=[requests.HTTPError(response=response), {"parsed": {"spans": []}}])
        with patch.object(ns["random"], "uniform", return_value=0):
            result = client.annotate_one(rollout_id="id", question="q", trace="trace", image="image")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual([c.kwargs["endpoint"] for c in client._request_once.call_args_list], ["http://a", "http://b"])
        self.assertEqual(client.events[0]["http_status"], 429)

    def test_permanent_service_error_not_retried(self):
        client = Client(config())
        response = requests.Response()
        response.status_code = 401
        client._request_once = Mock(side_effect=requests.HTTPError("secret-not-for-logs", response=response))
        with self.assertRaises(ns["AnnotationServiceError"]):
            client.annotate_one(rollout_id="id", question="q", trace="t", image="i")
        self.assertEqual(client._request_once.call_count, 1)
        self.assertTrue(client.stop.is_set())
        self.assertNotIn("secret-not-for-logs", json.dumps(client.events))

    def test_400_not_retried(self):
        client = Client(config())
        response = requests.Response()
        response.status_code = 400
        client._request_once = Mock(side_effect=requests.HTTPError(response=response))
        with self.assertRaises(RuntimeError):
            client.annotate_one(rollout_id="id", question="q", trace="t", image="i")
        self.assertEqual(client._request_once.call_count, 1)

    def test_403_recovers_on_third_retry(self):
        client = Client(config(pa_annotator_request_budget=5))
        response = requests.Response()
        response.status_code = 403
        client._request_once = Mock(side_effect=[
            *[requests.HTTPError(response=response) for _ in range(3)],
            {"parsed": {"spans": []}},
        ])
        with patch.object(ns["random"], "uniform", return_value=0):
            result = client.annotate_one(rollout_id="id", question="q", trace="t", image="i")
        self.assertEqual(result["attempts"], 4)
        self.assertFalse(client.stop.is_set())

    def test_exhausted_403_is_sample_failure_not_service_abort(self):
        pipe = Pipeline(config(pa_annotator_request_budget=5,
                               pa_annotation_batch_budget=5), TOKENIZER, 1)
        response = requests.Response()
        response.status_code = 403
        pipe.annotator._request_once = Mock(side_effect=requests.HTTPError(response=response))
        try:
            with patch.object(ns["random"], "uniform", return_value=0):
                pipe.enqueue(Batch([1]))
                results, errors, attempted = pipe.collect(["1"], ["1"])
            self.assertEqual(pipe.annotator._request_once.call_count, 4)
            self.assertEqual(results, [None])
            self.assertEqual(attempted, 1)
            self.assertIn("HTTP 403", errors[0])
            self.assertFalse(pipe.annotator.stop.is_set())
            self.assertEqual(pipe.diagnostics()["pa/request_http403_count"], 4)
        finally:
            pipe.close()

    def test_403_still_respects_request_budget(self):
        client = Client(config(pa_annotator_request_budget=.01))
        response = requests.Response()
        response.status_code = 403
        def slow(**kwargs):
            time.sleep(.02)
            raise requests.HTTPError(response=response)
        client._request_once = Mock(side_effect=slow)
        with self.assertRaises(RuntimeError):
            client.annotate_one(rollout_id="id", question="q", trace="t", image="i")
        self.assertEqual(client._request_once.call_count, 1)
        self.assertFalse(client.stop.is_set())

    def test_request_budget_stops_retries(self):
        client = Client(config(pa_annotator_request_budget=.01))
        def slow(**kwargs):
            time.sleep(.02)
            raise requests.Timeout()
        client._request_once = Mock(side_effect=slow)
        with self.assertRaises(RuntimeError):
            client.annotate_one(rollout_id="id", question="q", trace="t", image="i")
        self.assertEqual(client._request_once.call_count, 1)

    def test_http_timeout_and_payload(self):
        client = Client(config())
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": '{"spans": []}'}, "finish_reason": "stop"}]}
        session = Mock()
        session.post.return_value = response
        client._session = lambda: session
        client._request_once(prompt="q", image="i", endpoint="http://replica", remaining=.03)
        self.assertEqual(session.post.call_args.kwargs["timeout"], (.02, .03))
        self.assertFalse(session.post.call_args.kwargs["json"]["chat_template_kwargs"]["enable_thinking"])

    def test_streaming_order_concurrency_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = Pipeline(config(pa_annotation_log_dir=tmp, pa_annotation_batch_budget=2), TOKENIZER, 1)
            running, peak = 0, 0
            lock = threading.Lock()
            def request(**kwargs):
                nonlocal running, peak
                with lock:
                    running += 1
                    peak = max(peak, running)
                time.sleep(.01)
                with lock:
                    running -= 1
                return {"parsed": {"spans": []}}
            pipe.annotator._request_once = request
            try:
                pipe.enqueue(Batch([1, 2]))
                pipe.enqueue(Batch([3, 4]))
                results, errors, attempted = pipe.collect(["1", "2", "3", "4"], ["1", "2", "3", "4"])
                self.assertEqual(attempted, 4)
                self.assertFalse(errors)
                self.assertTrue(all(result is not None for result in results))
                self.assertLessEqual(peak, 2)
            finally:
                pipe.close()
            summary = json.loads((Path(tmp) / "step_1_request_summary.json").read_text())
            self.assertEqual(summary["metrics"]["pa/request_attempt_count"], 4)
            self.assertEqual(summary["unfinished"], 0)

    def test_order_mismatch_aborts(self):
        pipe = Pipeline(config(), TOKENIZER, 1)
        pipe.annotator._request_once = lambda **kwargs: {"parsed": {"spans": []}}
        try:
            pipe.enqueue(Batch([1, 2]))
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                pipe.collect(["2", "1"], ["2", "1"])
        finally:
            pipe.close(abort=True)

    def test_batch_timeout_does_not_wait_for_blocked_thread(self):
        pipe = Pipeline(config(pa_annotation_batch_budget=.02), TOKENIZER, 1)
        release = threading.Event()
        def request(**kwargs):
            release.wait(2)
            return {"parsed": {"spans": []}}
        pipe.annotator._request_once = request
        try:
            pipe.enqueue(Batch([1]))
            with self.assertRaisesRegex(TimeoutError, "not silently ungated"):
                pipe.collect(["1"], ["1"])
            started = time.monotonic()
            pipe.close(abort=True)
            self.assertLess(time.monotonic() - started, .1)
        finally:
            release.set()
            pipe.close(abort=True)

    def test_all_tokens_and_plain_training_keep_original_path(self):
        trainer = (ROOT / "verl/trainer/ray_trainer.py").read_text()
        self.assertIn('self.config.algorithm.pa_routing_mode == "direct"', trainer)
        self.assertIn("annotation_callback=pa_pipeline.enqueue if pa_pipeline is not None else None", trainer)


if __name__ == "__main__":
    unittest.main()
