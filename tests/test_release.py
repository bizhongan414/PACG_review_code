"""CPU checks of actual exported training gates, launcher wiring and SPD equations."""
import ast
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from pacg.annotate import align_exact
from pacg.annotation import load_helpers
from pacg.diagnose import intervene, score_row
from pacg.evaluate import summarize
from pacg.metrics import claim_metrics, paired_bootstrap, question_macro, token_metrics
from verl.trainer.config import AlgorithmConfig, PPOConfig

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_all_six_launcher_configs_merge_actual_schema(self):
        for parent in ('grpo', 'dapo', 'vppo'):
            for enabled in ('true', 'false'):
                with self.subTest(parent=parent, enabled=enabled):
                    env = {**os.environ, 'PARENT': parent, 'USE_PACG': enabled,
                           'MODEL_PATH': '/tmp/model with spaces', 'TRAIN_DATA': '/tmp/train', 'VAL_DATA': '/tmp/val'}
                    cmd = subprocess.check_output(['bash', 'scripts/train_qwen25vl7b.sh', '--dry-run'],
                                                  cwd=ROOT, env=env, text=True)
                    cli = OmegaConf.from_dotlist(shlex.split(cmd)[3:])
                    filename = cli.pop('config')
                    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PPOConfig()),
                                                                OmegaConf.load(ROOT / filename), cli))
                    with patch('os.path.exists', return_value=True):
                        config.deep_post_init()
                    self.assertEqual(config.algorithm.use_pa_grpo, enabled == 'true')
                    self.assertEqual(config.algorithm.use_vppo_on_perception, parent == 'vppo')
                    self.assertEqual(config.algorithm.online_filtering, parent != 'grpo')
                    self.assertEqual(config.worker.actor.loss_avg_mode, 'seq' if parent == 'grpo' else 'token')
                    self.assertEqual(config.worker.actor.model.model_path, '/tmp/model with spaces')
                    self.assertEqual(config.algorithm.pa_margin, .05)
                    self.assertEqual(config.worker.actor.entropy_penalty_coef, .1)
                    self.assertTrue(config.algorithm.disable_kl)
                    self.assertFalse(config.algorithm.use_kl_loss)
                    self.assertEqual(config.algorithm.kl_coef, 0.)
                    self.assertTrue(config.worker.actor.disable_kl)
                    self.assertFalse(config.worker.actor.use_kl_loss)
                    self.assertTrue(Path(config.algorithm.pa_annotator_script).is_file())
                    self.assertTrue(Path(config.worker.reward.reward_function).is_file())

    def test_actual_training_gate_calibration_and_overlap(self):
        path = ROOT / 'verl/trainer/pa_grpo.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_pa_token_multipliers')
        ns = {'torch': torch, 'np': np, 'Any': object, 'DataProto': SimpleNamespace, 'Path': Path, 'json': json}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        config = AlgorithmConfig()
        direct = torch.tensor([[1, 1, 1] + [0] * 9], dtype=torch.bool)
        def run(shift):
            batch = SimpleNamespace(batch={
                'old_log_probs': torch.zeros((1, 12)),
                'pa_cf_log_probs': torch.tensor([[-.10] * 3 + [-.20] * 9]) + shift,
                'response_mask': torch.ones((1, 12), dtype=torch.bool),
                'pa_visual_token_mask': direct, 'pa_direct_token_mask': direct,
                'pa_response_text_token_mask': torch.ones((1, 12), dtype=torch.bool),
                'pa_intervention_valid': torch.tensor([True]),
                'advantages': torch.tensor([[1., -1., 0.] + [1.] * 9])},
                non_tensor_batch={'pa_direct_spans': [[{'token_indices': [0, 1]}, {'token_indices': [1, 2]}]]})
            metrics = ns['compute_pa_token_multipliers'](batch, config)
            return batch, metrics
        original, _ = run(0.)
        shifted, _ = run(.4)
        expected = torch.tensor([[.3, .3, .3] + [1.] * 9])
        torch.testing.assert_close(original.batch['pa_token_multiplier'], expected)
        torch.testing.assert_close(shifted.batch['pa_token_multiplier'], expected)
        # Execute the positive-only assignment from the actual actor implementation.
        actor = ast.parse((ROOT / 'verl/workers/actor/dp_actor.py').read_text())
        block = next(n for n in ast.walk(actor) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'self.config.use_pa_grpo' and len(n.body) > 2)
        context = {'torch': torch, 'self': SimpleNamespace(config=SimpleNamespace(use_pa_grpo=True)),
                   'model_inputs': original.batch, 'advantages': original.batch['advantages'],
                   'response_mask': original.batch['response_mask'], 'metrics': defaultdict(list)}
        exec(compile(ast.Module(body=[block], type_ignores=[]), 'actor', 'exec'), context)
        torch.testing.assert_close(context['advantages'], torch.tensor([[.3, -1., 0.] + [1.] * 9]))
        original.batch['pa_intervention_valid'][:] = False
        ns['compute_pa_token_multipliers'](original, config)
        self.assertTrue(torch.all(original.batch['pa_token_multiplier'] == 1))

    def test_exact_full_kl_and_directional_persistence(self):
        original = np.log([[.8, .2], [.2, .8]])
        cf = np.log([[.2, .8], [.8, .2]])
        efs, delta = token_metrics(original, cf, [0, 0])
        np.testing.assert_allclose(efs, [.6 * np.log(4)] * 2)
        self.assertLess(delta[0], 0)
        self.assertGreater(delta[1], 0)

    def test_residual_excludes_derived_claims(self):
        spans = [{'text': 'a', 'char_start': 0, 'char_end': 1, 'span_group': 'direct_vef'},
                 {'text': 'b', 'char_start': 1, 'char_end': 2, 'span_group': 'derived_vef'}]
        result = claim_metrics([1.] * 4, [-.3, 9., -.2, -.2], spans,
                               [(0, 1), (1, 2), (2, 3), (3, 4)], min_residual=2)
        self.assertEqual(result['residual_tokens'], 2)
        self.assertAlmostEqual(result['claims'][0]['corrected_persistence'], -.1)
        self.assertIsNone(claim_metrics([1.] * 4, [0.] * 4, spans,
                                       [(i, i+1) for i in range(4)], 3)['claims'][0]['corrected_persistence'])

    def test_question_macro_and_paired_population_guard(self):
        rows = [{'question_id': 'q1', 'claims': [{'efs': 1.}] * 9},
                {'question_id': 'q2', 'claims': [{'efs': 3.}]}]
        macro = question_macro(rows, 'efs')
        self.assertEqual(np.mean(list(macro.values())), 2.)
        result = paired_bootstrap(macro, {k: v+.1 for k, v in macro.items()}, draws=100)
        np.testing.assert_allclose(result['ci95'], [.1, .1])
        with self.assertRaises(ValueError):
            paired_bootstrap(macro, {})

    def test_annotation_helpers_and_ambiguous_alignment(self):
        helpers = load_helpers()
        self.assertIn('trace', helpers.build_prompt({'question': 'q'}, 'trace', 10))
        self.assertEqual(helpers.parse_json_content('```json\n{"spans": []}\n```'), {'spans': []})
        with self.assertRaises(ValueError):
            align_exact('a a', [{'text': 'a'}])
        self.assertEqual(align_exact('a a', [{'text': 'a', 'char_start': 2}])[0]['char_end'], 3)
        # Dynamic path loading is the exact mechanism used in online training.
        spec = importlib.util.spec_from_file_location('test_prompt_helpers', ROOT / 'pacg/annotation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.load_helpers().build_prompt))

    def test_interventions_are_deterministic_and_do_not_modify_source(self):
        from PIL import Image
        image = Image.new('RGB', (112, 112), 'white')
        a, b = intervene(image, 42), intervene(image, 42)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(np.asarray(image) == 255))
        self.assertFalse(np.array_equal(image, a))

    def test_accuracy_is_mean_not_pass_at_k(self):
        rows = [{'benchmark': 'math', 'question_id': 'q1', 'accuracy': x} for x in [0, 1]]
        self.assertEqual(summarize(rows)['accuracy_percent']['math'], 50.)

    def test_teacher_forcing_keeps_prefixes_and_scores_correct_target_positions(self):
        from PIL import Image
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': list(map(ord, text)), 'offset_mapping': [(i, i+1) for i in range(len(text))]}
            def decode(self, ids, **kwargs):
                return ''.join(map(chr, ids))
        class Processor:
            tokenizer = Tokenizer()
            def apply_chat_template(self, *args, **kwargs):
                return 'prompt'
            def __call__(self, images, **kwargs):
                return {'input_ids': torch.tensor([[4, 5]]), 'image_grid_thw': torch.tensor([[1, 8, 8]]),
                        'pixel_values': torch.tensor(float(np.asarray(images[0]).mean()) / 255)}
        class Model:
            device = torch.device('cpu')
            def __init__(self):
                self.inputs = []
            def __call__(self, input_ids, pixel_values, **kwargs):
                self.inputs.append(input_ids.clone())
                logits = torch.zeros((1, input_ids.shape[1], 128))
                logits[:, :, ord('a')] = pixel_values * 2
                return SimpleNamespace(logits=logits)
        model = Model()
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'image.png'
            Image.new('RGB', (112, 112), 'white').save(image)
            row = {'sample_id': 's', 'question_id': 'q', 'question': 'q', 'image': str(image),
                   'response': 'abcdefghijkl', 'spans': [{'text': 'a', 'span_group': 'direct_vef',
                      'char_start': 0, 'char_end': 1, 'evidence_status': 'unsupported'}]}
            args = SimpleNamespace(min_pixels=1, max_pixels=1000000, seed=42, patch_size=14,
                   black_prob=.5, operator='random_patch', min_changed_ratio=.005,
                   max_changed_ratio=.95, min_residual=8, max_length=6144, chunk_tokens=3, model='fake')
            dataset = SimpleNamespace(process_image=lambda path, *args: Image.open(path).convert('RGB'))
            with patch.dict(sys.modules, {'verl.utils.dataset': dataset}):
                result = score_row(model, Processor(), row, args)
            torch.testing.assert_close(model.inputs[0], model.inputs[1])
            self.assertEqual(model.inputs[0].tolist()[0], [4, 5] + list(map(ord, row['response'])))
            self.assertTrue(result['intervention_valid'])
            self.assertLess(result['claims'][0]['corrected_persistence'], 0)
            self.assertGreater(result['claims'][0]['efs'], 0)

    def test_no_private_infrastructure_and_syntax(self):
        for path in ROOT.rglob('*.py'):
            text = path.read_text()
            ast.parse(text)
            if 'tests' in path.relative_to(ROOT).parts:
                continue
            for forbidden in ('from vppo_eval', 'import oss2', '/Users/', 'alipay.com', 'oss://'):
                self.assertNotIn(forbidden, text, str(path))


if __name__ == '__main__':
    unittest.main()

