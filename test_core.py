"""Synthetic tests only; no model outputs or paper measurements."""
import math
import unittest
from annotation import annotate, route
from metrics import token_scores, claim_metrics, reference_summary, benchmark_accuracy, nine_benchmark_mean
from rl import pacg_credit, clipped_token_mean_loss


class CoreTests(unittest.TestCase):
    def test_identical_distributions(self):
        efs, delta = token_scores([[2., 1.]], [[2., 1.]], [0])
        self.assertEqual(efs, [0.])
        self.assertEqual(delta, [0.])

    def test_direction_and_calibration(self):
        efs, delta = token_scores([[math.log(.8), math.log(.2)]],
                                  [[math.log(.4), math.log(.6)]], [0])
        self.assertAlmostEqual(delta[0], math.log(.5))
        self.assertGreater(efs[0], 0)
        result = claim_metrics([1.] * 10, [-.4, -.4] + [-.1]*8, [0, 1], range(2, 10))
        self.assertAlmostEqual(result['corrected_persistence'], -.3)
        self.assertIsNone(claim_metrics([0.], [0.], [0], []))

    def test_gate_sign_and_fallback(self):
        base = [2., -2., 0.] + [1.]*8
        credit, gates = pacg_credit(base, [.2]*3 + [0.]*8, [[0, 1]], range(3, 11), [True]*11)
        self.assertEqual(credit, [.6, -2., 0.] + [1.]*8)
        self.assertEqual(gates[:2], [.3, .3])
        self.assertEqual(pacg_credit(base, [0.]*11, [[0]], [], [True]*11)[0], base)
        self.assertEqual(pacg_credit(base, [0.]*11, [[0]], range(3, 11), [True]*11, valid=False)[0], base)

    def test_retraction_and_overlap(self):
        credit, _ = pacg_credit([1.]*10, [-.2]*2+[0.]*8, [[0, 1]], range(2, 10), [True]*10)
        self.assertEqual(credit, [1.]*10)
        _, gates = pacg_credit([1.]*11, [0., .2, .2]+[0.]*8,
                               [[0, 1], [1, 2]], range(3, 11), [True]*11)
        self.assertEqual(gates[1], .3)

    def test_routing_ignores_support(self):
        spans = [{'text': 'Red', 'start': 0, 'end': 3, 'span_group': 'direct_vef',
                  'evidence_status': 'contradicted'}]
        direct, residual, valid = route('Red ball', [(0, 3), (4, 8)], spans, [True, True])
        self.assertEqual((direct, residual, valid), ([[0]], [1], True))
        spans[0]['end'] = 4
        self.assertFalse(route('Red ball', [(0, 3), (4, 8)], spans, [True, True])[2])

    def test_request_has_no_gold(self):
        def router(request):
            self.assertEqual(set(request), {'instruction', 'question', 'trace', 'images'})
            return {'spans': []}
        self.assertEqual(annotate(router, 'Count?', 'Two.', []), [])

    def test_macro_not_claim_pooling(self):
        rows = [dict(question_id='q1', rollout_id=0, corrected_persistence=0., evidence_status='unsupported')]*3
        rows += [dict(question_id='q2', rollout_id=0, corrected_persistence=-1., evidence_status='contradicted')]
        self.assertEqual(reference_summary(rows, unsupported_only=True)['mean'], -.5)
        self.assertIsNone(reference_summary([])['mean'])

    def test_accuracy_not_pass(self):
        rows = [dict(question_id='q', rollout_id=i, correct=int(i == 0)) for i in range(8)]
        self.assertEqual(benchmark_accuracy(rows), 12.5)
        self.assertEqual(nine_benchmark_mean({str(i): 50. for i in range(9)}), 50.)
        with self.assertRaises(ValueError):
            benchmark_accuracy(rows[:-1])

    def test_clipped_objective(self):
        self.assertAlmostEqual(clipped_token_mean_loss([2., .5], [1., -1.], [True, True]), -.24)
        self.assertEqual(clipped_token_mean_loss([1., 2.], [1., 100.], [True, False]), -1.)


if __name__ == '__main__':
    unittest.main()
