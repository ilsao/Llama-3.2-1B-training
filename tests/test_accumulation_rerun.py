"""Regression checks use stubs only; no tensor/model work on the login node."""
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest

from slurm.slurm_accumulation_timing import observe_update

ROOT = Path(__file__).resolve().parents[1]


class AccumulationRerunTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "slurm/accumulation_entry_v2.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "observe_loss")
        self.result = object()
        self.grad_enabled = True
        self.calls = []
        self.env = dict(original_loss=lambda model, batch: self.calls.append(model) or self.result,
                        torch=SimpleNamespace(is_grad_enabled=lambda: self.grad_enabled,
                            cuda=SimpleNamespace(max_memory_allocated=lambda: 100,
                                                 max_memory_reserved=lambda: 200)),
                        counters={"stats": {"unique_graphs": 2}},
                        time=SimpleNamespace(monotonic=lambda: 1),
                        micro_calls=0, accumulation=2, observations=[])
        exec(compile(ast.Module(body=[function], type_ignores=[]), "observer", "exec"), self.env)

    def test_eager_evaluation_callable_is_excluded(self):
        self.grad_enabled = False
        eager_forward = lambda **kwargs: None
        result = self.env["observe_loss"](eager_forward, {})
        self.assertIs(result, self.result)
        self.assertEqual(self.env["micro_calls"], 0)
        self.assertEqual(self.env["observations"], [])
        self.assertEqual(self.calls, [eager_forward])

    def test_eval_between_microbatches_does_not_advance_observer(self):
        model = SimpleNamespace(training=True)
        batch = {"input_ids": SimpleNamespace(shape=(1, 8192))}
        self.env["observe_loss"](model, batch)
        self.grad_enabled = False
        self.env["observe_loss"](lambda **kwargs: None, {})
        self.grad_enabled = True
        self.env["observe_loss"](model, batch)
        self.assertEqual(self.env["micro_calls"], 2)
        self.assertEqual(len(self.env["observations"]), 1)
        self.assertEqual(self.env["observations"][0]["step"], 1)

    def test_nonaccumulating_geometry_records_each_update(self):
        self.env["accumulation"] = 1
        self.env["observe_loss"](SimpleNamespace(training=True),
            {"input_ids": SimpleNamespace(shape=(2, 8192))})
        self.assertEqual(self.env["observations"][0]["microbatch_size"], 2)

    def test_cold_compile_does_not_force_premature_stop(self):
        perf = SimpleNamespace(enabled=True, settings={"model_compile": True},
                               recent_steps=deque(maxlen=20), metadata={})
        initial = observe_update(perf, 318., 0.)
        self.assertEqual(initial, 30.)
        estimate = observe_update(perf, 1.4, initial)
        self.assertEqual(estimate, 1.4)
        self.assertEqual(perf.metadata["cold_start_update_seconds"], 318.)
        self.assertEqual(observe_update(perf, 40., estimate), 40.)

    def test_uncompiled_timing_keeps_conservative_history(self):
        perf = SimpleNamespace(enabled=False)
        self.assertEqual(observe_update(perf, 1., 12.), 12.)
        with self.assertRaises(ValueError):
            observe_update(perf, float("nan"), 12.)


if __name__ == "__main__":
    unittest.main()
