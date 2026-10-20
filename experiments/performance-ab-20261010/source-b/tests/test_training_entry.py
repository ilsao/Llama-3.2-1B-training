"""Launcher stop-code regression tests; never imports a real model or torch."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from slurm import training_entry


class TrainingEntryTests(unittest.TestCase):
    def check(self, original, expected):
        def fake_train(): raise SystemExit(original)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict('sys.modules',{'train':SimpleNamespace(main=fake_train)}),patch.dict(os.environ,{'JOB_RECORD':tmp,'RANK':'3'}):
                self.assertEqual(training_entry.main(),expected)
            self.assertEqual(json.loads((Path(tmp)/'worker-rank-3-exit.json').read_text())['exit_code'],original)

    def test_checkpointed_stop_succeeds_for_launcher_but_retains_75(self): self.check(75,0)
    def test_normal_completion_retains_zero(self): self.check(0,0)
    def test_real_training_failure_remains_a_failure(self): self.check(2,2)


if __name__=='__main__': unittest.main()
