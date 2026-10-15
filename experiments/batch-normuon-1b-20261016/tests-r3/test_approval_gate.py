import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

LIVE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LIVE/'source-r3'))
from training_state import atomic_json
spec=importlib.util.spec_from_file_location('corrected_controller',LIVE/'source-r3/slurm/batch1b_controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


class ApprovalGate(unittest.TestCase):
    def test_no_submission_or_cron_before_explicit_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            live=Path(tmp)
            atomic_json(live/'recovery-r3-authorization.json',dict(approved=False))
            with patch.object(controller,'LIVE',live),patch.object(controller,'check'),patch.object(controller,'ExperimentLedger') as ledger,patch.object(controller.subprocess,'run') as process:
                with self.assertRaisesRegex(ValueError,'explicit approval'):
                    controller.submit(dict(jobs={}),controller.VALIDATION,780,8,'validation')
                with self.assertRaisesRegex(ValueError,'explicit approval'):
                    controller.install()
                ledger.assert_not_called();process.assert_not_called()


if __name__=='__main__':unittest.main()
