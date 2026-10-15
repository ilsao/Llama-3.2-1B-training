"""Uncertain submissions retain one reservation; recovery never resubmits."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

LIVE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LIVE/'source-r1'))
from training_state import ExperimentLedger,atomic_json,read_json
from batch1b_contracts import STUDY

spec=importlib.util.spec_from_file_location('study_controller',LIVE/'source-r1/slurm/batch1b_controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


class Submission(unittest.TestCase):
    def test_timeout_is_reserved_once_and_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);live=base/'campaign';root=base/'storage'
            (live/'configs').mkdir(parents=True);root.mkdir()
            atomic_json(root/'experiment-ledger.json',dict(jobs={},runs={}))
            atomic_json(live/'plan.json',dict(excluded_nodes=[]))
            atomic_json(live/'validation-r1-authorization.json',dict(approved=True))
            cfg=json.loads((LIVE/'configs/control.json').read_text())
            atomic_json(live/'configs/control.json',cfg)
            state=dict(stage='test',jobs={})
            with patch.multiple(controller,LIVE=live,DEFAULT_ROOT=root,STATE=live/'state.json'),patch.object(controller,'check'):
                with patch.object(controller.subprocess,'run',side_effect=subprocess.TimeoutExpired('sbatch',30)) as runner:
                    controller.submit(state,'control',7200,8,'train')
                    controller.submit(state,'control',7200,8,'train')
                    self.assertEqual(runner.call_count,1)
                ledger=read_json(root/'experiment-ledger.json')
                self.assertEqual(len(ledger['jobs']),1)
                self.assertEqual(state['jobs']['control']['status'],'submission_uncertain')
                self.assertEqual(ledger['total_h200_hours_reserved_or_measured'],16)
                with patch.object(controller,'find_submission_job',return_value='777'),patch.object(controller.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'777|RUNNING|\n','')):
                    controller.reconcile(state)
                ledger=read_json(root/'experiment-ledger.json')
                self.assertEqual(set(ledger['jobs']),{'777'})
                self.assertEqual(ledger['jobs']['777']['run_id'],STUDY+'-control')
                self.assertEqual(ledger['total_h200_hours_reserved_or_measured'],16)
                self.assertEqual(state['jobs']['control']['job_id'],'777')


if __name__=='__main__':unittest.main()
