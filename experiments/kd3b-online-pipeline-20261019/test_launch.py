"""Small stdlib-only submission/accounting checks; no model imports."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('kd_launch', REPO/'slurm/kd3b_online_pipeline.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
CFG = launch.load_config(launch.LIVE/'config.json')


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.live = self.root/'review'
        self.live.mkdir()
        (self.live/'execution-seal.json').write_text('{}')
        ledger = dict(runs={f'formal-{i}':dict(category='full') for i in range(17)}, jobs={
            'old':dict(run_id='methods-parallel-20261019-failed', h200_hours=.25,
                       reserved_h200_hours=16, status='reconciled')})
        (self.root/'experiment-ledger.json').write_text(json.dumps(ledger))
        self.patches = [patch.object(launch,'LIVE',self.live),
                        patch.object(launch,'DEFAULT_ROOT',self.root),
                        patch.object(launch,'check',return_value=CFG)]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def test_reserve_before_submit_bind_and_preserve_costs(self):
        def sbatch(command, **kwargs):
            ledger=json.loads((self.root/'experiment-ledger.json').read_text())
            pending=[j for j in ledger['jobs'].values() if j['run_id']==launch.RUN]
            self.assertEqual(len(pending),1)
            self.assertEqual(pending[0]['status'],'reserved')
            return subprocess.CompletedProcess(command,0,'98765\n','')
        with patch.object(launch.subprocess,'run',side_effect=sbatch) as call:
            launch.submit()
            with self.assertRaises(ValueError):launch.submit()
            self.assertEqual(call.call_count,1)
        ledger=json.loads((self.root/'experiment-ledger.json').read_text())
        self.assertEqual(sum(r['category']=='full' for r in ledger['runs'].values()),17)
        self.assertEqual(ledger['jobs']['old']['h200_hours'],.25)
        self.assertEqual(ledger['jobs']['98765']['reserved_h200_hours'],16)

    def test_uncertain_submission_keeps_reservation_without_retry(self):
        with patch.object(launch.subprocess,'run',side_effect=subprocess.TimeoutExpired('sbatch',45)) as call:
            with self.assertRaises(RuntimeError):launch.submit()
            with self.assertRaises(ValueError):launch.submit()
            self.assertEqual(call.call_count,1)
        ledger=json.loads((self.root/'experiment-ledger.json').read_text())
        self.assertEqual(len(ledger['jobs']),2)
        self.assertTrue(json.loads((self.live/'submission.json').read_text())['recovery_required'])

    def test_study_cap_blocks_before_reservation(self):
        ledger=json.loads((self.root/'experiment-ledger.json').read_text())
        ledger['jobs']['old']['h200_hours']=113
        (self.root/'experiment-ledger.json').write_text(json.dumps(ledger))
        with patch.object(launch.subprocess,'run') as call:
            with self.assertRaises(ValueError):launch.submit()
            call.assert_not_called()
        self.assertFalse((self.live/'submission.json').exists())


if __name__=='__main__':unittest.main()
