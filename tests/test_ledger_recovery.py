"""Metadata-only recovery checks; every fixture stays in a temporary root."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('recovery',Path(__file__).resolve().parents[1]/'slurm/recover_training_ledger.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class RecoveryTests(unittest.TestCase):
    def test_merge_keeps_reconciled_cost_and_rejects_conflicting_identity(self):
        jobs={};r.merge_job(jobs,'1',dict(run_id='a',h200_hours=1,status='reconciled'))
        r.merge_job(jobs,'1',dict(run_id='a',status='submitted',reserved_h200_hours=2))
        self.assertEqual(jobs['1']['status'],'reconciled');self.assertEqual(jobs['1']['h200_hours'],1)
        with self.assertRaises(ValueError):r.merge_job(jobs,'1',dict(run_id='b'))
    def test_apply_requires_unchanged_empty_ledger_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);live=p/'experiment-ledger.json';r.atomic_json(live,dict(jobs={},runs={}))
            candidate=dict(jobs={'1':{'h200_hours':1}},runs={'a':{}})
            old=r.file_digest(live);r.apply(p,candidate,old,p/'backup.json')
            self.assertEqual(r.read_json(live),candidate);self.assertEqual(r.read_json(p/'backup.json')['jobs'],{})
            with self.assertRaises(ValueError):r.apply(p,candidate,r.file_digest(live),p/'other.json')
    def test_apply_refuses_concurrent_changes(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);r.atomic_json(p/'experiment-ledger.json',dict(jobs={},runs={}))
            with self.assertRaises(ValueError):r.apply(p,{},'incorrect',p/'backup.json')
    def test_actual_cost_requires_terminal_job_and_h200_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);d=p/'logs/job-1';d.mkdir(parents=True)
            (d/'gpus.txt').write_text('NVIDIA H200, GPU-unique\n')
            row='1|FAILED|60|gres/gpu=1|start|end|node\n'
            result=r.parse_accounting(row,p)['1'];self.assertAlmostEqual(result['h200_hours'],1/60)
            with self.assertRaises(ValueError):r.parse_accounting(row.replace('FAILED','RUNNING'),p)
            (d/'gpus.txt').write_text('NVIDIA A100, GPU-unique\n')
            with self.assertRaises(ValueError):r.parse_accounting(row,p)
    def test_inherited_cost_is_not_charged_twice_to_global_total(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);root=p/'work';project=p/'repo';project.mkdir()
            run=root/'runs/experiments/branch';run.mkdir(parents=True)
            r.atomic_json(run/'resource-accounting.json',dict(run_id='branch',category='pilot',max_h200_hours=10,inherited_h200_hours=2,jobs=[dict(job_id='1',run_id='branch',h200_hours=1,reserved_h200_hours=2,gpu_count=8,status='reconciled')]))
            candidate,_=r.build(root,project,{})
            self.assertEqual(candidate['total_h200_hours_measured'],1)
            self.assertEqual(candidate['runs']['branch']['logical_h200_hours_measured'],3)

if __name__=='__main__':unittest.main()
