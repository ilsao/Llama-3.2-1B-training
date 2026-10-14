"""Pure end-to-end orchestration simulation; no Slurm, torch or data calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('horizon_controller_under_test',ROOT/'slurm/horizon_campaign.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def test_full_stage_graph_has_exactly_eighteen_distinct_allocations(self):
        c=controller
        with tempfile.TemporaryDirectory() as d:
            tmp=Path(d);(tmp/'results').mkdir();(tmp/'to_human').mkdir()
            submitted=[]
            def fake_submit(s,key,name,path,cfg,minutes,mode='fresh',pause=0,parent='-',full_eval=False,test_only=False):
                if key in s['jobs']:return
                submitted.append((key,name,minutes,mode,pause,parent,full_eval))
                s['jobs'][key]=dict(run_id=c.CAMPAIGN+'-'+name,status='reconciled',slurm_state='COMPLETED',pause=pause,
                    accounting=dict(h200_hours=8*minutes/60))
                if full_eval:
                    run=tmp/'runs/experiments'/(c.CAMPAIGN+'-'+name);run.mkdir(parents=True,exist_ok=True)
                    (run/'full-evaluation.json').write_text(json.dumps({'training_global_step':15259,'eval/documents':50000,
                        'eval/perplexity':17.0,'eval/nll':2.8e6,'eval/prediction_tokens':1000000}))
            def summary(run_id,pause=0):
                return dict(data_provenance={'locked':'same'},last_evaluation={'eval/perplexity':20.,'eval/nll':3e6,'eval/prediction_tokens':1000000},
                    total_tokens_seen=700055552,wandb_url='test')
            with patch.object(c,'require_owner',return_value=True),patch.object(c,'DIRECTORY',tmp),patch.object(c,'STATE',tmp/'state.json'),patch.object(c,'DISABLED',tmp/'disabled'), \
                 patch.object(c,'CONFIGS',tmp/'configs'),patch.object(c,'REPO',tmp),patch.object(c,'DEFAULT_ROOT',tmp), \
                 patch.object(c,'plan',return_value=json.loads((ROOT/'experiments/horizon-wsd-20261014/plan.json').read_text())), \
                 patch.object(c,'check'),patch.object(c,'reconcile'),patch.object(c,'submit',side_effect=fake_submit), \
                 patch.object(c,'completed',side_effect=lambda s,keys:all(k in s['jobs'] for k in keys)), \
                 patch.object(c,'verify_job',side_effect=summary):
                c.run_once()
                final=json.loads((tmp/'state.json').read_text())
            self.assertEqual(final['stage'],'complete')
            self.assertEqual(len(submitted),18)
            self.assertEqual(len({row[0] for row in submitted}),18)
            self.assertAlmostEqual(sum(row[2]*8/60 for row in submitted),616/3)
            self.assertEqual(sum(row[3]=='resume' for row in submitted),2)
            self.assertEqual(sum(row[5]!='-' for row in submitted),5)
            self.assertEqual(sum(row[6] for row in submitted),2)
            self.assertTrue((tmp/'disabled').exists())

    def test_disabled_cycle_reconciles_without_submitting(self):
        c=controller
        with tempfile.TemporaryDirectory() as d:
            tmp=Path(d);stop=tmp/'disabled';stop.touch()
            with patch.object(c,'require_owner',return_value=True),patch.object(c,'DIRECTORY',tmp),patch.object(c,'STATE',tmp/'state.json'),patch.object(c,'DISABLED',stop), \
                 patch.object(c,'reconcile') as reconcile,patch.object(c,'submit') as submit,patch.object(c,'report'):
                c.run_once()
                reconcile.assert_called_once();submit.assert_not_called()


if __name__=='__main__':unittest.main()
