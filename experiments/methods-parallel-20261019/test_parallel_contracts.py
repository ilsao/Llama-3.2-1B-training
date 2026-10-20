"""Small stdlib-only protocol/submission tests. Never import torch or contact Slurm."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
spec=importlib.util.spec_from_file_location('methods_parallel_cli',REPO/'slurm/methods_parallel.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
from training_state import atomic_json,read_json,load_config,ExperimentLedger
from ppl_contracts import batch_plan
from parallel_contracts import Scheduler

class ProtocolTests(unittest.TestCase):
 def setUp(self):self.matrix=read_json(HERE/'matrix.json')
 def test_exact_independent_token_paths(self):
  ids=set();paths=set();baseline=None
  for row in self.matrix['jobs']:
   cfg=load_config(HERE/row['config']);rows=batch_plan(cfg)
   geometry=[(r['start_tokens'],r['end_tokens'],r['width']) for r in rows]
   if baseline is None:baseline=geometry
   self.assertEqual(geometry,baseline)
   self.assertEqual(len(rows),4775);self.assertEqual(rows[-1]['end_tokens'],1000079360)
   self.assertFalse(row['depends_on']);self.assertEqual(row['gpus'],8);self.assertEqual(row['cpus'],8)
   self.assertEqual(cfg['max_training_tokens'],rows[-1]['end_tokens'])
   self.assertLessEqual(cfg['eval_steps'],len(rows)//10)
   ids.add(row['run_id']);paths.add(row['config'])
  self.assertEqual(len(ids),8);self.assertEqual(len(paths),8)
 def test_same_seed_shared_control(self):
  configs={p.stem:load_config(p) for p in (HERE/'configs').glob('*.json')}
  for seed in (42,):
   base=configs[f'N0-seed{seed}']
   for arm in ('D1','D025','A1','S1','K1','F1','F2','C1'):
    cfg=configs[f'{arm}-seed{seed}']
    for k in ('seed','model_revision','sequence_length','max_training_tokens','batch_switch_tokens','betas','clipping','max_grad_norm','initialization','learning_rate'):
     self.assertEqual(cfg[k],base[k])
 def test_schedule_free_no_second_warmup_or_terminal_zero(self):
  for arm in ('F1','F2'):
   cfg=load_config(HERE/f'configs/{arm}-seed42.json')
   opt=SimpleNamespace(param_groups=[dict(name='muon_in2048',lr=.001)])
   s=Scheduler(opt,cfg)
   self.assertAlmostEqual(opt.param_groups[0]['lr'],.001/170)
   for tokens in (170*131072,6358*131072,1000079360):
    s.set_tokens(tokens);self.assertEqual(opt.param_groups[0]['lr'],.001)
  cfg=load_config(HERE/'configs/N0-seed42.json')
  opt=SimpleNamespace(param_groups=[dict(name='muon',lr=.008614891542046706)])
  s=Scheduler(opt,cfg);s.set_tokens(1000079360);self.assertEqual(opt.param_groups[0]['lr'],0.)
 def test_budget_selection_and_no_scheduler_dependencies(self):
  rows=cli.select(self.matrix,'all','42');self.assertEqual(len(rows),8)
  plan=cli.plan(rows);self.assertEqual(plan['max_new_h200_hours'],128)
  self.assertFalse(plan['concurrency_throttle'])
  for row in rows:
   cmd=cli.command(row,'reservation-test');self.assertNotIn('--hold',cmd)
   self.assertFalse(any('--dependency' in x or '--array' in x for x in cmd))
 def test_single_seed_rejects_43(self):
  for seeds in ('43','42,43'):
   with self.assertRaises(ValueError):cli.select(self.matrix,'all',seeds)
 def test_historical_reference_is_not_a_job(self):
  from control_reference import validate
  reference=validate();self.assertEqual(reference['key'],'N0-seed42')
  self.assertEqual(reference['new_h200_hours'],0)
  self.assertAlmostEqual(reference['ppl_10k'],18.915596244218058)
  with self.assertRaises(ValueError):cli.select(self.matrix,'N0','42')
 def test_changed_config_rejected(self):
  cfg=load_config(HERE/'configs/N0-seed42.json');cfg['learning_rate']*=2
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'cfg.json';atomic_json(p,cfg)
   with self.assertRaises(ValueError):load_config(p)
 def test_no_torch_import(self):self.assertNotIn('torch',sys.modules)

class SubmissionTests(unittest.TestCase):
 def setUp(self):
  self.td=tempfile.TemporaryDirectory();self.root=Path(self.td.name)
  self.patches=[patch.object(cli,'DEFAULT_ROOT',self.root),patch.object(cli,'RECORDS',self.root/'submissions')]
  for p in self.patches:p.start()
  self.historical={f'formal-{i}':{'category':'full','config_digest':str(i),'max_h200_hours':64} for i in range(17)}
  atomic_json(self.root/'experiment-ledger.json',{'runs':self.historical,'jobs':{}})
  self.row=cli.select(read_json(HERE/'matrix.json'),'D1','42')[0]
 def tearDown(self):
  for p in self.patches:p.stop()
  self.td.cleanup()
 def test_reserve_submit_bind_preserves_history(self):
  def fake(command,**kwargs):
   data=read_json(self.root/'experiment-ledger.json')
   if command[0]=='sbatch':
    self.assertIn(self.row['run_id'],data['runs']);self.assertNotIn('--hold',command)
    self.assertEqual(len(data['jobs']),1)
    return SimpleNamespace(returncode=0,stdout='991001\n',stderr='')
   self.fail('Unexpected Slurm mutation: '+str(command))
  with patch.object(cli.subprocess,'run',side_effect=fake),patch('builtins.print'):
   cli.submit([self.row],16)
  data=read_json(self.root/'experiment-ledger.json')
  for key,value in self.historical.items():self.assertEqual(data['runs'][key],value)
  self.assertEqual(data['total_h200_hours_reserved_or_measured'],16)
  with self.assertRaises(ValueError):cli.submit([self.row],16)
 def test_uncertain_submission_retains_reservation_without_retry(self):
  result=SimpleNamespace(returncode=1,stdout='',stderr='simulated submission uncertainty')
  with patch.object(cli.subprocess,'run',return_value=result) as process:
   with self.assertRaises(RuntimeError):cli.submit([self.row],16)
   self.assertEqual(process.call_count,1)
  data=read_json(self.root/'experiment-ledger.json');self.assertEqual(len(data['jobs']),1)
  self.assertEqual(data['total_h200_hours_reserved_or_measured'],16)
  state=read_json(cli.RECORDS/(self.row['key']+'.json'));self.assertTrue(state['recovery_required'])
  with self.assertRaises(ValueError):cli.submit([self.row],16)
 def test_wrong_budget_cannot_reserve(self):
  with self.assertRaises(ValueError):cli.submit([self.row],15)
  self.assertEqual(read_json(self.root/'experiment-ledger.json')['jobs'],{})

if __name__=='__main__':unittest.main()
