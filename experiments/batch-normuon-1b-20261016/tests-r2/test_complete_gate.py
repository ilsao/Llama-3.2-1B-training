import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

LIVE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LIVE/'source-r2'))
from training_state import atomic_json
spec=importlib.util.spec_from_file_location('gate_controller',LIVE/'source-r2/slurm/batch1b_controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


class CompleteGate(unittest.TestCase):
    def test_partial_native_success_never_releases_research(self):
        with tempfile.TemporaryDirectory() as tmp:
            live=Path(tmp)
            atomic_json(live/'native-evidence-r2.json',dict(native_pass=True,unchanged_native_payload=True))
            state=dict(jobs={controller.VALIDATION:dict(status='reconciled',slurm_state='COMPLETED')})
            with patch.object(controller,'LIVE',live):
                self.assertFalse(controller.complete_validation_gate(state))
                result=dict(normuon_pass=False,execution='eager',unique_compiled_graphs=0,processed_training_inputs=65536)
                atomic_json(live/'normuon-result-r2.json',result)
                self.assertFalse(controller.complete_validation_gate(state))
                result['normuon_pass']=True;atomic_json(live/'normuon-result-r2.json',result)
                self.assertTrue(controller.complete_validation_gate(state))
                state['jobs'][controller.VALIDATION]['slurm_state']='TIMEOUT'
                self.assertFalse(controller.complete_validation_gate(state))
                state['jobs'][controller.VALIDATION]['slurm_state']='COMPLETED'
                result['unique_compiled_graphs']=1;atomic_json(live/'normuon-result-r2.json',result)
                self.assertFalse(controller.complete_validation_gate(state))


if __name__=='__main__':unittest.main()
