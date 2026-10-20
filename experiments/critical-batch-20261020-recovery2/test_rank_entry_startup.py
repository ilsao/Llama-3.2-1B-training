"""Exercise the worker entrypoint without torch/GPU, including missing export."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import rank_entry


class WorkerStartupTests(unittest.TestCase):
    def exercise(self,explicit_record):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            env=dict(SLURM_JOB_ID='1234',SLURMD_NODENAME='test',LOCAL_RANK='0',DATA_ROOT=str(root))
            if explicit_record:env['JOB_RECORD']=str(root/'explicit-record')
            with patch.dict(os.environ,env,clear=True), \
                 patch.object(rank_entry,'private_paths',return_value={
                     'TORCHINDUCTOR_CACHE_DIR':str(root/'inductor'),
                     'TRITON_CACHE_DIR':str(root/'triton'),
                     'CUDA_CACHE_PATH':str(root/'cuda'), 'TMPDIR':str(root/'tmp')}), \
                 patch.object(rank_entry.sys,'argv',['rank_entry.py','compile_gate.py']), \
                 patch.object(rank_entry.runpy,'run_path') as run:
                rank_entry.main()
                run.assert_called_once()
                self.assertEqual(Path(run.call_args.args[0]).name,'compile_gate.py')
                record=root/'explicit-record' if explicit_record else root/'logs/job-1234'
                data=json.loads((record/'private-cache-rank0.json').read_text())
                self.assertEqual(data['TRITON_CACHE_DIR'],str(root/'triton'))
                self.assertEqual(os.environ['CUDA_CACHE_PATH'],str(root/'cuda'))

    def test_missing_job_record_export_reaches_gate(self):self.exercise(False)
    def test_explicit_job_record_reaches_gate(self):self.exercise(True)


if __name__=='__main__':unittest.main()
