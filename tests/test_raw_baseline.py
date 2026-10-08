"""Metadata and small packing checks only; no model imports or Slurm allocations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from raw_baseline import ProgressiveDataset, DataUnavailable, validate_append
from training_state import atomic_json
from dataset import TrainingPacker, pack_evaluation_document


class RawBaselineTests(unittest.TestCase):
    def test_raw_producer_running_does_not_duplicate_or_wait_for_clean(self):
        from slurm import auto_baseline as auto
        cfg=auto.load_campaign(auto.PROJECT/'configs/auto-baseline.json')
        state={'jobs':[{'phase':'data-clean','status':'submitted'},
                       {'phase':'raw-prepare','status':'submitted','kind':'preprocessing'}]}
        with patch.object(auto,'submit_worker') as submit, patch.object(auto,'manifest_ready') as ready:
            auto.data_advance(cfg,state)
            submit.assert_not_called(); ready.assert_not_called()
        self.assertEqual(auto.budget([{'kind':'preprocessing','status':'submitted','requested_seconds':21600,'gpu_count':1}])['h200_hours'],0)

    def test_published_prefix_cannot_change_or_shrink(self):
        a={'path':'a','sha256':'old','rows':3}
        validate_append([a],[a,{'path':'b','rows':2}])
        with self.assertRaises(ValueError): validate_append([a],[])
        with self.assertRaises(ValueError): validate_append([a],[{**a,'sha256':'new'}])

    def make_stream(self, root, files, status='active', receipt=None):
        (root/'raw').mkdir(exist_ok=True)
        atomic_json(root/'raw/progress.json',{'train_files':files,'status':status})
        path=root/'receipt.json'
        if receipt is not None: atomic_json(path,{'files':receipt,'cursor':3})
        return ProgressiveDataset(root,100,path,0)

    def test_consumer_checks_resume_prefix(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); a={'path':'a','rows':3}
            with self.assertRaises(ValueError): self.make_stream(root,[{**a,'rows':4}],receipt=[a])

    def test_pause_and_failure_stop_instead_of_reset(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for status in ('active','paused','failed','complete'):
                stream=self.make_stream(root,[],status)
                with self.assertRaises(DataUnavailable): stream[1]

    def test_checkpoint_crash_preserves_conservative_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); a={'path':'a','rows':3}; b={'path':'b','rows':3}
            stream=self.make_stream(root,[a,b],receipt=[a,b])
            stream.commit_receipt(2)
            self.assertEqual(json.loads((root/'receipt.json').read_text())['files'],[a,b])

    def test_variable_shard_offsets_and_append(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); files=[{'path':'a','rows':3},{'path':'b','rows':2}]
            stream=self.make_stream(root,files)
            stream.cache={0:['a0','a1','a2'],1:['b0','b1']}
            self.assertEqual(stream[2],'a2'); self.assertEqual(stream[3],'b0')
            stream.commit_receipt(5)
            atomic_json(root/'raw/progress.json',{'train_files':files+[{'path':'c','rows':7}],'status':'active'})
            stream.refresh(); self.assertEqual(stream.ends,[3,5,12])

    def test_progressive_packing_resumes_exact_carry(self):
        documents=[(10,[1,3,4,5,2]),(20,[1,6,7,8,9,2]),(30,[1,10,11,2])]
        whole=TrainingPacker(8,2); expected=[]
        for row,tokens in documents: expected.extend(whole.push(row,tokens))
        expected.extend(whole.finish())
        first=TrainingPacker(8,2); actual=list(first.push(*documents[0]))
        resumed=TrainingPacker(8,2,json.loads(json.dumps(first.state())))
        for row,tokens in documents[1:]: actual.extend(resumed.push(row,tokens))
        actual.extend(resumed.finish())
        self.assertEqual(actual,expected)
        self.assertEqual(sum(sum(x['attention_mask']) for x in actual),sum(len(x[1]) for x in documents))
        for sequence in actual:
            self.assertEqual(sequence['labels'][0],-100)
            for span in sequence['document_spans']:
                if span['document_start']==0:
                    self.assertEqual(sequence['labels'][span['sequence_start']],-100)

    def test_full_evaluation_document_targets_count_once(self):
        doc={'row_index':999,'evaluation_slot':0,'tokens':list(range(23))}
        rows=list(pack_evaluation_document(doc,8,2))
        self.assertEqual(sum(sum(v!=-100 for v in r['labels'][1:]) for r in rows),22)
        self.assertTrue(all(s['row_index']==999 for r in rows for s in r['document_spans']))


if __name__=='__main__': unittest.main()
