import tempfile
import unittest
from pathlib import Path
import monitor


class ComparisonTests(unittest.TestCase):
    def row(self, tokens=100, identity='fixed', count=10, nll=20):
        return dict(identity=identity,evaluations=[dict(tokens=tokens,predictions=count,
                    ppl=7.389,step=1,nll=nll)])

    def test_matched_tokens_subsets_and_counts_are_required(self):
        left=self.row()
        for other in (self.row(tokens=101),self.row(identity='different'),self.row(count=11)):
            self.assertEqual(monitor.matched(left,other),[])
        self.assertAlmostEqual(monitor.matched(left,self.row(nll=21))[0]['delta_nll'],-.1)

    def test_incomplete_run_cannot_enter_endpoint_table(self):
        self.assertIsNone(monitor.endpoint(dict(training_token_target_reached=False),{'present':True}))

    def test_confirmation_must_match_checkpoint(self):
        summary=dict(training_token_target_reached=True,total_tokens_seen=monitor.TARGET,
            global_step=4775,last_eval_step=4775,export='correct',preparation_fingerprint='same',
            last_evaluation={'eval/prediction_tokens':1479302,'eval/nll':3000000})
        confirm=dict(global_step=4775,export='wrong',preparation_fingerprint='same',
                     **{'eval/prediction_tokens':13666812,'eval/nll':28000000})
        with self.assertRaises(ValueError):monitor.endpoint(summary,confirm)
        confirm['export']='correct'
        self.assertAlmostEqual(monitor.endpoint(summary,confirm),monitor.math.exp(31000000/15146114))

    def test_live_partial_final_line_does_not_hide_valid_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metrics.jsonl'
            path.write_text('{"train/tokens_per_second": 123, "train/global_step": 10}\n{"partial":')
            _,last=monitor.metrics(path)
            self.assertEqual(last['train/global_step'],10)


if __name__=='__main__':unittest.main()
