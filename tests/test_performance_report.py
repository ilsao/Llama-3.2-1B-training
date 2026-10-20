"""Synthetic report contracts, without model loading or dataset processing."""
import copy
import unittest
from slurm.performance_report import measurements, compare
from training_state import PROJECT, load_config


class PerformanceReportTests(unittest.TestCase):
    def setUp(self):
        self.summary = dict(global_step=120,attempted_updates=120,total_tokens_seen=15728640,
            budget_tokens_charged=15728640,stop_reason='training_token_target',
            last_evaluation={'eval/perplexity':20.},configuration={'seed':42},
            initialization={'sample_signature':'same'},data_provenance={
                'fingerprint':'same','evaluation_indices_sha256':'same','pack_settings':{'mask':True}})
        self.rows = [dict(step=i,update_seconds=2.,compute_seconds=1.,optimizer_seconds=.2)
                     for i in range(1,121)]

    def test_end_to_end_not_compute_only(self):
        result = measurements(self.summary,self.rows)
        self.assertEqual(result['end_to_end_tokens_per_second'],65536)
        self.assertEqual(result['compute_tokens_per_second'],131072)
        self.assertEqual(result['measured_updates'],100)

    def test_warmup_cost_retained_separately(self):
        self.rows[0]['update_seconds']=102.
        self.assertEqual(measurements(self.summary,self.rows)['cold_start_excess_seconds_estimate'],100.)

    def test_incomplete_and_repeated_steps_are_rejected(self):
        for rows in (self.rows[:-1], self.rows[:-1]+[self.rows[0]]):
            with self.assertRaises(ValueError): measurements(self.summary,rows)
        bad=copy.deepcopy(self.summary); bad['budget_tokens_charged']+=1
        with self.assertRaises(ValueError): measurements(bad,self.rows)

    def test_recipe_and_initialization_are_matched(self):
        b=copy.deepcopy(self.summary); b['configuration']['performance']={'enabled':True}
        compare(self.summary,b)
        b['initialization']['sample_signature']='different'
        with self.assertRaises(ValueError): compare(self.summary,b)

    def test_actual_configs_preserve_original_horizon(self):
        a=load_config(PROJECT/'configs/performance-ab-20261010/A.json')
        b=load_config(PROJECT/'configs/performance-ab-20261010/B.json')
        b.pop('performance')
        self.assertEqual(a,b)
        self.assertEqual(a['warmup_steps'],170)
        self.assertEqual(a['scheduler_steps'],15259)
        self.assertEqual(a['target_training_tokens'],15728640)


if __name__=='__main__': unittest.main()
