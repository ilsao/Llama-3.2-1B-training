"""Lightweight checks for the train-only16-CPU execution override."""
import os
from unittest import TestCase
from unittest.mock import patch
from filtering_resources import worker_count, training_worker_count, evaluation_worker_count


class ResourceTests(TestCase):
    config=dict(preprocessing_workers=7,score_workers=7)

    def test_default_allocation_preserves_original_workers(self):
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'8','FILTERING_TRAIN_WORKERS':'7'}):
            self.assertEqual(training_worker_count(self.config),7)
            self.assertEqual(worker_count(self.config),7)

    def test_sixteen_cpu_override_only_expands_training_workers(self):
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'16','FILTERING_TRAIN_WORKERS':'15'}):
            self.assertEqual(training_worker_count(self.config),15)
            self.assertEqual(worker_count(self.config),7)
            self.assertEqual(worker_count(self.config,scoring=True),7)
            self.assertEqual(worker_count(dict(preprocessing_workers=3)),3)

    def test_twelve_cpu_fallback_preserves_evaluation_workers(self):
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'12','FILTERING_TRAIN_WORKERS':'11'}):
            self.assertEqual(training_worker_count(self.config),11)
            self.assertEqual(worker_count(self.config),7)

    def test_evaluation_override_works_without_expanding_training(self):
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'12','FILTERING_TRAIN_WORKERS':'7','FILTERING_EVAL_WORKERS':'11'}):
            self.assertEqual(training_worker_count(self.config),7)
            self.assertEqual(evaluation_worker_count(self.config),11)
            self.assertEqual(worker_count(self.config,scoring=True),7)
        with patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':'8','FILTERING_TRAIN_WORKERS':'7','FILTERING_EVAL_WORKERS':'11'}):
            with self.assertRaises(ValueError):evaluation_worker_count(self.config)

    def test_unapproved_geometry_fails_closed(self):
        for cpus,workers in (('16','7'),('8','15'),('32','15'),('16','16')):
            with self.subTest(cpus=cpus,workers=workers),patch.dict(os.environ,{'SLURM_CPUS_PER_TASK':cpus,'FILTERING_TRAIN_WORKERS':workers}):
                with self.assertRaises(ValueError):training_worker_count(self.config)
