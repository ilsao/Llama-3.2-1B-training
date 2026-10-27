"""Metadata-only checks: no model imports, workers or cluster allocations."""
import copy
import unittest
from recipes.sota6b.__main__ import ROOT, check_comparison, read, verify


class SotaRecipeTests(unittest.TestCase):
    def setUp(self):
        self.comparison = read(ROOT / 'results/6b-comparison.json')

    def test_selected_recipe_and_all_sealed_dependencies(self):
        manifest, best, count = verify()
        self.assertEqual(manifest['formal_id'], 19)
        self.assertEqual(best['run_id'], 'kd6b-r5-20261020-seed42')
        self.assertGreater(count, 0)

    def test_newest_run_is_not_automatically_selected(self):
        self.comparison['selected_run'] = self.comparison['runs'][-1]['run_id']
        with self.assertRaisesRegex(ValueError, 'lowest 10k PPL'):
            check_comparison(self.comparison)

    def test_arithmetic_mean_of_perplexities_is_rejected(self):
        row = self.comparison['runs'][0]
        row['combined_10k_ppl'] = (row['fixed_1k_ppl'] + row['additional_9k_ppl']) / 2
        with self.assertRaisesRegex(ValueError, 'aggregated NLL'):
            check_comparison(self.comparison)

    def test_unmatched_training_or_evaluation_tokens_are_rejected(self):
        for field in ('training_tokens', 'prediction_tokens'):
            with self.subTest(field=field):
                comparison = copy.deepcopy(self.comparison)
                comparison['runs'][0][field] -= 1
                with self.assertRaisesRegex(ValueError, 'Unmatched'):
                    check_comparison(comparison)

    def test_wrong_evaluation_scope_is_rejected(self):
        self.comparison['evaluation_documents'] = 50000
        with self.assertRaisesRegex(ValueError, 'scope'):
            check_comparison(self.comparison)


if __name__ == '__main__':
    unittest.main()
