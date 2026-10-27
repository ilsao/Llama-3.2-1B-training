"""Inspect and verify the selected recipe with only the Python standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_comparison(comparison):
    rows = comparison['runs']
    require(bool(rows), 'No completed runs to compare')
    require(len({r['run_id'] for r in rows}) == len(rows), 'Duplicate run identities')
    require(comparison['selection_metric'] == 'combined_10k_ppl', 'Unexpected ranking metric')
    require(comparison['evaluation_documents'] == 10000, 'Evaluation scope changed')
    for row in rows:
        require(row['training_tokens'] == 6000082944, 'Unmatched training endpoint')
        require(row['prediction_tokens'] == 15146114, 'Unmatched evaluation tokens')
        expected = math.exp(row['nll'] / row['prediction_tokens'])
        require(math.isfinite(expected) and math.isclose(
            row['combined_10k_ppl'], expected, rel_tol=1e-12), 'PPL must use aggregated NLL')
    best = min(rows, key=lambda row: row['combined_10k_ppl'])
    require(comparison['selected_run'] == best['run_id'], 'Selected run is not the lowest 10k PPL')
    return best


def verify(root=ROOT):
    manifest = read(root / 'recipes/sota6b/manifest.json')
    seal = read(root / manifest['source_seal'])
    require(seal['configuration_digest'] == manifest['configuration_digest'], 'Seal/config identity mismatch')
    for name, expected in seal['sha256'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        require(actual == expected, f'Sealed source changed: {name}')
    cfg = read(root / manifest['configuration'])
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == manifest['configuration_digest'], 'Resolved configuration changed')
    for alias, target in manifest['aliases'].items():
        require((root / alias).is_symlink(), f'Expected recipe symlink: {alias}')
        require((root / alias).resolve(strict=True) == (root / target).resolve(strict=True),
                f'Recipe points to the wrong implementation: {alias}')
    comparison = read(root / manifest['selection'])
    best = check_comparison(comparison)
    for name, evidence in comparison['evidence'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == evidence['sha256'],
                f'Result evidence changed: {name}')
    for row in comparison['runs']:
        require(row['summary'] in comparison['evidence'], 'Missing result evidence')
        summary = read(root / row['summary'])
        if row['formal_id'] == 17:
            fixed = summary['last_evaluation']
            extra = read(root / 'experiments/raw6b-normuon-adam2x-20261018/results.json')['confirmation_evaluation']
            steps, tokens = summary['export_global_step'], summary['export_tokens_seen']
        else:
            fixed, extra = summary['fixed1k'], summary['additional9k']
            steps, tokens = summary['global_step'], summary['confirmed_tokens']
            require(summary['complete'] and summary['run_id'] == row['run_id'], 'Incomplete/wrong run')
        require((steps, tokens) == (row['updates'], row['training_tokens']), 'Checkpoint endpoint mismatch')
        require(fixed['eval/documents'] == 1000 and extra['eval/documents'] == 9000, 'Evaluation document mismatch')
        require(row['nll'] == fixed['eval/nll'] + extra['eval/nll'], 'Result NLL disagrees with evidence')
        require(row['prediction_tokens'] == fixed['eval/prediction_tokens'] + extra['eval/prediction_tokens'],
                'Result token count disagrees with evidence')
        require(row['fixed_1k_ppl'] == fixed['eval/perplexity'] and row['additional_9k_ppl'] == extra['eval/perplexity'],
                'Subset PPL disagrees with evidence')
    completion = read(root / manifest['completion'])
    publication = read(root / manifest['publication'])
    require(best['run_id'] == manifest['run_id'] == completion['run_id'] == publication['parent_run'],
            'Selected run identity mismatch')
    require(manifest['formal_id'] == best['formal_id'], 'Formal identity mismatch')
    require(completion['state'] == 'COMPLETED' and completion['export']['reload_verified'], 'Run/export incomplete')
    require(completion['combined_10k_ppl'] == best['combined_10k_ppl'], 'Completion metric mismatch')
    require(manifest['checkpoint'] == completion['checkpoint'], 'Checkpoint pointer changed')
    require(manifest['export'] == completion['export']['path'] == publication['local_export'], 'Export mismatch')
    require(manifest['model_id'] == publication['model_id'] and manifest['model_revision'] == publication['revision'],
            'Published model identity mismatch')
    return manifest, best, len(seal['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('show', 'check'), default='show')
    args = parser.parse_args()
    if args.action == 'check':
        manifest, best, count = verify()
        print(f"PASS: {count} sealed files, config, recipe aliases, result evidence and model revision")
    else:
        manifest = read(ROOT / 'recipes/sota6b/manifest.json')
        best = check_comparison(read(ROOT / manifest['selection']))
    print(f"Selected: {manifest['run_id']} (formal {manifest['formal_id']})")
    print(f"10k PPL: {best['combined_10k_ppl']:.10f}; inputs: {best['training_tokens']:,}; updates: {best['updates']:,}")
    print(f"Implementation: {ROOT / 'recipes/sota6b/implementation/runner.py'}")
    print(f"Export: {manifest['export']}")
    print(f"Model: {manifest['model_id']} @ {manifest['model_revision']}")


if __name__ == '__main__':
    main()
