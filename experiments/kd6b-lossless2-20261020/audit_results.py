#!/usr/bin/env python3
"""Audit small saved evaluation receipts, never load models or launch jobs."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = Path('/work/asciibase64/Llama-3.2-1B-training/runs/experiments/kd6b-lossless2-validation-20261020')


def read(path):
    return json.loads(path.read_text())


def main():
    results = read(RUN / 'results.json')
    reference = read(RUN / 'R0_original/eval-documents-0128.json')
    base = results['arms'][0]
    reports = []
    for record in results['arms']:
        name = record['arm']['name']
        documents = read(RUN / name / 'eval-documents-0128.json')
        endpoint = record['endpoint_evaluation']
        assert len(documents) == len(reference) == 1000
        assert [r[1] for r in documents] == [r[1] for r in reference]
        count = sum(r[1] for r in documents)
        nll = math.fsum(r[0] for r in documents)
        assert count == endpoint['eval/prediction_tokens'] == 1479302
        assert math.isclose(nll, endpoint['eval/nll'], abs_tol=1e-6, rel_tol=1e-12)
        assert math.isclose(math.exp(nll / count), endpoint['eval/perplexity'], rel_tol=1e-12)
        evaluations = read(RUN / name / 'evaluations.json')
        assert [r['train/global_step'] for r in evaluations] == list(range(12, 128, 12)) + [128]
        assert record['quality_inputs'] == endpoint['eval/training_tokens'] == 16777216
        changes = [a[0] - b[0] for a, b in zip(documents, reference)]
        bs16_seconds = 131072 / record['tokens_per_second_bs16']
        bs64_seconds = 524288 / record['tokens_per_second_bs64']
        reports.append(dict(
            arm=name, matched_documents_and_tokens=True, reconstructed_ppl=math.exp(nll / count),
            paired_nll_change_per_token=math.fsum(changes) / count,
            documents_lower_nll=sum(x < 0 for x in changes),
            documents_equal_nll=sum(x == 0 for x in changes),
            documents_higher_nll=sum(x > 0 for x in changes),
            strict_short_prefix_ppl_pass=endpoint['eval/perplexity'] <= base['endpoint_evaluation']['eval/perplexity'],
            speedup_vs_R0=record['harmonic_tokens_per_second'] / base['harmonic_tokens_per_second'],
            optimizer_fraction_bs16=record['optimizer_seconds_bs16_rank0'] / bs16_seconds,
            optimizer_fraction_bs64=record['optimizer_seconds_bs64_rank0'] / bs64_seconds,
            peak_allocated_gib=record['peak_allocated_bytes'] / 2**30,
            six_b_training_only_h200_hours=8 * 6e9 / record['harmonic_tokens_per_second'] / 3600,
            six_b_with_overhead_h200_hours=record['projected_full6b_hours'],
        ))
    result = dict(complete=results['complete'], arms=reports,
                  final_six_b_quality_proven=False,
                  limitation='One seed, 128 updates during original warmup; document comparisons do not measure training-seed variability or final quality.')
    (HERE / 'quality-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
