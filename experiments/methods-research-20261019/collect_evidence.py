"""Summarize existing small metadata; never load a model or submit a job."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path('/work/asciibase64/Llama-3.2-1B-training/runs/experiments')
RUN = 'raw6b-normuon-adam2x-20261018'
SOURCES = {}


def read(path):
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def main():
    runs = []
    for run in ['raw4b-normuon-adam2x-20261017',
                'raw4b-normuon-adam2x-step30-20261018', RUN]:
        result = read(ROOT / 'experiments' / run / 'results.json')
        summary = read(RUNTIME / run / 'summary.json')
        ev1 = summary['last_evaluation']
        ev9 = result['confirmation_evaluation']
        assert ev1['eval/documents'] == 1000
        assert ev9['eval/documents'] == 9000
        assert ev9['evaluation_slots'] == [1000, 10000]
        tokens = ev1['eval/prediction_tokens'] + ev9['eval/prediction_tokens']
        nll = ev1['eval/nll'] + ev9['eval/nll']
        runs.append({
            'run_id': run, 'training_input_tokens': result['tokens'],
            'updates': result['updates'], 'logical_h200_hours': result['h200_hours'],
            'subset_1k_ppl': ev1['eval/perplexity'],
            'additional_9k_ppl': ev9['eval/perplexity'],
            'combined_10k_prediction_tokens': tokens,
            'combined_10k_nll_sum': nll,
            'combined_10k_mean_nll': nll / tokens,
            'combined_10k_ppl': math.exp(nll / tokens),
        })
    s = read(RUNTIME / RUN / 'summary.json')
    init = read(RUNTIME / RUN / 'initialization.json')
    groups = read(RUNTIME / RUN / 'optimizer-groups.json')
    config = init['architecture']['configuration']
    params = init['num_parameters']
    embedding = config['hidden_size'] * config['vocab_size']
    timing = s['timing_totals']
    f = timing['optimizer_seconds'] / timing['training_seconds']
    pilot = read(ROOT / 'experiments/ppl-next-20261017/selection.json')
    mask_pilot = read(ROOT / 'experiments/ppl-next-20261016/results.json')
    source_files = ['ppl_optimizer.py', 'recipe_runtime.py', 'train.py', 'performance.py']
    for name in source_files:
        path = ROOT / 'experiments' / RUN / 'source-v3' / name
        SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Existing metadata and scalar arithmetic only. New methods are untested proposals.',
        'runs': runs,
        'architecture': {
            'total_parameters': params, 'hidden_size': config['hidden_size'],
            'intermediate_size': config['intermediate_size'],
            'vocab_size': config['vocab_size'], 'layers': config['num_hidden_layers'],
            'tied_embeddings': config['tie_word_embeddings'],
            'embedding_parameters': embedding,
            'embedding_fraction': embedding / params,
            'training_inputs_per_parameter': s['total_tokens_seen'] / params,
            'optimizer_parameter_counts': groups['parameter_counts'],
            'initialization_method': init['method'],
            'pretrained_weights_loaded': init['pretrained_weights_loaded'],
        },
        'performance': s['performance'],
        'timing_totals': timing,
        'derived_estimates_not_benchmarks': {
            'optimizer_fraction_of_training_timer': f,
            'training_speedup_if_entire_optimizer_6x_rest_unchanged': 1 / (1-f+f/6),
            'training_speedup_if_optimizer_free_rest_unchanged': 1 / (1-f),
            'one_sequence_8192_full_vocab_logits_gib_bf16': 8192*config['vocab_size']*2/2**30,
            'one_sequence_8192_full_vocab_logits_gib_fp32': 8192*config['vocab_size']*4/2**30,
            'extra_full_matrix_fp32_second_moment_gib_per_replicated_rank': groups['parameter_counts']['muon']*4/2**30,
            'soap_two_gram_and_two_basis_matrices_gib_for_one_8192_by_2048_weight': 2*(8192**2+2048**2)*4/2**30,
            'full_6b_teacher_logits_fp32_decimal_petabytes': s['total_tokens_seen']*config['vocab_size']*4/10**15,
        },
        'adam_factorial_seed42': {
            k: {'run_id': v['run_id'], 'perplexity': v['perplexity'], 'mean_nll': v['mean_nll']}
            for k, v in pilot['records'].items()
        },
        'mask_and_normalization_pilots': {
            k: {'perplexity': v['evaluation']['eval/perplexity'],
                'observer_ppl': {name: ev['eval/perplexity'] for name, ev in v.get('observer_evaluations', {}).items()}}
            for k, v in mask_pilot['runs'].items()
        },
        'source_sha256': SOURCES,
    }
    out = Path(__file__).with_name('evidence.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'run_count': len(runs),
                      'combined_10k_ppl': runs[-1]['combined_10k_ppl'],
                      'optimizer_fraction': f, 'source_count': len(SOURCES)}))


if __name__ == '__main__':
    main()
