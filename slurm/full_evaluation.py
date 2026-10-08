"""Slurm-only full reserved-set evaluation of a standard HF export."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from train import require_compute, evaluate, prepare_data, on_main
    require_compute()
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    import pyarrow as pa
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    import datasets
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM
    from training_state import read_json, atomic_json, load_config, DEFAULT_ROOT
    from attention_backend import configure_attention, CALLS
    cfg = load_config(sys.argv[1]); run_id = sys.argv[2]; prep_id = sys.argv[3]
    root = Path(os.environ.get('DATA_ROOT', DEFAULT_ROOT))
    run = root / 'runs/experiments' / run_id
    prep = root / 'preprocessed' / prep_id
    accelerator = Accelerator(mixed_precision='bf16')
    # Reuse immutable training checksum receipt, while preserving subset provenance.
    _, _, provenance = prepare_data(prep, run, cfg, accelerator)
    summary = read_json(run / 'summary.json')
    model = AutoModelForCausalLM.from_pretrained(summary['export'], local_files_only=True,
                                               torch_dtype=torch.float32, attn_implementation='sdpa')
    model.config.use_cache = False
    configure_attention(model, cfg['attention_backend'])
    model.to(accelerator.device)
    if (prep / 'raw/ready.json').exists():
        from raw_baseline import packed_details
        packed = {'details': packed_details(prep)}
    else:
        packed = read_json(prep / 'pack/manifest.json')
    rows = datasets.concatenate_datasets([datasets.Dataset.from_file(item['path'])
                                          for item in packed['details']['eval_files']])
    result = evaluate(model, rows, {**cfg, 'eval_documents': 50000}, accelerator)
    if result['eval/prediction_tokens'] != packed['details']['evaluation_prediction_tokens']:
        raise ValueError('Full evaluation prediction-token count differs from sealed dataset')
    if not sum(CALLS.values()):
        raise ValueError('FA3 kernel did not execute')
    result.update(export=summary['export'], global_step=summary.get('export_global_step', summary['global_step']),
                  training_global_step=summary['global_step'],
                  preparation_fingerprint=provenance['fingerprint'],
                  slurm_job_id=os.environ['SLURM_JOB_ID'], kernel_calls=CALLS)
    on_main(accelerator, lambda: atomic_json(run / 'full-evaluation.json', result))
    def log():
        import wandb
        tracker = wandb.init(project='lab5-training-llama', id=read_json(run/'wandb.json')['id'],
                             resume='must', dir=str(run), config=None)
        wandb.define_metric('train/global_step')
        wandb.define_metric('eval/full_*', step_metric='train/global_step')
        wandb.log({'train/global_step': summary['global_step'],
                   'eval/full_perplexity': result['eval/perplexity'],
                   'eval/full_prediction_tokens': result['eval/prediction_tokens'],
                   'eval/full_documents': 50000,
                   'eval/full_checkpoint_step': result['global_step']})
        tracker.finish()
    on_main(accelerator, log)
    accelerator.end_training()


if __name__ == '__main__':
    main()
