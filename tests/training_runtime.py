#!/usr/bin/env python3
"""GPU-only distributed correctness checks; submit through Slurm, never discover.

Uses a tiny randomly initialized Llama to test the real HF causal loss, DDP,
AdamW, checkpoint/RNG restoration and unequal-length distributed evaluation.
The validate Slurm action additionally runs the full 1B/8192 packed-data smoke.
"""
import argparse
import contextlib
import copy
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import (causal_loss_sum, checkpoint_state, collate_rows, evaluate,
                   on_main, optimizer_groups, reduce_max, require_compute, restore_rng, save_checkpoint)
from training_state import atomic_json, storage_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    require_compute()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from transformers import LlamaConfig, LlamaForCausalLM

    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1,
                              kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])
    set_seed(42)
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=8192, tie_word_embeddings=True,
                         attention_dropout=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=2,
                         use_cache=False)
    model = LlamaForCausalLM(config).to(accelerator.device)
    reference = copy.deepcopy(model)
    groups, metadata = optimizer_groups(model, 0.1)
    assert metadata['tied_aliases']
    assert sum(len(g['params']) for g in groups) == len(list(model.parameters()))
    optimizer = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(scheduler)

    rows = []
    for index in range(accelerator.num_processes * 2 + 1):
        valid = 3 + index % 12
        tokens = [1] + [3 + (index + offset) % 29 for offset in range(valid - 2)] + [2]
        labels = [-100] + tokens[1:]
        if index % 2:
            labels[1] = -100  # structural masked token still counts as an input
        rows.append({'input_ids': tokens + [2] * (16 - valid),
                     'attention_mask': [1] * valid + [0] * (16 - valid),
                     'labels': labels + [-100] * (16 - valid)})
    accumulation_rows = rows[:-1]
    target_count = sum(sum(label != -100 for label in row['labels'][1:]) for row in accumulation_rows)
    input_count = sum(sum(row['attention_mask']) for row in accumulation_rows)
    assert input_count > target_count
    total_nll = torch.zeros((), dtype=torch.float64, device=accelerator.device)
    for micro in range(2):
        batch = collate_rows([accumulation_rows[micro * accelerator.num_processes + accelerator.process_index]], 16)
        batch = {key: value.to(accelerator.device) for key, value in batch.items()}
        context = accelerator.no_sync(model) if micro == 0 else contextlib.nullcontext()
        with context:
            loss = causal_loss_sum(model, batch)
            total_nll += loss.detach().double()
            accelerator.backward(loss * accelerator.num_processes / target_count)
    global_nll = accelerator.reduce(total_nll, reduction='sum')
    batch = collate_rows(accumulation_rows, 16)
    batch = {key: value.to(accelerator.device) for key, value in batch.items()}
    with accelerator.autocast():
        reference_nll = causal_loss_sum(reference, batch)
        (reference_nll / target_count).backward()
    torch.testing.assert_close(global_nll.float(), reference_nll.detach().float(), rtol=0.015, atol=0.05)
    for actual, expected in zip(accelerator.unwrap_model(model).parameters(), reference.parameters()):
        relative = (actual.grad - expected.grad).float().norm() / expected.grad.float().norm().clamp_min(1e-5)
        assert float(relative) < 0.08, f'Token-weighted DDP gradient mismatch: {float(relative)}'
    # MAX must use MAX, not the Accelerate helper's SUM fallback.
    maximum = reduce_max(torch.tensor(accelerator.process_index + 1, device=accelerator.device), accelerator)
    assert int(maximum) == accelerator.num_processes
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    # Uneven evaluation partition (2W+1 rows) must have no duplicated targets.
    metrics = evaluate(model, rows, {'sequence_length': 16, 'eval_documents': len(rows)}, accelerator)
    assert model.training
    batch = collate_rows(rows, 16)
    batch = {key: value.to(accelerator.device) for key, value in batch.items()}
    with torch.no_grad(), accelerator.autocast():
        reference_eval_nll = causal_loss_sum(accelerator.unwrap_model(model), batch)
    expected_predictions = int((batch['labels'][:, 1:] != -100).sum())
    assert metrics['eval/prediction_tokens'] == expected_predictions
    torch.testing.assert_close(torch.tensor(metrics['eval/nll']), reference_eval_nll.detach().float().cpu(), rtol=0.015, atol=0.05)

    root = storage_root(os.environ.get('DATA_ROOT', '/work/asciibase64/Llama-3.2-1B-training'))
    run_path = root / 'runs/runtime-validation' / args.run_id
    on_main(accelerator, lambda: run_path.mkdir(parents=True, exist_ok=True))
    state = {'global_step': 1, 'cursor': input_count, 'total_tokens': input_count}
    checkpoint = save_checkpoint(accelerator, run_path, state)
    expected_rng = torch.rand(8, device=accelerator.device)
    # Reference second update, then restore and recompute exactly the same batch.
    current = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    saved_lr = optimizer.param_groups[0]['lr']
    loss = causal_loss_sum(model, batch) / expected_predictions
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    expected_parameters = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    expected_next_lr = optimizer.param_groups[0]['lr']
    loaded_path, loaded_state = on_main(accelerator, lambda: checkpoint_state(run_path))
    assert loaded_path == checkpoint and loaded_state == state
    accelerator.load_state(checkpoint)
    restore_rng(checkpoint, accelerator)
    assert optimizer.param_groups[0]['lr'] == saved_lr
    for key, value in accelerator.unwrap_model(model).state_dict().items():
        torch.testing.assert_close(value, current[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(8, device=accelerator.device), expected_rng, rtol=0, atol=0)
    optimizer.zero_grad(set_to_none=True)
    accelerator.backward(causal_loss_sum(model, batch) / expected_predictions)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == expected_next_lr
    for key, value in accelerator.unwrap_model(model).state_dict().items():
        torch.testing.assert_close(value, expected_parameters[key], rtol=0, atol=0)
    on_main(accelerator, lambda: atomic_json(run_path / 'result.json', {
        'passed': True, 'world_size': accelerator.num_processes,
        'checks': ['tied parameter coverage', 'padding and structural masks',
                   'token-weighted accumulation gradients', 'global MAX',
                   'uneven held-out evaluation', 'checkpoint weights/optimizer/scheduler/RNG resume']}))
    accelerator.print('Distributed training correctness checks passed')
    accelerator.end_training()


if __name__ == '__main__':
    main()
