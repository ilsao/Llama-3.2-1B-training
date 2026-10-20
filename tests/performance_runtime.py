"""Slurm-only compiled FA3/DDP/transfer correctness and checkpoint validation."""
import argparse
import contextlib
import copy
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(cfg, run_id):
    from train import (build_optimizers, causal_loss_sum, collate_rows, evaluate,
                       on_main, require_compute, save_checkpoint, export_model, restore_rng)
    require_compute()
    import torch
    import torch.optim._muon as native
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    from attention_backend import configure_attention
    from performance import TrainingPerformance
    from training_state import DEFAULT_ROOT, atomic_json, schedule_multiplier
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)])
    set_seed(42)
    assert accelerator.num_processes == 8
    original_core = native._zeropower_via_newtonschulz
    matrix = torch.randn(64, 128, device=accelerator.device)
    expected_core = original_core(matrix.clone(), (3.4445, -4.775, 2.0315), 5, 1e-7)
    toy = LlamaConfig(vocab_size=64, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=8192, attention_dropout=0., tie_word_embeddings=True,
        use_cache=False, bos_token_id=1, eos_token_id=2, pad_token_id=2)
    raw = configure_attention(LlamaForCausalLM(toy).to(accelerator.device), 'flash_attention_3')
    raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    reference = copy.deepcopy(raw)
    optimizers, _ = build_optimizers(raw, cfg)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: schedule_multiplier(s, cfg)) for o in optimizers]
    perf = TrainingPerformance(cfg)
    perf.configure(raw)
    actual_core = native._zeropower_via_newtonschulz(matrix.clone(), (3.4445, -4.775, 2.0315), 5, 1e-7)
    core_error = float((actual_core-expected_core).norm()/expected_core.norm())
    assert core_error <= .03, core_error
    prepared = accelerator.prepare(raw, *optimizers)
    model, optimizers = prepared[0], list(prepared[1:])
    accelerator.register_for_checkpointing(*schedulers)
    errors = []

    def update(rows):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        local = [collate_rows([rows[micro*8+accelerator.process_index]], len(rows[0]['input_ids']))
                 for micro in range(2)]
        targets = sum(sum(x != -100 for x in row['labels'][1:]) for row in rows)
        perf.step_begin()
        staged = perf.stage(local, accelerator.device)
        nll = torch.zeros((), dtype=torch.float64, device=accelerator.device)
        for micro, (batch, event) in enumerate(staged):
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            sync = accelerator.no_sync(model) if micro == 0 else contextlib.nullcontext()
            with sync, perf.forward_context(model, accelerator, batch):
                loss = causal_loss_sum(model, batch)
                nll += loss.detach().double()
                accelerator.backward(loss*8/targets)
        return accelerator.reduce(nll, reduction='sum')/targets

    def rows(length, padded=False):
        values = []
        for rank in range(16):
            ids = [1]+[3+(i+rank)%61 for i in range(length-2)]+[2]
            valid = length-rank%7 if padded else length
            mask = [1]*valid+[0]*(length-valid)
            labels = [-100]+ids[1:]
            labels[2] = -100
            labels[valid:] = [-100]*(length-valid)
            values.append(dict(input_ids=ids, attention_mask=mask, labels=labels))
        return values

    last_rows = None
    for length, padded in ((64, False), (64, True), (8192, False), (64, False)):
        batch_rows = rows(length, padded)
        expected_batch = {key: value.to(accelerator.device) for key, value in
                          collate_rows(batch_rows, length).items()}
        reference.zero_grad(set_to_none=True)
        targets = int((expected_batch['labels'][:, 1:] != -100).sum())
        with accelerator.autocast():
            expected_loss = causal_loss_sum(reference, expected_batch)/targets
        expected_loss.backward()
        actual_loss = update(batch_rows)
        torch.testing.assert_close(actual_loss.float(), expected_loss.detach().float(), rtol=.005, atol=1e-4)
        worst = 0.
        for a, b in zip(accelerator.unwrap_model(model).parameters(), reference.parameters()):
            error = float((a.grad-b.grad).norm()/b.grad.norm().clamp_min(1e-6))
            worst = max(worst, error)
        assert worst <= .03, worst
        errors.append(dict(length=length, padded=padded, gradient_relative_error=worst))
        torch.cuda.synchronize()
        last_rows = batch_rows
    metrics = evaluate(model, rows(64, True)+rows(64, False)[:1],
                       {'sequence_length': 64, 'eval_documents': 33}, accelerator)
    assert model.training and metrics['eval/prediction_tokens'] > 0
    path = DEFAULT_ROOT / 'runs/runtime-validation' / run_id / 'compiled'
    on_main(accelerator, lambda: path.mkdir(parents=True, exist_ok=True))
    state = dict(global_step=0, cursor=0, total_tokens=0)
    checkpoint = save_checkpoint(accelerator, path, state, retain_all=True)
    expected_rng = torch.rand(8, device=accelerator.device)
    def step():
        update(last_rows)
        accelerator.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
        for optimizer in optimizers: optimizer.step()
        for scheduler in schedulers: scheduler.step()
        torch.cuda.synchronize()
    step()
    expected = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    rates = [g['lr'] for o in optimizers for g in o.param_groups]
    accelerator.load_state(checkpoint)
    restore_rng(checkpoint, accelerator)
    torch.testing.assert_close(torch.rand(8, device=accelerator.device), expected_rng, rtol=0, atol=0)
    step()
    assert rates == [g['lr'] for o in optimizers for g in o.param_groups]
    for name, value in accelerator.unwrap_model(model).state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert '_orig_mod' not in name
    # Save/export a toy with matching tokenizer vocabulary; full-size export is
    # additionally validated by each benchmark's --validate-export.
    raw = accelerator.unwrap_model(model)
    def reload_check():
        target = path / 'hf-toy'
        backend = raw.config._attn_implementation
        raw.config._attn_implementation = 'sdpa'
        try:
            raw.save_pretrained(target, safe_serialization=True)
        finally:
            raw.config._attn_implementation = backend
        loaded = LlamaForCausalLM.from_pretrained(target, local_files_only=True)
        assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
        for name, p in raw.named_parameters():
            torch.testing.assert_close(p.detach().cpu(), dict(loaded.named_parameters())[name], rtol=0, atol=0)
    on_main(accelerator, reload_check)
    result = dict(passed=True, world_size=8, compile_mode=cfg['performance']['compile_mode'],
                  core_relative_error=core_error, cases=errors, diagnostics=perf.diagnostics(),
                  checkpoint_replay=True, eager_evaluation=True, hf_reload=True)
    on_main(accelerator, lambda: atomic_json(path.parent / 'performance-result.json', result))
    accelerator.print('Performance correctness passed', result)
    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    from training_state import load_config
    check(load_config(args.config), args.run_id)
