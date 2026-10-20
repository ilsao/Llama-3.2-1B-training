"""Slurm-only kernel correctness, including GQA, padding, causal masking and backward."""
import copy
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from train import require_compute
    require_compute()
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    import attention_backend as backend
    from training_state import atomic_json
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    config = LlamaConfig(vocab_size=64, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=8192, attention_dropout=0.0,
                         tie_word_embeddings=True, use_cache=False)
    config._attn_implementation = 'sdpa'
    reference = LlamaForCausalLM(config).to(device)
    actual = backend.configure_attention(copy.deepcopy(reference), 'flash_attention_3')
    actual.train(); reference.train()
    for length, padded in ((17, False), (17, True), (8192, False)):
        ids = torch.randint(0, 64, (2 if padded else 1, length), device=device)
        mask = torch.ones_like(ids)
        if padded:
            mask[1, 11:] = 0
        labels = ids.clone(); labels[:, 0] = -100; labels[mask == 0] = -100
        reference.zero_grad(set_to_none=True); actual.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = reference(input_ids=ids, attention_mask=mask, labels=labels)
            result = actual(input_ids=ids, attention_mask=mask, labels=labels)
        valid = mask.bool()
        torch.testing.assert_close(result.logits[valid], expected.logits[valid], rtol=.04, atol=.04)
        torch.testing.assert_close(result.loss, expected.loss, rtol=.02, atol=.02)
        result.loss.backward(); expected.loss.backward()
        for a, b in zip(actual.parameters(), reference.parameters()):
            relative = (a.grad-b.grad).norm()/b.grad.norm().clamp_min(1e-5)
            assert float(relative) < .12, float(relative)
        if length == 17 and not padded:
            changed = ids.clone(); changed[:, 9:] = (changed[:, 9:]+1) % 64
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                future = actual(input_ids=changed, attention_mask=mask).logits
            torch.testing.assert_close(future[:, :9], result.logits[:, :9], rtol=0, atol=0)
    assert backend.CALLS['dense'] and backend.CALLS['varlen']
    if int(os.environ.get('RANK', 0)) == 0:
        atomic_json(Path(os.environ['JOB_RECORD']) / 'fa3-check.json',
                    {'passed': True, 'calls': backend.CALLS, 'checks': ['GQA', 'causal', 'right padding', '8192', 'BF16 backward']})


if __name__ == '__main__':
    main()
