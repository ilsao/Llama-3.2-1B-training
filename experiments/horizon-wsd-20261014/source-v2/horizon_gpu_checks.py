"""Small distributed tensor checks, run inside campaign validation allocation."""
import os
from pathlib import Path
import random


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from accelerate import Accelerator
    from train import require_compute, on_main, restore_rng
    from training_state import atomic_json
    from horizon_runtime import unclipped_gradient_norm
    require_compute()
    accelerator=Accelerator(mixed_precision='bf16')
    model=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.fill_(1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda i:1-i/100)
    model,optimizer=accelerator.prepare(model,optimizer)
    accelerator.register_for_checkpointing(scheduler)
    # Replicated gradients deliberately [3,4]: global unique-parameter norm=5,
    # not sqrt(world_size)*5. Disabling clipping leaves tensors unchanged.
    parameter=next(model.parameters())
    parameter.grad=torch.tensor([[3.,4.]],device=accelerator.device)
    before=parameter.grad.clone()
    measured=unclipped_gradient_norm(model,accelerator)
    if abs(float(measured)-5)>1e-6 or not torch.equal(parameter.grad,before):
        raise ValueError('Clipping-off norm mutated gradients or double-counted ranks')
    for threshold in (.5,1.,2.):
        parameter.grad=before.clone()
        measured=accelerator.clip_grad_norm_(model.parameters(),threshold)
        if abs(float(measured)-5)>1e-5 or abs(float(parameter.grad.norm())-threshold)>1e-5:
            raise ValueError('Clipping threshold semantics changed')
    parameter.grad=before.clone();optimizer.step();scheduler.step();optimizer.zero_grad()
    path=Path(os.environ['JOB_RECORD'])/'small-resume-state'
    accelerator.save_state(str(path),safe_serialization=True)
    expected=parameter.detach().clone()
    rng_expected=torch.rand(8,device=accelerator.device)
    with torch.no_grad():parameter.add_(9)
    accelerator.load_state(str(path));restore_rng(path,accelerator)
    if not torch.equal(parameter,expected) or not torch.equal(torch.rand(8,device=accelerator.device),rng_expected):
        raise ValueError('Checkpoint model/RNG restore failed')
    if scheduler.last_epoch!=1 or optimizer.param_groups[0]['lr']!=.0099:
        raise ValueError('Checkpoint scheduler restore failed')
    on_main(accelerator,lambda:atomic_json(Path(os.environ['JOB_RECORD'])/'horizon-runtime-checks.json',
        dict(clipping_thresholds=[.5,1.,2.],clipping_off=True,unique_parameter_norm=5.,
             checkpoint_model_optimizer_scheduler_rng=True,world_size=accelerator.num_processes)))
    accelerator.end_training()


if __name__=='__main__':main()
