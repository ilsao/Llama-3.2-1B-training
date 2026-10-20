"""Eight-rank toy gate; run with sbatch, never directly on the login node."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise SystemExit('Use allocated Slurm job')
from bootstrap import HERE
from contracts import RUN,CONFIG_DIGEST
from training_state import DEFAULT_ROOT, ExperimentLedger, atomic_json, file_digest
from gradient_probe import synchronize_and_measure


def main():
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator
    from launch import verify_source_seal
    verify_source_seal()
    allocation=ExperimentLedger(DEFAULT_ROOT).allocation(os.environ['SLURM_JOB_ID'],RUN)
    assert allocation['config_digest']==CONFIG_DIGEST
    acc=Accelerator();torch.set_num_threads(1)
    assert acc.num_processes==8
    torch.manual_seed(1234)
    model=acc.prepare(torch.nn.Linear(3,2,bias=False).to(acc.device))
    torch.manual_seed(4000+acc.process_index)
    n=acc.process_index+2  # unequal valid-unit counts
    x=torch.randn(n,3,device=acc.device);y=torch.randn(n,2,device=acc.device)
    total=int(acc.reduce(torch.tensor(n,device=acc.device),reduction='sum'))
    loss=(model(x)-y).square().sum()*acc.num_processes/total
    acc.backward(loss)
    ref=[p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    with acc.no_sync(model):
        loss=(model(x)-y).square().sum()*acc.num_processes/total
        acc.backward(loss)
    raw=acc.unwrap_model(model)
    local=[p.grad.clone() for p in raw.parameters()]
    gathered=[None]*acc.num_processes
    dist.all_gather_object(gathered,[(g.cpu().tolist(),n) for g in local])
    moments=synchronize_and_measure(raw.named_parameters(),n,total,chunk_elements=2)
    error=max(float((p.grad-g).abs().max()) for p,g in zip(raw.parameters(),ref))
    assert error<2e-6,error
    weighted=sum(total/(acc.num_processes**2*rank[0][1])*torch.tensor(rank[0][0],dtype=torch.float64).square().sum().item() for rank in gathered)
    assert abs(moments['global']['weighted_local_norm2']-weighted)<1e-6
    assert abs(moments['global']['mean_norm2']-sum(g.double().square().sum().item() for g in ref))<1e-6
    # A subsequent ordinary DDP backward must still synchronize correctly.
    model.zero_grad(set_to_none=True)
    acc.backward((model(x)-y).square().sum()*acc.num_processes/total)
    assert max(float((p.grad-g).abs().max()) for p,g in zip(raw.parameters(),ref))<2e-6
    if acc.is_main_process:
        atomic_json(DEFAULT_ROOT/'runs/experiments'/(RUN+'-startup-replacement2')/'gpu-gate.json',dict(
            passed=True,world_size=acc.num_processes,unequal_target_counts=True,
            source_seal_sha256=file_digest(HERE/'source-seal.json'),configuration_digest=CONFIG_DIGEST,
            gradient_max_abs_error=error,moments=moments,subsequent_ddp_sync_verified=True))
    acc.wait_for_everyone()


if __name__=='__main__':main()
