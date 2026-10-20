"""Compile/load/backward smoke plus existing DDP gate, inside replacement job."""
import os
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Use sbatch')
from bootstrap import HERE


def main():
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator
    from launch import verify_source_seal
    from training_state import DEFAULT_ROOT,atomic_json
    from contracts import RUN
    verify_source_seal()
    acc=Accelerator();torch.set_num_threads(1)
    def rms(x):return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5).to(x.dtype)
    x=torch.randn(2,64,128,device=acc.device,dtype=torch.bfloat16,requires_grad=True)
    y=torch.compile(rms,fullgraph=True,dynamic=False)(x)
    y.float().square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    paths=[None]*8
    dist.all_gather_object(paths,{k:os.environ[k] for k in ['TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TMPDIR']})
    assert all(len({p[k] for p in paths})==8 for k in paths[0])
    if acc.is_main_process:
        atomic_json(DEFAULT_ROOT/'runs/experiments'/RUN/'compile-cache-gate.json',
                    dict(passed=True,compiled_forward_backward_finite=True,rank_private_paths=paths))
    from gpu_gate import main as gradient_gate
    gradient_gate()


if __name__=='__main__':main()
