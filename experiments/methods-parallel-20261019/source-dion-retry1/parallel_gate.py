"""Eight-rank method, masking and real-data preflight; Slurm only."""
import copy,json,os,sys,time
from pathlib import Path


def check_method(device,model,cfg):
    import torch
    import torch.distributed as dist
    from training_state import atomic_json
    from parallel_runtime import build_optimizers,Scheduler,eval_weights
    from parallel_losses import kd_loss
    checks=[]
    # Historical B-seed42 supplies N0 quality results. Validate the new native
    # wrapper against the preserved original update before any candidate trains.
    from training_state import read_json
    from batch3b_runtime import build_optimizers as original_build, TokenScheduler as OriginalScheduler, configure_update as original_configure
    n0=read_json(Path(__file__).parent/'locked-configs.json')['N0-seed42']
    left,right=copy.deepcopy(model),copy.deepcopy(model)
    current,_=build_optimizers(left,n0);original,_=original_build(right,n0)
    current_s=[Scheduler(o,n0) for o in current]
    original_s=[OriginalScheduler(o,n0) for o in original]
    for unit in (0,169,3815,6358,7629):
        for scheduler in current_s+original_s:scheduler.set_tokens(unit*131072)
        original_configure(original,n0,16,unit*131072)
        for a,b in zip(left.parameters(),right.parameters()):
            a.grad=torch.randn_like(a);b.grad=a.grad.clone()
        for optimizer in current+original:optimizer.step()
        for a,b in zip(left.parameters(),right.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    checks.append('historical native update vs current wrapper: exact tiny-model parameter parity across warmup/batch/phase/end LR knots')
    del left,right,current,original,current_s,original_s
    opts,metadata=build_optimizers(model,cfg)
    schedulers=[Scheduler(o,cfg) for o in opts]
    assert sum(metadata['parameter_counts'].values())==sum(p.numel() for p in model.parameters())
    if cfg['parallel_method']['name'] in ('dion1','dion025'):
        assert os.environ.get('TORCH_COMPILE_DISABLE')=='1'
        for parameter in model.parameters():
            replica_exact(parameter, 'initial parameter')
    # Test changing token LR, zero gradient, tall/wide real module shapes, and
    # SOAP's first basis-only step plus a subsequent preconditioner refresh.
    for step in range(12):
        for p in model.parameters():
            p.grad=torch.randn_like(p) if step!=5 else torch.zeros_like(p)
            # DDP supplies common reduced gradients; explicitly model that here.
            if cfg['parallel_method']['name'] in ('dion1','dion025'):
                dist.broadcast(p.grad,src=0)
                replica_exact(p.grad, 'synthetic common gradient')
        for opt in opts:opt.step()
        for scheduler in schedulers:scheduler.set_tokens((step+1)*131072)
    for p in model.parameters():
        assert torch.isfinite(p).all()
        replicas=[torch.empty_like(p) for _ in range(8)]
        dist.all_gather(replicas,p.detach())
        for q in replicas:torch.testing.assert_close(p,q,rtol=0,atol=0)
    checks.append('12 finite selected-method updates, exact 8-rank replica identity, complete tied-parameter coverage')
    if cfg['parallel_method']['name'] in ('dion1','dion025'):
        dion_real_shapes(device,cfg)
        checks.append('official eager Dion: exact parameter/state replicas on actual Llama q/kv/up/down matrix shapes and state roundtrip')
    saved=[p.detach().clone() for p in model.parameters()]
    with eval_weights():
        for p in model.parameters():assert torch.isfinite(p).all()
    for p,before in zip(model.parameters(),saved):torch.testing.assert_close(p,before,rtol=0,atol=0)
    checks.append('evaluation weight context restores exact live training weights')
    # Serialization is only a synthetic optimizer test; training never loads a checkpoint.
    clone=copy.deepcopy(model);others,_=build_optimizers(clone,cfg)
    for old,new in zip(opts,others):new.load_state_dict(copy.deepcopy(old.state_dict()))
    for p,q in zip(model.parameters(),clone.parameters()):
        p.grad=torch.randn_like(p);q.grad=p.grad.clone()
    for opt in opts+others:opt.step()
    for p,q in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    checks.append('selected optimizer state roundtrip preserves next update')
    if cfg['parallel_method']['name'] in ('sf','observer'):
        from parallel_runtime import FeedbackOptimizer
        # Algebraically independent scalar Adam direction reference for the feedback.
        p=torch.nn.Parameter(torch.tensor([.7,-.3],device=device));z=p.detach().clone();x=z.clone()
        opt=FeedbackOptimizer([dict(params=[p],name='adamw_no_decay',lr=.01)],True)
        v=torch.zeros_like(p);weight_sum=0.
        for i in range(1,4):
            grad=torch.tensor([.1*i,-.2],device=device);p.grad=grad
            v=.95*v+.05*grad.square();eta=.01*(1-.95**i)**.5
            weight_sum+=eta*eta
            z=z-eta*grad/(torch.sqrt(v)+1e-8)
            x=x+(z-x)*(eta*eta/weight_sum)
            expected=.9*x+.1*z
            opt.step();torch.testing.assert_close(p,expected,rtol=2e-6,atol=2e-7)
        checks.append('SF x/z/y feedback agrees with independent scalar recurrence')
    if cfg['parallel_method']['name']=='kd':
        # Full-softmax reference checks the custom backward, ignore mask, chunk
        # boundaries, lambda endpoints, and teacher==student KL zero.
        for lam in (0.,.25,1.):
            h=torch.randn(2,7,16,device=device,requires_grad=True)
            w=torch.randn(32,16,device=device,requires_grad=True)
            th=torch.randn_like(h);tw=torch.randn_like(w)
            labels=torch.randint(0,32,(2,7),device=device);labels[0,3]=-100
            loss,ce,kl=kd_loss(h,w,th,tw,labels,lam=lam,chunk=5)
            logits=h[:,:-1]@w.T;teacher=th[:,:-1]@tw.T;target=labels[:,1:]
            ls=logits.log_softmax(-1);lt=teacher.log_softmax(-1)
            refce=torch.nn.functional.cross_entropy(logits.reshape(-1,32),target.reshape(-1),reduction='sum')
            refkl=((lt.exp()*(lt-ls)).sum(-1)*(target!=-100)).sum()
            reference=(1-lam)*refce+lam*refkl
            torch.testing.assert_close(loss,reference,rtol=2e-6,atol=2e-5)
            a=torch.autograd.grad(loss,(h,w));b=torch.autograd.grad(reference,(h,w))
            for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=5e-5,atol=1e-5)
            _,_,zero=kd_loss(h,w,h.detach(),w.detach(),labels,chunk=5)
            torch.testing.assert_close(zero,torch.zeros_like(zero),rtol=0,atol=1e-5)
        checks.append('full-vocabulary KD value/backward vs dense reference, teacher==student, lambda 0/.25/1')
    if cfg['parallel_method']['name']=='cce':
        from cut_cross_entropy import linear_cross_entropy
        for dtype in (torch.float32,torch.bfloat16):
            h=torch.randn(2,17,64,device=device,dtype=dtype,requires_grad=True)
            w=torch.randn(128,64,device=device,dtype=dtype,requires_grad=True)
            labels=torch.randint(0,128,(2,17),device=device);labels[0,4]=-100
            with torch.autocast('cuda',dtype=torch.bfloat16):
                got=linear_cross_entropy(h,w,labels,shift=True,reduction='sum',impl='cce_exact')
                ref=torch.nn.functional.cross_entropy((h[:,:-1]@w.T).float().reshape(-1,128),labels[:,1:].reshape(-1),reduction='sum')
            torch.testing.assert_close(got,ref,rtol=.005,atol=.05)
            a=torch.autograd.grad(got,(h,w));b=torch.autograd.grad(ref,(h,w))
            for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=.05,atol=.025)
        checks.append('CCE exact shifted loss and tied-head constituent gradients vs dense CE, FP32/BF16 storage')
    # Real loader check on every rank BEFORE actual 1B-model initialization.
    from accelerate import Accelerator
    import train
    from training_state import DEFAULT_ROOT
    accelerator=Accelerator()
    train.MASK_MODE='intradoc'
    gate_run=Path(os.environ['JOB_RECORD'])/'data-preflight'
    gate_run.mkdir(parents=True,exist_ok=True)
    rows,evaluation,provenance=train.prepare_data(DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008',gate_run,cfg,accelerator)
    end=cfg['target_training_tokens']//8192
    for index in (accelerator.process_index,end-8+accelerator.process_index):
        batch=train.collate_rows([rows[index]],8192)
        assert int(batch['attention_mask'].sum())==8192
        assert 'document_ids' in batch and batch['document_ids'].shape==(1,8192)
        assert int((batch['labels'][:,1:]!=-100).sum())>0
    checks.append('real canonical data first/last target batch across 8 ranks, input counts and intradoc masks')
    if accelerator.is_main_process:
        atomic_json(Path(os.environ['JOB_RECORD'])/'parallel-method-gate.json',
          dict(passed=True,checks=checks,method=cfg['parallel_method'],optimizer_groups=metadata,
               job_id=os.environ['SLURM_JOB_ID'],preparation_fingerprint=provenance['fingerprint'],
               runtime_validated_on_full_model=False,synthetic_gate_inputs_not_training=True))
    return checks


def replica_exact(value,label):
    import torch
    import torch.distributed as dist
    reference=value.detach().clone()
    dist.broadcast(reference,src=0)
    delta=(value.detach().float()-reference.float()).abs().max()
    dist.all_reduce(delta,op=dist.ReduceOp.MAX)
    if float(delta)!=0:
        raise AssertionError(f'{label}: cross-rank maximum absolute difference {float(delta)}')
    if not torch.isfinite(value).all():raise AssertionError(label+' nonfinite')


def dion_real_shapes(device,cfg):
    """Bounded numerical test on full layer dimensions, not full-model loading."""
    import torch
    import torch.distributed as dist
    from dion import Dion3
    from parallel_runtime import ns5
    from training_state import atomic_json
    fraction=1. if cfg['parallel_method']['name']=='dion1' else .25
    evidence=[]
    for shape in ((2048,2048),(512,2048),(8192,2048),(2048,8192)):
        p=torch.nn.Parameter(torch.randn(shape,device=device)*.01)
        dist.broadcast(p.data,src=0)
        def optimizer(q):
            return Dion3([q],distributed_mesh=None,lr=.001,fraction=fraction,mu=.95,muon_beta2=.95,
                weight_decay=0.,adjust_lr='rms_norm',use_triton=False,use_polar_express=False,
                use_gram_newton_schulz=False,newton_schulz_func=ns5)
        opt=optimizer(p)
        for step in range(4):
            p.grad=torch.randn_like(p) if step!=2 else torch.zeros_like(p)
            dist.broadcast(p.grad,src=0)
            opt.step();replica_exact(p,'real-shape parameter '+str(shape))
            for name,state in opt.state[p].items():
                if isinstance(state,torch.Tensor):replica_exact(state,'real-shape state '+name)
        q=torch.nn.Parameter(p.detach().clone());other=optimizer(q)
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        p.grad=torch.randn_like(p);dist.broadcast(p.grad,src=0);q.grad=p.grad.clone()
        opt.step();other.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        replica_exact(p,'real-shape restored update')
        evidence.append(dict(shape=list(shape),updates=5,parameter_and_optimizer_state_exact=True,
                             state_roundtrip_exact=True))
        del p,q,opt,other
    if dist.get_rank()==0:
        atomic_json(Path(os.environ['JOB_RECORD'])/'dion-eager-real-shapes.json',
            dict(passed=True,fraction=fraction,checks=evidence,torch_compile_disable=True,
                 input_gradients_explicitly_broadcast=True,replica_rtol=0,replica_atol=0))


def main():
    from train import require_compute
    require_compute()
    # Reuse established FA3 vs SDPA forward/backward and document-isolation tests.
    import cwd_gate,ppl_gate
    cwd_gate.check_cwd=check_method
    ppl_gate.main()

if __name__=='__main__':main()
