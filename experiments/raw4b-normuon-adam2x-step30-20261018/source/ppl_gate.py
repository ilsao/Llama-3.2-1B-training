"""Bounded numerical preflight inside each training allocation; no full LM training."""
import argparse,ast,copy,json,os,time
from pathlib import Path

def main():
    from train import require_compute
    require_compute()
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator
    from training_state import load_config,atomic_json,file_digest
    from ppl_optimizer import StudyMuon,polar_express,neuron_normalize
    from attention_backend import fa3_forward,fa3_mask,span_ids,configure_attention
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args()
    cfg=load_config(args.config);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    # Verify the complete real entrypoint import closure before numerical tests.
    import dataset,raw_baseline,preparation_parallel
    from third_party.dclm_fasttext import classify_fasttext_hq_prob
    accelerator=Accelerator();device=accelerator.device
    if accelerator.num_processes!=8 or 'H200' not in torch.cuda.get_device_name(device):raise ValueError('8 H200 gate required')
    torch.manual_seed(42);started=time.monotonic();checks=[]
    from types import SimpleNamespace
    for shape in ((64,64),(128,64),(64,128)):
        a=torch.nn.Parameter(torch.randn(shape,device=device));b=torch.nn.Parameter(a.detach().clone())
        native=torch.optim.Muon([a],lr=.01,momentum=.95,nesterov=True,weight_decay=.1,adjust_lr_fn='original')
        adapted=StudyMuon([dict(params=[b],lr=.01,momentum=.95,nesterov=True,weight_decay=.1,adjust_lr_fn='original')],'ns')
        for step in range(4):
            grad=torch.randn_like(a);a.grad=grad.clone();b.grad=grad.clone()
            native.step();adapted.step()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(native.state[a]['momentum_buffer'],adapted.state[b]['momentum_buffer'],rtol=0,atol=0)
    checks.append('native NS multi-step parameter/momentum exact parity')

    if cfg['ppl']['matrix']=='pe':
        # Execute only the archived reference function, stripping its compile decorator.
        ref=Path('/work/asciibase64/Llama-3.2-1B-training/environments/batch-normuon-1b-20261016/muonium-0.2.0/muonium/orthogonalize/polar_express.py')
        tree=ast.parse(ref.read_text());nodes=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLAR_EXPRESS_COEFFS' for t in node.targets):nodes.append(node)
            if isinstance(node,ast.FunctionDef) and node.name=='pe_loop':
                node.decorator_list=[];nodes.append(node)
        from typing import Annotated
        env=dict(torch=torch,Tensor=torch.Tensor,Annotated=Annotated)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ref),'exec'),env)
        for shape in ((64,64),(128,64),(64,128)):
            for kind in ('random','zero','rank_one'):
                g=torch.randn(shape,device=device)
                if kind=='zero':g.zero_()
                if kind=='rank_one':g=g[:,:1].expand(shape).clone()
                expected=env['pe_loop'](g.bfloat16().unsqueeze(0)).squeeze(0)
                result=polar_express(g)
                torch.testing.assert_close(result,expected,rtol=.03,atol=.03)
                assert torch.isfinite(result).all()
        checks.append('PE5 archived reference incl square/tall/wide/zero/rank-one')

    if cfg['ppl']['matrix']=='nor':
        for shape in ((64,64),(128,64),(64,128)):
            for scale in (0.,1e-12,1.):
                u=torch.randn(shape,device=device)*scale;v=torch.zeros((shape[0],1),device=device)
                for _ in range(3):
                    old=v.clone();expected_v=.95*old+.05*u.square().mean(-1,keepdim=True)
                    y=u/(expected_v.sqrt()+1e-8)
                    expected=y*(u.norm()/y.norm().clamp_min(1e-8))
                    result=neuron_normalize(u,v)
                    torch.testing.assert_close(v,expected_v,rtol=1e-6,atol=1e-8)
                    torch.testing.assert_close(result,expected,rtol=1e-5,atol=1e-7)
                    torch.testing.assert_close(result.norm(),u.norm(),rtol=1e-5,atol=1e-8)
                    assert torch.isfinite(result).all()
        p=torch.nn.Parameter(torch.randn(128,64,device=device));o=StudyMuon([p],'nor')
        p.grad=torch.randn_like(p);o.step();saved=copy.deepcopy(o.state_dict())
        q=torch.nn.Parameter(p.detach().clone());o2=StudyMuon([q],'nor');o2.load_state_dict(saved)
        grad=torch.randn_like(p);p.grad=grad.clone();q.grad=grad.clone();o.step();o2.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        assert o.state[p]['variance_neuron'].shape==(128,1)
        checks.append('NorMuon reference, norm/axis, zero/tiny inputs and state roundtrip')

    # Attention block layout includes a singleton, padding and a continuation.
    rows=[dict(attention_mask=[1]*23+[0]*9,document_spans=[
        dict(sequence_start=0,length=7,document_start=19,row_index=1),
        dict(sequence_start=7,length=1,document_start=0,row_index=2),
        dict(sequence_start=8,length=15,document_start=0,row_index=3)])]
    ids=span_ids(rows,32).to(device)
    mask=fa3_mask(None,ids,None,None,None)
    tensors=[torch.randn(1,h,32,64,device=device,dtype=torch.bfloat16,requires_grad=True) for h in (4,2,2)]
    q,k,v=tensors
    out,_=fa3_forward(None,q,k,v,mask,scaling=1/8)
    blocked=(ids[:,:,None]==ids[:,None,:]) & (ids[:,:,None]>0)
    blocked &= torch.ones(32,32,device=device,dtype=torch.bool).tril()[None]
    # Pin the independent reference to FP32 math. Default fused SDPA in this
    # installation produced nonzero outputs for fully masked padding queries.
    # Padding has no targets; explicitly zero those reference query outputs and
    # their gradients, matching varlen unpack semantics, while testing all valid
    # queries (including singleton document starts) without relaxing tolerance.
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        ref=torch.nn.functional.scaled_dot_product_attention(q.float(),k.float().repeat_interleave(2,1),v.float().repeat_interleave(2,1),attn_mask=blocked[:,None],scale=1/8).transpose(1,2)
    ref=ref*(ids>0)[:,:,None,None]
    assert torch.count_nonzero(out[:,23:])==0
    torch.testing.assert_close(out,ref,rtol=.03,atol=.03,check_dtype=False)
    probe=torch.randn_like(out)
    ga=torch.autograd.grad((out*probe).sum(),tensors,retain_graph=True)
    gb=torch.autograd.grad((ref*probe).sum(),tensors)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=.05,atol=.05)
    changed=[x.detach().clone() for x in tensors]
    for x in changed:x[:,:,:7]+=2
    perturbed,_=fa3_forward(None,*changed,mask,scaling=1/8)
    torch.testing.assert_close(perturbed[:,8:23],out.detach()[:,8:23],rtol=0,atol=0)
    checks.append('FA3 block causal SDPA forward/backward GQA singleton padding continuation isolation')

    # Check Transformers mask transport and non-reentrant recomputation.
    from transformers import LlamaConfig,LlamaForCausalLM
    config=LlamaConfig(vocab_size=128,hidden_size=128,intermediate_size=256,num_hidden_layers=2,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=8192,
                       attention_dropout=0.,use_cache=False,tie_word_embeddings=True)
    config._attn_implementation='sdpa'
    model=LlamaForCausalLM(config).to(device);reference=copy.deepcopy(model)
    configure_attention(model,'flash_attention_3')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    inputs=torch.randint(0,128,(1,32),device=device);labels=inputs.clone();labels[:,0]=-100;labels[:,23:]=-100
    labels[:,7]=-100;labels[:,8]=-100
    additive=torch.zeros((1,1,32,32),device=device);additive.masked_fill_(~blocked[:,None],torch.finfo(torch.float32).min)
    # Fully padded query outputs are not used in loss; ignore them in logits check.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        got=model(input_ids=inputs,attention_mask=ids,labels=labels)
        with sdpa_kernel(SDPBackend.MATH):
            expected=reference(input_ids=inputs,attention_mask=additive,labels=labels)
    torch.testing.assert_close(got.loss,expected.loss,rtol=.01,atol=.01)
    got.loss.backward();expected.loss.backward()
    for (_,a),(_,b) in zip(model.named_parameters(),reference.named_parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=.08,atol=.015)
    checks.append('tiny Llama HF forward/backward and activation checkpoint span-mask transport')

    from ppl_observers import Observers
    toy=torch.nn.Linear(4,4,bias=False).to(device);observer=Observers(toy,cfg)
    initial=toy.weight.detach().clone();rng=torch.cuda.get_rng_state(device)
    ticks=sorted(set([observer.knots['ema_start']]+observer.knots['mean']))
    snapshots=[]
    for tick in ticks:
        with torch.no_grad():toy.weight.add_(.01)
        if tick in observer.knots['mean']:snapshots.append(toy.weight.detach().clone())
        before=toy.weight.detach().clone();observer.update(tick)
        torch.testing.assert_close(toy.weight,before,rtol=0,atol=0)
    assert torch.equal(rng,torch.cuda.get_rng_state(device)) and observer.mean_count==5
    with observer.weights('finite_window_mean'):
        torch.testing.assert_close(toy.weight,torch.stack(snapshots).mean(0),rtol=1e-6,atol=1e-6)
    observer2=Observers(toy,cfg);observer2.load_state_dict(observer.state_dict())
    assert observer2.last_tokens==observer.last_tokens
    gathered=[torch.empty_like(toy.weight) for _ in range(8)]
    dist.all_gather(gathered,toy.weight.detach())
    for item in gathered:torch.testing.assert_close(item,toy.weight,rtol=0,atol=0)
    checks.append('observer no feedback/RNG, finite mean, state and eight-rank identity')
    # Exercise the actual scheduler with a CUDA tensor LR on all eight ranks.
    from batch3b_runtime import StepScheduler
    from ppl_contracts import token_prefix
    assert cfg['ppl']['mask']=='intradoc' and cfg['warmup_clock']=='nonpadding_tokens'
    probe=SimpleNamespace(param_groups=[dict(name='probe',lr=torch.tensor(.005,device=device))])
    scheduler=StepScheduler(probe,cfg)
    expected={0:1/170,169:1.,12463:1.,16023:.2,16024:.2,16914:.2,17803:0.,
              17804:.1/170,17847:.1,18697:.1,18951:.02,18952:.02,19016:.02,19079:0.}
    for step,value in expected.items():
        scheduler.set_step(step,token_prefix()[step])
        torch.testing.assert_close(probe.param_groups[0]['lr'],torch.tensor(.005*value,device=device),rtol=1e-6,atol=1e-10)
    saved=scheduler.state_dict();scheduler.set_step(0,0);scheduler.load_state_dict(saved)
    assert scheduler.completed_updates==19079 and scheduler.tokens==token_prefix()[19079]
    checks.append('actual-step30 decay, unchanged token warmup, CUDA tensor LR and state roundtrip')
    torch.cuda.synchronize()
    result=dict(passed=True,checks=checks,configuration=cfg,job_id=os.environ['SLURM_JOB_ID'],
                seconds=time.monotonic()-started,allocated_gpus=8,training_input_tokens=0,
                note='Synthetic numerical checks only; fresh actual LM initialization occurs in training process',
                gate_sha256=file_digest(__file__))
    if accelerator.is_main_process:
        atomic_json(Path(os.environ['JOB_RECORD'])/'ppl-gate.json',result)
        print(json.dumps({k:result[k] for k in ('passed','checks','seconds')}),flush=True)
    accelerator.wait_for_everyone();accelerator.end_training()

if __name__=='__main__':main()
