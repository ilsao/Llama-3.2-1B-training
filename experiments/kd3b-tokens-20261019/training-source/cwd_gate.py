"""Compute-node numerical tests for the exact native-compatible CWD update."""
def check_cwd(device,model,cfg):
    import copy,time,torch
    import torch.distributed as dist
    from ppl_optimizer import StudyMuon,neuron_normalize
    from torch.optim._muon import _zeropower_via_newtonschulz,_adjust_lr
    checks=[]
    for shape in ((64,64),(128,64),(64,128)):
        p=torch.nn.Parameter(torch.randn(shape,device=device))
        opt=StudyMuon([dict(params=[p],lr=.01,weight_decay=.1)],'nor',True)
        for _ in range(4):
            old=p.detach().clone();g=torch.randn_like(p);p.grad=g
            group=opt.param_groups[0];state=opt.state[p]
            buf=state.get('momentum_buffer',torch.zeros_like(p)).clone()
            buf.lerp_(g,1-group['momentum']);u=g.lerp(buf,group['momentum'])
            u=_zeropower_via_newtonschulz(u,group['ns_coefficients'],group['ns_steps'],group['eps'])
            v=state.get('variance_neuron',torch.zeros((shape[0],1),device=device)).clone()
            u=neuron_normalize(u,v);mask=(old*u>=0)
            expected=old-.01*.1*mask*old-_adjust_lr(.01,'original',shape)*u
            opt.step();torch.testing.assert_close(p,expected,rtol=2e-6,atol=2e-7)
        q=torch.nn.Parameter(p.detach().clone());other=StudyMuon([dict(params=[q],lr=.01,weight_decay=.1)],'nor',True)
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        g=torch.randn_like(p);p.grad=g.clone();q.grad=g.clone();opt.step();other.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        for wd in (0.,.1):
            # Positive all-one first update produces an all-one CWD mask.
            a=torch.nn.Parameter(torch.ones(shape,device=device));b=torch.nn.Parameter(a.detach().clone())
            oa=StudyMuon([dict(params=[a],lr=.01,weight_decay=wd)],'nor',False)
            ob=StudyMuon([dict(params=[b],lr=.01,weight_decay=wd)],'nor',True)
            g=torch.ones_like(a) if wd else torch.randn_like(a)
            a.grad=g.clone();b.grad=g.clone();oa.step();ob.step()
            torch.testing.assert_close(a,b,rtol=1e-6,atol=2e-7)
    checks.append('CWD explicit pre-W/post-NorMuon formula, shape LR, zero/all-one mask and state roundtrip')
    from batch3b_runtime import build_optimizers
    opts,groups=build_optimizers(model,cfg)
    assert sum(groups['parameter_counts'].values())==sum(p.numel() for p in model.parameters())
    for opt in opts:opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all()
        replicas=[torch.empty_like(p) for _ in range(8)]
        dist.all_gather(replicas,p.detach())
        for q in replicas:torch.testing.assert_close(p,q,rtol=0,atol=0)
    checks.append('tiny Llama tied-group coverage, finite CWD/aux2x step and exact eight-rank identity')
    return checks
