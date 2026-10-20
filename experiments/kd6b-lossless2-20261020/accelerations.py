"""Opt-in execution changes; no optimizer formula or attention-mask changes."""
import contextlib
import types
import torch
import torch.distributed as dist


class WholeMatrixOwners:
    """Update whole NorMuon matrices on one owner; publish FP32 weights exactly.

    DDP still reduces full gradients. Caller must zero and clip ALL model
    parameters, not just the owner optimizer groups. Never shard an NS matrix.
    """
    def __init__(self, optimizer, names):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.params = [p for g in optimizer.param_groups for p in g['params']]
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('Duplicate matrix assignment')
        assert all(p.ndim == 2 and p.dtype == torch.float32 for p in self.params)
        costs = [0]*self.world
        owner = {}
        # Largest NS work first; ties preserve parameter order.
        order = sorted(range(len(self.params)), key=lambda i: -self.params[i].numel()*min(self.params[i].shape))
        for i in order:
            r = min(range(self.world), key=lambda j: (costs[j], j))
            owner[id(self.params[i])] = r
            costs[r] += self.params[i].numel()*min(self.params[i].shape)
        self.by_rank = [[p for p in self.params if owner[id(p)] == r] for r in range(self.world)]
        self.counts = [sum(p.numel() for p in ps) for ps in self.by_rank]
        self.width = max(self.counts)
        self.local = torch.zeros(self.width, device=self.params[0].device, dtype=torch.float32)
        self.gathered = torch.empty(self.world*self.width, device=self.local.device, dtype=self.local.dtype)
        self.local_views=[]; self.dest=[]; self.remote_views=[]
        for r, ps in enumerate(self.by_rank):
            offset=0
            for p in ps:
                self.dest.append(p)
                self.remote_views.append(self.gathered[r*self.width+offset:r*self.width+offset+p.numel()].view_as(p))
                if r==self.rank: self.local_views.append(self.local[offset:offset+p.numel()].view_as(p))
                offset += p.numel()
        self.owner = owner
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if owner[id(p)] == self.rank]
        self.metadata=dict(whole_matrices=True,transport_dtype='float32',rank_parameter_counts=self.counts,
                           padded_elements_per_rank=self.width,
                           assignments={names[id(p)]:owner[id(p)] for p in self.params},
                           requires_global_zero_grad_and_clipping=True)

    @torch.no_grad()
    def publish(self):
        torch._foreach_copy_(self.local_views, self.by_rank[self.rank])
        dist.all_gather_into_tensor(self.gathered, self.local)
        torch._foreach_copy_(self.dest, self.remote_views)


class CompiledParts:
    """Compile RMSNorm/MLP/RoPE only; evaluation can use the same eager path."""
    def __init__(self, *models):
        import transformers.models.llama.modeling_llama as llama
        self.llama=llama
        self.original_rope=llama.apply_rotary_pos_emb
        options={'triton.cudagraphs':False}
        self.compiled_rope=torch.compile(self.original_rope, fullgraph=True, dynamic=False, options=options)
        self.parts=[]
        for model in models:
            for module in model.modules():
                if isinstance(module,(llama.LlamaRMSNorm,llama.LlamaMLP)):
                    original=module.forward
                    compiled=torch.compile(original,fullgraph=True,dynamic=False,options=options)
                    self.parts.append((module,original,compiled))
        self.enabled=False

    def set_enabled(self, enabled):
        self.enabled=enabled
        for module,original,compiled in self.parts:
            module.forward=compiled if enabled else original
        self.llama.apply_rotary_pos_emb=self.compiled_rope if enabled else self.original_rope

    @contextlib.contextmanager
    def eager(self):
        previous=self.enabled
        self.set_enabled(False)
        try: yield
        finally: self.set_enabled(previous)


def owner_update_gate(device):
    """12 nonzero-LR steps: identical gradients, full NS5/EMA/ordinary WD."""
    from ppl_optimizer import StudyMuon
    gen=torch.Generator(device=device).manual_seed(4321)
    shapes=[(48,32),(32,48),(24,24),(16,48)]*4
    ref=[torch.nn.Parameter(torch.randn(s,generator=gen,device=device)*.02) for s in shapes]
    got=[torch.nn.Parameter(p.detach().clone()) for p in ref]
    def make(params):
        opt=StudyMuon([dict(params=params,lr=.003,momentum=.95,nesterov=True,
                            weight_decay=0.,ns_steps=5,ns_coefficients=(3.4445,-4.775,2.0315),
                            eps=1e-7,adjust_lr_fn=None)],'nor')
        def decay(optimizer,args,kwargs):
            with torch.no_grad():
                for group in optimizer.param_groups:
                    for p in group['params']:p.mul_(1-group['lr']*.1)
        opt.register_step_pre_hook(decay)
        return opt
    ro,go=make(ref),make(got)
    owners=WholeMatrixOwners(go,{id(p):str(i) for i,p in enumerate(got)})
    max_abs=0.; state_abs=0.
    for step in range(12):
        ro.param_groups[0]['lr']=go.param_groups[0]['lr']=.003*(step+1)/12
        for a,b in zip(ref,got):
            g=torch.randn(a.shape,generator=gen,device=device)
            a.grad=g; b.grad=g.clone()
        ro.step();go.step();owners.publish()
        for a,b in zip(ref,got):
            max_abs=max(max_abs,float((a-b).abs().max()))
            if owners.owner[id(b)]==dist.get_rank():
                for key in ('momentum_buffer','variance_neuron'):
                    state_abs=max(state_abs,float((ro.state[a][key]-go.state[b][key]).abs().max()))
    errors=torch.tensor([max_abs,state_abs],device=device,dtype=torch.float64)
    dist.all_reduce(errors,op=dist.ReduceOp.MAX)
    assert float(errors.max())==0., errors
    return dict(nonzero_lr_steps=12,weights_max_absolute_difference=float(errors[0]),
                optimizer_state_max_absolute_difference=float(errors[1]),bitwise_equal=True)
