"""Pinned torch 2.9.1 flash varlen backend for a frozen head-dim128 teacher.

Uses the same ATen op as torch.nested SDPA. This private API is version-pinned,
GPU checked against dense masked attention, and deliberately inference-only.
"""
from types import MethodType
import torch


def forward(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
    if torch.is_grad_enabled() or dropout!=0 or kwargs.get('output_attentions',False):
        raise ValueError('Frozen teacher only: no gradients/dropout/attention weights')
    q,k,v=[x.transpose(1,2).contiguous() for x in (query,key,value)]
    if q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError('BF16 teacher attention required')
    batch,length=q.shape[:2]
    if isinstance(attention_mask,dict):
        indices=attention_mask['indices'];cu=attention_mask['cumulative'];maximum=attention_mask['maximum']
    else:
        mask=torch.ones((batch,length),device=q.device,dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        indices=mask.flatten().nonzero(as_tuple=False).flatten()
        lengths=mask.sum(-1,dtype=torch.int32)
        cu=torch.cat((lengths.new_zeros(1),lengths.cumsum(0,dtype=torch.int32)));maximum=int(lengths.max())
    packed=[x.flatten(0,1).index_select(0,indices) for x in (q,k,v)]
    result=torch.ops.aten._flash_attention_forward(*packed,cu,cu,maximum,maximum,0.,True,False,scale=scaling)[0]
    return q.new_zeros(q.shape).flatten(0,1).index_copy(0,indices,result).view(q.shape),None


def configure(teacher):
    from transformers import AttentionInterface
    from attention_backend import fa3_mask
    if not torch.__version__.startswith('2.9.1'):raise ValueError('Private teacher flash backend pinned to torch2.9.1')
    if any(p.requires_grad for p in teacher.parameters()) or teacher.config.use_cache or teacher.config.attention_dropout!=0:
        raise ValueError('Frozen, uncached, zero-dropout teacher required')
    AttentionInterface.register('kd_teacher_torch_flash',forward)
    teacher.config._attn_implementation='kd_teacher_torch_flash'
    teacher.model._update_causal_mask=MethodType(fa3_mask,teacher.model)


def check(device):
    from torch.nn import functional as F
    from attention_backend import fa3_mask
    with torch.no_grad():
        q=torch.randn(1,6,11,128,device=device,dtype=torch.bfloat16)
        k=torch.randn(1,2,11,128,device=device,dtype=torch.bfloat16);v=torch.randn_like(k)
        ids=torch.tensor([[1,1,1,1,2,2,2,2,2,0,0]],device=device)
        mask=fa3_mask(None,ids,None,None,None)
        actual=forward(None,q,k,v,mask,scaling=128**-.5)[0]
        ref=torch.zeros_like(actual,dtype=torch.float32)
        for start,end in ((0,4),(4,9)):
            kk=k[:,:,start:end].float().repeat_interleave(3,dim=1)
            vv=v[:,:,start:end].float().repeat_interleave(3,dim=1)
            result=F.scaled_dot_product_attention(q[:,:,start:end].float(),kk,vv,is_causal=True,scale=128**-.5)
            ref[:,start:end]=result.transpose(1,2)
        error=float((actual.float()-ref).norm()/ref.norm())
        assert error<.005,error
        changed=v.clone();changed[:,:,:4].add_(3)
        altered=forward(None,q,k,changed,mask,scaling=128**-.5)[0]
        torch.testing.assert_close(actual[:,4:],altered[:,4:],rtol=0,atol=0)
        assert float(actual[:,9:].abs().max())==0
        return {'head_dimension':128,'GQA':True,'document_boundary_isolation':True,'padding_zero':True,'relative_l2_vs_fp32_sdpa':error}
