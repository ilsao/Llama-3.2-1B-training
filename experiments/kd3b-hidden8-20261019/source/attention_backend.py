"""FA3 adapter for pinned Transformers 4.51 Llama; imports kernels only on compute."""
from types import MethodType

CALLS = {'dense': 0, 'varlen': 0}

def span_ids(rows, length):
    """Validate archived spans in original sequence coordinates; never infer from IDs."""
    import torch
    ids=torch.zeros((len(rows),length),dtype=torch.int64)
    for b,row in enumerate(rows):
        cursor=0
        for number,span in enumerate(row['document_spans'],1):
            start,size=int(span['sequence_start']),int(span['length'])
            if start!=cursor or size<=0 or start+size>length:
                raise ValueError('Invalid or noncontiguous document spans')
            ids[b,start:start+size]=number
            cursor+=size
        if cursor!=sum(row['attention_mask']):raise ValueError('Spans do not cover actual inputs')
    return ids


def fa3_forward(module, query, key, value, attention_mask, dropout=0.0,
                scaling=None, **kwargs):
    import torch
    from flash_attn_3.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    if dropout != 0 or kwargs.get('output_attentions', False):
        raise ValueError('FA3 requires zero attention dropout and no attention weights')
    if not torch.compiler.is_compiling() and (query.device.type != 'cuda' or torch.cuda.get_device_capability(query.device)[0] != 9):
        raise RuntimeError('FA3 requires an allocated Hopper GPU')
    if query.shape[-2] != key.shape[-2]:
        raise ValueError('Training FA3 adapter does not support KV-cache decoding')
    q, k, v = [x.transpose(1, 2).contiguous() for x in (query, key, value)]
    # FP32 RMSNorm/master parameters can produce FP32 QKV despite BF16 autocast.
    if q.dtype == torch.float32:
        if not torch.is_autocast_enabled('cuda'):
            raise ValueError('FA3 FP32 QKV requires CUDA autocast')
        dtype = torch.get_autocast_dtype('cuda')
        q, k, v = [x.to(dtype) for x in (q, k, v)]
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('This FA3 build supports BF16 QKV only')
    if isinstance(attention_mask,dict):
        indices=attention_mask['indices'];cu=attention_mask['cumulative'];maximum=attention_mask['maximum']
        packed=[x.flatten(0,1).index_select(0,indices) for x in (q,k,v)]
        out=flash_attn_varlen_func(*packed,cu,cu,maximum,maximum,softmax_scale=scaling,causal=True)
        CALLS['varlen']+=1
        restored=out.new_zeros(q.shape).flatten(0,1).index_copy(0,indices,out)
        return restored.view(q.shape),None
    if attention_mask is None:
        if not torch.compiler.is_compiling():
            CALLS['dense'] += 1
        return flash_attn_func(q, k, v, softmax_scale=scaling, causal=True), None
    if attention_mask.ndim != 2 or attention_mask.shape != q.shape[:2]:
        raise ValueError('FA3 expects a 2D right-padding input mask')
    valid = attention_mask.bool()
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()) or not bool(valid[:, 0].all()):
        raise ValueError('FA3 supports nonempty right-padded sequences only')
    lengths = valid.sum(-1, dtype=torch.int32)
    indices = valid.flatten().nonzero(as_tuple=False).flatten()
    packed = [x.flatten(0, 1).index_select(0, indices) for x in (q, k, v)]
    cumulative = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
    maximum = int(lengths.max())
    CALLS['varlen'] += 1
    output = flash_attn_varlen_func(*packed, cumulative, cumulative, maximum, maximum,
                                  softmax_scale=scaling, causal=True)
    restored = output.new_zeros(q.shape).flatten(0, 1).index_copy(0, indices, output)
    return restored.view(q.shape), None


def fa3_mask(self, attention_mask, input_tensor, cache_position,
             past_key_values, output_attentions=False):
    if past_key_values is not None or output_attentions:
        raise ValueError('FA3 training adapter requires use_cache=False, output_attentions=False')
    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError('Unsupported FA3 attention mask')
        if bool((attention_mask>1).any()):
            import torch
            # Segment IDs increase within each packed row. Prepare metadata once
            # per model forward and retain it through checkpoint recomputation.
            valid=attention_mask>0
            indices=valid.flatten().nonzero(as_tuple=False).flatten()
            ids=attention_mask.flatten().index_select(0,indices)
            starts=torch.ones_like(ids,dtype=torch.bool)
            starts[1:]=(ids[1:]!=ids[:-1]) | (indices[1:]//attention_mask.shape[1]!=indices[:-1]//attention_mask.shape[1])
            offsets=starts.nonzero(as_tuple=False).flatten().to(torch.int32)
            cu=torch.cat((offsets,offsets.new_tensor([ids.numel()])))
            maximum=int((cu[1:]-cu[:-1]).max())
            return dict(indices=indices,cumulative=cu,maximum=maximum)
        if bool((attention_mask == 0).any()):
            return attention_mask
    return None


def configure_attention(model, backend):
    if backend == 'sdpa':
        model.config._attn_implementation = backend
        return model
    if backend != 'flash_attention_3' or model.config.model_type != 'llama':
        raise ValueError('Unknown attention backend/model')
    from transformers import AttentionInterface
    from flash_attn_3 import flash_attn_interface  # fail closed before training
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError('FA3 requires an allocated Hopper GPU')
    if model.config.attention_dropout != 0 or model.config.use_cache:
        raise ValueError('FA3 adapter requires zero dropout and disabled cache')
    AttentionInterface.register(backend, fa3_forward)
    model.config._attn_implementation = backend
    model.model._update_causal_mask = MethodType(fa3_mask, model.model)
    return model
