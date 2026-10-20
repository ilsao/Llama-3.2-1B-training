"""Two-slot, online teacher prefetch on a separate CUDA stream.

The caller plans canonical batches BEFORE enqueueing and commits the student
cursor only after consumption. Call release after backward; drain before eval,
checkpointing or shutdown. Pending teacher inputs count even if discarded.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


class LogitsKD(torch.autograd.Function):
    """T=1 full-vocabulary KD; teacher projection has already run asynchronously."""
    @staticmethod
    def forward(ctx,h,w,teacher_logits,labels,lam,chunk):
        if not 0<=lam<=1 or not isinstance(chunk,int) or chunk<1:
            raise ValueError('Invalid lambda/chunk')
        if teacher_logits.shape!=(h.shape[0],w.shape[0]) or labels.shape!=(h.shape[0],):
            raise ValueError('Teacher logits/shifted labels do not match student positions and vocabulary')
        ce=torch.zeros((),device=h.device,dtype=torch.float32);kl=torch.zeros_like(ce)
        dh=torch.zeros_like(h);dw=torch.zeros_like(w)
        for start in range(0,len(h),chunk):
            x=h[start:start+chunk];target=labels[start:start+chunk];valid=target!=-100
            ls=F.log_softmax((x@w.T).float(),-1)
            lt=F.log_softmax(teacher_logits[start:start+chunk].float(),-1);pt=lt.exp()
            ce.add_(F.nll_loss(ls,target,ignore_index=-100,reduction='sum'))
            kl.add_(((pt*(lt-ls)).sum(-1)*valid).sum())
            dl=ls.exp().sub_(pt,alpha=lam)
            dl[torch.arange(len(target),device=target.device),target.clamp_min(0)]-=1-lam
            dl.mul_(valid[:,None])
            dh[start:start+chunk]=dl@w
            dw.add_((dl.T@x).to(dw.dtype))
        ctx.save_for_backward(dh,dw);ctx.mark_non_differentiable(ce,kl)
        return (1-lam)*ce+lam*kl,ce,kl

    @staticmethod
    @once_differentiable
    def backward(ctx,grad,_ce,_kl):
        dh,dw=ctx.saved_tensors
        return dh*grad,dw*grad,None,None,None,None


def logits_kd(hidden,weight,logits,labels,lam=.25,chunk=1024):
    return LogitsKD.apply(hidden[:,:-1].reshape(-1,hidden.shape[-1]),weight,
                          logits,labels[:,1:].reshape(-1),lam,chunk)


@dataclass
class Ticket:
    key: int
    batch: dict
    logits: torch.Tensor
    ready: torch.cuda.Event
    slot: int
    versions: dict
    inputs: int
    teacher_start: torch.cuda.Event
    acquired: bool=False
    released: bool=False


class OnlineTeacherPipeline:
    def __init__(self,teacher,*,positions,vocab_size,chunk=1024,overlap=True):
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError('Teacher must be eval and frozen')
        self.teacher=teacher;self.device=next(teacher.parameters()).device
        if self.device.type!='cuda':raise ValueError('CUDA teacher required')
        if teacher.config.vocab_size!=vocab_size:raise ValueError('Vocabulary mismatch')
        self.chunk=chunk;self.overlap=overlap
        self.consumer=torch.cuda.current_stream(self.device)
        self.producer=torch.cuda.Stream(device=self.device) if overlap else self.consumer
        self.buffers=[torch.empty((positions,vocab_size),device=self.device,dtype=torch.bfloat16) for _ in range(2)]
        self.producer.wait_stream(self.consumer)
        self.free=[None,None];self.slots=[None,None];self.queue=[];self.last_key=-1
        self.teacher_inputs=0;self.student_inputs=0;self.discarded_teacher_inputs=0
        self.closed=False

    def prefetch(self,key,batch,*,nonpadding_inputs):
        if self.closed:raise RuntimeError('Pipeline closed')
        if key<=self.last_key:raise ValueError('Keys must strictly increase in canonical order')
        if len(self.queue)>=2 or all(x is not None for x in self.slots):
            raise RuntimeError('Two-slot bound reached; release the consumed batch')
        for name in ('input_ids','labels','document_ids'):
            if name not in batch or batch[name].device!=self.device:raise ValueError('GPU batch/document spans required')
        expected=batch['input_ids'].shape[0]*(batch['input_ids'].shape[1]-1)
        if expected!=self.buffers[0].shape[0]:raise ValueError('Fixed microbatch/context geometry required')
        if not 0<nonpadding_inputs<=batch['input_ids'].numel():raise ValueError('Invalid input counter')
        slot=next(i for i,x in enumerate(self.slots) if x is None)
        # Tensor writes on the input stream must finish before teacher reads.
        inputs_ready=torch.cuda.Event();inputs_ready.record(self.consumer)
        self.producer.wait_event(inputs_ready)
        if self.free[slot] is not None:self.producer.wait_event(self.free[slot])
        start=torch.cuda.Event(enable_timing=True);ready=torch.cuda.Event(enable_timing=True)
        self.teacher_inputs+=nonpadding_inputs;self.last_key=key
        with torch.cuda.stream(self.producer),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            start.record()
            for tensor in batch.values():
                if isinstance(tensor,torch.Tensor):tensor.record_stream(self.producer)
            h=self.teacher.model(input_ids=batch['input_ids'],attention_mask=batch['document_ids'],
                                  use_cache=False,return_dict=True).last_hidden_state
            h=h[:,:-1].reshape(-1,h.shape[-1]);weight=self.teacher.get_output_embeddings().weight
            out=self.buffers[slot]
            out.record_stream(self.producer)
            for pos in range(0,expected,self.chunk):out[pos:pos+self.chunk].copy_(h[pos:pos+self.chunk]@weight.T)
            ready.record()
        ticket=Ticket(key,batch,out,ready,slot,{k:v._version for k,v in batch.items()},nonpadding_inputs,start)
        self.slots[slot]=ticket;self.queue.append(ticket)
        return ticket

    def acquire(self,key):
        if not self.queue or self.queue[0].key!=key:raise ValueError('Out-of-order or missing batch')
        ticket=self.queue[0]
        if ticket.acquired:raise RuntimeError('Batch already acquired')
        if any(v._version!=ticket.versions[k] for k,v in ticket.batch.items()):raise RuntimeError('Prefetched batch was mutated')
        self.consumer.wait_event(ticket.ready)
        ticket.logits.record_stream(self.consumer)
        ticket.acquired=True;self.student_inputs+=ticket.inputs
        return ticket

    def release(self,ticket):
        if not ticket.acquired or ticket.released or not self.queue or self.queue[0] is not ticket:
            raise RuntimeError('Release requires the current acquired batch exactly once')
        event=torch.cuda.Event();event.record(self.consumer)
        self.free[ticket.slot]=event;self.slots[ticket.slot]=None
        self.queue.pop(0);ticket.released=True

    def drain(self,*,discard_pending=False):
        if self.queue and not discard_pending:raise RuntimeError('Consume or explicitly discard pending teacher work')
        # Ensure no consumer read or teacher write is outstanding before teardown.
        self.consumer.synchronize();self.producer.synchronize()
        self.discarded_teacher_inputs+=sum(t.inputs for t in self.queue if not t.acquired)
        self.queue.clear();self.slots=[None,None];self.closed=True
        return dict(teacher_inputs=self.teacher_inputs,student_inputs=self.student_inputs,
                    discarded_teacher_inputs=self.discarded_teacher_inputs)


def attach_pipeline_loss(student,method,tokenizer_receipt):
    """Teacher tickets are mandatory on training forwards; DDP wrapper stays intact."""
    import types
    from transformers.modeling_outputs import CausalLMOutputWithPast
    if not tokenizer_receipt.get('equivalent_tokenization_verified'):raise ValueError('Tokenizer verification required')
    if method['temperature']!=1.:raise ValueError('Only T=1 supported')
    original=student.forward
    def forward(self,input_ids=None,attention_mask=None,labels=None,teacher_ticket=None,**kwargs):
        if not self.training or labels is None:
            return original(input_ids=input_ids,attention_mask=attention_mask,labels=labels,**kwargs)
        t=teacher_ticket
        if t is None or not t.acquired or t.released:raise ValueError('Live acquired teacher ticket required')
        for a,b in [(input_ids,t.batch['input_ids']),(labels,t.batch['labels']),(attention_mask,t.batch['document_ids'])]:
            if a is not b:raise ValueError('Student batch differs from the teacher ticket')
        for key in ('num_items_in_batch','use_cache','logits_to_keep','num_logits_to_keep'):kwargs.pop(key,None)
        hidden=self.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True,**kwargs).last_hidden_state
        loss,_,_=logits_kd(hidden,self.get_output_embeddings().weight,t.logits,labels,
                          lam=method['teacher_weight'],chunk=method['logit_chunk'])
        return CausalLMOutputWithPast(loss=loss,logits=None)
    student.forward=types.MethodType(forward,student)
