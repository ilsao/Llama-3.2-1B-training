"""Meaningful GPU gates for loss equivalence, ticket ordering and slot reuse."""
import torch
from torch.nn import functional as F
from types import SimpleNamespace
from online_pipeline import LogitsKD,OnlineTeacherPipeline


def run_checks(device):
    records=[]
    for amp in (False,True):
        for lam,all_ignored,scale in ((0.,False,1.),(.25,False,3e-5),(1.,False,1.),(.25,True,1.)):
            h=torch.randn(23,32,device=device,requires_grad=True)
            w=(torch.randn(71,32,device=device)*.1).requires_grad_()
            th=torch.randn(23,48,device=device);tw=torch.randn(71,48,device=device)*.1
            labels=torch.randint(0,71,(23,),device=device);labels[::4]=-100
            if all_ignored:labels.fill_(-100)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                logits=th@tw.T;ls=F.log_softmax((h@w.T).float(),-1);lt=F.log_softmax(logits.float(),-1)
                reference=(1-lam)*F.nll_loss(ls,labels,ignore_index=-100,reduction='sum')+lam*((lt.exp()*(lt-ls)).sum(-1)*(labels!=-100)).sum()
                actual=LogitsKD.apply(h,w,logits,labels,lam,7)[0]
            rg=torch.autograd.grad(reference*scale,(h,w));ag=torch.autograd.grad(actual*scale,(h,w))
            errors=[float((a-b).norm()/a.norm().clamp_min(1e-12)) for a,b in zip(rg,ag)]
            assert max(errors)<(.02 if amp else 2e-5),errors
            torch.testing.assert_close(actual,reference,rtol=2e-6,atol=3e-5)
            records.append(dict(bf16=amp,lam=lam,ignored=all_ignored,relative_l2=errors))

    class TinyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(71,48,device=device,dtype=torch.bfloat16)
            self.config=SimpleNamespace(vocab_size=71)
        @property
        def model(self):return self
        def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        def get_output_embeddings(self):return self.embedding
    teacher=TinyTeacher().requires_grad_(False).eval()
    def reference_logits(ids):
        with torch.no_grad():
            h=teacher.embedding(ids)[:,:-1].reshape(-1,48)
            return torch.cat([h[i:i+7]@teacher.embedding.weight.T for i in range(0,len(h),7)])
    def batch(i):
        ids=(torch.arange(24,device=device)[None,:]+i)%71
        return dict(input_ids=ids,labels=ids.clone(),document_ids=torch.ones_like(ids))
    pipe=OnlineTeacherPipeline(teacher,positions=23,vocab_size=71,chunk=7)
    b0,b1,b2=batch(0),batch(1),batch(2)
    pipe.prefetch(0,b0,nonpadding_inputs=24);pipe.prefetch(1,b1,nonpadding_inputs=24)
    try:pipe.prefetch(2,b2,nonpadding_inputs=24)
    except RuntimeError:pass
    else:raise AssertionError('Missing two-slot bound')
    try:pipe.acquire(1)
    except ValueError:pass
    else:raise AssertionError('Missing ordering check')
    first=pipe.acquire(0);captured=first.logits.clone()
    pipe.release(first)
    try:pipe.release(first)
    except RuntimeError:pass
    else:raise AssertionError('Double release accepted')
    pipe.prefetch(2,b2,nonpadding_inputs=24)
    for i in (1,2):
        ticket=pipe.acquire(i)
        expected=reference_logits(ticket.batch['input_ids'])
        torch.testing.assert_close(ticket.logits,expected,rtol=0,atol=0)
        pipe.release(ticket)
    counts=pipe.drain();assert counts==dict(teacher_inputs=72,student_inputs=72,discarded_teacher_inputs=0)
    expected0=reference_logits(b0['input_ids'])
    torch.testing.assert_close(captured,expected0,rtol=0,atol=0)
    pipe=OnlineTeacherPipeline(teacher,positions=23,vocab_size=71,chunk=7)
    pending=batch(3);pipe.prefetch(0,pending,nonpadding_inputs=24)
    # Mutate only after producer finishes, to test detection without a data race.
    pipe.producer.synchronize();pending['labels'].add_(1)
    try:pipe.acquire(0)
    except RuntimeError:pass
    else:raise AssertionError('Mutated batch accepted')
    counts=pipe.drain(discard_pending=True);assert counts['discarded_teacher_inputs']==24 and counts['student_inputs']==0
    return {'loss_cases':records,'different_teacher_hidden_size':48,'student_hidden_size':32,
            'two_slot_bound':True,'ordering_and_mutation_checks':True,'reuse_without_overwrite':True,'discard_accounting':True}
