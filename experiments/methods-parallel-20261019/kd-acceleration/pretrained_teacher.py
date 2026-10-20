"""Opt-in pretrained teacher adapter; model loading is compute-node work.

The caller must pin weight files and verify tokenizer vocabulary/normalization/
special-token equivalence before passing the verification receipt. The teacher
keeps its own hidden size, depth, head geometry and RoPE configuration.
"""
import types
import torch
from fast_kd import configured_loss


def attach(student, teacher, method, tokenizer_receipt):
    from transformers.modeling_outputs import CausalLMOutputWithPast
    if not tokenizer_receipt.get('equivalent_tokenization_verified'):
        raise ValueError('Verify actual tokenization and token-ID mapping before logit KD')
    if student.config.vocab_size != teacher.config.vocab_size:
        raise ValueError('Aligned vocabulary required; vocabulary size alone is not proof of alignment')
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.config.use_cache=False
    loss_function=configured_loss(method)
    original=student.forward
    stats=[]

    def forward(self,input_ids=None,attention_mask=None,labels=None,**kwargs):
        if not self.training or labels is None:
            return original(input_ids=input_ids,attention_mask=attention_mask,labels=labels,**kwargs)
        for key in ('num_items_in_batch','use_cache','logits_to_keep','num_logits_to_keep'):
            kwargs.pop(key,None)
        h=self.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,
                     return_dict=True,**kwargs).last_hidden_state
        with torch.no_grad():
            th=teacher.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,
                             return_dict=True,**kwargs).last_hidden_state
        loss,ce,kl=loss_function(h,self.get_output_embeddings().weight,th,
                                teacher.get_output_embeddings().weight,labels)
        # Caller consumes/clears this bounded scalar receipt after each microbatch.
        stats[:]=[ce.detach(),kl.detach(),(labels[:,1:]!=-100).sum().detach()]
        return CausalLMOutputWithPast(loss=loss,logits=None)

    student.forward=types.MethodType(forward,student)
    # Explicit handle for profiling; teacher is not a registered student parameter.
    return {'teacher':teacher,'last_microbatch_stats':stats,'original_forward':original}
