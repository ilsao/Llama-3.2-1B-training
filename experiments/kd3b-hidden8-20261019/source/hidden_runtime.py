"""Student integration: frozen teacher head, bounded async reads, ACK after backward."""
import concurrent.futures
import os
import time
import types
from pathlib import Path
import torch
from safetensors import safe_open
from training_state import atomic_json, read_json
from spool import Spool, wait_for_roles
from hidden_contracts import SHAPE, BYTES, SLOTS, identity, ready_record, validate_peers
from hidden_common import byteview, verify_teacher, loader_preflight, numerical_check, projection_digest


class PipelineRuntime:
    def __init__(self, cfg, run, rows, evaluation, provenance, accelerator):
        import train
        self.cfg,self.run,self.accelerator,self.provenance=cfg,run,accelerator,provenance
        self.root=run/'transport';self.root.mkdir(parents=True,exist_ok=True)
        self.out=self.root/'consumer';self.out.mkdir(exist_ok=True)
        self.next_key=0;self.transaction=None;self.ticket=None;self.pending=None
        self.consumed_inputs=0;self.window_inputs=0;self.read_seconds=0.;self.wait_seconds=0.
        self.manifest=verify_teacher(cfg,accelerator,run)
        checks=dict(real_loader=loader_preflight(rows,cfg,accelerator),
                    hidden_kd=numerical_check(accelerator.device))
        train.on_main(accelerator,lambda:atomic_json(run/'pipeline-preflight.json',checks))
        teacher_root=Path(cfg['parallel_method']['teacher']['export'])
        index=read_json(teacher_root/'model.safetensors.index.json')['weight_map']
        with safe_open(str(teacher_root/index['model.embed_tokens.weight']),framework='pt',device='cpu') as f:
            self.head=f.get_tensor('model.embed_tokens.weight').to(accelerator.device)
        if self.head.shape!=(128256,3072) or self.head.dtype!=torch.bfloat16:
            raise ValueError('Wrong frozen teacher head')
        self.head.requires_grad_(False)
        self.pinned=[torch.empty(SHAPE,dtype=torch.bfloat16,pin_memory=True) for _ in range(2)]
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.training_end_epoch=int(os.environ['TRAIN_END_EPOCH'])-900
        self.deadline=self.training_end_epoch-300

    def attach(self, student):
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from fast_kd import PrecomputedKD
        import online_pipeline
        original=student.forward;runtime=self;method=self.cfg['parallel_method']
        def forward(model,input_ids=None,attention_mask=None,labels=None,teacher_ticket=None,**kwargs):
            if not model.training or labels is None:
                return original(input_ids=input_ids,attention_mask=attention_mask,labels=labels,**kwargs)
            t=teacher_ticket
            if t is None or t is not runtime.ticket or t.released:
                raise ValueError('Current unconsumed hidden ticket required')
            for key,value in (('input_ids',input_ids),('document_ids',attention_mask),('labels',labels)):
                if value is not t.batch[key]:raise ValueError('Teacher/student batch mismatch')
            for key in ('num_items_in_batch','use_cache','logits_to_keep','num_logits_to_keep'):kwargs.pop(key,None)
            h=model.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True,**kwargs).last_hidden_state
            loss,ce,kl=PrecomputedKD.apply(h[:,:-1].reshape(-1,h.shape[-1]),model.get_output_embeddings().weight,
                t.hidden,runtime.head,labels[:,1:].reshape(-1),method['teacher_weight'],method['logit_chunk'])
            stats=torch.stack((ce.detach().double(),kl.detach().double(),(labels[:,1:]!=-100).sum().double()))
            online_pipeline._STATS=stats if online_pipeline._STATS is None else online_pipeline._STATS+stats
            return CausalLMOutputWithPast(loss=loss,logits=None)
        student.forward=types.MethodType(forward,student)
        if any(p is self.head for p in student.parameters()):raise ValueError('Teacher head registered in student')

    def start(self):
        from train import on_main
        self.accelerator.wait_for_everyone()
        on_main(self.accelerator,lambda:atomic_json(self.out/'ready.json',ready_record(self.cfg,self.provenance)))
        peers=wait_for_roles(self.root,('producer','consumer'),int(os.environ['TRAIN_START_EPOCH'])+480)
        self.peers=peers
        self.producer_done=None
        common_end=validate_peers(peers)
        self.training_end_epoch=common_end-900
        self.deadline=common_end-300
        if common_end-time.time()<5.856118962333093*3600:
            raise RuntimeError('Peer scheduling delay leaves insufficient conservative completion margin')
        self.queue=Spool(self.root,self.accelerator.process_index,'hidden',BYTES,self.deadline,slots=SLOTS)
        on_main(self.accelerator,lambda:atomic_json(self.out/'paired.json',dict(peers=peers,common_end=common_end)))

    def reserve_window(self, inputs):
        self.boundary();self.transaction=True;self.window_inputs=inputs

    def begin_window(self, local):
        if self.transaction is None:raise ValueError('Student input reservation missing')
        self.base_key=self.next_key
        self.cpu=local
        self.identities=[identity(self.base_key+i,self.accelerator.process_index,b,self.cfg,self.provenance) for i,b in enumerate(local)]
        self.batches=[{k:v.to(self.accelerator.device) for k,v in b.items()} for b in local]
        if sum(int(b['attention_mask'].sum()) for b in local)*8!=self.window_inputs:
            raise ValueError('Reserved input count differs')
        self.next_key+=len(local)
        self.pending=self.pool.submit(self.read,0)

    def read(self, index):
        return self.queue.read_into(self.base_key+index,byteview(self.pinned[index%2]),self.identities[index])

    def batch(self, index):
        if self.ticket is not None:raise ValueError('Unreleased prior ticket')
        start=time.perf_counter();io=self.pending.result();self.wait_seconds+=time.perf_counter()-start
        self.read_seconds+=io['read_seconds'];self.pending=None
        payload=self.pinned[index%2].to(self.accelerator.device,non_blocking=True)
        torch.cuda.synchronize()
        if index+1<len(self.batches):self.pending=self.pool.submit(self.read,index+1)
        key=self.base_key+index
        if key==0:
            expected=read_json(self.queue.directory/'projection-check.json')['sha256']
            if projection_digest(payload,self.head)!=expected:raise ValueError('Transported hidden projection checksum differs')
            atomic_json(self.out/f'projection-check-rank{self.accelerator.process_index}.json',dict(passed=True,sha256=expected))
        self.ticket=types.SimpleNamespace(key=key,hidden=payload,batch=self.batches[index],released=False)
        return {**self.batches[index],'teacher_ticket':self.ticket}

    def release(self):
        if self.ticket is None:raise ValueError('No current ticket')
        torch.cuda.synchronize()
        self.queue.acknowledge(self.ticket.key)
        self.ticket.released=True;self.ticket=None

    def confirm_window(self):
        if self.ticket is not None or self.pending is not None:raise ValueError('Unconsumed window')
        self.consumed_inputs+=self.window_inputs
        self.transaction=None;self.batches=[];self.cpu=[];self.identities=[]

    def boundary(self):
        if self.transaction is not None or self.ticket is not None or self.pending is not None:
            raise ValueError('Outstanding student hidden consumption at boundary')

    def accounting(self):
        from train import on_main
        self.boundary()
        teacher=on_main(self.accelerator,lambda:read_json(self.root/'producer/teacher-input-journal.json'))
        return dict(teacher_inputs_confirmed=teacher['confirmed_tokens'],teacher_inputs_charged=teacher['charged_tokens'],
            student_hidden_inputs_consumed=self.consumed_inputs,teacher_prefetch_may_be_ahead=True,
            local_read_seconds=self.read_seconds,local_exposed_wait_seconds=self.wait_seconds)

    def stop_training(self, reason):
        from train import on_main
        self.boundary()
        on_main(self.accelerator,lambda:atomic_json(self.root/'consumer-stopped.json',
            dict(reason=reason,student_inputs=self.consumed_inputs,timestamp=time.time())))

    def progress(self):
        from train import on_main, reduce_max
        teacher=on_main(self.accelerator,lambda:read_json(self.root/'producer/teacher-input-journal.json'))
        timings=reduce_max(torch.tensor([self.read_seconds,self.wait_seconds],device=self.accelerator.device),self.accelerator).cpu().tolist()
        return {'train/teacher_inference_inputs':teacher['confirmed_tokens'],
            'train/teacher_charged_inputs':teacher['charged_tokens'],
            'train/conservative_total_model_inputs':teacher['charged_tokens']+self.consumed_inputs,
            'train/hidden_read_seconds_cumulative_max_rank':timings[0],
            'train/hidden_wait_seconds_cumulative_max_rank':timings[1]}

    def resource_hours(self):
        now=time.time()
        done=self.root/'producer/done.json'
        if self.producer_done is None and done.exists():self.producer_done=read_json(done)
        teacher_end=self.producer_done['timestamp'] if self.producer_done else now
        return 8*((now-self.peers['consumer']['start_epoch'])+
            (teacher_end-self.peers['producer']['start_epoch']))/3600

    def close(self):
        self.boundary();self.pool.shutdown(wait=True)
