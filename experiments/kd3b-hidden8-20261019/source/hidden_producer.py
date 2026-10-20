"""Eight-rank pretrained teacher. Canonical dense rows, separate durable input journal."""
import os
import sys
import time
from pathlib import Path


def main():
    import train
    train.require_compute()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import AutoModelForCausalLM
    from training_state import DEFAULT_ROOT, load_config, TokenJournal, atomic_json, ExperimentLedger, digest
    from hidden_contracts import RUN, SHAPE, BYTES, SLOTS, identity, ready_record, validate_peers
    from hidden_common import verify_teacher, loader_preflight, byteview, projection_digest
    from spool import Spool, wait_for_roles
    from ppl_contracts import window
    import teacher_attention
    cfg=load_config(sys.argv[1]);accelerator=Accelerator(mixed_precision='bf16')
    if accelerator.num_processes!=8 or 'H200' not in torch.cuda.get_device_name():
        raise ValueError('Eight H200 ranks required')
    rank=accelerator.process_index
    root=DEFAULT_ROOT/'runs/experiments'/RUN/'transport';out=root/'producer';out.mkdir(parents=True,exist_ok=True)
    allocation=train.on_main(accelerator,lambda:ExperimentLedger(DEFAULT_ROOT).allocation(os.environ['SLURM_JOB_ID'],RUN))
    if allocation['config_digest']!=digest(cfg) or allocation.get('role')!='producer':raise ValueError('Producer reservation mismatch')
    train.MASK_MODE='intradoc'
    rows,evaluation,provenance=train.prepare_data(DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008',out,cfg,accelerator)
    loader_preflight(rows,cfg,accelerator)
    set_seed(42)
    checks=teacher_attention.check(accelerator.device)
    train.on_main(accelerator,lambda:atomic_json(out/'preflight.json',dict(real_loader=True,attention=checks)))
    manifest=verify_teacher(cfg,accelerator,out)
    teacher=AutoModelForCausalLM.from_pretrained(cfg['parallel_method']['teacher']['export'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(accelerator.device)
    teacher.requires_grad_(False).eval();teacher.config.use_cache=False;teacher_attention.configure(teacher)
    head=teacher.get_output_embeddings().weight
    # The teacher evaluation is separately counted held-out inference, never student initialization.
    os.environ['EVAL_DOCUMENT_OUTPUT']=str(out/'teacher-selection-documents.json')
    measured=train.evaluate(teacher,evaluation,cfg,accelerator)
    if measured['eval/prediction_tokens']!=1479302:raise ValueError('Teacher evaluation membership changed')
    train.on_main(accelerator,lambda:atomic_json(out/'teacher-evaluation.json',dict(measured,model_id=manifest['model_id'],revision=manifest['revision'])))
    journal=None
    def open_journal():
        nonlocal journal
        if (out/'teacher-input-journal.json').exists():raise ValueError('No producer restart')
        journal=TokenJournal(out/'teacher-input-journal.json',cfg['max_training_tokens'])
    train.on_main(accelerator,open_journal)
    accelerator.wait_for_everyone()
    train.on_main(accelerator,lambda:atomic_json(out/'ready.json',ready_record(cfg,provenance)))
    peers=wait_for_roles(root,('producer','consumer'),int(os.environ['TRAIN_START_EPOCH'])+480)
    end=validate_peers(peers)
    if end-time.time()<5.856118962333093*3600:raise RuntimeError('Insufficient paired completion margin')
    queue=Spool(root,rank,'hidden',BYTES,end-300,slots=SLOTS)
    cpu_buffer=torch.empty(SHAPE,dtype=torch.bfloat16,pin_memory=True)
    tokens=0;key=0;updates=0;compute_seconds=0.;write_seconds=0.;wait_seconds=0.
    started=time.time()
    while tokens<cfg['target_training_tokens']:
        # Collective decision only at complete teacher-window boundaries.
        stop=train.on_main(accelerator,lambda:(root/'consumer-stopped.json').exists())
        if stop:break
        if time.time()>=end-300:raise TimeoutError('Producer stopping margin')
        width,accumulation=window(cfg,tokens)
        local=[train.collate_rows([rows[(tokens//8192)+i*8+rank]],8192) for i in range(accumulation)]
        local_inputs=sum(int(b['attention_mask'].sum()) for b in local)
        inputs=int(accelerator.reduce(torch.tensor(local_inputs,device=accelerator.device),reduction='sum'))
        if inputs!=width*8192:raise ValueError('Dense canonical teacher window changed')
        transaction=train.on_main(accelerator,lambda:journal.reserve(inputs))
        for cpu in local:
            queue.guard()
            if key>=SLOTS:
                from spool import wait_for
                wait_seconds+=wait_for(queue.path(key-SLOTS,'ack'),queue.deadline,root)
            mark=time.perf_counter()
            batch={k:v.to(accelerator.device) for k,v in cpu.items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                hidden=teacher.model(input_ids=batch['input_ids'],attention_mask=batch['document_ids'],
                    use_cache=False,return_dict=True).last_hidden_state[:,:-1].reshape(SHAPE)
                if key==0:
                    atomic_json(queue.directory/'projection-check.json',dict(sha256=projection_digest(hidden,head)))
                cpu_buffer.copy_(hidden,non_blocking=True)
            torch.cuda.synchronize();compute_seconds+=time.perf_counter()-mark
            io=queue.publish(key,byteview(cpu_buffer),identity(key,rank,cpu,cfg,provenance))
            write_seconds+=io['write_seconds'];wait_seconds+=io['backpressure_seconds']
            key+=1
            del hidden,batch
        accelerator.wait_for_everyone()
        train.on_main(accelerator,lambda:journal.confirm(transaction))
        tokens+=inputs;updates+=1
        if updates==1 or updates%10==0 or tokens==cfg['target_training_tokens']:
            service=compute_seconds+write_seconds
            service_max=float(train.reduce_max(torch.tensor(service,device=accelerator.device),accelerator))
            record=dict(teacher_inputs=tokens,produced_updates=updates,rank_payloads=key,wall_seconds=time.time()-started,
                service_seconds_max_rank=service_max,service_tokens_per_second=tokens/service_max,
                local_write_seconds=write_seconds,local_backpressure_seconds=wait_seconds,timestamp=time.time())
            train.on_main(accelerator,lambda:atomic_json(out/'progress.json',record))
            if updates%100==0 and rank==0:print(record,flush=True)
    complete=tokens==cfg['target_training_tokens']
    train.on_main(accelerator,lambda:atomic_json(out/'done.json',dict(complete=complete,teacher_inputs=tokens,
        job_id=os.environ['SLURM_JOB_ID'],timestamp=time.time(),exit_after_last_publication=True)))
    accelerator.wait_for_everyone()
    return 0 if complete else 75


if __name__=='__main__':
    import traceback
    from training_state import DEFAULT_ROOT,atomic_json
    from hidden_contracts import RUN
    code=0
    try:code=main()
    except BaseException as error:
        root=DEFAULT_ROOT/'runs/experiments'/RUN/'transport';root.mkdir(parents=True,exist_ok=True)
        atomic_json(root/f'abort-producer-{os.environ.get("RANK","unknown")}.json',dict(error=repr(error),timestamp=time.time()))
        traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
