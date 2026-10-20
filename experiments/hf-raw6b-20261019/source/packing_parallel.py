"""Canonical shard planning and bounded parallel packing/verification workers.

Only metadata and <=one sequence of carry cross process boundaries. Workers
write unique immutable files; the coordinator alone commits original state.
"""
import contextlib
import multiprocessing
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import dataset as pipeline
from preparation_parallel import bounded_map, configure_threads, worker_count

CONTEXT={}


def initialize_worker(context,cpus):
    global CONTEXT
    configure_threads()
    number=(multiprocessing.current_process()._identity[-1]-1)%len(cpus)
    os.sched_setaffinity(0,{cpus[number]})
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    CONTEXT=context


class Workers:
    def __init__(self,config,context=None):
        pipeline.require_compute();self.count=worker_count(config)
        self.allowed=sorted(os.sched_getaffinity(0));self.pids=set()
        if len(self.allowed)<self.count+1:raise ValueError('Need workers plus coordinator CPUs')
        self.executor=ProcessPoolExecutor(max_workers=self.count,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(context or {},self.allowed[1:self.count+1]))
    def __enter__(self):
        os.sched_setaffinity(0,{self.allowed[0]});return self
    def run(self,function,tasks):
        for value in bounded_map(self.executor,function,tasks,2*self.count,pipeline.stop_if_requested):
            if isinstance(value,dict) and 'pid' in value:self.pids.add(value['pid'])
            yield value
    def __exit__(self,*args):
        try:self.executor.shutdown(wait=True,cancel_futures=True)
        finally:os.sched_setaffinity(0,set(self.allowed))
    @property
    def stats(self):return dict(workers=self.count,worker_pids=sorted(self.pids),maximum_outstanding_tasks=2*self.count)


def document_metadata(files,cursor=(0,0)):
    import pyarrow as pa
    import pyarrow.compute as pc
    for shard in range(cursor[0],len(files)):
        pipeline.stop_if_requested()
        with pa.memory_map(files[shard]['path'],'r') as source:
            table=pa.ipc.open_stream(source).read_all()
            if len(table)!=files[shard]['rows']:raise ValueError('Document shard row count changed')
            lengths=pc.list_value_length(table['tokens']).to_pylist()
            rows=table['row_index'].to_pylist()
            first=cursor[1] if shard==cursor[0] else 0
            for row in range(first,len(table)):
                pipeline.stop_if_requested()
                after=[shard+1,0] if row+1==len(table) else [shard,row+1]
                yield rows[row],lengths[row],[shard,row],after


def advance_carry(refs,row,count,location,length):
    """Token-count-only counterpart of original TrainingPacker.push."""
    offset=sum(ref['length'] for ref in refs)
    combined=refs+[dict(row_index=row,document_start=0,sequence_start=offset,
                       length=count,location=location,token_offset=0)]
    sequences=(offset+count)//length;cut=sequences*length;result=[]
    for ref in combined:
        taken=min(ref['length'],max(0,cut-ref['sequence_start']))
        if ref['length']>taken:
            result.append({**ref,'length':ref['length']-taken,
                'sequence_start':max(0,ref['sequence_start']-cut),
                'document_start':ref['document_start']+taken,'token_offset':ref['token_offset']+taken})
    return sequences,result


def decode_carry(refs,files,initial):
    import pyarrow as pa
    tokens=[];labels=[];spans=[]
    for ref in refs:
        start=ref['token_offset'];count=ref['length']
        if ref['location'] is None:
            piece=initial['tokens'][start:start+count];targets=initial['labels'][start:start+count]
        else:
            shard,row=ref['location']
            with pa.memory_map(files[shard]['path'],'r') as source:
                table=pa.ipc.open_stream(source).read_all()
                piece=table['tokens'][row].values.slice(start,count).to_pylist()
            targets=([-100]+piece[1:]) if start==0 else list(piece)
        tokens.extend(piece);labels.extend(targets)
        spans.append({k:ref[k] for k in ('row_index','document_start','sequence_start','length')})
    return dict(tokens=tokens,labels=labels,spans=spans)


def training_plans(files,cursor,carry,length,shard_rows,metadata=None,decoder=None):
    initial={k:list(carry.get(k,[])) for k in ('tokens','labels','spans')}
    refs=[{**s,'location':None,'token_offset':s['sequence_start']} for s in initial['spans']]
    decode=decoder or (lambda r:decode_carry(r,files,initial))
    begin=list(cursor);end=list(cursor);start_carry=initial;outputs=0;documents=0
    source=document_metadata(files,cursor) if metadata is None else metadata
    for row,count,location,end in source:
        if count<2:raise ValueError('Selected document lacks BOS/EOS')
        emitted,refs=advance_carry(refs,row,count,location,length)
        outputs+=emitted;documents+=1
        if outputs>=shard_rows:
            after=decode(refs)
            yield dict(start=begin,end=list(end),carry=start_carry,carry_after=after,
                       final=False,expected_rows=outputs,documents=documents)
            begin=list(end);start_carry=after;outputs=0;documents=0
    if documents or refs:
        yield dict(start=begin,end=list(end),carry=start_carry,
                   carry_after=dict(tokens=[],labels=[],spans=[]),final=True,
                   expected_rows=outputs+bool(refs),documents=documents)


def evaluation_plans(files,cursor,length,shard_rows):
    begin=list(cursor);end=list(cursor);outputs=0;documents=0
    for row,count,location,end in document_metadata(files,cursor):
        outputs+=(count-2)//(length-1)+1;documents+=1
        if outputs>=shard_rows:
            yield dict(start=begin,end=list(end),expected_rows=outputs,documents=documents)
            begin=list(end);outputs=0;documents=0
    if documents:yield dict(start=begin,end=list(end),expected_rows=outputs,documents=documents)


def task_input(files,plan):
    first,offset=plan['start'];last,last_row=plan['end']
    stop=last+(last_row>0)
    return dict(files=files[first:stop],first_row=offset,last_row=last_row)


def task_documents(task):
    import pyarrow as pa
    files=task['input']['files']
    for number,item in enumerate(files):
        pipeline.stop_if_requested()
        with pa.memory_map(item['path'],'r') as source:
            table=pa.ipc.open_stream(source).read_all()
            start=task['input']['first_row'] if number==0 else 0
            end=task['input']['last_row'] if number+1==len(files) and task['input']['last_row'] else len(table)
            for batch in table.slice(start,end-start).to_batches(max_chunksize=32):
                for row in batch.to_pylist():yield row


def sequence_task(task):
    started=time.monotonic();plan=task['plan'];length=task['length'];pad=task['pad_id']
    if task['phase']=='train':
        packer=pipeline.TrainingPacker(length,pad,{k:list(v) for k,v in plan['carry'].items()})
        def expected():
            for doc in task_documents(task):yield from packer.push(doc['row_index'],doc['tokens'])
            if plan['final']:yield from packer.finish()
    else:
        def expected():
            for doc in task_documents(task):yield from pipeline.pack_evaluation_document(doc,length,pad)
    if task.get('actual') is not None:
        stats=pipeline.validate_sequences(pipeline.iter_arrow([task['actual']]),expected(),length)
        if stats['sequences']!=plan['expected_rows']:raise ValueError('Planned verification row count changed')
        return dict(**stats,pid=os.getpid(),seconds=time.monotonic()-started)
    rows=list(expected());computed=time.monotonic()
    if len(rows)!=plan['expected_rows']:raise ValueError('Parallel shard boundary differs from original packing')
    if task['phase']=='train' and packer.state()!=plan['carry_after']:
        raise ValueError('Parallel packing carry differs from canonical plan')
    stats=dict(input_tokens=sum(sum(r['attention_mask']) for r in rows),
        prediction_tokens=sum(sum(v!=-100 for v in r['labels'][1:]) for r in rows),
        subset_prediction_tokens=sum(sum(v!=-100 for v in r['labels'][1:]) for r in rows
            if 0<=r['evaluation_slot']<task['subset_documents']))
    file=pipeline.write_arrow(Path(task['output']),rows,pipeline.sequence_schema(length))
    return dict(file=file,**stats,cursor=plan['end'],carry=plan.get('carry_after'),pid=os.getpid(),
                compute_seconds=computed-started,write_seconds=time.monotonic()-computed)


def make_tasks(files,plans,phase,config,pad,output=None,actual=None):
    count=0
    for count,plan in enumerate(plans,1):
        item=dict(plan=plan,input=task_input(files,plan),phase=phase,length=config['sequence_length'],
                  pad_id=pad,subset_documents=config['evaluation_subset_documents'])
        if output is not None:item['output']=str(output/f'{phase}-{count-1:06d}.arrow')
        if actual is not None:
            if count>len(actual):raise ValueError('Extra planned packed shard')
            item['actual']=actual[count-1]
        yield item
    if actual is not None and count!=len(actual):raise ValueError('Missing planned packed shard')


def checksum_task(item):
    pipeline.check_artifact(item,True);return dict(pid=os.getpid())


def packing(prep,directory,identity):
    cfg=prep.config;selected=prep.manifest('select')['details'];path=prep.directory('pack')/'state.json'
    state=pipeline.state_for(path,dict(train_files=[],eval_files=[],train_cursor=[0,0],eval_cursor=[0,0],
        carry={},train_complete=False,training_tokens=0,evaluation_input_tokens=0,
        evaluation_prediction_tokens=0,subset_prediction_tokens=0))
    initial=state['training_tokens'];started=time.monotonic();last_log=started
    output=Path(directory)/('outputs-'+os.environ['SLURM_JOB_ID']);output.mkdir(exist_ok=False)
    with Workers(cfg) as workers:
        list(workers.run(checksum_task,state['train_files']+state['eval_files']))
        def commit(phase):
            nonlocal last_log
            pipeline.atomic_json(path,state)
            now=time.monotonic();rate=(state['training_tokens']-initial)/(now-started)
            record=dict(identity=identity,job_id=os.environ['SLURM_JOB_ID'],phase=phase,
                timestamp=time.time(),session_seconds=now-started,training_tokens=state['training_tokens'],
                evaluation_input_tokens=state['evaluation_input_tokens'],training_tokens_per_second=rate,
                remaining_training_seconds=(selected['train_tokens']-state['training_tokens'])/rate if rate>0 else None,
                committed_train_shards=len(state['train_files']),committed_eval_shards=len(state['eval_files']),**workers.stats)
            pipeline.atomic_json(Path(directory)/'progress.json',record)
            pipeline.atomic_json(prep.path/'packing-access/parallel-v1/progress.json',record)
            if now-last_log>=20:
                print('parallel pack progress:',record,flush=True);last_log=now
        if not state['train_complete']:
            plans=training_plans(selected['train_files'],state['train_cursor'],state['carry'],cfg['sequence_length'],cfg['sequence_shard_rows'])
            tasks=make_tasks(selected['train_files'],plans,'train',cfg,selected['pad_token_id'],output=output)
            for value in workers.run(sequence_task,tasks):
                state['train_files'].append(value['file']);state['training_tokens']+=value['input_tokens']
                state['train_cursor']=value['cursor'];state['carry']=value['carry'];commit('train')
            state['train_complete']=True;commit('train-complete')
        if state['training_tokens']!=selected['train_tokens']:raise ValueError('Packing changed training token count')
        plans=evaluation_plans(selected['eval_files'],state['eval_cursor'],cfg['sequence_length'],cfg['sequence_shard_rows'])
        tasks=make_tasks(selected['eval_files'],plans,'eval',cfg,selected['pad_token_id'],output=output)
        for value in workers.run(sequence_task,tasks):
            state['eval_files'].append(value['file']);state['eval_cursor']=value['cursor']
            state['evaluation_input_tokens']+=value['input_tokens']
            state['evaluation_prediction_tokens']+=value['prediction_tokens']
            state['subset_prediction_tokens']+=value['subset_prediction_tokens'];commit('eval')
        commit('complete')
    paths=[item['path'] for item in state['train_files']+state['eval_files']]
    records=[{k:item[k] for k in ('path','size','sha256')} for item in state['train_files']+state['eval_files']]
    prep.finish('pack',paths,{**state,'sequence_length':cfg['sequence_length'],'pad_token_id':selected['pad_token_id'],
        'training_sequence_count':sum(item['rows'] for item in state['train_files']),
        'evaluation_sequence_count':sum(item['rows'] for item in state['eval_files']),
        'training_attention':'ordinary causal attention across EOS-separated training documents',
        'training_loss':'ignore structural BOS, first input position of each sequence, and padding',
        'evaluation_loss':'independent documents, stride 8191; each within-document target once'},records=records)
