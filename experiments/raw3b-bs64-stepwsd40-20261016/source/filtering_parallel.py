"""Pinned CPU workers: one processing batch plus one prefetched batch each.

Workers never write artifacts or keep a raw shard cache. The coordinator returns
completed batches strictly in canonical order; dataset.py remains unchanged.
"""
import atexit
import multiprocessing
import os
from pathlib import Path
import queue
import threading
import time

import dataset as pipeline
from filtering_batch_reader import BatchRaw
from preparation_parallel import configure_threads
from filtering_resources import worker_count, training_worker_count, evaluation_worker_count

BATCH_DOCUMENTS=32


class FixtureTokenizer:
    bos_token_id=1;eos_token_id=9
    def encode(self,text,add_special_tokens=False):
        time.sleep(.003);number=int(text.rsplit(' ',1)[1]);return [2+number%5]*((number%7)+1)
    def save_pretrained(self,path):
        Path(path).mkdir(exist_ok=True);(Path(path)/'tokenizer.json').write_text('{}')


class EphemeralRaw(BatchRaw):
    """At most one active mapping while reading, zero retained shard mappings."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.max_open=1
    def _index(self,shard,source):
        if not (self.directory/f'shard-{shard:06d}.json').exists():
            raise ValueError('Workers require a complete prebuilt IPC index')
        return super()._index(shard,source)
    def fetch(self,indices,columns=None):
        if len(indices)>2048:raise ValueError('Raw selection exceeds bounded document batch')
        # Close between shard groups; only detached Python values enter prefetch.
        groups={};output=[None]*len(indices)
        for slot,row in enumerate(indices):
            shard,_=self.locate(row);groups.setdefault(shard,[]).append((slot,row))
        for group in groups.values():
            try:records=super().fetch([row for _,row in group],columns)
            finally:self.close()
            for (slot,_),record in zip(group,records):output[slot]=record
        return output


def put_until(target,value,stop):
    while not stop.is_set():
        try:target.put(value,timeout=.2);return True
        except queue.Full:pass
    return False


def get_until(target,stop):
    while not stop.is_set():
        try:return target.get(timeout=.2)
        except queue.Empty:pass
    return None


def worker_main(number,cpu,inbox,outbox,stop,index,directory,identity,config,fixture):
    os.sched_setaffinity(0,{cpu});configure_threads()
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    raw=EphemeralRaw(index,directory,identity)
    ready=queue.Queue(maxsize=1)
    try:
        if fixture:tokenizer=FixtureTokenizer()
        else:
            pipeline.initialize_tokenizer(config);tokenizer=pipeline._TOKENIZER
        def prefetch():
            try:
                while not stop.is_set():
                    task=get_until(inbox,stop)
                    if task is None:return
                    sequence,metadata=task;pipeline.stop_if_requested()
                    started=time.monotonic()
                    if fixture:time.sleep(.02)
                    documents=raw.fetch([row[0] for row in metadata])
                    loaded=(sequence,metadata,documents,time.monotonic()-started,dict(raw.stats),started,time.monotonic())
                    # A single ready slot; dispatcher grants at most two credits.
                    if not put_until(ready,loaded,stop):return
            except BaseException as error:
                put_until(ready,('error',type(error).__name__+': '+str(error)),stop)
            finally:raw.close()
        thread=threading.Thread(target=prefetch,name='one-batch-prefetch',daemon=True);thread.start()
        while not stop.is_set():
            loaded=get_until(ready,stop)
            if loaded is None:break
            if loaded[0]=='error':raise RuntimeError(loaded[1])
            sequence,metadata,documents,read_seconds,stats,read_start,read_end=loaded
            started=time.monotonic();results=[]
            for original,(row,pos,source,slot) in zip(documents,metadata):
                pipeline.stop_if_requested()
                if stop.is_set():return
                results.append(pipeline.make_document(original,row,pos,source,tokenizer,slot))
            del documents,loaded
            if not put_until(outbox,(sequence,results,dict(worker=number,pid=os.getpid(),cpu=cpu,
                    sequence=sequence,read_seconds=read_seconds,read_start=read_start,read_end=read_end,
                    tokenize_start=started,tokenize_end=time.monotonic(),tokenize_seconds=time.monotonic()-started,
                    reader_stats=stats)),stop):break
    except BaseException as error:
        put_until(outbox,('error',type(error).__name__+': '+str(error)),stop)
    finally:raw.close()


class OrderedBatches:
    """Two unconsumed batch credits per worker, including completed results."""
    def __init__(self,index,directory,identity,config,fixture=False,workers=None):
        pipeline.require_compute();self.index=index;self.directory=directory;self.identity=identity
        self.config=config;self.fixture=fixture;self.count=worker_count(config) if workers is None else workers
        if type(self.count) is not int or not 1<=self.count<=max(training_worker_count(config),evaluation_worker_count(config)):
            raise ValueError('Worker count exceeds authorized allocation')
        self.allowed=sorted(os.sched_getaffinity(0))
        if len(self.allowed)<self.count+1:raise ValueError('Need one allocated CPU per worker plus coordinator')
        self.ctx=multiprocessing.get_context('spawn');self.stop=self.ctx.Event()
        self.inputs=[self.ctx.Queue(maxsize=2) for _ in range(self.count)]
        self.outputs=[self.ctx.Queue(maxsize=2) for _ in range(self.count)]
        self.processes=[];self.pending=[0]*self.count;self.maximum=[0]*self.count
        self.metrics={};self.closed=False
        # Start children from the full Slurm affinity, then dedicate a coordinator CPU.
        for number in range(self.count):
            process=self.ctx.Process(target=worker_main,args=(number,self.allowed[number+1],
                self.inputs[number],self.outputs[number],self.stop,index,str(directory),identity,config,fixture))
            process.start();self.processes.append(process)
        os.sched_setaffinity(0,{self.allowed[0]})

    def ordered(self,items):
        iterator=iter(items);submitted=0;consumed=0;exhausted=False
        def dispatch():
            nonlocal submitted,exhausted
            if exhausted:return False
            pipeline.stop_if_requested()
            try:metadata=next(iterator)
            except StopIteration:exhausted=True;return False
            if not 0<len(metadata)<=BATCH_DOCUMENTS:raise ValueError('Invalid worker document batch')
            worker=submitted%self.count
            if self.pending[worker]>=2:raise ValueError('Worker batch credits exceeded')
            if not put_until(self.inputs[worker],(submitted,metadata),self.stop):raise RuntimeError('Worker stopped')
            self.pending[worker]+=1;self.maximum[worker]=max(self.maximum[worker],self.pending[worker]);submitted+=1
            return True
        for _ in range(2*self.count):
            if not dispatch():break
        while consumed<submitted:
            worker=consumed%self.count
            while True:
                pipeline.stop_if_requested()
                try:result=self.outputs[worker].get(timeout=.2);break
                except queue.Empty:
                    if not self.processes[worker].is_alive():raise RuntimeError('Worker exited without a result')
            if result[0]=='error':raise RuntimeError(result[1])
            sequence,documents,metrics=result
            if sequence!=consumed:raise ValueError('Worker returned out-of-order batch')
            self.metrics[worker]=metrics
            yield documents
            # Return the credit only after the caller consumed the complete batch.
            self.pending[worker]-=1;consumed+=1;dispatch()

    @property
    def stats(self):
        values=list(self.metrics.values())
        return dict(workers=self.count,documents_per_batch=BATCH_DOCUMENTS,
            max_batches_per_worker=max(self.maximum,default=0),per_worker_max_batches=list(self.maximum),
            outstanding_batches=sum(self.pending),retained_raw_shard_cache=0,
            max_active_mapping_per_prefetch_worker=1,prefetch_slots_per_worker=1,
            workers_observed=len(values),worker_pids=[p.pid for p in self.processes],
            worker_metrics=values)

    def close(self):
        if self.closed:return
        self.closed=True;self.stop.set()
        deadline=time.monotonic()+10
        for process in self.processes:process.join(max(0,deadline-time.monotonic()))
        for process in self.processes:
            if process.is_alive():process.terminate()
        for process in self.processes:process.join(2)
        for pipe in self.inputs+self.outputs:
            pipe.cancel_join_thread();pipe.close()
        os.sched_setaffinity(0,set(self.allowed))


class PreparedDocument(dict):pass


def prepared_documents(ordered,tokenizer,tasks,workers):
    for original,row,pos,source,slot in tasks:
        if not isinstance(original,PreparedDocument):raise ValueError('Expected worker-prepared document')
        if (original['row_index'],original['shuffle_position'],original['source'],original['evaluation_slot'])!=(row,pos,source,slot):
            raise ValueError('Worker document identity/order differs from canonical selection')
        yield dict(original)


class ParallelRaw:
    """Adapt unchanged stage_select: speculative batches, faithful ordered commits."""
    def __init__(self,prep,raw,fixture=False):
        self.prep=prep;self.raw=raw;self.fixture=fixture;self.session=None;self.iterator=None
        self.phase=None;self.history=[];self.db=None;self.evaluation=None
        self.index=dict(row_count=len(raw),files=raw.files)
        self.start_state=pipeline.read_json(prep.path/'select/state.json') if (prep.path/'select/state.json').exists() else dict(last_position=-1,next_evaluation_slot=0)
    def _start(self,phase):
        self.close();self.phase=phase
        if phase=='train':
            self.db=pipeline.readonly(self.prep.path/'select/candidates.sqlite')
            cursor=self.db.execute('SELECT row_index,position,source FROM candidates WHERE position>? ORDER BY position',(self.start_state['last_position'],))
            def batches():
                while True:
                    rows=cursor.fetchmany(BATCH_DOCUMENTS)
                    if not rows:return
                    yield [(row,pos,source,-1) for row,pos,source in rows]
        else:
            split=self.prep.manifest('split')['details'];ranges=split['ranges'];ends=[item['end'] for item in ranges]
            def batches():
                for offset in range(self.start_state['next_evaluation_slot'],len(self.evaluation),BATCH_DOCUMENTS):
                    yield [(int(row),split['train_document_count']+offset+i,
                            pipeline.source_at(ranges,ends,int(row)),offset+i)
                           for i,row in enumerate(self.evaluation[offset:offset+BATCH_DOCUMENTS])]
        self.raw.close()
        workers=worker_count(self.prep.config) if self.fixture else (
            training_worker_count(self.prep.config) if phase=='train' else evaluation_worker_count(self.prep.config))
        self.session=OrderedBatches(self.index,self.raw.directory,self.raw.identity,self.prep.config,self.fixture,workers=workers)
        self.iterator=self.session.ordered(batches())
    def select(self,indices):
        import numpy as np
        if self.evaluation is None:
            self.evaluation=np.load(self.prep.path/'split/evaluation.npy',mmap_mode='r')
            self.eval_set={int(row) for row in self.evaluation}
        phase='eval' if indices and indices[0] in self.eval_set else 'train'
        if phase!=self.phase:self._start(phase)
        try:documents=next(self.iterator)
        except StopIteration:raise ValueError('Parallel stream exhausted before canonical request') from None
        if [doc['row_index'] for doc in documents]!=list(indices):raise ValueError('Parallel batch differs from original canonical cursor')
        result=[PreparedDocument(doc) for doc in documents]
        if phase=='eval' and indices[-1]==int(self.evaluation[-1]):self.close()
        return result
    @property
    def stats(self):
        return dict(current=self.session.stats if self.session else None,completed_phases=self.history)
    def close(self):
        if self.iterator is not None:self.iterator.close();self.iterator=None
        if self.session is not None:
            self.history.append(self.session.stats);self.session.close();self.session=None
        if self.db is not None:self.db.close();self.db=None
