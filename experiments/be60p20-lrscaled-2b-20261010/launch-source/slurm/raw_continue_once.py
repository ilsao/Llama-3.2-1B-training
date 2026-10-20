"""Append raw canonical sequences for the explicitly authorized one-hour resume."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import signal
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dataset as pipeline
from training_state import read_json,atomic_json,locked,file_digest


def main(receipt):
    pipeline.require_compute()
    from preparation_parallel import configure_threads,configure_processes
    configure_threads();configure_processes()
    import numpy as np
    from filtering_reader import build_index
    from filtering_batch_reader import BatchRaw
    from filtering_parallel import OrderedBatches
    from packing_parallel import Workers,checksum_task
    authorization=read_json(receipt);root=Path(authorization['root'])
    source=pipeline.Preparation(root,'dolma-dclm10-005',read_json(root/'preprocessed/dolma-dclm10-005/configuration.json')['config'])
    rawdir=root/'preprocessed/dolma-raw-baseline-20261008/raw'
    config=source.config;split=source.manifest('split')['details']
    for sig in (signal.SIGUSR1,signal.SIGTERM):
        signal.signal(sig,lambda *_:setattr(pipeline,'STOP',True))
    def sealed():
        for name,expected in authorization['raw_metadata_hashes'].items():
            if file_digest(Path(name))!=expected:raise ValueError('Immutable raw preparation changed')
    sealed()
    output=rawdir/'continuations'/authorization['id']/os.environ['SLURM_JOB_ID']
    output.mkdir(parents=True,exist_ok=False)
    with locked(rawdir/'producer.lock',blocking=False):
        state_path=rawdir/'producer-state.json';state=read_json(state_path)
        if state['train_files']!=read_json(rawdir/'progress.json')['train_files']:
            raise ValueError('Published raw prefix and producer checkpoint differ')
        if state['train_files']!=authorization['raw_prefix']:
            raise ValueError('Raw producer prefix changed since authorization')
        if not state['eval_complete'] or state['train_complete']:raise ValueError('Unexpected raw producer state')
        with Workers(config) as workers:list(workers.run(checksum_task,state['train_files']))
        index=build_index(source);identity=dict(preparation_fingerprint=source.key,split_sha256=file_digest(source.path/'split/manifest.json'))
        directory=source.path/'filtering-access/batch-reader-v1/indexes'
        perm=np.load(source.path/'split/permutation.npy',mmap_mode='r')
        inverse=np.load(source.path/'split/inverse.npy',mmap_mode='r')
        ends=[x['end'] for x in split['ranges']]
        def metadata(start,stop):
            for pos in range(start,stop,32):
                batch=[]
                for position in range(pos,min(pos+32,stop)):
                    row=int(perm[position])
                    if int(inverse[row])!=position:raise ValueError('Canonical raw cursor changed')
                    batch.append((row,position,pipeline.source_at(split['ranges'],ends,row),-1))
                yield batch
        # Compare the actual next batch with serial original tokenization before
        # publishing any new artifacts. Worker implementation already has 1/7 gates.
        first=next(metadata(state['next_position'],split['train_document_count']))
        reader=BatchRaw(index,directory,identity)
        try:originals=reader.fetch([x[0] for x in first])
        finally:reader.close()
        pipeline.initialize_tokenizer(config)
        reference=[]
        for record,(row,pos,src,slot) in zip(originals,first):
            original=pipeline.original_document(record)
            if original['text']!=record['text'] or original.get('id')!=record['id']:
                raise ValueError('Raw JSON fields changed')
            reference.append(pipeline.tokenize_document((original,row,pos,src,slot)))
        with OrderedBatches(index,directory,identity,config) as batches:
            actual=next(batches.ordered([first]))
        if actual!=reference:raise ValueError('Parallel raw tokenizer differs from original')
        atomic_json(output/'validation.json',dict(passed=True,documents=len(first),workers=7,
            original_tokenization_equal=True,raw_metadata_hashes=authorization['raw_metadata_hashes']))
        packer=pipeline.TrainingPacker(8192,pipeline._TOKENIZER.eos_token_id,copy.deepcopy(state['carry']))
        schema=pipeline.sequence_schema(8192);buffer=[]
        def publish():
            state.update(status='active',job_id=os.environ['SLURM_JOB_ID'],updated_at=__import__('time').time())
            atomic_json(state_path,state)
            atomic_json(rawdir/'progress.json',{k:state[k] for k in ('status','train_files','training_tokens','next_position','job_id','updated_at')})
        def flush():
            nonlocal buffer
            path=output/f'train-{len(state["train_files"]):06d}.arrow'
            state['train_files'].append(pipeline.write_arrow(path,buffer,schema));buffer=[]
            state['carry']=packer.state();publish()
            print(json.dumps({k:state[k] for k in ('training_tokens','next_position','job_id')}),flush=True)
        try:
            with OrderedBatches(index,directory,identity,config) as batches:
                for docs in batches.ordered(metadata(state['next_position'],split['train_document_count'])):
                    for doc in docs:
                        pipeline.stop_if_requested();state['next_position']=doc['shuffle_position']+1
                        if len(doc['tokens'])>6_000_000_000-state['training_tokens']:continue
                        state['training_tokens']+=len(doc['tokens']);buffer.extend(packer.push(doc['row_index'],doc['tokens']))
                        if len(buffer)>=256:
                            flush()
                            if state['training_tokens']>=authorization['raw_target_tokens']:
                                sealed();atomic_json(output/'completed.json',dict(target_met=True,training_tokens=state['training_tokens'],completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
                                state['status']='paused';atomic_json(state_path,state)
                                atomic_json(rawdir/'progress.json',{k:state[k] for k in ('status','train_files','training_tokens','next_position','job_id','updated_at')})
                                return 0
            raise ValueError('Unexpected exhaustion before bounded raw target')
        except pipeline.StopRequested:
            durable=read_json(state_path);durable['status']='paused';atomic_json(state_path,durable)
            atomic_json(rawdir/'progress.json',{k:durable[k] for k in ('status','train_files','training_tokens','next_position','job_id','updated_at')})
            return 75


if __name__=='__main__':sys.exit(main(Path(sys.argv[1])))
