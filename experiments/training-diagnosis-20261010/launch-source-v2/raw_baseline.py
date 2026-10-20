"""Progressive, append-only packing of the existing canonical raw training split.

The filtered preparation remains untouched. No corpus or model imports on import.
"""
from bisect import bisect_right
from pathlib import Path
import hashlib
import json
import os
import signal
import sys
import time
from types import SimpleNamespace
from training_state import atomic_json, digest, file_digest, read_json, locked


class DataUnavailable(RuntimeError):
    pass


def validate_append(previous, current):
    if len(current) < len(previous) or current[:len(previous)] != previous:
        raise ValueError('Published raw baseline shards changed; append-only stream required')


def packed_details(prep):
    return read_json(Path(prep)/'raw/ready.json')['packed']


class ProgressiveDataset:
    def __init__(self, prep, target_rows, receipt, stop_epoch):
        self.prep, self.target_rows = Path(prep), target_rows
        self.receipt, self.stop_epoch = Path(receipt), stop_epoch
        self.files, self.ends, self.cache = [], [], {}
        if self.receipt.exists():
            old = read_json(self.receipt)
            validate_append(old['files'], read_json(self.prep/'raw/progress.json')['train_files'])
        self.refresh()

    def refresh(self):
        value = read_json(self.prep/'raw/progress.json')
        validate_append(self.files, value['train_files'])
        self.files = value['train_files']
        self.ends = []
        total = 0
        for item in self.files:
            total += item['rows']; self.ends.append(total)
        self.status = value['status']

    def __len__(self):
        return self.target_rows

    def __getitem__(self, index):
        if not 0 <= index < self.target_rows:
            raise IndexError(index)
        started = time.monotonic()
        while not self.ends or index >= self.ends[-1]:
            self.refresh()
            if self.ends and index < self.ends[-1]:
                break
            if self.status in ('failed', 'complete'):
                raise DataUnavailable(f'Raw producer {self.status} before sequence {index}')
            if time.time() >= self.stop_epoch or time.monotonic()-started >= 60:
                raise DataUnavailable(f'Raw producer has not supplied sequence {index}; checkpoint safely')
            time.sleep(.5)
        shard = bisect_right(self.ends, index)
        if shard not in self.cache:
            from dataset import check_artifact
            import datasets
            check_artifact(self.files[shard], True)
            self.cache[shard] = datasets.Dataset.from_file(self.files[shard]['path'])
            # Each rank retains just two shard mappings; committed descriptors persist.
            for key in list(self.cache):
                if key < shard-1: del self.cache[key]
        start = self.ends[shard-1] if shard else 0
        return self.cache[shard][index-start]

    def commit_receipt(self, cursor):
        self.refresh()
        count = bisect_right(self.ends, max(0,cursor-1))+1 if cursor else 0
        files = self.files[:count]
        if self.receipt.exists():
            old = read_json(self.receipt)
            if len(old['files']) > len(files):
                validate_append(files, old['files'])
                files = old['files']  # A checkpoint crash may leave a conservative later receipt.
            else:
                validate_append(old['files'], files)
        atomic_json(self.receipt, {'files': files, 'cursor':cursor})


def prepare_data(prep, run_path, cfg, accelerator):
    from train import on_main
    from dataset import check_artifact
    import datasets
    import numpy as np
    def verify():
        ready = read_json(prep/'raw/ready.json')
        identity = read_json(prep/'configuration.json')
        if ready['fingerprint'] != identity['fingerprint'] or ready['identity_sha256'] != file_digest(prep/'configuration.json'):
            raise ValueError('Raw preparation seal changed')
        config = identity['config']
        if (config['sequence_length'],config['tokenizer_id'],config['tokenizer_revision']) != (cfg['sequence_length'],cfg['model_id'],cfg['model_revision']):
            raise ValueError('Raw baseline context/tokenizer mismatch')
        if cfg['eval_documents'] > config['evaluation_subset_documents']:
            raise ValueError('Evaluation exceeds the fixed subset')
        for item in ready['tokenizer_files']:
            check_artifact(item, True)
        # Reuse a receipt only for identical immutable evaluation files and stats.
        files = ready['packed']['eval_files']
        signature = digest([{**i,'mtime_ns':Path(i['path']).stat().st_mtime_ns,'ctime_ns':Path(i['path']).stat().st_ctime_ns} for i in files])
        receipt = prep/'raw/eval-checksum-receipt.json'
        with locked(receipt.with_suffix('.lock')):
            cached = receipt.exists() and read_json(receipt)['signature'] == signature
            for item in files: check_artifact(item, not cached)
            atomic_json(receipt, {'signature':signature})
        evaluation = np.load(ready['evaluation_subset_path'], mmap_mode='r')[:cfg['eval_documents']].tolist()
        provenance = {'preparation_id':prep.name,'fingerprint':identity['fingerprint'],
            'dataset_revision':config['dataset_revision'],'raw_identity_sha256':ready['identity_sha256'],
            'evaluation_manifest_sha256':file_digest(prep/'raw/ready.json'),
            'evaluation_indices':evaluation,'evaluation_indices_sha256':digest(evaluation),
            'evaluation_selection':f'first {cfg["eval_documents"]} ordered documents of fixed subset',
            'pack_settings':{k:ready['packed'][k] for k in ('training_attention','training_loss','evaluation_loss','pad_token_id')},
            'tokenizer_directory':ready['tokenizer_directory'], 'bos_token_id':ready['bos_token_id'],
            'eos_token_id':ready['eos_token_id'], 'selection':'unfiltered canonical shuffled training documents; no repeats',
            'split_provenance':identity['source_split']}
        old=run_path/'data-provenance.json'
        if old.exists() and read_json(old)!=provenance:
            raise ValueError('Raw baseline resume data identity changed')
        atomic_json(old,provenance)
        return provenance
    provenance=on_main(accelerator,verify)
    ready=read_json(prep/'raw/ready.json')
    pieces=[datasets.Dataset.from_file(i['path']) for i in ready['packed']['eval_files']]
    full=datasets.concatenate_datasets(pieces) if len(pieces)>1 else pieces[0]
    count=0
    for row in full.select_columns(['evaluation_slot']):
        if row['evaluation_slot']>=cfg['eval_documents']: break
        count+=1
    if not count: raise ValueError('Empty fixed evaluation subset')
    stop_epoch=min(int(os.environ['TRAIN_END_EPOCH']),int(os.environ.get('CAMPAIGN_END_EPOCH',os.environ['TRAIN_END_EPOCH'])))
    stop_epoch-=cfg.get('final_evaluation_reserve_seconds',0)+cfg['stop_margin_seconds']
    train=ProgressiveDataset(prep,ready['packed']['training_sequence_count'],run_path/'raw-prefix-receipt.json',stop_epoch)
    return train,full.select(range(count)),provenance


def produce(root, source_id, prep_id, initial_rows=1024):
    from dataset import (require_compute, setup_storage, check_artifact, artifact,
        load_raw, load_tokenizer, initialize_tokenizer, tokenize_document, make_document,
        TrainingPacker, pack_evaluation_document, sequence_schema, write_arrow,
        iter_arrow, source_at, stop_if_requested, StopRequested)
    from preparation_parallel import configure_processes, configure_threads, ordered_executor
    require_compute(); configure_threads(); configure_processes(); setup_storage(root)
    import numpy as np
    import pyarrow as pa
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    import dataset
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:setattr(dataset,'STOP',True))
    source=root/'preprocessed'/source_id
    prep=root/'preprocessed'/prep_id
    rawdir=prep/'raw'; rawdir.mkdir(parents=True,exist_ok=True)
    with locked(rawdir/'producer.lock',blocking=False):
        original=read_json(source/'configuration.json')
        split=read_json(source/'split/manifest.json')
        details=split['details']; config=original['config']
        if details['seed']!=42 or details['evaluation_raw_documents']!=50000:
            raise ValueError('Need the complete canonical seed-42 split')
        print('Checking canonical split index artifacts',flush=True)
        for item in split['artifacts']:
            if Path(item['path']).suffix=='.npy': check_artifact(item,True)
        permutation=np.load(source/'split/permutation.npy',mmap_mode='r')
        inverse=np.load(source/'split/inverse.npy',mmap_mode='r')
        evaluation=np.load(source/'split/evaluation.npy',mmap_mode='r')
        subset=np.load(source/'split/evaluation_subset.npy',mmap_mode='r')
        ntrain=details['train_document_count']
        if len(permutation)!=details['row_count'] or ntrain+50000!=len(permutation) or not np.array_equal(permutation[ntrain:],evaluation) or not np.array_equal(evaluation[:len(subset)],subset):
            raise ValueError('Canonical evaluation membership changed')
        for offset in range(0,len(permutation),1000000):
            stop_if_requested()
            rows=permutation[offset:offset+1000000]
            if not np.array_equal(inverse[rows],np.arange(offset,offset+len(rows),dtype=np.uint64)):
                raise ValueError('Canonical split is not a bijection')
        identity={'config':config,'selection':'raw unfiltered canonical training prefix, whole documents, <=6B input tokens',
            'source_split':{'preparation_id':source_id,'fingerprint':original['fingerprint'],
                'manifest_sha256':file_digest(source/'split/manifest.json'),
                'permutation_sha256':file_digest(source/'split/permutation.npy'),
                'evaluation_sha256':file_digest(source/'split/evaluation.npy'),
                'row_count':len(permutation),'seed':42,'evaluation_documents':50000},
            'source_hashes':{'raw_baseline.py':file_digest(Path(__file__)), 'dataset.py':file_digest(Path(__file__).with_name('dataset.py'))}}
        identity['fingerprint']=digest(identity)
        identity_path=prep/'configuration.json'
        if identity_path.exists() and read_json(identity_path)!=identity:
            raise ValueError('Raw baseline preparation identity changed')
        atomic_json(identity_path,identity)
        tokenizer=load_tokenizer(SimpleNamespace(config=config))
        token_dir=rawdir/'tokenizer'; tokenizer.save_pretrained(token_dir)
        token_files=[artifact(p) for p in sorted(token_dir.iterdir()) if p.is_file()]
        schema=sequence_schema(8192)
        state_path=rawdir/'producer-state.json'
        state=read_json(state_path) if state_path.exists() else {'train_files':[],'eval_files':[],
            'eval_cursor':[0,0],'eval_documents':0,'evaluation_prediction_tokens':0,
            'training_tokens':0,'next_position':0,'carry':{},'train_complete':False,'eval_complete':False}
        for item in state['train_files']+state['eval_files']: check_artifact(item,True)
        def publish(status='active'):
            state['status']=status; state['job_id']=os.environ['SLURM_JOB_ID']; state['updated_at']=time.time()
            atomic_json(state_path,state)
            atomic_json(rawdir/'progress.json',{'status':status,'train_files':state['train_files'],
                'training_tokens':state['training_tokens'],'next_position':state['next_position'],
                'job_id':state['job_id'],'updated_at':state['updated_at']})
        publish()
        try:
            with ordered_executor(7,initializer=initialize_tokenizer,initargs=(config,),check=stop_if_requested) as ordered:
                if not state['eval_complete']:
                    print('Packing reserved original evaluation documents (no filtering)',flush=True)
                    def eval_tasks():
                        for rec,cursor in iter_arrow(details['evaluation_raw_files'],state['eval_cursor']):
                            row=int(rec['row_index']); slot=int(rec['evaluation_slot'])
                            if row!=int(evaluation[slot]) or int(inverse[row])!=ntrain+slot:
                                raise ValueError('Evaluation original differs from reserved membership')
                            original_doc=json.loads(rec['document_json'])
                            if hashlib.sha256(original_doc['text'].encode()).digest()!=rec['text_sha256']:
                                raise ValueError('Evaluation text authentication failed')
                            yield (original_doc,row,int(inverse[row]),rec['source'],slot)
                    buffer=[]
                    for doc in ordered(tokenize_document,eval_tasks()):
                        rows=list(pack_evaluation_document(doc,8192,tokenizer.eos_token_id))
                        buffer.extend(rows)
                        state['evaluation_prediction_tokens']+=sum(sum(x!=-100 for x in r['labels'][1:]) for r in rows)
                        state['eval_documents']+=1
                        # Archived raw shards have exactly 256 documents except the last.
                        state['eval_cursor']=list(divmod(state['eval_documents'],256))
                        if len(buffer)>=256:
                            state['eval_files'].append(write_arrow(rawdir/f'eval-{len(state["eval_files"]):06d}.arrow',buffer,schema))
                            buffer=[]; publish()
                    if buffer: state['eval_files'].append(write_arrow(rawdir/f'eval-{len(state["eval_files"]):06d}.arrow',buffer,schema))
                    if state['eval_documents']!=50000: raise ValueError('Incomplete full evaluation set')
                    state['eval_complete']=True; publish()
                print('Mapping existing downloaded raw Arrow files',flush=True)
                raw=load_raw(SimpleNamespace(manifest=lambda _:split))
                ends=[i['end'] for i in details['ranges']]
                packer=TrainingPacker(8192,tokenizer.eos_token_id,state['carry']); buffer=[]
                limit=6000000000
                def tasks():
                    for pos in range(state['next_position'],ntrain):
                        stop_if_requested(); row=int(permutation[pos])
                        if int(inverse[row])!=pos or pos>=ntrain: raise ValueError('Training/evaluation split violation')
                        rec=raw[row]
                        original_doc=json.loads(rec['document_json'])
                        if original_doc['text']!=rec['text'] or str(original_doc.get('id'))!=str(rec['id']):
                            raise ValueError('Raw original fields changed')
                        yield (original_doc,row,pos,source_at(details['ranges'],ends,row),-1)
                def flush():
                    nonlocal buffer
                    if buffer:
                        state['train_files'].append(write_arrow(rawdir/f'train-{len(state["train_files"]):06d}.arrow',buffer,schema))
                        buffer=[]
                    state['carry']=packer.state(); publish()
                    if not (rawdir/'ready.json').exists() and sum(i['rows'] for i in state['train_files'])>=initial_rows:
                        packed={'eval_files':state['eval_files'],'evaluation_prediction_tokens':state['evaluation_prediction_tokens'],
                            'training_sequence_count':limit//8192,'sequence_length':8192,'pad_token_id':tokenizer.eos_token_id,
                            'training_attention':'ordinary causal attention across EOS-separated training documents',
                            'training_loss':'ignore structural BOS, first input position of each sequence, and padding',
                            'evaluation_loss':'independent documents, stride 8191; each within-document target once'}
                        atomic_json(rawdir/'ready.json',{'fingerprint':identity['fingerprint'],'identity_sha256':file_digest(identity_path),
                            'packed':packed,'tokenizer_directory':str(token_dir),'tokenizer_files':token_files,
                            'bos_token_id':tokenizer.bos_token_id,'eos_token_id':tokenizer.eos_token_id,
                            'evaluation_subset_path':str(source/'split/evaluation_subset.npy')})
                        print('RAW BASELINE READY: training can start while remaining shards are produced',flush=True)
                    print(json.dumps({'training_tokens':state['training_tokens'],'next_position':state['next_position'],
                                      'sequences':sum(i['rows'] for i in state['train_files'])}),flush=True)
                if not state['train_complete']:
                    for doc in ordered(tokenize_document,tasks()):
                        state['next_position']=doc['shuffle_position']+1
                        if len(doc['tokens'])>limit-state['training_tokens']: continue
                        state['training_tokens']+=len(doc['tokens'])
                        buffer.extend(packer.push(doc['row_index'],doc['tokens']))
                        if len(buffer)>=256: flush()
                        if limit-state['training_tokens']<8192: break
                    buffer.extend(packer.finish()); state['train_complete']=True; flush()
                publish('complete')
        except StopRequested:
            # Resume rolls back uncommitted documents and partial buffer to the last published state.
            durable=read_json(state_path); durable['status']='paused'; atomic_json(state_path,durable)
            atomic_json(rawdir/'progress.json',{k:durable[k] for k in ('status','train_files','training_tokens','next_position','job_id','updated_at')})
            return 75
        except BaseException:
            durable=read_json(state_path); durable['status']='failed'; atomic_json(state_path,durable)
            atomic_json(rawdir/'progress.json',{k:durable[k] for k in ('status','train_files','training_tokens','next_position','job_id','updated_at')})
            raise
    return 0


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-id',required=True)
    p.add_argument('--prep-id',required=True)
    args=p.parse_args()
    sys.exit(produce(args.root,args.source_id,args.prep_id))
