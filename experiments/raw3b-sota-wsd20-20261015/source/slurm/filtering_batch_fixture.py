"""Tiny real-Arrow and production select interruption fixtures; Slurm only."""
import contextlib
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch
import dataset as pipeline
from filtering_batch_reader import BatchRaw


def run(directory):
    pipeline.require_compute()
    import pyarrow as pa
    import numpy as np
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    schema=pa.schema([('id',pa.string()),('text',pa.string()),('document_json',pa.string())])
    files=[];rows=[dict(id=str(i),text='臺灣 '+str(i),document_json=json.dumps({'id':str(i),'text':'臺灣 '+str(i)})) for i in range(9)]
    for number,piece in enumerate((rows[:4],[],rows[4:])):
        path=directory/f'raw-{number}.arrow'
        with pa.OSFile(str(path),'wb') as sink,pa.ipc.new_stream(sink,schema) as writer:
            for part in (piece[:2],[],piece[2:]):
                writer.write_batch(pa.RecordBatch.from_pylist(part,schema=schema))
        files.append(dict(**pipeline.artifact(path),rows=len(piece)))
    index=dict(row_count=len(rows),files=files);identity=dict(fixture=True)
    raw=BatchRaw(index,directory/'indexes',identity)
    for indices in ([8,0,4,8,3,0],[0,3,4,8],[],[2,1]):
        assert list(raw.select(indices))==[rows[i] for i in indices]
        assert raw.select(indices)['text']==[rows[i]['text'] for i in indices]
    assert raw[-1]==rows[-1] and raw[2:5]['id']==['2','3','4']
    try: raw.fetch([-1])
    except IndexError: pass
    else: raise AssertionError('negative fetch accepted')
    try: raw.select(range(2049))
    except ValueError: pass
    else: raise AssertionError('unbounded fetch accepted')
    handles=[v['source'] for v in raw.cache.values()];raw.close();assert all(h.closed for h in handles)
    sidecar=directory/'indexes/shard-000000.json';original=sidecar.read_text();bad=json.loads(original);bad['batches'][0]['offset']+=1
    sidecar.write_text(json.dumps(bad));raw.indexes.clear()
    try: raw[0]
    except ValueError: pass
    else: raise AssertionError('corrupt index accepted')
    sidecar.write_text(original)
    changed=BatchRaw(index,directory/'indexes',dict(other=True))
    try: changed[0]
    except ValueError: pass
    else: raise AssertionError('changed identity accepted')
    raw.close();changed.close()
    native=raw.shuffle(seed=42,load_from_cache_file=False,indices_cache_file_name=str(directory/'canonical.arrow'))
    actual=np.asarray(native._indices.column(0),dtype=np.uint64)
    assert np.array_equal(actual,np.random.default_rng(42).permutation(len(rows)))
    # Exercise the unchanged stage_select checkpoint logic with tiny fake tokenizer.
    class Tokenizer:
        bos_token_id=1;eos_token_id=9
        def encode(self,text,add_special_tokens=False):return [2+int(text.rsplit(' ',1)[1])]
        def save_pretrained(self,path):Path(path).mkdir(exist_ok=True);(Path(path)/'tokenizer.json').write_text('{}')
    cfg=dict(preprocessing_workers=1,document_shard_tokens=3,max_training_tokens=9,common_crawl_keep_fraction=0.1)
    def prep_at(name):
        path=directory/name;path.mkdir();(path/'select').mkdir();(path/'split').mkdir()
        np.save(path/'split/evaluation.npy',np.array([7,8],dtype=np.uint64))
        with sqlite3.connect(path/'select/candidates.sqlite') as db:
            db.execute('CREATE TABLE candidates(row_index INTEGER,position INTEGER,source TEXT)')
            db.executemany('INSERT INTO candidates VALUES (?,?,?)',[(i,i,'finemath') for i in range(7)])
            db.execute('CREATE TABLE progress(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            pipeline.set_progress(db,'complete',True)
            pipeline.set_progress(db,'common_crawl_kept',0)
            pipeline.set_progress(db,'boundary',None)
        def finish(stage,paths,details):pipeline.atomic_json(path/'select/result.json',details)
        return SimpleNamespace(path=path,config=cfg,directory=lambda stage:path/stage,finish=finish,
            manifest=lambda stage:{'details':dict(scored_documents=0,train_document_count=7,ranges=[dict(start=0,end=9,source='finemath')])})
    full,partial=prep_at('uninterrupted'),prep_at('interrupted')
    @contextlib.contextmanager
    def inline(*args,**kwargs):yield lambda f,items:map(f,items)
    real_atomic=pipeline.atomic_json
    def interrupt(path,value):
        real_atomic(path,value)
        if Path(path)==partial.path/'select/state.json' and value['train_documents']==1:
            raise pipeline.StopRequested('fixture checkpoint')
    with patch.object(pipeline,'load_raw',return_value=raw),patch.object(pipeline,'load_tokenizer',return_value=Tokenizer()),patch.object(pipeline,'ordered_executor',inline):
        pipeline.stage_select(full)
        with patch.object(pipeline,'atomic_json',interrupt):
            try:pipeline.stage_select(partial)
            except pipeline.StopRequested:pass
            else:raise AssertionError('fixture failed to interrupt')
        pipeline.stage_select(partial)
    states=[pipeline.read_json(p.path/'select/state.json') for p in (full,partial)]
    for key in ('train_tokens','train_documents','last_position','next_evaluation_slot','evaluation_tokens','budget_skipped_documents'):
        assert states[0][key]==states[1][key],key
    for key in ('train_files','eval_files'):
        documents=[[row for row,_ in pipeline.iter_arrow(s[key])] for s in states]
        assert documents[0]==documents[1]
        sequences=[list(pipeline.expected_sequences(iter((doc,None) for doc in docs),4,9,key=='eval_files')) for docs in documents]
        assert sequences[0]==sequences[1]
    raw.close()
    result=dict(passed=True,real_ipc_boundaries=True,corrupt_index_rejected=True,maximum_cached_shards=raw.maximum_open,
                interrupted_select_equal=True,packing_masks_equal=True,slurm_job_id=os.environ['SLURM_JOB_ID'])
    pipeline.atomic_json(directory/'result.json',result)
    return result
