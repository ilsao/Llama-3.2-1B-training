"""Allocated-only reweighting of the immutable old filtered document pool.

No tokenization or quality rescoring. Original TrainingPacker/masks are reused.
"""
import bisect
from collections import Counter,OrderedDict
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from training_state import atomic_json,digest,file_digest,read_json
TARGET=3000107008
TOLERANCE=0.0001  # absolute source token share,0.01 percentage points


def choose_source(counts,weights):
    total=sum(counts.values())
    return max(sorted(weights),key=lambda name:weights[name]*total-counts.get(name,0))


def check_mix(actual,reference,tolerance=TOLERANCE):
    a=sum(actual.values());b=sum(reference.values())
    if a!=TARGET or b!=TARGET:raise ValueError('Both mixtures must count the actual3B processed-input target')
    error={s:abs(actual.get(s,0)/a-reference.get(s,0)/b) for s in set(actual)|set(reference)}
    if max(error.values())>tolerance:raise ValueError('Whole-document mixture outside prespecified tolerance: '+str(error))
    return error


def context():
    import numpy as np
    import packing_parallel as packing
    c=packing.CONTEXT
    if 'inverse' not in c:c['inverse']=np.load(c['inverse_path'],mmap_mode='r')
    return c


def scan_task(task):
    import pyarrow as pa
    import pyarrow.compute as pc
    import dataset as pipeline
    c=context();shard,item=task['shard'],task['file'];pipeline.check_artifact(item,True)
    with pa.memory_map(item['path'],'r') as stream:table=pa.ipc.open_stream(stream).read_all()
    if len(table)!=item['rows']:raise ValueError('Parent document shard row count changed')
    if task['kind']=='raw':
        counts=Counter();table=table.slice(0,task['rows'])
        for spans in table['document_spans'].to_pylist():
            if sum(s['length'] for s in spans)!=8192:raise ValueError('Raw target prefix contains padding')
            for span in spans:
                row=span['row_index']
                if int(c['inverse'][row])>=c['train_documents']:raise ValueError('Evaluation document in raw prefix')
                counts[c['ranges'][bisect.bisect_right(c['ends'],row)]['source']]+=span['length']
        return dict(counts=dict(counts),pid=os.getpid())
    lengths=pc.list_value_length(table['tokens']).to_pylist();rows=table['row_index'].to_pylist();positions=table['shuffle_position'].to_pylist();sources=table['source'].to_pylist();slots=table['evaluation_slot'].to_pylist()
    result=[]
    for offset,(row,pos,src,slot,n) in enumerate(zip(rows,positions,sources,slots,lengths)):
        if slot!=-1 or int(c['inverse'][row])!=pos or pos>=c['train_documents'] or c['ranges'][bisect.bisect_right(c['ends'],row)]['source']!=src or n<2:
            raise ValueError('Filtered retained document split/source provenance changed')
        result.append((src,pos,row,shard,offset,n))
    return dict(rows=result,pid=os.getpid())


def reference_documents(task):
    import pyarrow as pa
    c=context();cache=c.setdefault('doc_cache',OrderedDict())
    for shard,offset,row,count in task['references']:
        if shard not in cache:
            if len(cache)>=8:cache.popitem(last=False)
            with pa.memory_map(c['parent_files'][shard]['path'],'r') as source:cache[shard]=pa.ipc.open_stream(source).read_all()
        cache.move_to_end(shard);doc=cache[shard].slice(offset,1).to_pylist()[0]
        if doc['row_index']!=row or len(doc['tokens'])!=count:raise ValueError('Reordered document reference differs')
        if doc['tokens'][0]!=c['bos'] or doc['tokens'][-1]!=c['eos']:raise ValueError('BOS/EOS changed')
        yield doc


def pack_task(task):
    import pyarrow as pa
    import packing_parallel as packing
    packing.task_documents=reference_documents
    result=packing.sequence_task(task)
    # Independent count from written provenance spans for exactly the consumed prefix.
    c=context();counts=Counter()
    with pa.memory_map(result['file']['path'],'r') as source:table=pa.ipc.open_stream(source).read_all()
    for spans in table['document_spans'].slice(0,task['prefix_rows']).to_pylist():
        if sum(s['length'] for s in spans)!=8192:raise ValueError('Reweighted consumed prefix contains padding')
        for span in spans:counts[c['ranges'][bisect.bisect_right(c['ends'],span['row_index'])]['source']]+=span['length']
    result['prefix_counts']=dict(counts)
    if task.get('gate'):
        # First shard also checks independently assembled serial packing and masks.
        import dataset as pipeline
        packer=pipeline.TrainingPacker(8192,task['pad_id'],task['plan']['carry'])
        def expected():
            for doc in reference_documents(task):yield from packer.push(doc['row_index'],doc['tokens'])
            if task['plan']['final']:yield from packer.finish()
        pipeline.validate_sequences(pipeline.iter_arrow([result['file']]),expected(),8192)
        result['serial_reconstruction_verified']=True
    return result


def selected_documents(db,weights,statistics):
    cursors={s:iter(db.execute('SELECT pos,row,shard,offset,n FROM docs WHERE source=? ORDER BY pos',(s,))) for s in weights}
    counts=Counter();cycles=Counter();repeated=Counter();consumed_repeated=Counter();total=0
    while total<TARGET+8192:
        source=choose_source(counts,weights)
        try:pos,row,shard,offset,n=next(cursors[source])
        except StopIteration:
            cycles[source]+=1;cursors[source]=iter(db.execute('SELECT pos,row,shard,offset,n FROM docs WHERE source=? ORDER BY pos',(source,)))
            try:pos,row,shard,offset,n=next(cursors[source])
            except StopIteration:raise ValueError('Old filtered pool lacks source '+source)
        if cycles[source]:
            repeated[source]+=n;consumed_repeated[source]+=min(n,max(0,TARGET-total))
        counts[source]+=n;total+=n;statistics['documents']+=1
        statistics.update(selected_tokens=total,selected_source_tokens=dict(counts),completed_source_cycles=dict(cycles),selected_repeated_source_tokens=dict(repeated),consumed_repeated_source_tokens=dict(consumed_repeated))
        yield row,n,[shard,offset],[statistics['documents'],0]


def plans(db,weights,statistics,files,out,cfg,pad):
    import packing_parallel as packing
    refs=[];start_carry=dict(tokens=[],labels=[],spans=[]);documents=[];outputs=0;sequence_offset=0;number=0
    for row,count,location,end in selected_documents(db,weights,statistics):
        emitted,refs=packing.advance_carry(refs,row,count,location,8192);outputs+=emitted;documents.append([*location,row,count])
        if outputs>=128:
            after=packing.decode_carry(refs,files,{})
            plan=dict(carry=start_carry,carry_after=after,final=False,expected_rows=outputs,end=end)
            yield dict(plan=plan,references=documents,phase='train',length=8192,pad_id=pad,subset_documents=10000,
                output=str(out/f'train-{number:06d}.arrow'),prefix_rows=min(outputs,max(0,TARGET//8192-sequence_offset)),gate=number==0)
            number+=1;sequence_offset+=outputs;outputs=0;documents=[];start_carry=after
    if documents or refs:
        plan=dict(carry=start_carry,carry_after=dict(tokens=[],labels=[],spans=[]),final=True,expected_rows=outputs+bool(refs),end=end)
        yield dict(plan=plan,references=documents,phase='train',length=8192,pad_id=pad,subset_documents=10000,
            output=str(out/f'train-{number:06d}.arrow'),prefix_rows=min(plan['expected_rows'],max(0,TARGET//8192-sequence_offset)),gate=number==0)


def prepare(root,prep_id):
    import dataset as pipeline
    from packing_parallel import Workers
    from preparation_parallel import configure_threads,configure_processes
    pipeline.require_compute();configure_threads();configure_processes()
    parent=root/'preprocessed/dolma-dclm10-005';raw=root/'preprocessed/dolma-raw-baseline-20261008';dest=root/'preprocessed'/prep_id
    if dest.exists():raise ValueError('No automatic retry/overwrite of reweight preparation')
    dest.mkdir();(dest/'pack').mkdir();(dest/'select').mkdir();(dest/'split').mkdir();(dest/'verify').mkdir()
    split=read_json(parent/'split/manifest.json')['details'];select=read_json(parent/'select/manifest.json');configuration=read_json(parent/'configuration.json');rawready=read_json(raw/'raw/ready.json');progress=read_json(raw/'raw/progress.json')
    cfg=configuration['config'];oldfiles=select['details']['train_files']
    verified=read_json(parent/'verify/manifest.json')
    if select['fingerprint']!=configuration['fingerprint'] or verified['fingerprint']!=configuration['fingerprint']:
        raise ValueError('Old filtered selection/verification identity differs')
    if read_json(raw/'configuration.json')['config']['dataset_revision']!=cfg['dataset_revision']:
        raise ValueError('Raw and filtered dataset revisions differ')
    c=dict(ranges=split['ranges'],ends=[x['end'] for x in split['ranges']],inverse_path=str(parent/'split/inverse.npy'),train_documents=split['train_document_count'],parent_files=oldfiles,bos=select['details']['bos_token_id'],eos=select['details']['eos_token_id'])
    db=sqlite3.connect(dest/'select/retained-index.sqlite');db.execute('CREATE TABLE docs(source TEXT,pos INTEGER,row INTEGER,shard INTEGER,offset INTEGER,n INTEGER,PRIMARY KEY(source,pos)) WITHOUT ROWID')
    rawcounts=Counter();stats=dict(documents=0);trainfiles=[];actual=Counter();input_tokens=0;serial_gate=False
    with Workers(cfg,c) as workers:
        def rawtasks():
            remaining=TARGET//8192
            for number,item in enumerate(progress['train_files']):
                if remaining<=0:break
                n=min(remaining,item['rows']);remaining-=n
                yield dict(kind='raw',shard=number,file=item,rows=n)
            if remaining:raise ValueError('Raw producer has not supplied complete3B prefix')
        for value in workers.run(scan_task,rawtasks()):rawcounts.update(value['counts'])
        if sum(rawcounts.values())!=TARGET:raise ValueError('Raw token histogram incomplete')
        atomic_json(dest/'raw-token-reference.json',dict(target_tokens=TARGET,counts=dict(rawcounts),raw_prefix_sha256=digest(progress['train_files']),raw_preparation_fingerprint=rawready['fingerprint']))
        for number,value in enumerate(workers.run(scan_task,(dict(kind='filtered',shard=i,file=f) for i,f in enumerate(oldfiles))),1):
            db.executemany('INSERT INTO docs VALUES(?,?,?,?,?,?)',value['rows'])
            if number%100==0:db.commit();print('indexed filtered shards',number,flush=True)
        db.commit();pool={s:n for s,n in db.execute('SELECT source,SUM(n) FROM docs GROUP BY source')};weights={s:n/TARGET for s,n in rawcounts.items() if n}
        if set(weights)-set(pool):raise ValueError('Positive raw source absent from old retained pool')
        tasks=plans(db,weights,stats,oldfiles,dest/'pack',cfg,select['details']['pad_token_id'])
        for i,value in enumerate(workers.run(pack_task,tasks),1):
            trainfiles.append(value['file']);actual.update(value['prefix_counts']);input_tokens+=value['input_tokens'];serial_gate|=value.get('serial_reconstruction_verified',False)
            if i%50==0:print('packed shards',i,'tokens',input_tokens,flush=True)
        resource=workers.stats
    db.close();errors=check_mix(actual,rawcounts)
    if not serial_gate:raise ValueError('Missing serial packer verification gate')
    identity=dict(config=cfg,selection='Old dolma-dclm10-005 retained tokenized documents; token-deficit source mixing, canonical order within source, cycle only on source exhaustion',source_filtered_fingerprint=configuration['fingerprint'],source_select_manifest_sha256=file_digest(parent/'select/manifest.json'),source_verify_manifest_sha256=file_digest(parent/'verify/manifest.json'),raw_target_tokens=TARGET,raw_source_token_counts=dict(rawcounts),token_mix_tolerance=TOLERANCE,source_sha256=file_digest(__file__))
    fp=digest(identity);atomic_json(dest/'configuration.json',dict(**identity,fingerprint=fp))
    for name in ['evaluation_subset.npy','evaluation.npy']:shutil.copyfile(parent/'split'/name,dest/'split'/name)
    tokenizer_items=[a for a in select['artifacts'] if Path(a['path']).parent==Path(select['details']['tokenizer_directory'])]
    atomic_json(dest/'select/manifest.json',dict(fingerprint=fp,artifacts=tokenizer_items,details=dict(tokenizer_directory=select['details']['tokenizer_directory'],bos_token_id=c['bos'],eos_token_id=c['eos'],pad_token_id=select['details']['pad_token_id'],train_tokens=stats['selected_tokens'],train_documents=stats['documents'])))
    packed={**rawready['packed'], 'train_files':trainfiles,'training_tokens':input_tokens,'training_sequence_count':sum(f['rows'] for f in trainfiles)}
    # ready.json advertises potential raw length; actual reweighted length is explicit.
    atomic_json(dest/'pack/manifest.json',dict(fingerprint=fp,artifacts=trainfiles,details=packed))
    report=dict(passed=True,exact_consumed_tokens=TARGET,raw_source_tokens=dict(rawcounts),reweighted_consumed_source_tokens=dict(actual),absolute_share_errors=errors,maximum_absolute_share_error=max(errors.values()),pool_source_tokens=pool,selection=stats,resources=resource,serial_first_shard_reconstruction=True,all_selected_documents_parent_checksums_verified=True,all_parent_documents_excluded_from_evaluation=True,packing='Original TrainingPacker; full documents, unchanged BOS/EOS/loss masking; only final run token boundary truncates consumed stream',evaluation_files_unchanged=True,parent_select_sha256=identity['source_select_manifest_sha256'],parent_verify_sha256=identity['source_verify_manifest_sha256'],pack_manifest_sha256=file_digest(dest/'pack/manifest.json'))
    atomic_json(dest/'verify/report.json',report)
    atomic_json(dest/'verify/manifest.json',dict(fingerprint=fp,artifacts=[pipeline.artifact(dest/'verify/report.json')],details=report))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    prepare(Path(sys.argv[1]),sys.argv[2])
