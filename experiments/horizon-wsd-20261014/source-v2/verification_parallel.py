"""Full verification with independent worker reads and canonical reconstruction."""
import contextlib
import hashlib
import os
from pathlib import Path
import time

import dataset as pipeline
import packing_parallel as parallel
from filtering_parallel import EphemeralRaw

RAW=None
INVERSE=None
PERM=None
CANONICAL=None


def indexes():
    global INVERSE,PERM
    if INVERSE is None:
        import numpy as np
        p=Path(parallel.CONTEXT['preparation_path'])/'split'
        INVERSE=np.load(p/'inverse.npy',mmap_mode='r');PERM=np.load(p/'permutation.npy',mmap_mode='r')
    return PERM,INVERSE


def raw_reader():
    global RAW
    if RAW is None:
        c=parallel.CONTEXT;RAW=EphemeralRaw(c['raw_index'],c['raw_directory'],c['raw_identity'])
    return RAW


def permutation_task(task):
    global CANONICAL
    import numpy as np
    import pyarrow as pa
    perm,inverse=indexes();first,last=task
    if CANONICAL is None:
        # Arrow buffers retain their mapping; this is an index, not a raw shard cache.
        source=pa.memory_map(parallel.CONTEXT['canonical_path'],'r')
        CANONICAL=pa.ipc.open_stream(source).read_all()
    values=perm[first:last];reference=np.asarray(CANONICAL.column(0).slice(first,last-first),dtype=np.uint64)
    if not np.array_equal(values,reference):raise ValueError('Split differs from canonical native seed42 shuffle')
    if np.any(values>=len(perm)) or not np.array_equal(inverse[values],np.arange(first,last,dtype=np.uint64)):
        raise ValueError('Split permutation is not a bijection')
    return dict(pid=os.getpid(),rows=last-first)


def provenance_task(task):
    import numpy as np
    context=parallel.CONTEXT;split=context['split'];selected=context['selected'];cfg=context['config']
    perm,inverse=indexes();raw=raw_reader();ranges=split['ranges'];ends=[x['end'] for x in ranges]
    evaluation=np.load(Path(context['preparation_path'])/'split/evaluation.npy',mmap_mode='r')
    phase=task['phase'];slot=task['slot'];tokens=docs=0;previous=-1;first_position=None
    candidates=pipeline.readonly(Path(context['preparation_path'])/'select/candidates.sqlite') if phase=='train' else None
    def check(batch):
        nonlocal slot,tokens,docs,previous,first_position
        originals=raw.fetch([d['row_index'] for d in batch])
        for doc,original in zip(batch,originals):
            row=doc['row_index'];position=doc['shuffle_position']
            if not 0<=row<split['row_count']:raise ValueError('Document row outside full split')
            if int(inverse[row])!=position:raise ValueError('Document shuffle position differs')
            if phase=='archive':
                if slot>=len(evaluation) or row!=int(evaluation[slot]) or doc['evaluation_slot']!=slot:
                    raise ValueError('Evaluation archive membership/order changed')
                if (doc['document_json']!=original['document_json']
                        or pipeline.original_document(doc)!=pipeline.original_document(original)):
                    raise ValueError('Independent evaluation original archive changed')
            else:
                if doc['tokens'][0]!=selected['bos_token_id'] or doc['tokens'][-1]!=selected['eos_token_id']:
                    raise ValueError('Document BOS/EOS changed')
                tokens+=len(doc['tokens'])
                if phase=='train':
                    if position>=split['train_document_count'] or position<=previous or doc['evaluation_slot']!=-1:
                        raise ValueError('Duplicate/evaluation/out-of-order training document')
                    if not candidates.execute('SELECT 1 FROM candidates WHERE row_index=? AND position=? AND source=?',
                            (row,position,doc['source'])).fetchone():raise ValueError('Training candidate eligibility changed')
                    if first_position is None:first_position=position
                    previous=position
                elif slot>=len(evaluation) or row!=int(evaluation[slot]) or doc['evaluation_slot']!=slot:
                    raise ValueError('Evaluation membership/order changed')
            # The immutable archive stores the complete JSON, not flattened id/text
            # fields. Its id was already checked by the full original comparison.
            if ((phase!='archive' and original['id']!=doc['id']) or doc['source']!=pipeline.source_at(ranges,ends,row)
                    or hashlib.sha256(original['text'].encode()).digest()!=doc['text_sha256']):
                raise ValueError('Document text/id/source provenance mismatch')
            docs+=1;slot+=1
    try:
        batch=[]
        for doc,_ in pipeline.iter_arrow([task['file']]):
            batch.append(doc)
            if len(batch)==32:check(batch);batch=[]
        if batch:check(batch)
    finally:
        raw.close()
        if candidates is not None:candidates.close()
    if docs!=task['file']['rows']:raise ValueError('Document file row count differs')
    return dict(pid=os.getpid(),documents=docs,tokens=tokens,first_position=first_position,last_position=previous)


def ranking_task(task):
    """Indexed SQL anti-joins check every winner; rank boundary includes tie order."""
    first,last=task;context=parallel.CONTEXT;p=Path(context['preparation_path']);boundary=context['score_boundary']
    with contextlib.closing(pipeline.readonly(p/'select/candidates.sqlite')) as db:
        db.execute('ATTACH DATABASE ? AS clean',(f'file:{p}/clean/documents.sqlite?mode=ro',))
        db.execute('ATTACH DATABASE ? AS scored',(f'file:{p}/score/scores.sqlite?mode=ro',))
        score_count=db.execute('SELECT COUNT(*) FROM scored.scores WHERE row_index>=? AND row_index<?',(first,last)).fetchone()[0]
        predicate='0' if boundary is None else '(s.score>? OR (s.score=? AND s.position<=?))'
        parameters=(first,last) if boundary is None else (first,last,boundary[0],boundary[0],boundary[1])
        qualified=db.execute('SELECT COUNT(*) FROM scored.scores s WHERE s.row_index>=? AND s.row_index<? AND '+predicate,parameters).fetchone()[0]
        missing=db.execute("SELECT COUNT(*) FROM scored.scores s LEFT JOIN candidates c ON c.row_index=s.row_index "
            "AND c.position=s.position AND c.source='common_crawl' WHERE s.row_index>=? AND s.row_index<? AND "+predicate+' AND c.row_index IS NULL',parameters).fetchone()[0]
        missing_other=db.execute("SELECT COUNT(*) FROM clean.winners w LEFT JOIN candidates c ON c.row_index=w.row_index "
            "AND c.position=w.position AND c.source=w.source WHERE w.row_index>=? AND w.row_index<? "
            "AND w.source!='common_crawl' AND c.row_index IS NULL",(first,last)).fetchone()[0]
        kept=db.execute("SELECT COUNT(*) FROM candidates WHERE row_index>=? AND row_index<? AND source='common_crawl'",(first,last)).fetchone()[0]
        others=db.execute("SELECT COUNT(*) FROM candidates WHERE row_index>=? AND row_index<? AND source!='common_crawl'",(first,last)).fetchone()[0]
    if missing or missing_other:raise ValueError('Global quality ranking/nonCC membership changed')
    return dict(pid=os.getpid(),score_count=score_count,qualified=qualified,kept=kept,others=others)


def provenance_inputs(files,phase):
    slot=0
    for file in files:
        yield dict(file=file,phase=phase,slot=slot);slot+=file['rows']


def ranking_totals_task(_):
    p=Path(parallel.CONTEXT['preparation_path'])
    with contextlib.closing(pipeline.readonly(p/'score/scores.sqlite')) as scores,contextlib.closing(pipeline.readonly(p/'select/candidates.sqlite')) as candidates:
        return dict(pid=os.getpid(),score_count=scores.execute('SELECT COUNT(*) FROM scores').fetchone()[0],
            kept=candidates.execute("SELECT COUNT(*) FROM candidates WHERE source='common_crawl'").fetchone()[0],
            others=candidates.execute("SELECT COUNT(*) FROM candidates WHERE source!='common_crawl'").fetchone()[0])


def verify(prep,raw,index,directory,identity):
    import numpy as np
    cfg=prep.config;split=prep.manifest('split')['details'];selected=prep.manifest('select')['details']
    packed=prep.manifest('pack')['details'];clean=prep.manifest('clean')['details'];n=split['row_count']
    canonical_path=prep.directory('verify')/'canonical-shuffle.arrow'
    canonical=raw.shuffle(seed=cfg['shuffle_seed'],load_from_cache_file=False,indices_cache_file_name=str(canonical_path))
    perm=np.load(prep.path/'split/permutation.npy',mmap_mode='r');inverse=np.load(prep.path/'split/inverse.npy',mmap_mode='r')
    if perm.shape!=(n,) or inverse.shape!=(n,):raise ValueError('Split index length mismatch')
    evaluation=np.load(prep.path/'split/evaluation.npy');subset=np.load(prep.path/'split/evaluation_subset.npy')
    if not np.array_equal(evaluation,perm[-cfg['evaluation_documents']:]):raise ValueError('Evaluation is not canonical last50000')
    if not np.array_equal(subset,evaluation[:cfg['evaluation_subset_documents']]):raise ValueError('Fixed evaluation subset changed')
    del canonical;raw.close()
    expected_kept=pipeline.top_count(clean['source_counts'].get('common_crawl',0),cfg['common_crawl_keep_fraction'])
    with contextlib.closing(pipeline.readonly(prep.path/'score/scores.sqlite')) as db:
        boundary=db.execute('SELECT score,position FROM scores ORDER BY score DESC,position ASC LIMIT 1 OFFSET ?',(expected_kept-1,)).fetchone() if expected_kept else None
    context=dict(preparation_path=str(prep.path),config=cfg,split=split,selected=selected,score_boundary=boundary,
        raw_index=index,raw_directory=str(raw.directory),raw_identity=raw.identity,canonical_path=str(canonical_path))
    started=time.monotonic()
    def progress(phase,done=None,total=None):
        value=dict(identity=identity,phase=phase,completed_tasks=done,total_tasks=total,job_id=os.environ['SLURM_JOB_ID'],
            timestamp=time.time(),elapsed_seconds=time.monotonic()-started,**workers.stats)
        pipeline.atomic_json(Path(directory)/'verification-progress.json',value)
        pipeline.atomic_json(prep.path/'packing-access/parallel-v1/verification-progress.json',value)
        print('parallel verify:',value,flush=True)
    with parallel.Workers(cfg,context) as workers:
        artifacts=[]
        for stage in pipeline.STAGES[:-1]:
            artifacts.extend(prep.manifest(stage)['artifacts'])
        by_path={}
        for item in artifacts:
            old=by_path.get(item['path'])
            if old and any(old[k]!=item[k] for k in ('size','sha256')):raise ValueError('Conflicting immutable artifact checksums')
            by_path[item['path']]=item
        artifacts=list(by_path.values());progress('checksums',0,len(artifacts))
        for done,_ in enumerate(workers.run(parallel.checksum_task,artifacts),1):
            if done%256==0:progress('checksums',done,len(artifacts))
        progress('canonical-split')
        list(workers.run(permutation_task,((i,min(n,i+1_000_000)) for i in range(0,n,1_000_000))))
        archived=split.get('evaluation_raw_files',[])
        if not archived:raise ValueError('Independent evaluation original archive missing')
        counts={}
        for phase,files in (('archive',archived),('train',selected['train_files']),('eval',selected['eval_files'])):
            progress('provenance-'+phase,0,len(files));documents=tokens=0;previous=-1
            for done,result in enumerate(workers.run(provenance_task,provenance_inputs(files,phase)),1):
                documents+=result['documents'];tokens+=result['tokens']
                if phase=='train':
                    if result['first_position'] is not None and result['first_position']<=previous:
                        raise ValueError('Cross-shard training order/duplicate changed')
                    previous=result['last_position']
                if done%32==0:progress('provenance-'+phase,done,len(files))
            counts[phase]=dict(documents=documents,tokens=tokens)
        if counts['archive']['documents']!=cfg['evaluation_documents']:raise ValueError('Incomplete evaluation archive')
        if (counts['train']['tokens']!=selected['train_tokens'] or counts['train']['tokens']>cfg['max_training_tokens']
                or counts['train']['documents']!=selected['train_documents']):raise ValueError('Training budget/counters mismatch')
        if counts['eval']['documents']!=cfg['evaluation_documents'] or counts['eval']['tokens']!=selected['evaluation_tokens']:
            raise ValueError('Incomplete evaluation documents')
        reconstruction={}
        for phase in ('train','eval'):
            progress('reconstruction-'+phase)
            files=selected['train_files' if phase=='train' else 'eval_files']
            plans=parallel.training_plans(files,[0,0],{},cfg['sequence_length'],cfg['sequence_shard_rows']) if phase=='train' else parallel.evaluation_plans(files,[0,0],cfg['sequence_length'],cfg['sequence_shard_rows'])
            tasks=parallel.make_tasks(files,plans,phase,cfg,selected['pad_token_id'],actual=packed['train_files' if phase=='train' else 'eval_files'])
            stats=dict(input_tokens=0,prediction_tokens=0,sequences=0)
            for result in workers.run(parallel.sequence_task,tasks):
                for key in stats:stats[key]+=result[key]
            reconstruction[phase]=stats
        train=reconstruction['train'];eval_stats=reconstruction['eval']
        if train['input_tokens']!=counts['train']['tokens'] or train['input_tokens']!=packed['training_tokens']:
            raise ValueError('Packed training tokens disagree')
        if (eval_stats['prediction_tokens']!=counts['eval']['tokens']-counts['eval']['documents']
                or eval_stats['prediction_tokens']!=packed['evaluation_prediction_tokens']):raise ValueError('Evaluation target coverage changed')
        progress('global-quality-ranking')
        totals=dict(score_count=0,qualified=0,kept=0,others=0)
        for result in workers.run(ranking_task,((i,min(n,i+1_000_000)) for i in range(0,n,1_000_000))):
            for key in totals:totals[key]+=result[key]
        global_counts=next(workers.run(ranking_totals_task,[None]))
        if any(totals[k]!=global_counts[k] for k in ('score_count','kept','others')):
            raise ValueError('Quality score/candidate rows outside full canonical split')
        if totals['score_count']!=clean['source_counts'].get('common_crawl',0):raise ValueError('Missing eligibleCC scores')
        if totals['qualified']!=expected_kept or totals['kept']!=expected_kept:raise ValueError('Common Crawl top10% differs')
        if totals['others']!=clean['eligible_documents']-totals['score_count']:raise ValueError('NonCC candidates differ')
        progress('complete')
    report=prep.directory('verify')/'report.json'
    pipeline.atomic_json(report,dict(fingerprint=prep.key,pack_manifest_sha256=pipeline.sha256(prep.path/'pack/manifest.json'),
        training=train,evaluation=eval_stats,evaluation_documents=counts['eval']['documents'],
        evaluation_raw_documents=counts['archive']['documents'],evaluation_raw_files=archived,subset_documents=len(subset),
        manifest_sha256={s:pipeline.sha256(prep.path/s/'manifest.json') for s in pipeline.STAGES[:-1]},
        training_limit=cfg['max_training_tokens'],scope='artifact checksums, split bijection, provenance, global quality ranking, budget, full sequence reconstruction and loss masks',
        execution_overlay=identity))
    prep.finish('verify',[report,canonical_path]+[item['path'] for item in archived],pipeline.read_json(report))
    return pipeline.read_json(report)
