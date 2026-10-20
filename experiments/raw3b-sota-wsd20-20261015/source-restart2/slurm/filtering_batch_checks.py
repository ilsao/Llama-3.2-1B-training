"""Compute-only equivalence, saved-token checks and matched-input speed gate."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import dataset as pipeline
from filtering_reader import IndexedRaw
from preparation_parallel import ordered_executor
from filtering_resources import worker_count, training_worker_count, evaluation_worker_count


def rss_upper_bytes(workers):
    # Linux ru_maxrss is KiB. Conservatively charge the largest child to every worker.
    return 1024 * (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss +
                   workers * resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


def saved_tokens(prep, raw):
    state = pipeline.read_json(prep.path / 'select/state.json')
    for item in state['train_files'] + state['eval_files']:
        pipeline.stop_if_requested(); pipeline.check_artifact(item, True)
    recent = []
    for item in reversed(state['train_files']):
        rows = [row for row, _ in pipeline.iter_arrow([item])]
        recent = rows[-(16-len(recent)):] + recent
        if len(recent) >= 16: break
    tokenizer = pipeline.load_tokenizer(prep)
    for doc in recent:
        actual = pipeline.make_document(raw[doc['row_index']], doc['row_index'],
            doc['shuffle_position'], doc['source'], tokenizer, doc['evaluation_slot'])
        if actual != doc:
            raise ValueError('Saved tokenized document differs from new reader/tokenizer')
    return dict(documents=len(recent), state_sha256=pipeline.sha256(prep.path/'select/state.json'),
                committed_documents=state['train_documents'], committed_tokens=state['train_tokens'])


def benchmark(prep, raw, directory, evidence_identity):
    target = directory / 'performance.json'
    if target.exists():
        saved = pipeline.read_json(target)
        if saved['identity'] != evidence_identity or not saved['passed']:
            raise ValueError('Existing performance gate failed or changed identity')
        return saved
    state = pipeline.read_json(prep.path/'select/state.json')
    with contextlib.closing(pipeline.readonly(prep.path/'select/candidates.sqlite')) as db:
        rows = db.execute('SELECT row_index,position,source FROM candidates WHERE position>? ORDER BY position LIMIT 1024',
                          (state['last_position'],)).fetchall()
    if len(rows) != 1024:
        raise ValueError('Need 1024 matched candidate documents for performance gate')
    index = dict(row_count=len(raw), files=raw.files)
    old = IndexedRaw(index)
    tokenizer = pipeline.load_tokenizer(prep); workers = worker_count(prep.config)
    measurements = {}; reference = None
    # Same original 32-document input batches and seven ordered tokenizer workers.
    # New reader runs first so its measured cost includes lazy index construction.
    for label, reader in (('batch', raw), ('legacy', old)):
        read_seconds = 0.; outputs = []; started = time.monotonic()
        try:
            with ordered_executor(workers, initializer=pipeline.initialize_tokenizer,
                                  initargs=(prep.config,), check=pipeline.stop_if_requested) as ordered:
                for offset in range(0, len(rows), 32):
                    pipeline.stop_if_requested(); chunk = rows[offset:offset+32]
                    before = time.monotonic(); documents = list(reader.select([r[0] for r in chunk])); read_seconds += time.monotonic()-before
                    tasks = ((doc, row, pos, source, -1) for (row,pos,source),doc in zip(chunk,documents))
                    outputs.extend(pipeline.tokenize_documents(ordered, tokenizer, tasks, workers))
            seconds = time.monotonic()-started
            tokens = sum(len(doc['tokens']) for doc in outputs)
            measurements[label] = dict(seconds=seconds, reader_seconds=read_seconds, tokens=tokens,
                tokens_per_second=tokens/seconds, documents=len(outputs))
            if reference is None: reference = outputs
            elif outputs != reference: raise ValueError('Matched benchmark tokens/order/provenance differ')
        finally:
            reader.close()
    rss = rss_upper_bytes(workers)
    speedup = measurements['legacy']['seconds']/measurements['batch']['seconds']
    result = dict(identity=evidence_identity, measurements=measurements, speedup=speedup,
        memory_upper_bytes=rss, memory_limit_bytes=160*1024**3,
        passed=speedup>=2 and rss<160*1024**3, checkpoint_sha256=pipeline.sha256(prep.path/'select/state.json'),
        input_sha256=pipeline.fingerprint(rows), lazy_index_cost_included=True,
        reader_stats=raw.stats, slurm_job_id=os.environ['SLURM_JOB_ID'], timestamp=time.time())
    pipeline.atomic_json(target,result); print('reader benchmark:',json.dumps(result),flush=True)
    if not result['passed']:
        raise ValueError('Performance/memory gate failed; production selection left unchanged')
    return result


def progress_writer(prep, raw, evidence_identity):
    """Wrap only durable state commits; never alter selection or checkpoint contents."""
    original = pipeline.atomic_json; state_path=prep.path/'select/state.json'
    initial=pipeline.read_json(state_path); started=time.monotonic(); last=[started, initial['train_tokens']]
    progress_directory=prep.path/'filtering-access/batch-reader-v1'/pipeline.fingerprint(evidence_identity)
    progress_directory.mkdir(parents=True,exist_ok=True)
    progress_path=progress_directory/'progress.json'
    def write(path, value):
        original(path,value)
        if Path(path) != state_path: return
        now=time.monotonic(); elapsed=now-started; tokens=value['train_tokens']-initial['train_tokens']
        rate=tokens/elapsed if elapsed else 0
        record=dict(identity=evidence_identity,slurm_job_id=os.environ['SLURM_JOB_ID'],
            timestamp=time.time(), committed_tokens=value['train_tokens'], committed_documents=value['train_documents'],
            evaluation_documents=value['next_evaluation_slot'], evaluation_tokens=value['evaluation_tokens'],
            session_tokens=tokens, session_seconds=elapsed, tokens_per_second=rate,
            last_commit_tokens_per_second=(value['train_tokens']-last[1])/(now-last[0]) if now>last[0] else 0,
            remaining_training_seconds=(prep.config['max_training_tokens']-value['train_tokens'])/rate if rate>0 else None,
            reader_stats=dict(raw.stats), process_memory_upper_bytes=rss_upper_bytes(max(training_worker_count(prep.config),evaluation_worker_count(prep.config))),
            timing_scope='production pipeline elapsed, including reader, tokenization, writing and checkpoints',
            checkpoint_sha256=pipeline.sha256(state_path))
        original(progress_path,record)
        original(prep.path/'filtering-access/batch-reader-v1/progress.json',record)
        if (training_worker_count(prep.config)>7
                and prep.config['max_training_tokens']-value['train_tokens']<2
                and value['next_evaluation_slot']==0):
            raise pipeline.StopRequested('Training tokenization complete; resume evaluation in its own allocation')
        last[:]=[now,value['train_tokens']]
        if now-started>=900 and not (progress_path.parent/'fifteen-minute.json').exists():
            original(progress_path.parent/'fifteen-minute.json',record)
        print('select progress:',json.dumps(record),flush=True)
    return original,write
