"""Immutable run-local completion of the last padded raw shard; compute only."""
from pathlib import Path
import os,sys,copy
from training_state import read_json,atomic_json,file_digest,DEFAULT_ROOT
from ppl_contracts import STUDY,TARGET,PREP
LIVE=Path(__file__).resolve().parent.parent
DIRECTORY=DEFAULT_ROOT/'runs/data-overlays'/STUDY
MANIFEST=DIRECTORY/'manifest.json'

def metadata_check():
    for name,sha in read_json(LIVE/'raw-metadata-seal.json').items():
        if file_digest(Path(name))!=sha:raise ValueError('Canonical raw metadata changed: '+name)

def check():
    metadata_check();m=read_json(MANIFEST);prefix=read_json(LIVE/'raw-prefix.json')['files']
    if (m['run_id']!=STUDY or m['target_tokens']!=TARGET or not m['validated'] or
        m['train_files'][:-1]!=prefix[:-1] or m['original_last_shard']!=prefix[-1] or
        sum(x['rows'] for x in m['train_files'])*8192!=TARGET):raise ValueError('Invalid isolated6B tail')
    from dataset import check_artifact
    check_artifact(m['train_files'][-1],True)
    return m

def main():
    import dataset as p
    p.require_compute()
    from preparation_parallel import configure_threads
    configure_threads()
    import numpy as np
    import datasets
    from filtering_reader import build_index
    from filtering_batch_reader import BatchRaw
    metadata_check()
    if MANIFEST.exists():
        check();print('Reusing verified immutable canonical tail; no training checkpoint loaded',flush=True);return
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    if any(DIRECTORY.iterdir()):raise ValueError('Unpublished tail artifacts exist; preserve for inspection')
    prefix=read_json(LIVE/'raw-prefix.json')['files'];last=prefix[-1]
    p.check_artifact(last,True)
    rows=list(datasets.Dataset.from_file(last['path']));tail=rows[-1]
    valid=sum(tail['attention_mask'])
    if not 0<valid<8192 or tail['attention_mask']!=[1]*valid+[0]*(8192-valid):raise ValueError('Unexpected padding layout')
    if any(sum(r['attention_mask'])!=8192 for r in rows[:-1]):raise ValueError('Earlier padding in final shard')
    if sum(x['length'] for x in tail['document_spans'])!=valid:raise ValueError('Invalid document spans')
    source=p.Preparation(DEFAULT_ROOT,'dolma-dclm10-005',read_json(DEFAULT_ROOT/'preprocessed/dolma-dclm10-005/configuration.json')['config'])
    split=source.manifest('split')['details'];config=source.config
    index=build_index(source)
    identity=dict(preparation_fingerprint=source.key,split_sha256=file_digest(source.path/'split/manifest.json'))
    reader=BatchRaw(index,source.path/'filtering-access/batch-reader-v1/indexes',identity)
    perm=np.load(source.path/'split/permutation.npy',mmap_mode='r');inverse=np.load(source.path/'split/inverse.npy',mmap_mode='r')
    ends=[x['end'] for x in split['ranges']]
    p.initialize_tokenizer(config)
    final_span=tail['document_spans'][-1];last_position=int(inverse[final_span['row_index']])
    original=p.original_document(reader.fetch([final_span['row_index']])[0])
    final_doc=p.tokenize_document((original,final_span['row_index'],last_position,p.source_at(split['ranges'],ends,final_span['row_index']),-1))
    if len(final_doc['tokens'])!=final_span['document_start']+final_span['length']:raise ValueError('Final old document truncated')
    if final_doc['tokens'][final_span['document_start']:]!=tail['input_ids'][final_span['sequence_start']:valid]:raise ValueError('Final original document mismatch')
    start=last_position+1
    saved=read_json(LIVE/'producer-state-before.json')
    if start!=saved['next_position']:raise ValueError('Canonical producer cursor has skipped trailing documents')
    packer=p.TrainingPacker(8192,p._TOKENIZER.eos_token_id,copy.deepcopy(dict(tokens=tail['input_ids'][:valid],labels=tail['labels'][:valid],spans=tail['document_spans'])))
    output=rows[:-1];needed=TARGET//8192-sum(x['rows'] for x in prefix[:-1]);positions=[]
    try:
        for position in range(start,min(split['train_document_count'],start+4096)):
            p.stop_if_requested();row=int(perm[position])
            if int(inverse[row])!=position or position>=split['train_document_count']:raise ValueError('Reserved membership violation')
            record=reader.fetch([row])[0];original=p.original_document(record)
            doc=p.tokenize_document((original,row,position,p.source_at(split['ranges'],ends,row),-1))
            positions.append(dict(position=position,row_index=row,tokens=len(doc['tokens'])))
            output.extend(packer.push(row,doc['tokens']))
            if len(output)>=needed:break
        else:raise ValueError('Insufficient bounded canonical tail')
    finally:reader.close()
    output=output[:needed]
    assert output[:len(rows)-1]==rows[:-1]
    filled=output[len(rows)-1]
    assert filled['input_ids'][:valid]==tail['input_ids'][:valid] and filled['labels'][:valid]==tail['labels'][:valid]
    assert filled['document_spans'][:len(tail['document_spans'])]==tail['document_spans']
    for row in output:
        assert len(row['input_ids'])==8192 and row['attention_mask']==[1]*8192
        assert sum(s['length'] for s in row['document_spans'])==8192
        cursor=0
        for span in row['document_spans']:
            assert span['sequence_start']==cursor and int(inverse[span['row_index']])<split['train_document_count']
            cursor+=span['length']
    item=p.write_arrow(DIRECTORY/'train-final-extended.arrow',output,p.sequence_schema(8192))
    p.check_artifact(item,True);metadata_check()
    atomic_json(MANIFEST,dict(run_id=STUDY,job_id=os.environ['SLURM_JOB_ID'],target_tokens=TARGET,validated=True,
        train_files=prefix[:-1]+[item],original_last_shard=last,original_valid_tail_tokens=valid,
        replaced_padding_tokens=8192-valid,original_nonpadding_prefix_preserved=True,
        original_full_rows_preserved=True,canonical_next_position=start,appended_documents=positions,
        metadata_hashes=read_json(LIVE/'raw-metadata-seal.json'),status='complete'))
    check();print('6B canonical tail complete: '+str(TARGET)+' nonpadding inputs; original files unchanged',flush=True)

if __name__=='__main__':
    if sys.argv[1:]==['--check']:check();print('6B tail metadata/checksum verified',flush=True)
    else:main()
