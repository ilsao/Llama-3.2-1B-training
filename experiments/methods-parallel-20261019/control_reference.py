"""Immutable historical control metadata. No model, checkpoint or dataset loading."""
import hashlib,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('/work/asciibase64/Llama-3.2-1B-training')
RUN='ppl-next-20261017-B-seed42'
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def recipe(cfg):
    result={k:v for k,v in cfg.items() if k not in ('sweep_id','sweep_stage','sweep_purpose','parent_runs','parallel_method')}
    result['ppl']={k:v for k,v in cfg['ppl'].items() if k!='arm'}
    return result
def build(n0):
    run=ROOT/'runs/experiments'/RUN
    summary=read(run/'summary.json');config=read(run/'resolved-config.json')
    initial=read(run/'initialization.json');provenance=read(run/'data-provenance.json')
    confirmation_path=REPO/'experiments/ppl-next-20261017/confirmation.json'
    confirmation=read(confirmation_path);confirm=confirmation['records']['B']
    documents=run.parent/'ppl-next-20261017-confirm/B.json'
    assert recipe(n0)==recipe(config)
    assert initial['seed']==42 and initial['pretrained_weights_loaded'] is False
    assert summary['training_token_target_reached'] and summary['global_step']==4775
    assert summary['total_tokens_seen']==1000079360
    assert confirm['export']==summary['export'] and confirm['evaluation_slots']==[1000,10000]
    assert confirm['original_subset_identity']==provenance['evaluation_indices_sha256']
    selection=summary['last_evaluation']
    assert selection['eval/prediction_tokens']==1479302 and confirm['eval/prediction_tokens']==13666812
    per_doc=read(documents)
    assert len(per_doc)==10000 and all(x[1]>0 for x in per_doc[1000:])
    assert sum(x[1] for x in per_doc[1000:])==confirm['eval/prediction_tokens']
    assert math.isclose(sum(x[0] for x in per_doc[1000:]),confirm['eval/nll'],abs_tol=.01)
    original=REPO/'experiments/ppl-next-20261017/source'
    core=('ppl_optimizer.py','optimizers.py','attention_backend.py','recipe_runtime.py','dataset.py','requirements-training.txt')
    for name in core:assert sha(original/name)==sha(HERE/'source'/name),name
    files=[run/name for name in ('summary.json','resolved-config.json','initialization.json','data-provenance.json',
           'environment.json','optimizer-groups.json','execution-identity.json','resource-accounting.json')]
    files += [confirmation_path,documents,*[original/name for name in core]]
    ledger=read(ROOT/'experiment-ledger.json')
    shared_cost=ledger['jobs'][confirmation['job_id']].get('h200_hours')
    return dict(key='N0-seed42',arm='N0',seed=42,kind='historical_reference',run_id=RUN,
       summary_path=str(run/'summary.json'),confirmation_path=str(confirmation_path),confirmation_record='B',
       document_nll_path=str(documents),new_training_allocation=False,new_h200_hours=0,
       source_files_sha256={str(p):sha(p) for p in files},semantic_recipe_matches=True,
       common_core_files=list(core),native_update_equivalence='synthetic old/new gate added; GPU check not yet run',
       training_inputs=summary['total_tokens_seen'],updates=summary['global_step'],
       preparation_fingerprint=summary['preparation_fingerprint'],initialization_signature=initial['sample_signature'],
       nll_1k=selection['eval/nll']/selection['eval/prediction_tokens'],ppl_1k=selection['eval/perplexity'],
       nll_9k=confirm['eval/nll']/confirm['eval/prediction_tokens'],ppl_9k=confirm['eval/perplexity'],
       ppl_10k=math.exp((selection['eval/nll']+confirm['eval/nll'])/(selection['eval/prediction_tokens']+confirm['eval/prediction_tokens'])),
       historical_training_h200_hours=summary['h200_hours_final'],
       historical_shared_confirmation_h200_hours=shared_cost,shared_confirmation_job_id=confirmation['job_id'],
       cost_note='Historical shared confirmation evaluated multiple models; preserve its whole existing ledger cost, do not recharge or attribute all to this control.',
       comparison_scope='Historical quality reference at matched tokens; speed/memory records descriptive, not contemporaneous controlled evidence',
       confidence_scope='Single training seed; document uncertainty conditional on this trained model')
def validate():
    item=read(HERE/'reference-control.json')
    for path,value in item['source_files_sha256'].items():
        if sha(path)!=value:raise ValueError('Historical reference artifact changed: '+path)
    if recipe(read(HERE/'configs/N0-seed42.json'))!=recipe(read(Path(item['summary_path']).parent/'resolved-config.json')):
        raise ValueError('Historical control recipe no longer matches')
    return item
