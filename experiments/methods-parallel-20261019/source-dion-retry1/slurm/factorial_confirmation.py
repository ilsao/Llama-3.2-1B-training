"""Read-only model evaluation of the locked winner and control, slots 1000:10000."""
import sys,os,math,gc
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1];LIVE=SOURCE.parent
sys.path.insert(0,str(SOURCE))

def main():
    from train import require_compute,evaluate,prepare_data,on_main
    require_compute()
    import torch,numpy as np,datasets,pyarrow as pa
    torch.set_num_threads(1);torch.set_num_interop_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM
    from training_state import DEFAULT_ROOT,read_json,load_config,atomic_json
    from raw_baseline import packed_details
    from attention_backend import configure_attention
    selection=read_json(LIVE/'selection.json');winner=selection['winner']
    if not selection['qualified']:raise ValueError('Winner not qualified')
    accelerator=Accelerator(mixed_precision='bf16')
    prep=DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008'
    records={};diagnostics=DEFAULT_ROOT/'runs/experiments/ppl-next-20261017-confirm'
    on_main(accelerator,lambda:diagnostics.mkdir(parents=True,exist_ok=True))
    for arm in ('A',winner):
        cfg=load_config(LIVE/'configs'/f'{arm}-seed42.json')
        run=DEFAULT_ROOT/'runs/experiments'/f'ppl-next-20261017-{arm}-seed42'
        _,_,provenance=prepare_data(prep,run,cfg,accelerator)
        details=packed_details(prep)
        rows=datasets.concatenate_datasets([datasets.Dataset.from_file(x['path']) for x in details['eval_files']])
        rows=rows.filter(lambda x:1000<=x<10000,input_columns=['evaluation_slot'],load_from_cache_file=False,keep_in_memory=True)
        if set(rows['evaluation_slot'])!=set(range(1000,10000)):raise ValueError('Incorrect confirmation membership')
        summary=read_json(run/'summary.json')
        model=AutoModelForCausalLM.from_pretrained(summary['export'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa')
        model.config.use_cache=False;configure_attention(model,cfg['attention_backend']);model.to(accelerator.device)
        os.environ['EVAL_DOCUMENT_OUTPUT']=str(diagnostics/f'{arm}.json')
        result=evaluate(model,rows,{**cfg,'eval_documents':10000},accelerator)
        result['eval/documents']=9000
        result.update(export=summary['export'],evaluation_slots=[1000,10000],original_subset_identity=provenance['evaluation_indices_sha256'])
        records[arm]=result
        del model;gc.collect();torch.cuda.empty_cache()
    def finish():
        a=np.asarray(read_json(diagnostics/'A.json'))[1000:10000]
        b=np.asarray(read_json(diagnostics/f'{winner}.json'))[1000:10000]
        if not np.array_equal(a[:,1],b[:,1]) or np.any(a[:,1]<=0):raise ValueError('Unmatched prediction tokens')
        delta=float((b[:,0].sum()-a[:,0].sum())/a[:,1].sum())
        rng=np.random.default_rng(42);samples=[]
        for _ in range(2000):
            idx=rng.integers(0,9000,9000)
            samples.append(float((b[idx,0].sum()-a[idx,0].sum())/a[idx,1].sum()))
        result=dict(status='complete',winner=winner,delta_nll=delta,paired_document_ci95=np.quantile(samples,[.025,.975]).tolist(),
                    bootstrap_replicates=2000,bootstrap_seed=42,records=records,
                    prefix128={arm:read_json(diagnostics/f'{arm}.json.prefix.json') for arm in records},
                    note='Reused canonical validation subset; document uncertainty does not measure seed variance',job_id=os.environ['SLURM_JOB_ID'])
        atomic_json(LIVE/'confirmation.json',result)
    on_main(accelerator,finish);accelerator.end_training()
if __name__=='__main__':main()
