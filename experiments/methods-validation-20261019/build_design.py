"""Build and check an experiment DESIGN. No imports of training code, no jobs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/ppl-next-20261017/configs/B-seed42.json'
BATCH = ROOT / 'experiments/ppl-next-20261017/batch-plan.json'
EVIDENCE = ROOT / 'experiments/methods-research-20261019/evidence.json'

def read(p):
    return json.loads(p.read_text())

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def main():
    base = read(BASE)
    rows = read(BATCH)['rows']
    assert len(rows) == 4775
    for i, row in enumerate(rows):
        assert row['update'] == i+1
        assert row['start_tokens'] == (rows[i-1]['end_tokens'] if i else 0)
        assert row['end_tokens']-row['start_tokens'] == row['width']*8192
        assert row['width'] == row['accumulation']*8
    assert rows[-1]['end_tokens'] == base['target_training_tokens'] == 1000079360
    assert rows[-1]['lr_multiplier'] == 0
    assert base['eval_steps'] <= len(rows)//10
    shared = {
        'architecture': base['model_id'], 'model_revision': base['model_revision'],
        'sequence_length': 8192, 'preparation': 'dolma-raw-baseline-20261008',
        'fresh_random_initialization': 'residual_half', 'training_checkpoint_load': False,
        'seed_screen': 42, 'seed_confirm': 43, 'world_size': 8, 'cpus_total': 8,
        'microbatch_size': 1, 'target_nonpadding_inputs': 1000079360,
        'actual_updates': len(rows), 'reference_batch_tokens': 131072,
        'batch_switch_inputs': 500039680, 'phase_split_inputs': 833355776,
        'global_batch_sequences': [16,64], 'boundary_batches': 'Use ALL archived rows, including widths 16/32/48 at knots.',
        'phase_second_peak_ratio': .1, 'warmup_reference_units_per_phase': 170,
        'schedule_clock': 'nonpadding_inputs', 'tail_fraction_per_phase': .4,
        'tail_ramp_hold_final': [4,1,1], 'plateau_ratio': .2,
        'mask': 'intradoc_document_spans', 'attention_backend': 'flash_attention_3',
        'parameter_dtype': 'float32', 'autocast_dtype': 'bfloat16',
        'activation_checkpointing': True, 'global_clip': .5,
        'matrix_peak_lr_reference': base['muon']['learning_rate'],
        'aux_adam_peak_lr': base['learning_rate'], 'aux_betas': [.9,.95], 'aux_eps': 1e-8,
        'aux_weight_decay': 0, 'matrix_reference_wd': .1,
        'matrix_decay_policy': 'External pre-update decay q(t)=1-reference_matrix_peak_lr*0.1*schedule(t), fixed across LR multipliers; native optimizer WD disabled. SF exception: decay on z, documented separately.',
        'optimizer_groups': {'hidden_matrices':973078528,'shared_embedding_and_norms':262735872},
        'tied_embedding_updated_once': True, 'evaluation': {'fixed_subset_documents':1000,'every_updates':400,'also':'archived token/batch/phase knots and endpoint','confirmation_slots':[1000,10000]},
        'wandb_project':'lab5-training-llama','logging_steps':10,
        'checkpoint_policy':'final_only','auto_retry_resume_requeue':False,
    }
    arms=[]
    def arm(id, module, method, controls, overrides, hypothesis, activation='module selected and correctness gates pass'):
        arms.append({'id':id,'module':module,'method':method,'controls':controls,
                     'status':'design_only','execution_ready':False,'activation':activation,
                     'overrides':overrides,'hypothesis':hypothesis,
                     'source_commit':None,'resolved_lr_mapping':None,
                     'unresolved_before_execution':['Pin adapter/source and dependencies','Resolve method-specific LR units on training-only gradients','Pass eight-rank numerical/memory/timing gates','Seal full resolved configuration and accounting reservation']})
    arm('N0','core','current native NS5 NorMuon + AdamW x2',[],{},'Reproduce the current method in the new sealed environment.')
    arm('D1','core','Dion3 NorMuon flavor f=1 bridge',['N0'],{'fraction':1,'momentum':.95,'row_second_moment_beta':.95,'ns_steps':5,'ns_coefficients':[3.4445,-4.775,2.0315],'implementation':'author implementation; document momentum and scaling mapping','aux_optimizer':'same local AdamW'},'Measure implementation migration; equivalence to N0 is a test, not an assumption.')
    arm('D025','core','Dion3 NorMuon flavor f=1/4',['N0','D1'],{'fraction':.25,'momentum':.95,'row_second_moment_beta':.95,'ns_steps':5,'ns_coefficients':[3.4445,-4.775,2.0315],'error_feedback':True,'aux_optimizer':'same local AdamW','lr_center':'D1 matched RMS center / sqrt(f), applied after translating author units'},'Fractional updates improve quality or cost beyond the full-update bridge.')
    arm('A0','geometry','fresh native NorMuon control',[],{},'Concurrent control for geometry module.')
    arm('A1','geometry','AdaMuon v3',['A0'],{'version':'2507.11005v3','sign_before_ns':True,'elementwise_second_moment':True,'rms_alignment':True,'momentum':.95,'second_moment_beta':.95,'aux_optimizer':'unchanged'},'Coordinate-wise adaptive updates improve over row-wise normalization.')
    arm('S1','geometry','full SOAP on hidden matrices',['A0'],{'betas':[.95,.95],'shampoo_beta':.95,'precondition_frequency':10,'max_precond_dim':10000,'merge_dims':False,'precondition_1d':False,'normalize_grads':False,'correct_bias':True,'aux_optimizer':'unchanged','first_gradient_only_initializes_basis':'charge tokens; no replacement batch'},'Historical rotated geometry offsets its time and memory cost.')
    arm('K0','distillation','fresh native NorMuon hard-CE control',[],{},'Control for teacher supervision.')
    arm('K1','distillation','hard CE plus forward KL from existing teacher',['K0'],{'lambda':.25,'temperature':1,'teacher_run':'raw6b-normuon-adam2x-20261018','teacher_revision':'13125ad091ca25c914e1528af6dce22d5f6efa3a','teacher_frozen':True,'student_initialization':'random','kl':'teacher || student','mask':'same shifted valid labels and intradoc context','implementation':'full-vocabulary chunked exact KL first','input_token_counters':['student_training_inputs','teacher_inference_inputs','total_model_input_processing']},'Context-dependent soft targets improve student sample efficiency.')
    arm('K2','distillation_mechanism','hard CE plus uniform smoothing',['K0','K1'],{'epsilon':.1,'lambda_or_entropy_matching':False,'interpretation':'generic smoothing control, not entropy-matched causal isolation'},'Determine whether simple smoothing explains part of a positive KD result.','K1 wins; separate followup budget')
    arm('F0','schedule_free','fresh scheduled native NorMuon control',[],{},'Fixed-horizon endpoint control.')
    arm('F1','schedule_free','SF-NorMuon hidden matrices + SF-AdamW auxiliary',['F0','F2'],{'beta_interpolation':.9,'matrix_momentum':.95,'second_moment_beta':.95,'eta_scale':.2,'warmup_reference_units':170,'schedule':'one warmup, then constant LR; no second-phase restart or cooldown','averaging_clock':'optimizer updates, weight proportional to lr squared; no reset at batch switch','matrix_decay_coefficient':'q(t)=1-reference_matrix_peak_lr*0.1*warmup(t), external on z; fixed across LR factors','eval_weights':'x','training_weights':'y','decay_location':'z','method_label':'local 1B adaptation, not an exact reproduction of paper defaults'},'Feedback averaging yields useful anytime checkpoints with acceptable endpoint regret.')
    arm('F2','schedule_free','constant-LR bridge plus observer averaging',['F0'],{'inner_update':'same momentum/RMS-scaled matrix direction and Adam direction as F1','schedule':'same warmup then constant LR as F1','matrix_decay_coefficient':'same q(t) as F1, on fast weights','gradient_location':'current fast weights, not interpolated y','observer':'same lr-squared average, never fed back','evaluation_variants':['live','observer'],'shared_budget':'two evaluations from one training run'},'Separate observer averaging from the feedback effect of SF training.')
    arm('C0','cce','native CE with native NorMuon',[],{},'Loss-kernel equivalence control.')
    arm('C1','cce','cce_exact with native NorMuon',['C0'],{'cross_entropy':'cce_kahan_full_c_full_e','gradient_filtering':False,'microbatch_size':1,'global_batch':'same archived rows'},'Reduce memory/time while preserving the full objective and gradients.')
    arm('I00','interaction','baseline',[],{},'2x2 interaction reference.','D025 and K1 independently confirmed')
    arm('I10','interaction','confirmed optimizer only',['I00'],{'optimizer':'locked D025 winner'},'Optimizer main effect.','D025 and K1 independently confirmed')
    arm('I01','interaction','confirmed distillation only',['I00'],{'distillation':'locked K1'},'Distillation main effect.','D025 and K1 independently confirmed')
    arm('I11','interaction','confirmed optimizer plus distillation',['I00','I10','I01'],{'optimizer':'locked D025 winner','distillation':'locked K1'},'Test whether independent gains combine.','D025 and K1 independently confirmed')
    budgets=[]
    def cost(module,stage,jobs,minutes,gpus=8):
        budgets.append({'module':module,'stage':stage,'jobs':jobs,'minutes_per_job_cap':minutes,'gpus_per_job':gpus,'cpus_per_job':8,'h200_hours_cap':jobs*minutes*gpus/60})
    for m,val,cal,train in [('core',3,9,3),('geometry',2,9,3),('distillation',2,6,2),('schedule_free',2,9,3)]:
        cost(m,'runtime_validation',val,15)
        cost(m,'bounded_lr_calibration',cal,20)
        cost(m,'matched_1b_seed42',train,120)
        cost(m,'additional_9k_confirmation',1,15)
    cost('replication_per_winner','matched_1b_seed43_control_and_winner',2,120)
    cost('replication_per_winner','additional_9k_confirmation',1,15)
    cost('cce','correctness_and_abba_timing',2,15)
    cost('cce','matched_1b_seed42',2,120)
    cost('cce','additional_9k_confirmation',1,15)
    cost('distillation_mechanism','runtime_validation',1,15)
    cost('distillation_mechanism','bounded_lr_calibration',3,20)
    cost('distillation_mechanism','matched_1b_seed42_K2',1,120)
    cost('distillation_mechanism','confirmation',1,15)
    cost('interaction','combination_runtime_validation',1,15)
    cost('interaction','fresh_2x2_seed42',4,120)
    cost('interaction','confirmation',1,15)
    cost('compute_matched_pair','fresh_budget_matched_control_and_winner',2,90)
    cost('compute_matched_pair','confirmation',1,15)
    totals={m:sum(x['h200_hours_cap'] for x in budgets if x['module']==m) for m in dict.fromkeys(x['module'] for x in budgets)}
    ids={a['id'] for a in arms}
    assert len(ids)==len(arms)
    assert all(c in ids for a in arms for c in a['controls'])
    assert totals['core']==80 and totals['replication_per_winner']==34
    assert all(b['gpus_per_job']<=8 and b['cpus_per_job']<=8 and b['minutes_per_job_cap']<=120 for b in budgets)
    plan={
        'id':'methods-validation-20261019','status':'designed_not_implemented_not_submitted',
        'objective':'Validate structural training methods against a strong matched-token NorMuon baseline, separating migration, algorithm, supervision and systems effects.',
        'execution_ready':False,'allocation_reserved':False,'automatic_stage_promotion':False,
        'default_parallel_jobs':1,'common':shared,
        'immutable_references':{str(p.relative_to(ROOT)):digest(p) for p in [BASE,BATCH,EVIDENCE]},
        'base_configuration_snapshot':base,
        'calibration':{'input_target':1526*131072,'actual_updates':1526,'global_batch_sequences':16,'warmup_updates':50,'eval_steps':152,'schedule':'single-phase token-clock 40% tail, same 4:1:1 shape; SF/constant families instead preserve their constant-LR definition','seed':42,'lr_factors':[.7,1,1.4],'trials_per_method':3,'cap_minutes_per_trial':20,'scope':'numerical and LR-scale screen; short-horizon NLL is not evidence of superiority','selection':'minimum final fixed-1k mean NLL among complete finite trials; ties within .002 choose factor closest to 1; edge optimum flagged as under-calibrated','freeze_before_1b':True,'v0_evaluation_cadence':'For any short model-training check, eval at least every max(1,floor(planned_actual_updates/10)) updates, at endpoint, and every update for very short checks','aux_lr':'fixed at base 1B value except full SF family mapping','wd':'fixed reference shrinkage, not candidate-LR times .1','target_adjustment':'If projected runtime cannot fit before any family trial, pre-register one lower shared target for the whole comparison; never change after observing NLL.'},
        'gate_thresholds_proposed':{'tiny_fp64_reference_loss_atol':1e-6,'tiny_reference_grad_relative_l2_max':1e-5,'full_bf16_ce_loss_abs_max':2e-4,'full_bf16_ce_grad_relative_l2_max':.01,'full_bf16_ce_grad_cosine_min':.999,'gram_ns_bf16_direction_relative_l2_max':.05,'gram_ns_bf16_direction_cosine_min':.995,'diagnostics_step_overhead_max':.03,'runtime_projection_with_margin_minutes_max':110,'memory_free_fraction_min':.1,'nonfinite_allowed':0,'interpretation':'engineering gates fixed before measurements; failing exactness does not permit relabeling an approximation as exact'},
        'quality_decision':{'primary_metric':'sum document NLL / sum valid prediction tokens','delta':'candidate minus control','practical_mean_delta_max':-.005,'seed42_screen':True,'confirmation_documents':9000,'document_bootstrap_resamples':10000,'core_contrasts':['D025-N0','D025-D1'],'core_simultaneous_ci_level_each':.975,'geometry_simultaneous_ci_level_each':.975,'single_locked_contrast_ci_level':.95,'quality_confirmation':'paired interval upper < 0; practical delta <= -.005; fresh seed43 same direction; mean across seeds <= -.005','speed_noninferiority':'paired interval upper < +.002 and >=5% reduction in actual comparable allocation cost; label efficiency result, not quality improvement','cost_increase_policy':'over 5% cost increase is a Pareto tradeoff until compute-matched confirmation','multiple_methods':'lock one winner per family before seed43; retain all failed and nonwinning runs','uncertainty':'document CI is conditional on trained weights; two seeds do not establish population-level seed confidence'},
        'teacher_accounting':{'student_training_cap':1000079360,'teacher_forward_inputs':'separately charged and reported','student_cap_counts_repeated_student_forward_passes':True,'combined_processing_reported':True,'conservative_logical_input_charge':'student forwards + teacher forwards + any extra model forward on training data','conservative_logical_input_cap':6000000000,'schedule_clock':'student nonpadding inputs; distinct from conservative logical charge','teacher_historical_training_h200_hours':64.70666666666668,'teacher_historical_training_inputs':6000082944,'teacher_cost_not_erased_or_recharged_as_new_allocation':True},
        'budget_totals_h200_hours_caps':totals,
        'recommended_first_decision_cap_h200_hours':80,
        'recommended_first_decision_with_one_replication_cap_h200_hours':114,
        'first_decision_estimate_assumptions':{'historical_1b_N0_h200_hours':10.726666666666667,'three_1b_arms_if_similar_cost':32.18,'three_runtime_validation_jobs_max':6,'nine_calibration_jobs_max':24,'confirmation_max':2,'illustrative_total_if_training_costs_match_N0':64.18,'not_a_measured_new_method_cost':True},
        'optional_modules_not_automatically_summed_or_activated':True,
        'ledger_snapshot_reference':'experiments/raw6b-normuon-adam2x-20261018/ledger-latest.json',
        'historical_measured_h200_hours_at_snapshot':1194.6611111111113,
        'remaining_authorized_campaign_budget':None,
        'authorization_scope':'Design only. Historical run-specific exceptions do not authorize these new allocations.',
        'slurm_reference':'slurm/ppl-factorial-train.sbatch',
        'implementation_contract':['Keep original source-v3 and preparation immutable','Isolated source snapshot and environment','Do not emit runnable train configs until method adapters and LR mappings are resolved','Full-model checks run through Slurm','Every allocation, including failure and compilation, enters shared ledger','No automatic retry/resume/requeue, HF publication or cleanup'],
        'deferred_methods':[
            {'method':'Newton-Muon','experiment':'fresh N0 versus activation-covariance preconditioned direction; activate only after covariance diagnosis and use same geometry-module cost gates'},
            {'method':'DoReMi/DoGE','experiment':'train-only proxy learns frozen domain mixture, then compare fresh uniform/native-mixture and learned-mixture students at matched tokens; original split untouched; proxy and packing separate charged costs','requires':'new data-mixture protocol; not included in core or given a fabricated cost estimate'},
            {'method':'MARS','experiment':'compare approximate one-gradient implementation with N0; exact variant needs an explicit repeated-input and compute budget','requires':'separate method adapter and resource design'},
            {'method':'Cautious optimizer','experiment':'change auxiliary Adam update mask only; preserve ordinary weight decay; compare to fresh N0','requires':'separate bounded module'},
            {'method':'multi-token prediction','experiment':'same backbone with additional training heads, masked across document boundaries','requires':'architecture-scope change; excluded from current fixed-architecture core'}],
        'sources':['https://arxiv.org/html/2608.11612v1','https://github.com/microsoft/dion','https://arxiv.org/html/2507.11005v3','https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py','https://arxiv.org/html/2605.23061v1','https://arxiv.org/abs/2502.08606','https://github.com/apple-aiml-research/ml-cross-entropy'],
    }
    dump('plan.json',plan)
    dump('arms.json',{'status':'design_blueprints_not_training_configs','common_reference':'plan.json#/common','arms':arms})
    with (OUT/'budget.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(budgets[0]));w.writeheader();w.writerows(budgets)
    dump('design-check.json',{'status':'static_design_checks_passed','batch_rows_validated':len(rows),'target_tokens':rows[-1]['end_tokens'],'eval_interval_valid':True,'unique_arm_ids':len(ids),'all_comparison_references_resolved':True,'budget_caps_recomputed':totals,'core_plus_replication_h200_hours':totals['core']+totals['replication_per_winner'],'models_loaded':False,'jobs_submitted':0,'new_method_runtime_checks':'not performed'})
    print(json.dumps({'arms':len(arms),'budget_caps':totals,'core_plus_replication':114},ensure_ascii=False))

if __name__=='__main__':
    main()
