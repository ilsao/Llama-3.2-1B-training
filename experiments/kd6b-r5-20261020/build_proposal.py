#!/usr/bin/env python3
"""Build a reviewable proposal using existing receipts; no Slurm submission."""
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD6 = HERE.parent / 'raw6b-normuon-adam2x-20261018'
KD3 = HERE.parent / 'kd3b-hidden8-20261019'
VALID = HERE.parent / 'kd6b-lossless2-20261020'
sys.path.insert(0, str(OLD6 / 'source-v3'))
from ppl_contracts import batch_plan, boundaries, evaluation_units, Q, SPLIT
from training_state import hold_schedule_boundaries


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    source = read(OLD6 / 'config.json')
    cfg = copy.deepcopy(source)
    cfg.update(activation_checkpointing=False, microbatch_size=2,
               gradient_accumulation_steps=1, max_h200_hours=72,
               sweep_id='kd6b-r5-20261020',
               sweep_purpose='PROPOSAL ONLY: fresh 6B KD using validated R5 execution and established 6B horizon recipe',
               sweep_stage='proposed-full',
               parent_runs=['kd3b-hidden8-20261019-seed42',
                            'kd6b-lossless2-validation-20261020',
                            'raw6b-normuon-adam2x-20261018'])
    cfg['parallel_method'] = copy.deepcopy(read(KD3 / 'config.json')['parallel_method'])
    cfg['parallel_method']['pipeline'] = dict(scope='same_rank_online', overlap=False,
                                            disk_spool=False, teacher_devices='same 8 GPUs as student')
    cfg['ppl'] = dict(arm='R5-KD-6BT', cautious_decay=False, mask='intradoc', matrix='nor', stage=3)
    plan = batch_plan(cfg)
    # Preserve the old 6B event-aware global windows, including BS16/32/48
    # shortened windows at schedule/evaluation boundaries. Only micro grouping changes.
    assert plan == batch_plan(source)
    assert all(r['width'] % 16 == 0 and r['accumulation'] // 2 == r['width'] // 16 for r in plan)
    assert len(plan) == 28619 and plan[-1]['end_tokens'] == 6000082944
    assert sum(r['width'] * 8192 for r in plan) == 6000082944
    assert plan[-1]['lr_multiplier'] == 0.0
    assert cfg['eval_steps'] <= len(plan) // 10
    boundaries_detail = []
    for phase in cfg['phase_schedule']:
        start, end = phase['start'], phase['end']
        decay, hold, final, _ = hold_schedule_boundaries(end-start, phase['warmup_steps'], phase['hold_schedule'])
        boundaries_detail.append(dict(
            start_tokens=start*Q, end_tokens=end*Q, peak_ratio=phase['peak_ratio'],
            warmup_end_tokens=(start+phase['warmup_steps'])*Q,
            decay_start_tokens=(start+decay)*Q, plateau_start_tokens=(start+hold)*Q,
            final_decay_start_tokens=(start+final)*Q,
            actual_updates=sum(start*Q <= r['start_tokens'] < end*Q for r in plan)))
    counts = collections.Counter(r['width'] for r in plan)
    verification = read(VALID / 'verification.json')
    status = read(VALID / 'status.json')
    best = next(r for r in status['comparisons'] if r['arm']['name'] == 'R5_combined')
    proposal = dict(
        status='proposal_only_not_submitted', run_id='kd6b-r5-20261020-seed42',
        formal_number_if_authorized=19, training_configuration='training-config.proposed.json',
        training_configuration_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        selection=dict(arm='R5_combined', criterion='fastest tested arm passing strict short-prefix PPL screen',
                       validation_steps=128, validation_student_tokens=16777216,
                       endpoint_ppl=best['endpoint_evaluation']['eval/perplexity'],
                       full6b_quality_proven=False, new_6b_lr_schedule_quality_unmeasured=True),
        horizon_transfer=dict(
            source='existing raw6B table and event-aware schedule',
            note='6B LRs differ from the 3B warmup-prefix benchmark; this is explicit horizon transfer, not an acceleration-only comparison',
            matrix_peak_lr=cfg['muon']['learning_rate'], adamw_peak_lr=cfg['learning_rate'],
            adamw_table_multiplier=2, nominal_tokens=6000000000,
            proposed_complete_batch_endpoint=6000082944, excess_inputs=82944,
            endpoint_exception_not_yet_granted=True),
        batch_plan=dict(actual_updates=len(plan), phase_actual_updates=[p['actual_updates'] for p in boundaries_detail],
                        reference_unit_tokens=Q, reference_units=cfg['scheduler_steps'],
                        global_sequence_counts=dict(sorted(counts.items())),
                        microbatch_per_gpu=2, gpu_count=8,
                        accumulation_by_global_batch={str(n):n//16 for n in counts},
                        batch_switch_tokens=cfg['batch_switch_tokens'],
                        schedule_events_tokens=[n*Q for n in boundaries()],
                        additional_eval_events_tokens=[n*Q for n in evaluation_units()],
                        phases=boundaries_detail, final_update=plan[-1],
                        final_update_accumulation_micro2=plan[-1]['width']//16,
                        event_windows_equal_to_existing_6b=True),
        execution=dict(student_parameters='FP32', teacher_parameters='BF16 frozen', autocast='BF16',
                       loss='0.75 CE + 0.25 KL(teacher || student)', temperature=1.0,
                       vocab_size=128256, loss_chunk_tokens=1024, fused_full_vocab_ce_kl=True,
                       compile_modules=['teacher/student RMSNorm', 'teacher/student MLP', 'RoPE'],
                       compile_options=dict(fullgraph=True,dynamic=False,cudagraphs=False),
                       student_attention='existing pinned FA3 head_dim64',
                       teacher_attention='existing pinned torch2.9.1 ATen flash varlen head_dim128',
                       optimizer_distribution='whole NorMuon matrix owners, FP32 all-gather; replicated fused AdamW',
                       global_gradient_clipping=0.5,
                       evaluation='same eager evaluator for all comparisons'),
        parameter_groups=dict(muon=dict(parameters=973078528, assignment='all hidden attention/MLP Linear weights',
                                        momentum=.95,nesterov=True,ns_steps=5,neuron_ema_beta=.95,
                                        ns_coefficients=[3.4445,-4.775,2.0315],eps=1e-7,
                                        ordinary_weight_decay=.1,decay_lr='nominal matrix LR times scheduler multiplier',
                                        shape_lr_adjustment='original, unchanged'),
                              adamw_no_decay=dict(parameters=262735872,assignment='tied embedding/head once, RMSNorm and any biases',
                                                  betas=[.9,.95],epsilon=1e-8,weight_decay=0.0),
                              adamw_decay=dict(parameters=0)),
        data=dict(dataset='allenai/dolma3_mix-150B-1025',revision='afa92bfb22366821c5e6cd427cdd036b34b713ef',
                  preparation='dolma-raw-baseline-20261008',split='full canonical train document shuffle seed42, last50000 reserved',
                  training='canonical unfiltered order, no repeats',sequence_length=8192,
                  attention='intradoc document-span causal',labels='ignore structural BOS, first sequence position and padding; retain EOS targets',
                  tail_overlay='/work/asciibase64/Llama-3.2-1B-training/runs/data-overlays/raw6b-normuon-adam2x-20261018',
                  tail_policy='read-only reuse after original manifest/checksum and actual 8-rank TailDataset preflight'),
        evaluation=dict(fixed_documents=1000,eval_steps=2800,extra_token_boundary_evaluations=True,
                        endpoint_additional_documents=9000,combined_endpoint_documents=10000,
                        full_reserved_50000_evaluation=False,teacher_not_needed=True),
        checkpoint=dict(policy='final or early wall-time stop',
                        contents=['student weights','all owner matrix optimizer states with ownership map',
                                  'replicated AdamW state once','scheduler','each rank RNG','data cursor',
                                  'teacher and student token journals','W&B identity','cumulative resource ledger'],
                        export='HF-compatible student weights/config/tokenizer; reload verification inside allocation',
                        upload=False,automatic_resume=False,automatic_retry=False,automatic_requeue=False),
        resources=dict(proposed_gpus=8,proposed_cpus_total=8,nodes=1,launcher_tasks=1,rank_threads=1,
                       memory='200G',account='ACD115198',partition='8gpus',constraint='H200',
                       proposed_seconds=32400,proposed_h200_hours=72,
                       currently_requested_h200_ceiling=64,resource_exception_not_yet_granted=True,
                       predicted_compute_only_h200_hours=60.854033224160275,
                       conservative_full_run_h200_hours=best['projected_full6b_hours'],
                       stopping_margin_seconds=300,final_evaluation_reserve_seconds=900,
                       study_used_h200_hours=verification['study_h200_hours'],study_existing_cap=191.419,
                       proposed_study_maximum=verification['study_h200_hours']+72,
                       prior_acceleration_validation_h200_hours=3.0888888888888886),
        environment=dict(torch='2.9.1+cu126',transformers='4.51.3',accelerate='1.6.0',
                         fa3_commit='94e22c906678e5483fa0e9e24d8e787bc2c0ed4c',
                         path='/work/asciibase64/Llama-3.2-1B-training/environments/training-muon-torch291-fa3-94e22c9'),
        validation_remaining=['formal runner integration and owner checkpoint/export support',
                              'end-to-end runtime including data/journal/evaluation/checkpoint overhead',
                              '6B endpoint quality; short 3B-schedule-prefix screen is not proof'],
        authorization_note='No allocation reserved or submitted. The old 3B/6B exceptions are scoped and are not inherited by this proposal.'
    )
    write('training-config.proposed.json', cfg)
    write('proposal.json', proposal)
    print(json.dumps(dict(status=proposal['status'],digest=proposal['training_configuration_sha256'],
                          updates=len(plan),batch_counts=dict(counts),phases=boundaries_detail,
                          proposed_h200_hours=72,study_maximum=proposal['resources']['proposed_study_maximum']),indent=2))


if __name__ == '__main__':
    main()
