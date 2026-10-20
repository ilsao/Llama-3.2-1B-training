#!/usr/bin/env python3
"""Read small validation receipts only; no model work or job submission."""
import datetime
import json
import math
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=Path('/work/asciibase64/Llama-3.2-1B-training')
RUN=WORK/'runs/experiments/kd6b-lossless2-validation-20261020'
def read(p):return json.loads(p.read_text())
state={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'full6b_started':False,'full6b_ppl_noninferiority_proven':False}
for name in ('owner-update-gate.json','teacher-attention-gate.json','gradient-gates.json','results.json','complete.json','compiler.json'):
    if (RUN/name).exists():state[name]=read(RUN/name)
for name in ('teacher','student'):
    p=RUN/(name+'-input-journal.json')
    if p.exists():state[name+'_journal']=read(p)
if (HERE/'submission.json').exists():
    state['submission']=read(HERE/'submission.json')
    job=state['submission']['job_id']
    assert job.isdigit()
    result=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2',
                           '--format=JobIDRaw,State,ElapsedRaw,Start,End'],capture_output=True,text=True,timeout=20,check=True)
    state['slurm']=result.stdout.strip()
cost=4.;cost_status='reserved maximum'
if (HERE/'accounting.json').exists():
    state['accounting']=read(HERE/'accounting.json');cost=state['accounting']['h200_hours'];cost_status='actual'
state['all_acceleration_validation_h200_hours']=.8+cost
state['validation_cost_status']=cost_status
state['rate_needed_including_validation']=1.1*6e9/(64*3600/8-1800-(.8+cost)*3600/8)
state['progress']={}
for p in sorted(RUN.glob('R*/metrics.jsonl')):
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    metrics=[r for r in rows if 'train/total_tokens_seen' in r]
    if metrics:state['progress'][p.parent.name]=metrics[-1]
arms=state.get('results.json',{}).get('arms',[])
if arms:
    base=arms[0]
    for r in arms:
        ppl=r['endpoint_evaluation']['eval/perplexity'];ref=base['endpoint_evaluation']['eval/perplexity']
        r['ppl_difference_vs_R0']=ppl-ref
        r['nll_difference_vs_R0']=math.log(ppl/ref)
        r['ppl_percent_change_vs_R0']=100*(ppl/ref-1)
        r['forecast_including_validation_h200_hours']=r['projected_full6b_hours']+.8+cost
state['comparisons']=arms
(HERE/'status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
lines=['# Lossless加速：配對品質與速度結果','',
       '此頁由collect.py讀取實際GPU收據產生。最終6B PPL不退步仍未證明。',
       '',f"Slurm：`{state.get('slurm','unknown')}`",
       '',f'本輪成本：{cost:.6f} H200-hours（{cost_status}）；連前輪驗證共{cost+.8:.6f}H。',
       '',f"若64H包含這些驗證，需至少{state['rate_needed_including_validation']:,.0f}加權tokens/s（含10%時間餘量及30分鐘收尾）。",'',
       '| 組別 | 固定1k PPL | ΔPPL對R0 | 短程PPL不升 | BS16 tok/s | BS64 tok/s | 加權tok/s | 6B預估H含餘量 |',
       '|---|---:|---:|---|---:|---:|---:|---:|']
for r in arms:
    lines.append(f"| {r['arm']['name']} | {r['endpoint_evaluation']['eval/perplexity']:.9f} | {r['ppl_difference_vs_R0']:+.9f} | {'通過' if r['short_prefix_ppl_not_worse_than_R0'] else '未通過'} | {r['tokens_per_second_bs16']:,.0f} | {r['tokens_per_second_bs64']:,.0f} | {r['harmonic_tokens_per_second']:,.0f} | {r['projected_full6b_hours']:.3f} |")
lines+=['','品質訓練為同初始化／同順序的128步、16,777,216 tokens，原3B schedule前綴，沒有縮短cooldown。全部使用相同eager evaluation。',
        '', 'BS16速度來自非零LR真實訓練；BS64在品質評估之後，以13個後續真實窗口做零LR測速，前3個為warmup。',
        '', 'PPL門檻未添加容忍退步幅度。短程通過只代表此配對前綴的測量結果，不代表最終6B效果。',
        '', '完整契約見[protocol.md](protocol.md)，逐文件NLL與W&B連結位於各arm工作目錄。']
(HERE/'results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(slurm=state.get('slurm'),completed_arms=[r['arm']['name'] for r in arms],
                     progress={k:v['train/global_step'] for k,v in state['progress'].items()},
                     gradient_gates=state.get('gradient-gates.json'),comparisons=[{
                         'arm':r['arm']['name'],'ppl':r['endpoint_evaluation']['eval/perplexity'],
                         'ppl_delta':r['ppl_difference_vs_R0'],'rate':r['harmonic_tokens_per_second'],
                         'forecast_h':r['projected_full6b_hours']} for r in arms]),indent=2))
