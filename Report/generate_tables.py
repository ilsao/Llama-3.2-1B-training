#!/usr/bin/env python3
"""Generate LaTeX tables and numeric macros from the frozen evidence snapshot."""
import collections
import csv
import json
import math
from pathlib import Path

P = Path(__file__).resolve().parent
D = json.loads((P / 'data/evidence.json').read_text())
R = {r['formal']: r for r in D['runs'] if r['formal']}
OUT = P / 'tables'
OUT.mkdir(exist_ok=True)
NAMES = {
 1:'AdamW baseline',2:'Raw Muon + continuation',3:'Filtered Muon',4:'BE60P20 + LR scaling',
 5:'WSD standard',6:'WSD resplit identity',7:'Initialization recipe',8:'Residual-half + resplit',
 9:'Raw 3B reference',10:'3B selected recipe',11:'3B WSD20',12:'3B batch + token40',
 13:'3B batch + update40',14:'4B intradoc + NorMuon',15:r'4B AdamW LR $2\times$',
 16:'4B update-tail30',17:'6B non-KD',18:'3B KD, 8+8 GPUs',19:'6B R5 KD',20:'6B C2 + early decay'}


def fmt(v, n=4):
    return '--' if v is None else f'{v:.{n}f}'


def table(name, caption, label, spec, header, rows, note=''):
    text = '\\begin{table}[tbp]\n\\centering\n\\small\n'
    text += '\\caption{' + caption + '}\n\\label{' + label + '}\n'
    text += '\\begin{tabular}{' + spec + '}\n\\toprule\n'
    text += header + ' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, r)) + ' \\\\' for r in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    if note:
        text += '\\par\\vspace{3pt}\\begin{minipage}{\\linewidth}\\footnotesize ' + note + '\\end{minipage}\n'
    text += '\\end{table}\n'
    (OUT / name).write_text(text)


def main():
    table('formal_results.tex',
          'Registry of all 20 formal identities. PPL columns retain their distinct evaluation scopes; lower is better. B denotes billions of nonpadding student input tokens.',
          'tab:formal', 'rlrrrrr',
          'ID & Recipe & Inputs (B) & 1k PPL & 10k PPL & 50k PPL & H200-h',
          [[i,NAMES[i],fmt(r['export_inputs']/1e9,3),fmt(r['ppl_1k']),fmt(r['ppl_10k']),
            fmt(r['ppl_50k']) if r['full_eval_matches_export'] else '--',fmt(r['h200_hours'],2)]
           for i,r in R.items()],
          r'Inputs refer to the exported checkpoint; all processed inputs remain charged. ID 1 processed 1.166934016B but exported the earlier best checkpoint at 1.159725056B. ID 2\textquotesingle s stored 50k result belongs to step 8500, not its final export at step 13717, and is omitted here. ID 6 includes checkpoint forks. ID 18 includes both teacher and student allocations. Missing scopes were not measured or not checkpoint-aligned.')
    rows=[]
    for i in [17,19,20]:
        r=R[i];rows.append([NAMES[i],f"{r['updates']:,}",fmt(r['ppl_1k']),fmt(r['ppl_9k']),fmt(r['ppl_10k']),fmt(r['h200_hours'],3)])
    table('final6b.tex','Six-billion-input recipe comparison at the identical 6,000,082,944-input endpoint, seed42 and evaluation provenance.',
          'tab:final6b','lrrrrr','Recipe & Updates & 1k PPL & Extra 9k & Combined 10k & H200-h',rows,
          r'The 10k metric is $\exp[(\mathrm{NLL}_{1k}+\mathrm{NLL}_{9k})/(N_{1k}+N_{9k})]$, not an average of perplexities. Runtime implementation differs between non-KD and R5. C2 also changes the first decay boundary; it is a combined intervention.')
    table('critical_screen.tex','Adaptive-batch screen: all attempts finish at 1,000,079,360 confirmed student inputs using the original 6B schedule prefix, without a compressed cooldown.',
          'tab:cbs','lrrrr','Arm & Updates & 1k PPL & Combined 10k & Logical H200-h',
          [[name, f"{r['updates']:,}",fmt(r['ppl_1k']),fmt(r['ppl_10k']),fmt(r['h200_hours'],3)]
           for name,r in zip(['C0 shadow','C1 half','C2 one'],D['runs'][-3:])],
          r'C0/C1 costs include both earlier failed startups. Each retains 131,072 prior charged-but-unconfirmed inputs per role in addition to its successful attempt. The three-arm correctness gate costs another 0.05778 H200-h. These endpoints must not be ranked against complete 1B schedules from other studies.')
    methods=[]
    labels={'N0':'NorMuon reference','D1':'Dion3 full','D025':'Dion3 fraction 0.25','A1':'AdaMuon','S1':'SOAP',
            'K1':'Initial local teacher','F1':'Schedule-free feedback','F2':'Weight-average observer','C1':'CCE exact','K3':'Official-teacher pipeline'}
    for k in ['N0','D1','D025','A1','S1','K1','F1','F2','C1','K3']:
        v=D['methods_status']['runs'][k];state=v['state'].split(' by ')[0].title()
        methods.append([labels[k],state,fmt(v['inputs']/1e9,3),fmt(v.get('ppl_10k')),fmt(v['h200_hours_measured'],3)])
    table('methods.tex','Method exploration with the later NorMuon baseline. Only complete matched 1B endpoints receive a combined-10k PPL.',
          'tab:methods','llrrr','Method & Outcome & Inputs (B) & 10k PPL & H200-h',methods,
          r'N0 is a historical control already charged elsewhere; it is not charged a second time to this campaign. Early stops and implementation failures do not establish an endpoint quality ranking or the theoretical failure of a method.')
    common=[
      ['Architecture','Llama 3.2 1B; 1,235,814,400 trainable parameters'],
      ['Layers / dimensions','16 layers; hidden 2048; MLP 8192; 32 query / 8 KV heads'],
      ['Vocabulary / context','128,256 tokens; training context fixed at 8192'],
      ['Shared parameters','Tied embedding / LM head; AdamW group counted once'],
      ['Canonical split','104,020,922 documents; seed42 global shuffle; final 50,000 held out'],
      ['Evaluation','Fixed first 1k; next 9k confirmation; full 50k when recorded'],
      ['Raw preparation',r'\texttt{dolma-raw-baseline-20261008}'],
      ['Late initialization',r'Fresh Gaussian origin; residual output projections scaled by $1/2$'],
      ['Late matrix optimizer','NorMuon, NS5, momentum 0.95, Nesterov, row second moment 0.95'],
      ['Late AdamW','Betas (0.9, 0.95); epsilon $10^{-8}$; no decay in this group'],
      ['Matrix decay / clipping','Ordinary decoupled WD 0.1; global gradient clip 0.5'],
      ['6B peak LRs',r'Matrix 0.0042596113; auxiliary AdamW 0.0002555767'],
      ['R5 resources','8 H200 / 8 CPUs; teacher shares the same devices'],
      ['KD objective',r'$0.75\,\mathrm{CE}+0.25\,\mathrm{KL}(p_T\Vert p_S)$; temperature 1'],
    ]
    table('setup.tex','Common protocol and the late selected recipe. Earlier baselines differ as described in the text; architecture positional capacity is not the training context length.',
          'tab:setup',r'p{.23\linewidth}p{.69\linewidth}','Item & Setting',common)
    # Machine-generated provenance registry: hyperlinks point to the real run identities.
    s=r'\begin{longtable}{rp{.64\linewidth}p{.22\linewidth}}'+'\n'+r'\caption{Formal identities, persistent run links and configuration hashes. Full hashes and artifact paths are retained in the evidence snapshot.}\label{tab:registry}\\'+'\n'+r'\toprule ID & Logical run ID & W\&B / config \\ \midrule \endfirsthead'+'\n'+r'\toprule ID & Logical run ID & W\&B / config \\ \midrule \endhead'+'\n'
    for i,r in R.items():
        link=r'\href{'+r['wandb_url']+r'}{run link}' if r['wandb_url'] else '--'
        digest=(r['config_digest'] or '')[:12]
        s+=str(i)+' & '+r'\path{'+r['run_id']+'} & '+link+r' / \texttt{'+digest+r'} \\'+'\n'
    s+=r'\bottomrule\end{longtable}'+'\n';(OUT/'registry.tex').write_text(s)
    totals=D['ledger_totals'];macros={
      'ExperimentHours':f"{totals['experiments']:,.3f}",'PreprocessingHours':f"{totals['preprocessing']:,.3f}",
      'CombinedHours':f"{totals['combined']:,.3f}",'KDImprovement':f"{100*(1-R[19]['ppl_10k']/R[17]['ppl_10k']):.2f}",
      'CtwoRegression':f"{100*(R[20]['ppl_10k']/R[19]['ppl_10k']-1):.2f}",
      'CtwoUpdateReduction':f"{100*(1-R[20]['updates']/R[19]['updates']):.2f}",
      'CtwoCostIncrease':f"{100*(R[20]['h200_hours']/R[19]['h200_hours']-1):.2f}",
    }
    (P/'numbers.tex').write_text('% Generated from data/evidence.json; do not edit by hand.\n'+'\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    print('Generated',len(list(OUT.glob('*.tex'))),'tables and numeric macros')


if __name__=='__main__':main()
