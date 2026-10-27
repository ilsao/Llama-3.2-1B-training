#!/usr/bin/env python3
"""Reproduce report figures from the frozen metadata snapshot (CPU only).

Architecture drawings are editable vector primitives, used because no Gemini
credential is configured. No generated or interpolated values are substituted
for missing measurements. Run from any directory.
"""
import csv
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
D = json.loads((DATA / "evidence.json").read_text())
RUNS = {r["formal"]: r for r in D["runs"] if r["formal"]}
COL = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
       "pink": "#CC79A7", "sky": "#56B4E9", "gray": "#777777", "ink": "#263238"}
PALE = {"blue": "#E8F0F6", "green": "#E7F2EC", "orange": "#FCF3E2", "red": "#F9EBE5", "gray": "#F1F3F5"}
METHOD_COLOR={1:'gray',4:'orange',12:'sky',17:'blue',18:'pink',19:'green',20:'red'}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .16,
                     "lines.linewidth": 1.7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
MANIFEST = []


def save(fig, name, kind, sources):
    # Fit diagram labels inside their boxes before exporting at report size.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for t in ax.texts:
            if not hasattr(t, '_report_box'):
                continue
            x,y,w,h=t._report_box
            bounds=t.get_window_extent(renderer)
            lower=ax.transData.transform((x,y)); upper=ax.transData.transform((x+w,y+h))
            factor=min(1.,.92*(upper[0]-lower[0])/max(bounds.width,1),.90*(upper[1]-lower[1])/max(bounds.height,1))
            if factor<1: t.set_fontsize(t.get_fontsize()*factor)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(HERE / ("fig_" + name + "." + ext), facecolor="white")
    MANIFEST.append({"name": name, "kind": kind, "source": sources,
                     "outputs": ["fig_" + name + "." + e for e in ["pdf", "svg", "png"]]})
    plt.close(fig)


def canvas(w=7.2, h=3.3):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    return fig, ax


def box(ax, xy, w, h, label, color="blue", size=9, weight="normal"):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                       facecolor=PALE[color], edgecolor="none", zorder=2)
    ax.add_patch(p)
    label='\n'.join(textwrap.fill(line,width=max(10,int(w*.92)),break_long_words=False,break_on_hyphens=False) for line in label.split('\n'))
    t=ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center",
              fontsize=size, color=COL["ink"], fontweight=weight, zorder=3)
    t._report_box=(*xy,w,h)


def arrow(ax, a, b, label=None, color="gray", dashed=False, rad=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                linewidth=1.1, color=COL[color], linestyle="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+2, label, ha="center", fontsize=7.5, color=COL["gray"])


def panel(ax, text):
    ax.text(0, 1.03, text, transform=ax.transAxes, fontsize=9, weight="bold", va="bottom")


def csvrows(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def fig_history():
    fig, ax = canvas(h=3.4)
    stages = ["AdamW\nbaseline", "Muon +\nWSD", "LR / init /\ndata studies", "Batch / mask /\nNorMuon", "3--6B-token\ntransfers", "KD +\nadaptive batch"]
    xs = [1, 18, 35, 52, 69, 86]
    for x, text, day in zip(xs, stages, ["Oct 8", "Oct 9", "Oct 10--15", "Oct 16--17", "Oct 15--19", "Oct 19--20"]):
        box(ax, (x, 62), 13, 21, text, "blue", 8.5)
        ax.text(x+6.5, 91, day, ha="center", fontsize=8, color=COL["gray"])
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a+13.5, 72), (b-.5, 72))
    notes = [(18, "Filtered recipe\nnot promoted", "orange"), (35, "Proxy LR transfer\nrecalibrated", "orange"),
             (52, "Averaging / PE\nnot selected", "orange"), (69, "Shorter tail\ndid not improve", "orange"),
             (86, "R5 retained over\nC2 + early decay", "green")]
    for x, text, color in notes:
        box(ax, (x, 22), 13, 23, text, color, 8)
        arrow(ax, (x+6.5, 61), (x+6.5, 46), dashed=True)
    ax.text(50, 3, "Arrows show research decisions and recipe influence.\nCheckpoint continuations are identified separately.", ha='center',fontsize=8)
    save(fig, "history", "vector schematic", "outline.tex; data/formal_runs.csv; experimental protocols")


def fig_data():
    fig, ax = canvas(h=3.4)
    box(ax, (1, 76), 24, 18, "104,020,922 documents\nPinned canonical train", size=9)
    box(ax, (36, 76), 27, 18, "Global shuffle (seed 42)\nSplit before tokenization", size=9)
    arrow(ax, (26, 85), (35, 85))
    box(ax, (1, 43), 36, 19, "Training: 103,970,922 documents\nRaw or training-only filtering", "green", 9)
    box(ax, (62, 43), 36, 19, "Reserved evaluation: 50,000\nOriginal documents retained", "orange", 9)
    arrow(ax, (44, 75), (26, 63)); arrow(ax, (56, 75), (78, 63))
    box(ax, (1, 9), 36, 21, "BOS / EOS + tokenizer\n8192-token packing + document spans", "green", 8.5)
    arrow(ax, (19, 42), (19, 31))
    box(ax, (51, 9), 22, 21, "Fixed 1k\nDevelopment", "orange", 9)
    box(ax, (76, 9), 22, 21, "Additional 9k\nConfirmation", "orange", 9)
    arrow(ax, (75, 42), (62, 31)); arrow(ax, (84, 42), (87, 31))
    ax.text(75, -2, "1k + 9k form 10k; full 50k also\nincludes the remaining 40k.", ha="center", fontsize=7.5)
    save(fig, "data_protocol", "vector schematic", "data/evidence.json: dataset_config; canonical split metadata")


def fig_routing():
    fig, ax = canvas(h=3.6)
    labels = [(1, "Input IDs", 13), (22, "Tied token\nembedding", 17), (48, "Llama blocks\nAttention + MLP", 22), (79, "Tied LM\nhead", 16)]
    for x, label, w in labels:
        box(ax, (x, 66), w, 19, label, "blue", 9)
    for a, b in [((14,75),(21,75)),((39,75),(47,75)),((70,75),(78,75))]: arrow(ax,a,b)
    box(ax,(3,16),39,24,"AdamW\nEmbedding / shared head + norms\nTied tensor counted once","orange",9)
    box(ax,(55,16),40,24,"Muon / NorMuon\nHidden projection matrices\nNS5 + ordinary weight decay","green",9)
    arrow(ax,(30,65),(23,41),color="orange",dashed=True)
    arrow(ax,(87,65),(34,41),color="orange",dashed=True,rad=-.16)
    arrow(ax,(55,65),(70,41),color="green",dashed=True)
    ax.text(53,95,"Shared embedding / head storage",ha="center",fontsize=8.5)
    ax.add_patch(FancyArrowPatch((31,89),(86,89),arrowstyle='<->',mutation_scale=10,linewidth=1.1,color=COL['blue']))
    ax.text(50,3,"Solid: forward path. Dashed: parameter routing (norms inside blocks use AdamW).",ha="center",fontsize=8)
    save(fig,"optimizer_routing","vector schematic","resolved configurations and optimizer-groups.json receipts")


def fig_masks():
    ids=np.repeat(np.arange(3),[5,7,5]); n=len(ids)
    causal=np.tril(np.ones((n,n))); intra=causal*(ids[:,None]==ids[None,:])
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),layout="constrained")
    from matplotlib.colors import ListedColormap
    for ax,m,title in zip(axs,[causal,intra],["(a) Dense causal mask","(b) Intra-document causal mask"]):
        ax.imshow(m,cmap=ListedColormap(["#F2F3F5",COL["blue"]]),vmin=0,vmax=1,origin="upper")
        ax.set(xticks=[2,8,14],xticklabels=["Doc A","Doc B","Doc C"],yticks=[2,8,14],yticklabels=["A","B","C"],xlabel="Key positions",ylabel="Query positions")
        ax.grid(False);panel(ax,title)
        for boundary in [4.5,11.5]: ax.axhline(boundary,color="white",lw=1.5);ax.axvline(boundary,color="white",lw=1.5)
    save(fig,"attention_masks","exact binary schematic","document-span mask definition; no empirical attention values")


def fig_kd():
    fig,ax=canvas(h=4)
    ax.text(24,96,"(a) 8 + 8 hidden-state transport",ha="center",weight="bold",fontsize=10)
    ax.text(76,96,"(b) R5: teacher + student on 8 GPUs",ha="center",weight="bold",fontsize=10)
    box(ax,(3,71),42,16,"Frozen 3B teacher: 8 GPUs","blue",9)
    box(ax,(3,43),42,17,"BF16 hidden states\nBounded READY / ACK spool","gray",8.5)
    box(ax,(3,15),42,18,"1B student: 8 GPUs\n+ frozen teacher output head","green",8.5)
    arrow(ax,(24,70),(24,61));arrow(ax,(24,42),(24,34))
    box(ax,(56,74),39,12,"Identical IDs + intradoc context","gray",8.5)
    box(ax,(53,43),20,18,"Frozen 3B\nNo gradient","blue",8.5)
    box(ax,(78,43),20,18,"Random 1B\nTrainable","green",8.5)
    arrow(ax,(68,73),(63,62));arrow(ax,(83,73),(88,62))
    box(ax,(55,14),42,18,"Exact full-vocabulary CE + KL\n0.75 CE + 0.25 KL; T = 1","orange",8.5)
    arrow(ax,(63,42),(69,33));arrow(ax,(88,42),(83,33))
    arrow(ax,(97,25),(97.5,52),color="green",dashed=True,rad=.25)
    ax.text(50,1,"Student and teacher inputs are journaled separately; the teacher is never optimized.",ha="center",fontsize=8)
    save(fig,"kd_architecture","vector schematic","kd3b-hidden8 and kd6b-r5 protocols; data/evidence.json configs")


def fig_curves():
    curves=csvrows("evaluation_curves.csv")
    fig=plt.figure(figsize=(7.2,4.0),layout="constrained")
    gs=fig.add_gridspec(2,2,width_ratios=[1.6,1]);ax=fig.add_subplot(gs[:,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,1])
    specs=[(1,"AdamW baseline","gray"),(4,"2B BE60P20","orange"),(12,"3B batch / token40","sky"),(17,"6B non-KD","blue"),(19,"6B R5 KD","green"),(20,"6B C2 + early decay","red")]
    for num,label,color in specs:
        rr=RUNS[num];z=[q for q in curves if q["run_id"]==rr["run_id"]]
        ax.plot([float(q['tokens'])/1e9 for q in z],[float(q['perplexity']) for q in z],label=label,color=COL[color],marker="o",markersize=2.3)
    ax.set(xlabel="Cumulative student input tokens (B)",ylabel="Fixed-1k perplexity",yscale="log",ylim=(11.5,110))
    ax.set_yticks([12,15,20,30,50,100]);ax.set_yticklabels([12,15,20,30,50,100]);ax.minorticks_off();ax.legend(loc="upper right",fontsize=7)
    panel(ax,"(a) Measured development curves")
    for aa,field,title in [(b,'ppl_10k','(b) Combined-10k endpoints'),(c,'ppl_50k','(c) Full-50k endpoints')]:
        for n,r in RUNS.items():
            if r[field] is None or (field=='ppl_50k' and not r['full_eval_matches_export']):continue
            color=COL[METHOD_COLOR.get(n,'blue')]
            aa.scatter(r['export_inputs']/1e9,r[field],color=color,s=24,zorder=3)
            if n in [7,8,12,14,17,19,20]:
                offset={7:(-14,8),8:(-14,-8),12:(4,-9)}.get(n,(4,3)) if field=='ppl_50k' else (4,3)
                aa.annotate(str(n),(r['export_inputs']/1e9,r[field]),xytext=offset,textcoords='offset points',fontsize=7)
        aa.set(xlabel="Export input tokens (B)",ylabel="Perplexity");panel(aa,title)
    save(fig,"training_progress","data chart","data/evaluation_curves.csv; data/formal_runs.csv; numerals denote formal IDs")


def fig_sweep():
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8),layout="constrained")
    schedules=['cosine','linear','wsd']
    for ax,opt,lrs in zip(axs,['adamw','muon'],[[.0003,.0006,.0012],[.02,.04,.08]]):
        grid=np.full((3,3),np.nan)
        for r in D['sweep']:
            if r['optimizer']!=opt or r['status']!='complete':continue
            lr=float(r['recipe'].split('lr')[1]);sched=r['id'].split('-')[-1]
            i=int(np.argmin(np.abs(np.array(lrs)-lr)));j=schedules.index(sched);grid[i,j]=r['nll_100']
        im=ax.imshow(grid,cmap='cividis',vmin=4,vmax=6.3,aspect='auto')
        for i in range(3):
            for j in range(3):
                val=grid[i,j];ax.text(j,i,f'{math.exp(val):.1f}' if np.isfinite(val) else 'N/A',ha='center',va='center',fontsize=9,color='white' if val<5.2 else '#111111')
        ax.set(xticks=range(3),xticklabels=['Cosine','Linear','WSD'],yticks=range(3),yticklabels=[f'{v:g}' for v in lrs],ylabel='Optimizer peak learning rate');ax.grid(False)
        panel(ax,'(a) AdamW' if opt=='adamw' else '(b) Muon + auxiliary AdamW')
    fig.colorbar(im,ax=axs,label='Mean NLL (cell labels: PPL)',shrink=.86)
    save(fig,"initial_sweep","data chart","data/initial_sweep.csv; all endpoints 104,857,600 inputs, fixed1k")


def fig_calibration():
    cal=D['calibration'];fig,axs=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for (h,obs),color in zip(sorted(cal['observations'].items(),key=lambda kv:int(kv[0])),['blue','orange','green']):
        obs=sorted(obs,key=lambda x:x['muon_lr']);v=min(x['mean_nll'] for x in obs)
        axs[0].plot([x['muon_lr'] for x in obs],[x['mean_nll']-v for x in obs],'-o',color=COL[color],label=f'{int(h)/1e6:.0f}M inputs',markersize=4)
    axs[0].set(xscale='log',xlabel='Matrix peak learning rate',ylabel='NLL above best measured at that horizon');axs[0].legend()
    law=cal['direct_wsd_law'];points=np.array(law['points']);a=law['muon_lr_at_1b'];beta=law['beta']
    xx=np.geomspace(.2,6,150);yy=a*xx**beta
    axs[1].plot(xx[xx<=.6],yy[xx<=.6],color=COL['blue']);axs[1].plot(xx[xx>=.6],yy[xx>=.6],'--',color=COL['blue'],label='Extrapolation')
    axs[1].scatter(points[:,0]/1e9,points[:,1],color=COL['blue'],label='Local fitted optima',zorder=3)
    axs[1].set(xscale='log',yscale='log',xlabel='Training horizon (B input tokens)',ylabel='Matrix peak learning rate');axs[1].legend(fontsize=7)
    axs[1].text(.05,.08,rf'$\eta_{{\rm matrix}}={a:.5f}(D/10^9)^{{{beta:.3f}}}$',transform=axs[1].transAxes,fontsize=8)
    panel(axs[0],'(a) Complete-WSD calibration');panel(axs[1],'(b) Calibrated law and extrapolation')
    save(fig,"lr_calibration","data chart","data/evidence.json: calibration; fitted curves are explicitly distinguished")


def fig_schedule():
    metrics=csvrows('training_metrics.csv');fig,axs=plt.subplots(3,1,figsize=(7.2,4.7),sharex=True,layout='constrained')
    for num,color,label in [(19,'green','R5 fixed 16 -> 64'),(20,'red','C2 + early decay')]:
        rows=[q for q in metrics if q['run_id']==RUNS[num]['run_id']];cfg=D['configs'][RUNS[num]['run_id']]
        xx=np.array([float(q['train/total_tokens_seen'])/1e9 for q in rows])
        lr=np.array([float(q['train/lr_muon']) if q['train/lr_muon'] else np.nan for q in rows])
        axs[0].plot(xx,lr/cfg['muon']['learning_rate'],color=COL[color],label=label)
        bs=[float(q['train/global_batch_sequences']) if q['train/global_batch_sequences'] else (16 if x<3.000107008 else 64) for x,q in zip(xx,rows)]
        axs[1].step(xx,bs,where='post',color=COL[color],label=label)
        axs[2].plot(xx,[float(q['train/global_step']) for q in rows],color=COL[color])
    for ax in axs:
        for x in [2.399928320,3.000107008,5.000003584]:ax.axvline(x,color='#999999',ls=':',lw=.8)
    axs[0].set(ylabel='Matrix LR / first peak');axs[0].legend(ncol=2,loc='upper right')
    axs[1].set(ylabel='Actual global batch',yticks=[16,32,64,128]);axs[2].set(xlabel='Cumulative student input tokens (B)',ylabel='Optimizer updates')
    save(fig,'token_schedule','data chart','data/training_metrics.csv; C2 exact token trigger receipt in evidence.json')


def fig_ablations():
    a={r['run_id']:r for r in D['ablations']};n=lambda k:a[k]['mean_nll']
    root='ppl-next-20261016-'
    rows=[('Intradoc vs dense (1B, 1k)',n(root+'s1-intradoc')-n(root+'s1-dense'),'blue'),
          ('NorMuon vs selected NS5 (1B, 1k)',n(root+'s2-nor')-n(root+'s1-intradoc'),'blue'),
          ('PE vs selected NS5 (1B, 1k)',n(root+'s2-pe')-n(root+'s1-intradoc'),'blue'),
          ('Locked candidate, seed43 (1B, 1k)',n('replication-candidate')-n('replication-control'),'sky')]
    for aa,bb,label in [(11,10,'20% vs 60% tail (3B, 50k)'),(12,10,'Batch + token40 vs reference (3B, 50k)'),(13,12,'Update40 vs token40 (3B, 50k)')]:
        rows.append((label,math.log(RUNS[aa]['ppl_50k']/RUNS[bb]['ppl_50k']),'orange'))
    for aa,bb,label,field in [(15,14,'Auxiliary AdamW LR x2 (4B, 1k)','ppl_1k'),(16,15,'Update-tail30 vs token40 (4B, 10k)','ppl_10k'),(19,17,'R5 KD vs non-KD (6B, 10k)','ppl_10k'),(20,19,'C2 + early decay vs R5 (6B, 10k)','ppl_10k')]:
        rows.append((label,math.log(RUNS[aa][field]/RUNS[bb][field]),'green' if aa==19 else 'red' if aa==20 else 'orange'))
    fig,ax=plt.subplots(figsize=(7.2,4),layout='constrained');yy=np.arange(len(rows))
    ax.axvline(0,color='#555555',lw=1)
    for y,(label,delta,color) in zip(yy,rows):
        ax.plot([0,delta],[y,y],color=COL[color],lw=1.2);ax.scatter(delta,y,color=COL[color],s=30,zorder=3)
    ax.set(yticks=yy,yticklabels=[r[0] for r in rows],xlabel='Change in mean held-out NLL (negative is better)');ax.invert_yaxis();ax.grid(axis='y',visible=False)
    save(fig,'ablations','data chart','data/evidence.json: ablations; formal endpoint comparisons; distinct controls and evaluation scopes')


def fig_pareto():
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),layout='constrained')
    for ax,field,title in zip(axs,['ppl_1k','ppl_10k','ppl_50k'],['(a) Fixed 1k','(b) Combined 10k','(c) Full 50k']):
        for num,r in RUNS.items():
            if r[field] is None or (field=='ppl_50k' and not r['full_eval_matches_export']):continue
            color=METHOD_COLOR.get(num,'blue')
            ax.scatter(r['h200_hours'],r[field],color=COL[color],marker='s' if num>=18 else 'o',s=24,zorder=3)
            if num in [1,4,8,12,14,17,18,19,20]:ax.annotate(str(num),(r['h200_hours'],r[field]),xytext=(3,3),textcoords='offset points',fontsize=6.5)
        ax.set(xlabel='Logical-run H200-hours',ylabel='Perplexity');panel(ax,title)
    save(fig,'quality_cost','data chart','data/formal_runs.csv; all teacher allocations included for formal18; no cross-subset ranking')


def fig_performance():
    fig,axs=plt.subplots(1,4,figsize=(7.2,2.5),layout='constrained');p=D['performance']
    metrics=[('end_to_end_tokens_per_second','Training interval\n(k input tokens/s)',1000),('median_update_seconds','Median update\n(seconds)',1),('peak_allocated_bytes','Peak allocated\n(GiB)',2**30),('peak_reserved_bytes','Peak reserved\n(GiB)',2**30)]
    for ax,(key,label,denom) in zip(axs,metrics):
        values=[p[k][key]/denom for k in ['A','B']];bars=ax.bar(['A','B'],values,color=[COL['gray'],COL['blue']],width=.6)
        ax.set(ylabel=label,ylim=(0,max(values)*1.22));ax.grid(axis='x',visible=False)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+max(values)*.04,f'{v:.2f}' if denom==1 else f'{v:.1f}',ha='center',fontsize=8)
    save(fig,'performance','data chart','performance-ab-20261010; 100 measured updates; compile startup excluded from interval throughput')


def fig_cbs():
    probes=json.loads((DATA/'critical_batch_probes.json').read_text());fig,axs=plt.subplots(3,1,figsize=(7.2,4.6),sharex=True,layout='constrained')
    for rid,color,label in [(D['runs'][-3]['run_id'],'blue','C0 shadow'),(D['runs'][-2]['run_id'],'orange','C1 half'),(D['runs'][-1]['run_id'],'green','C2 one')]:
        zz=[p for p in probes if p['run_id']==rid];xx=[p['window_end_tokens']/1e9 for p in zz]
        axs[0].plot(xx,[p['controller'].get('noise_scale_sequence_equivalent',np.nan) for p in zz],color=COL[color],label=label)
        axs[1].step(xx,[16*p['state']['units'] for p in zz],where='post',color=COL[color])
        ev=[z for z in csvrows('evaluation_curves.csv') if z['run_id']==rid]
        axs[2].plot([float(z['tokens'])/1e9 for z in ev],[math.log(float(z['perplexity'])) for z in ev],color=COL[color])
    axs[0].set(ylabel='Smoothed noise proxy\n(sequence equivalents)',yscale='log');axs[0].legend(ncol=3)
    axs[1].set(ylabel='Controller batch',yticks=[16,32,64,128]);axs[2].set(ylabel='Fixed-1k mean NLL',xlabel='Cumulative student input tokens (B)',ylim=(2.9,3.7))
    save(fig,'critical_batch','data chart','data/critical_batch_probes.json; common 1B prefix of the original 6B schedule; no compressed cooldown')


def fig_costs():
    rows=csvrows('allocations.csv');full={r['run_id'] for r in RUNS.values()};groups=['Preprocessing','Formal logical runs','Other experiment allocations']
    vals={g:[0.,0.] for g in groups}
    for r in rows:
        group='Preprocessing' if r['ledger']=='preprocessing' else 'Formal logical runs' if r['run_id'] in full else 'Other experiment allocations'
        vals[group][0 if r['state']=='COMPLETED' else 1]+=float(r['h200_hours'])
    fig,ax=plt.subplots(figsize=(7.2,2.6),layout='constrained');yy=np.arange(3)
    good=np.array([vals[g][0] for g in groups]);other=np.array([vals[g][1] for g in groups])
    ax.barh(yy,good,color=COL['blue'],label='Slurm COMPLETED');ax.barh(yy,other,left=good,color=COL['orange'],label='Failed / cancelled / timeout states')
    for y,v in zip(yy,good+other):ax.text(v+5,y,f'{v:.2f}',va='center',fontsize=8)
    ax.set(yticks=yy,yticklabels=groups,xlabel='Allocated H200-hours, including startup, evaluation, saving and idle time',xlim=(0,max(good+other)*1.12));ax.invert_yaxis();ax.legend(loc='lower right');ax.grid(axis='y',visible=False)
    save(fig,'allocation_costs','data chart','data/allocations.csv; 327 disjoint job IDs individually verified against sacct')


def main():
    for fn in [fig_history,fig_data,fig_routing,fig_masks,fig_kd,fig_curves,fig_sweep,fig_calibration,fig_schedule,fig_ablations,fig_pareto,fig_performance,fig_cbs,fig_costs]:
        fn();print('generated',MANIFEST[-1]['name'],flush=True)
    (HERE/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2)+'\n')


if __name__=='__main__':main()
