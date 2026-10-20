"""Small metadata-only fitting and configuration contracts for token LR transfer."""
import copy
import json
import math
from pathlib import Path

STUDY='lr-transfer-20261015'
POOL='init-recipe-20261014'
PREFIX=POOL+'-transfer-'
BATCH=131072
MULTIPLIERS=(.35,.7,1.4,2.8)
ANCHOR_MUON=.005974384411214128
ANCHOR_ADAM=.00017923153233642382
RATIO=ANCHOR_ADAM/ANCHOR_MUON
MINUTES={'validation':15,'screen':15,'proxy':94,'decay':45}
FAILED_VALIDATION_HOURS=601*8/3600
MAX_STUDY_HOURS=FAILED_VALIDATION_HOURS+(15+4*15+4*94+3*45)*8/60


def align_tokens(tokens):
    return math.ceil(tokens/BATCH)*BATCH


def configuration(base,kind,multiplier=1.):
    if kind not in MINUTES or not math.isfinite(multiplier) or multiplier<=0:raise ValueError('Invalid transfer case')
    cfg=copy.deepcopy(base)
    cfg.pop('phase_schedule',None)
    if kind!='decay':cfg.pop('hold_schedule',None)
    tokens={'validation':32*BATCH,'screen':align_tokens(120_000_000),'proxy':align_tokens(1_000_000_000),'decay':align_tokens(400_000_000)}[kind]
    cfg.update(category='validation' if kind=='validation' else 'pilot',sweep_id=STUDY,sweep_stage=kind,
        sweep_purpose='One-dimensional LR transfer at fixed AdamW/Muon ratio; single WSD target',parent_runs=[],
        initialization='residual_half',seed=42,eval_documents=16 if kind=='validation' else 1000,ema=[.6] if kind in ('validation','screen','proxy') else [],
        learning_rate=ANCHOR_ADAM*multiplier,world_size=8,microbatch_size=1,gradient_accumulation_steps=2,
        max_steps=tokens//BATCH,scheduler_steps=tokens//BATCH,target_training_tokens=tokens,max_training_tokens=tokens,
        max_h200_hours=MINUTES[kind]*8/60,logging_steps=10 if kind=='proxy' else 1,
        eval_steps=384 if kind=='proxy' else (3 if kind=='validation' else max(1,tokens//BATCH//10)),
        warmup_steps=2 if kind=='validation' else 170,export_on_completion=False,retain_all_checkpoints=True,
        checkpoint_policy='final_only',fixed_horizon=True,min_lr_ratio=0,final_evaluation_reserve_seconds=0,
        stop_margin_seconds=120 if kind in ('screen','validation') else 180,
        lr_scheduler='wsd_hold_zero' if kind=='decay' else 'warmup_constant',
        lr_transfer={'study':STUDY,'kind':kind,'multiplier':multiplier,'ema_stride_steps':2 if kind=='validation' else 32})
    cfg['muon']['learning_rate']=ANCHOR_MUON*multiplier
    return cfg


def solve(matrix,vector):
    """Pivoted elimination for at most3x3 systems; no numerical runtime import."""
    a=[list(map(float,row))+[float(v)] for row,v in zip(matrix,vector)];n=len(a)
    for col in range(n):
        pivot=max(range(col,n),key=lambda row:abs(a[row][col]))
        if abs(a[pivot][col])<1e-12:raise ValueError('Rank-deficient regression')
        a[col],a[pivot]=a[pivot],a[col]
        scale=a[col][col];a[col]=[v/scale for v in a[col]]
        for row in range(n):
            if row!=col:
                scale=a[row][col];a[row]=[v-scale*w for v,w in zip(a[row],a[col])]
    return [row[-1] for row in a]


def r_squared(observed,predicted):
    mean=sum(observed)/len(observed);total=sum((y-mean)**2 for y in observed)
    error=sum((y-z)**2 for y,z in zip(observed,predicted))
    return 1-error/total if total>1e-16 else (1. if error<1e-16 else 0.)


def parabola(points,min_points=4):
    points=sorted(points)
    if len(points)<min_points or len({p[0] for p in points})!=len(points):raise ValueError('Insufficient distinct LR points')
    if any(lr<=0 or not math.isfinite(lr+loss) for lr,loss in points):raise ValueError('Invalid LR/loss')
    center=sum(math.log(p[0]) for p in points)/len(points)
    xs=[math.log(lr)-center for lr,_ in points];ys=[loss for _,loss in points]
    design=[[x*x,x,1.] for x in xs]
    normal=[[sum(row[i]*row[j] for row in design) for j in range(3)] for i in range(3)]
    target=[sum(row[i]*y for row,y in zip(design,ys)) for i in range(3)]
    a,b,c=solve(normal,target)
    if a<=0:raise ValueError('LR curve is not convex')
    vertex=-b/(2*a);optimum=math.exp(vertex+center)
    if not points[0][0]<optimum<points[-1][0]:raise ValueError('Optimal LR is not bracketed')
    if min(range(len(ys)),key=ys.__getitem__) in (0,len(ys)-1):raise ValueError('Measured winner lies on LR boundary')
    predicted=[a*x*x+b*x+c for x in xs]
    return {'muon_lr':optimum,'adamw_lr':optimum*RATIO,'r_squared':r_squared(ys,predicted),
        'curvature':a,'coefficients_centered':[a,b,c],'log_lr_center':center,'points':points}


def power_law(points):
    if len(points)<4:raise ValueError('Need at least four valid token horizons')
    xs=[math.log(t/1e9) for t,lr in points];ys=[math.log(lr) for t,lr in points]
    mx=sum(xs)/len(xs);my=sum(ys)/len(ys);variance=sum((x-mx)**2 for x in xs)
    if variance<=0:raise ValueError('No token-horizon variation')
    beta=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/variance;intercept=my-beta*mx
    predicted=[intercept+beta*x for x in xs]
    return {'beta':beta,'log_muon_lr_at_1b':intercept,'muon_lr_at_1b':math.exp(intercept),'r_squared':r_squared(ys,predicted),'points':points}


def predict(fit,tokens):
    return math.exp(fit['log_muon_lr_at_1b']+fit['beta']*math.log(tokens/1e9))


def analyze(cohorts):
    """Fit early horizons, test untouched later horizons, then refit all valid points."""
    if set(cohorts)!=set(MULTIPLIERS):raise ValueError('Require the complete prespecified four-LR cohort')
    steps=set.intersection(*(set(rows) for rows in cohorts.values()))
    curves=[];rejected=[]
    for step in sorted(steps):
        rows=[cohorts[m][step] for m in MULTIPLIERS];tokens={r['tokens'] for r in rows};identities={r['evaluation_identity'] for r in rows}
        if len(tokens)!=1 or len(identities)!=1:raise ValueError('LR cases have mismatched token/evaluation endpoint')
        tokens=tokens.pop()
        if tokens<200_000_000:continue
        try:
            q=parabola([(ANCHOR_MUON*m,cohorts[m][step]['nll']) for m in MULTIPLIERS])
            if q['r_squared']<.9:raise ValueError('Quadratic R2 below0.9')
            curves.append({'step':step,'tokens':tokens,**q})
        except ValueError as e:rejected.append({'step':step,'tokens':tokens,'reason':str(e)})
    early=[(q['tokens'],q['muon_lr']) for q in curves if q['tokens']<=650_000_000]
    held=[q for q in curves if q['tokens']>=750_000_000]
    fit=power_law(early)
    if len(held)<4:raise ValueError('Need at least four held-out late horizons')
    validation=[{'tokens':q['tokens'],'predicted_muon_lr':predict(fit,q['tokens']),'observed_fit_muon_lr':q['muon_lr'],'relative_error':abs(predict(fit,q['tokens'])/q['muon_lr']-1)} for q in held]
    errors=sorted(v['relative_error'] for v in validation);median=errors[len(errors)//2]
    passed=fit['r_squared']>=.8 and median<=.15 and max(errors)<=.25
    final=power_law([(q['tokens'],q['muon_lr']) for q in curves])
    # Bounds measure fit-window sensitivity, not statistical95% confidence.
    alternatives=[final,fit]
    middle=[(q['tokens'],q['muon_lr']) for q in curves if q['tokens']>=350_000_000]
    if len(middle)>=4:alternatives.append(power_law(middle))
    rows=[]
    for billion in range(1,7):
        tokens=billion*1_000_000_000;muon=predict(final,tokens);bounds=[predict(f,tokens) for f in alternatives]
        rows.append({'tokens':tokens,'muon_peak_lr':muon,'adamw_peak_lr':muon*RATIO,
            'muon_fit_window_range':[min(bounds),max(bounds)],'adamw_fit_window_range':[min(bounds)*RATIO,max(bounds)*RATIO],
            'extrapolation_factor':tokens/max(q['tokens'] for q in curves),
            'status':'proxy_estimate_pending_wsd_check' if passed else 'unvalidated_extrapolation'})
    return {'proxy_validation_passed':passed,'curves':curves,'rejected_curves':rejected,'early_fit':fit,
        'held_out_validation':validation,'median_held_out_relative_error':median,'final_fit':final,
        'lr_ratio_adamw_to_muon':RATIO,'targets':rows,'scope':'fixed-model fixed-ratio joint LR path, single WSD; no width-transfer claim',
        'uncertainty':'fit-window sensitivity only; checkpoints are correlated; one seed'}


def projected_allocation(config,step,recent_seconds,seconds_left):
    """Reject sustained slow nodes early using complete update timing, not peakT/s."""
    if not recent_seconds or any(not math.isfinite(x) or x<=0 for x in recent_seconds):raise ValueError('Invalid timing sample')
    values=sorted(recent_seconds);median=values[len(values)//2]
    updates=config['max_steps']-step
    evaluations=math.ceil(updates/config['eval_steps'])
    required=updates*median+evaluations*6+config['stop_margin_seconds']
    return {'median_update_seconds':median,'remaining_updates':updates,'required_seconds':required,
        'seconds_left':seconds_left,'fits':required<=seconds_left*.98}
