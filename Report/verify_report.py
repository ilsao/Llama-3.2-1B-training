#!/usr/bin/env python3
"""Check scientific scope, asset completeness, and the rendered report."""
import csv
import json
import math
from pathlib import Path
import re

P = Path(__file__).resolve().parent
D = json.loads((P / 'data/evidence.json').read_text())


def main():
    checks = []
    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)
    runs={r['formal']:r for r in D['runs'] if r['formal']}
    check('all 20 formal identities are preserved', set(runs)==set(range(1,21)))
    check('three final6B student endpoints match', all(runs[i]['processed_inputs']==6000082944 for i in [17,19,20]))
    check('R5 is better than nonKD on the same10k scope', runs[19]['ppl_10k']<runs[17]['ppl_10k'])
    check('C2 is worse than R5 despite fewer updates', runs[20]['ppl_10k']>runs[19]['ppl_10k'] and runs[20]['updates']<runs[19]['updates'])
    check('C2 does not reduce allocated cost', runs[20]['h200_hours']>runs[19]['h200_hours'])
    check('old Muon50k checkpoint is explicitly unmatched', runs[2]['full_eval_matches_export'] is False)
    check('the authorized earlyMuon continuation remains explicit',runs[2]['checkpoint_loaded_for_training'] is True)
    check('early AdamW export differs from final processed count', runs[1]['export_inputs']<runs[1]['processed_inputs'])
    check('final6B full50k scores are absent', all(runs[i]['ppl_50k'] is None for i in [17,19,20]))
    check('both finalKD exports passed reload checks',all(runs[i]['export_reload_verified'] for i in [19,20]))
    for i in [15,16,17,18,19,20]:
        r=runs[i]
        combined=math.exp((math.log(r['ppl_1k'])*1479302+math.log(r['ppl_9k'])*13666812)/15146114)
        check(f'formal{i} combines NLL with valid-token weights',abs(combined-r['ppl_10k'])<1e-8)
    check('every allocation is independently Slurm verified',D['ledger_totals']['sacct_verified_jobs']==D['ledger_totals']['job_count']==327)
    with (P/'data/allocations.csv').open() as f:
        jobs=list(csv.DictReader(f))
    check('allocation IDs are unique',len({r['job_id'] for r in jobs})==327)
    check('all-inclusive allocated hours agree with CSV',abs(sum(float(r['h200_hours']) for r in jobs)-D['ledger_totals']['combined'])<1e-7)
    receipt=D['c2_decay_receipt']
    check('decay is bound to exact input tokens',receipt['tokens_before_update']==2399928320 and receipt['trigger']=='cumulative_nonpadding_input_tokens')
    text='\n'.join(p.read_text() for p in [P/'main.tex',*sorted((P/'sections').glob('*.tex')),*sorted((P/'tables').glob('*.tex'))])
    check('no manuscript TODOs or citation placeholders',not re.search(r'\b(TODO|PLACEHOLDER|CITATION NEEDED|First Author|Second Author)\b',text))
    figs=json.loads((P/'figures/figure_manifest.json').read_text())
    check('all14 planned core and supplementary figures exist',len(figs)==14 and all((P/'figures'/o).is_file() for x in figs for o in x['outputs']))
    citations=set(re.findall(r'\\cite\w*\{([^}]+)\}',text))
    cited={k.strip() for group in citations for k in group.split(',')}
    verified=json.loads((P/'data/citation_verification.json').read_text())
    check('every manuscript citation has an arXiv/DataCite receipt',cited<=set(verified))
    check('report style is unchanged', (P/'cerulean_report.sty').read_bytes()==(P/'template-original/cerulean_report.sty').read_bytes())
    log=(P/'main.log').read_text(errors='replace')
    check('no unresolved citations or cross references',not re.search(r'undefined|There were undefined|Rerun to get cross-references right',log,re.I))
    check('no LaTeX overfull boxes',not re.search(r'Overfull \\[hv]box',log))
    import fitz
    doc=fitz.open(P/'main.pdf')
    pages=[p.get_text() for p in doc]
    check('compiled PDF contains complete report and appendices',len(doc)>=15 and all(x in '\n'.join(pages) for x in ['Conclusion','Reproducing the Report','12.3989','12.5380']))
    check('no blank PDF pages',all(len(t.strip())>80 for t in pages))
    report={'passed':True,'checks':checks,'pdf_pages':len(doc),'pdf_bytes':(P/'main.pdf').stat().st_size,
            'figure_count':len(figs),'verified_citations':len(cited)}
    (P/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
