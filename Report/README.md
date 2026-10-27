# Llama training technical report

The English report follows the full experimental history and the decisions made from it. Start with [main.pdf](main.pdf); edit [main.tex](main.tex), [sections/main_text.tex](sections/main_text.tex), and [sections/appendix.tex](sections/appendix.tex).

The manuscript includes 20 formal run identities, 14 figures, 6 tables, and 11 references whose BibTeX was fetched from arXiv and whose DOI/title metadata was checked against DataCite. The supplied Cerulean style is unchanged. The original formatting example and assets are preserved in `template-original/`.

## Main verified outcomes

All three final recipes below use 6,000,082,944 student nonpadding inputs, seed42, and the same combined-10k evaluation provenance.

| Recipe | Updates | Combined-10k PPL | Logical H200-hours |
|---|---:|---:|---:|
| Non-KD | 28,619 | 12.9078 | 64.707 |
| R5 online KD | 28,619 | 12.3989 | 64.711 |
| C2 adaptive batch + early decay | 22,060 | 12.5380 | 65.009 |

R5 is the best verified 6B recipe on this evaluation scope. C2 uses fewer updates but has worse quality and slightly greater allocation cost. It changes both batch policy and the first decay boundary, so the result does not isolate their effects. Full-50k results for these final exports are unmeasured in the snapshot.

The full local project accounting is 1,538.533889 experiment H200-hours plus 27.837500 separately recorded preprocessing H200-hours, totaling 1,566.371389 across 327 distinct Slurm allocations. This includes failures and both roles of the 8+8 KD run. The external teacher's original pretraining cost is unknown. These totals describe an expanded study, not compliance with the original three-run/64-hour assignment scope.

## Rebuild from the frozen snapshot

Use Python 3.11 or newer and a TeX distribution with `latexmk` or Tectonic. The build reads only small report data; it does not recollect evidence, contact W&B, run models, or submit jobs.

```bash
python -m pip install -r requirements-report.txt
bash build.sh
```

On nano4, `build.sh` reuses the isolated reporting wheels and portable Tectonic under `/work/asciibase64/Llama-3.2-1B-training/report-tools-20261021/`. It does not install into any training environment. Override `REPORT_TOOLS_ROOT` and `REPORT_PYTHON` if needed. Limit scientific-library threads to one, as the script does.

To regenerate only selected components:

```bash
python generate_tables.py
python figures/gen_fig_report.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
python verify_report.py
```

The 5 schematics use editable vector primitives because no Gemini credential was configured. Data figures are deterministic Matplotlib charts. Each figure has PDF, SVG, and 300-DPI PNG outputs. `figures/figure_manifest.json` records its evidence source. Observed values, fitted summaries, and extrapolations are distinguished explicitly.

## Evidence and validation

- `data/evidence.json` is the frozen result/configuration snapshot.
- `data/formal_runs.csv` and `tables/registry.tex` preserve all 20 formal identities and persistent run links.
- `data/allocations.csv` contains the 327 jobs and their independent Slurm checks, with no overlap between the two ledgers.
- `data/source_manifest.json` stores hashes of the metadata sources, and `data/citation_verification.json` stores the bibliography verification receipts.
- `validation.json` records the final numerical-scope, asset, citation, PDF, and layout checks.

Recollecting cluster metadata is optional and separate from rendering:

```bash
python collect_evidence.py --work-root /work/asciibase64/Llama-3.2-1B-training
python fetch_references.py
```

These commands update the snapshot or bibliography and should be followed by rebuilding and reviewing the report. The collector never modifies the source experiment ledgers. Large model/data artifacts remain under `/work`; no new training, evaluation allocation, checkpoint continuation, cleanup, or publication was performed to write this report.
