#!/usr/bin/env python3
"""Read small experiment metadata; never load a model, dataset, or checkpoint.

The frozen JSON/CSVs are sufficient for plotting without access to /work.
Slurm queries are read-only. Authoritative experiment ledgers are never changed.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCES = {}


def read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    raw = path.read_bytes()
    SOURCES[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)


def dump(name, value):
    (HERE / "data" / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / "data" / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def compact_eval(e):
    if not isinstance(e, dict) or "eval/perplexity" not in e:
        return None
    return {
        "documents": e.get("eval/documents"),
        "nll": e.get("eval/nll"),
        "prediction_tokens": e.get("eval/prediction_tokens"),
        "perplexity": e["eval/perplexity"],
        "step": e.get("train/global_step", e.get("global_step")),
        "tokens": e.get("eval/training_tokens", e.get("eval/total_training_tokens_seen")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, default=Path("/work/asciibase64/Llama-3.2-1B-training"))
    ap.add_argument("--no-sacct", action="store_true")
    args = ap.parse_args()
    base = args.work_root
    (HERE / "data").mkdir(exist_ok=True)
    training = read(base / "experiment-ledger.json")
    prep = read(base / "preprocessing-ledger.json")
    assert not (set(training["jobs"]) & set(prep["jobs"])), "Deduplicate overlapping ledgers first"
    jobs = {k: dict(v, ledger="experiments") for k, v in training["jobs"].items()}
    jobs.update({k: dict(v, ledger="preprocessing") for k, v in prep["jobs"].items()})
    sacct = {}
    if not args.no_sacct:
        result = subprocess.run([
            "sacct", "-X", "-j", ",".join(jobs), "--format=JobID,State,ElapsedRaw,AllocTRES,Start,End",
            "-n", "-P"], text=True, capture_output=True, timeout=40)
        if result.returncode:
            raise RuntimeError("Read-only Slurm accounting query failed: " + result.stderr)
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 6 or parts[0] not in jobs:
                continue
            jid, state, elapsed, tres, start, end = parts[:6]
            if not elapsed.isdigit():
                continue
            gpu = jobs[jid].get("gpu_count")
            if gpu is None:
                continue
            sacct[jid] = {"state": state, "elapsed_seconds": int(elapsed), "alloc_tres": tres,
                          "start": start, "end": end, "h200_hours": gpu * int(elapsed) / 3600}
    ledger_rows = []
    for jid, j in jobs.items():
        live = sacct.get(jid, {})
        old = j.get("h200_hours")
        row = {"job_id": jid, "run_id": j.get("run_id", j.get("preparation-id", "")),
               "ledger": j["ledger"], "category": j.get("category", ""),
               "state": live.get("state", j.get("state", "UNKNOWN")), "gpu_count": j.get("gpu_count"),
               "elapsed_seconds": live.get("elapsed_seconds", j.get("elapsed_seconds")),
               "h200_hours": live.get("h200_hours", old), "start": live.get("start", j.get("start")),
               "end": live.get("end", j.get("end")), "sacct_verified": jid in sacct,
               "archived_h200_hours": old}
        ledger_rows.append(row)
    totals = {kind: sum(r["h200_hours"] or 0 for r in ledger_rows if r["ledger"] == kind)
              for kind in ["experiments", "preprocessing"]}
    totals.update(combined=sum(totals.values()), job_count=len(jobs),
                  sacct_verified_jobs=len(sacct), archived_experiment_total=training["total_h200_hours_measured"],
                  archived_experiment_reserved_or_measured=training["total_h200_hours_reserved_or_measured"])
    costs = collections.defaultdict(float)
    for r in ledger_rows:
        costs[r["run_id"]] += r["h200_hours"] or 0

    formal_order = [
        "baseline-fa3-20261009-baseline", "baseline-muon-wsd-raw-20261009", "muon-wsd-filtered-2b-20261009",
        "be60p20-lrscaled-2b-20261010", "horizon-wsd-20261014-final-standard", "horizon-wsd-20261014-final-resplit",
        "init-recipe-20261014-final-2b", "init-recipe-20261014-followup-rh-resplit-2b", "raw-filtered-3b-20261015-raw",
        "raw3b-sota-20261015", "raw3b-sota-wsd20-20261015", "raw3b-bs64-wsd40-20261016",
        "raw3b-bs64-stepwsd40-20261016", "raw4b-intradoc-normuon-20261017", "raw4b-normuon-adam2x-20261017",
        "raw4b-normuon-adam2x-step30-20261018", "raw6b-normuon-adam2x-20261018",
        "kd3b-hidden8-20261019-seed42", "kd6b-r5-20261020-seed42", "kd6b-c2-earlydecay-20261020-seed42"]
    selected = formal_order + [
        "critical-batch-20261020-c0-shadow-startup-replacement2",
        "critical-batch-20261020-c1-cbs-half-startup-replacement2", "critical-batch-20261020-c2-cbs-one"]
    runs, curve_rows, train_rows, configs = [], [], [], {}
    metric_keys = ["train/global_step", "train/total_tokens_seen", "train/lr_muon", "train/learning_rate",
                   "train/global_batch_sequences", "train/schedule_tokens_before_update", "train/grad_norm",
                   "train/tokens_per_second", "train/step_seconds", "resource/h200_hours_elapsed"]
    for rid in selected:
        p = base / "runs/experiments" / rid
        s = read(p / "summary.json", {})
        c = read(p / "resolved-config.json", {})
        configs[rid] = c
        formal = formal_order.index(rid) + 1 if rid in formal_order else None
        ev = compact_eval(s.get("fixed1k", s.get("last_evaluation", {})))
        full = read(p / "full-evaluation.json", {})
        add = read(p / "additional9k.json", {}) or read(p / "confirmation-evaluation.json", {})
        combined = read(p / "combined10k.json", {})
        tokens = s.get("confirmed_tokens", s.get("total_tokens_seen"))
        fixed_step = s.get("export_global_step", s.get("global_step"))
        row = {"formal": formal, "run_id": rid, "processed_inputs": tokens,
               "export_inputs": s.get("export_tokens_seen", tokens), "updates": s.get("global_step"),
               "export_step": fixed_step, "h200_hours": costs.get(rid) or costs.get(rid.split("-startup-replacement")[0]) or training["runs"].get(rid, {}).get("h200_hours_measured"),
               "ppl_1k": ev["perplexity"] if ev else None, "ppl_9k": add.get("eval/perplexity"),
               "ppl_10k": combined.get("perplexity"), "ppl_50k": full.get("eval/perplexity"),
               "full_eval_step": full.get("global_step"),
               "full_eval_matches_export": full.get("global_step") == fixed_step if full else None,
               "stop_reason": s.get("stop_reason"),
               "checkpoint_loaded_for_training": s.get("checkpoint_loaded_for_training",
                   True if s.get("continuation_id") else c.get("parallel_method", {}).get("training_checkpoint_load")),
               "checkpoint": s.get("checkpoint"), "export": s.get("export"),
               "wandb_url": s.get("wandb_url", s.get("wandb", {}).get("url")),
               "config_digest": s.get("config_digest", s.get("configuration_digest"))}
        # Old summaries use last_evaluation; use the exported checkpoint's 1k
        # evaluation whenever it differs from the final training checkpoint.
        curves = []
        for e in read(p / "evaluations.json", []):
            z = compact_eval(e)
            if z:
                curves.append(z)
        metrics = p / "metrics.jsonl"
        if metrics.exists():
            raw = metrics.read_bytes()
            SOURCES[str(metrics)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for line in raw.splitlines():
                m = json.loads(line)
                z = compact_eval(m)
                if z:
                    curves.append(z)
                if "train/learning_rate" in m:
                    train_rows.append(dict(run_id=rid, **{k: m[k] for k in metric_keys if k in m}))
        seen = {}
        for z in curves:
            if z["documents"] == 1000 and z["tokens"] is not None:
                seen[(z["step"], z["tokens"])] = z
            if z["documents"] == 1000 and z["step"] == fixed_step:
                row["ppl_1k"] = z["perplexity"]
        curve_rows.extend(dict(run_id=rid, **z) for z in sorted(seen.values(), key=lambda x: x["tokens"]))
        if row["ppl_10k"] is None and row["ppl_9k"] is not None and ev:
            combined_nll = ev["nll"] + add["eval/nll"]
            combined_tokens = ev["prediction_tokens"] + add["eval/prediction_tokens"]
            row["ppl_10k"] = math.exp(combined_nll / combined_tokens)
        if row["ppl_9k"] is None and s.get("confirmation_evaluation"):
            row["ppl_9k"] = s["confirmation_evaluation"].get("eval/perplexity")
        row["export_reload_verified"] = read(p / "export.json", {}).get("reload_verified")
        runs.append(row)

    # Existing non-KD 9k metadata is stored in a separate evaluation file.
    for row in runs:
        rid = row["run_id"]
        local_dir = REPO / "experiments" / rid
        report = read(local_dir / "results.json", {})
        if isinstance(report, dict) and report.get("confirmation_evaluation"):
            e = report["confirmation_evaluation"]
            row["ppl_9k"] = e.get("eval/perplexity")
            p1k = next((z for z in curve_rows if z["run_id"] == rid and z["step"] == row["export_step"]), None)
            if p1k:
                row["ppl_10k"] = math.exp((p1k["nll"] + e["eval/nll"]) /
                                           (p1k["prediction_tokens"] + e["eval/prediction_tokens"]))

    sweep = read(base / "runs/optimizer-sweep/optimizer-scheduler-raw-20261008/summary.json")
    ranking = [{k: r.get(k) for k in ["id", "run_id", "optimizer", "recipe", "tokens", "charged_tokens",
                                    "nll_100", "nll_90", "h200_hours", "status", "wandb_url"]}
               for r in sweep["ranking"]]
    init = read(REPO / "experiments/init-recipe-20261014/results/ranking.json")
    ab = read(REPO / "experiments/ppl-next-20261016/results.json")
    replication = read(REPO / "experiments/ppl-next-20261017/replication.json")
    ablations = [{"run_id": k, "mean_nll": v["mean_nll"],
                  "perplexity": v["evaluation"]["eval/perplexity"], "seed": 42,
                  "observers": {a: {"nll": e["eval/nll"], "prediction_tokens": e["eval/prediction_tokens"],
                                      "perplexity": e["eval/perplexity"]}
                                for a, e in v.get("observer_evaluations", {}).items()}}
                 for k, v in ab["runs"].items()]
    for name in ["control", "candidate"]:
        v = replication[name]
        ablations.append({"run_id": "replication-" + name, "mean_nll": v["mean_nll"],
                          "perplexity": v["perplexity"], "seed": 43})
    probes = []
    for rid in [formal_order[-1], selected[-3], selected[-2], selected[-1]]:
        p = base / "runs/experiments" / rid / "critical-batch.jsonl"
        if p.exists():
            raw = p.read_bytes()
            SOURCES[str(p)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for line in raw.splitlines():
                probes.append(dict(run_id=rid, **json.loads(line)))
    data = {"captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "work_root": str(base),
            "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "runs": runs, "ledger_totals": totals, "sweep": ranking,
            "sweep_ranking_stage": sweep.get("ranking_stage"),
            "init_rows": init["rows"], "init_selections": init["selections"],
            "ablations": ablations,
            "calibration": read(REPO / "experiments/lr-transfer-horizons-20261015/calibration-result.json"),
            "performance": read(REPO / "experiments/performance-ab-20261010/results.json"),
            "dataset_config": read(REPO / "configs/dataset.json"), "configs": configs,
            "batch_study": read(REPO / "experiments/batch-normuon-1b-20261016/results.json"),
            "methods_status": read(REPO / "experiments/methods-parallel-20261019/tracking/status.json"),
            "resplit_comparison": read(REPO / "experiments/horizon-wsd-20261014/results/resplit-lr-full.json"),
            "c2_decay_receipt": read(base / "runs/experiments" / formal_order[-1] / "decay-start-phase1.json"),
            "scope_note": "No model weights or training data were read. Ledgers are unchanged; sacct overrides apply only to this report snapshot."}
    # Keep the repository snapshot small: deterministic thinning for LR/batch
    # plots, retaining every observed batch transition and each run's endpoints.
    grouped_metrics = collections.defaultdict(list)
    for row in train_rows:
        grouped_metrics[row["run_id"]].append(row)
    sampled_metrics = []
    for rows in grouped_metrics.values():
        stride = max(1, math.ceil(len(rows) / 400))
        keep = set(range(0, len(rows), stride)) | {len(rows) - 1}
        for i in range(1, len(rows)):
            if rows[i].get("train/global_batch_sequences") != rows[i - 1].get("train/global_batch_sequences"):
                keep.update([i - 1, i])
        sampled_metrics.extend(rows[i] for i in sorted(keep))
    data["training_metric_sampling"] = "At most approximately 400 regularly spaced records per run, plus all observed batch transitions and both endpoints. Full-source hashes are retained."
    dump("evidence.json", data)
    dump("source_manifest.json", SOURCES)
    dump("critical_batch_probes.json", probes)
    write_csv("formal_runs.csv", [r for r in runs if r["formal"]])
    write_csv("evaluation_curves.csv", curve_rows)
    write_csv("training_metrics.csv", sampled_metrics)
    write_csv("allocations.csv", ledger_rows)
    write_csv("initial_sweep.csv", ranking)
    print(json.dumps({"formal_runs": len(formal_order), "eval_records": len(curve_rows),
                      "train_records": len(train_rows), "source_files": len(SOURCES), "totals": totals,
                      "latest": runs[19]}, indent=2))


if __name__ == "__main__":
    main()
