"""Execute prespecified K/input-window/quality-retention comparisons."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
import pandas as pd
from run_suite import schedule, SEEDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--suite", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--windows", type=Path)
    p.add_argument("--quality", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    args = p.parse_args()
    suite = args.suite or args.root / "results/suite"
    base_data = args.data or args.root / "data/prepared"
    windows = args.windows or args.root / "data/window_comparisons"
    quality = args.quality or args.root / "data/quality_comparisons"
    if len(set(args.gpus)) != len(args.gpus) or any(g not in [0, 1, 2, 3] for g in args.gpus):
        raise ValueError("Use distinct allocated GPUs 0–3")
    if not (suite / "suite_status.json").exists():
        raise RuntimeError("Wait for the main suite to finish before sharing its GPUs")
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    for line in usage.strip().splitlines():
        gpu, memory, utilization = map(int, line.split(","))
        if gpu in args.gpus and (memory > 512 or utilization > 5):
            raise RuntimeError("A selected GPU is busy")
    args.epochs, args.patience = 100, 15
    args.output = args.output or args.root / "results/parameters"
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((suite / "selected_configs.json").read_text())["RAIF"]["config"]
    records = []
    groups = [("K", k, base_data, ["--k", str(k)]) for k in [1, 3, 10, 20]]
    groups += [("L", length, windows / f"L{length}", []) for length in [5, 10, 20]]
    groups += [("quality_quantile", cutoff / 100, quality / f"q{cutoff}", []) for cutoff in [25, 50, 75]]
    for parameter, value, data, extra in groups:
        args.data = data
        jobs = [{"path": f"{parameter}/{value}/seed{seed}", "model": "RAIF", "seed": seed,
                 "config": config, "extra": extra} for seed in SEEDS]
        schedule(jobs, args)
        for seed in SEEDS:
            path = args.output / f"{parameter}/{value}/seed{seed}"
            metrics = pd.read_csv(path / "overall_metrics.csv")
            metrics["parameter"], metrics["parameter_value"] = parameter, value
            metrics["run_path"] = str(path)
            metrics["validation_mae"] = json.loads((path / "status.json").read_text())["validation_standardized_mae"]
            run_config = json.loads((path / "config.json").read_text())
            metrics["n_train"], metrics["n_test"] = run_config["n_train"], run_config["n_test"]
            metrics["memory_size"] = run_config["n_memory"]
            metrics["latency_ms"] = pd.read_csv(path / "timing.csv").inference_batch_ms.iloc[0]
            metrics["latency_scope"] = "Per-run observed batch timing during concurrent training; separate serial benchmark required for method efficiency comparison"
            records.append(metrics)
        pd.concat(records, ignore_index=True).to_csv(args.output / "sweep_metrics.csv", index=False)
    # Reuse the original K=5/q=0 runs; no duplicate fitting or test-driven selection.
    for parameter, value in [("K", 5), ("quality_quantile", 0)]:
        for seed in SEEDS:
            path = suite / f"main/RAIF/seed{seed}"
            metrics = pd.read_csv(path / "overall_metrics.csv")
            metrics["parameter"], metrics["parameter_value"] = parameter, value
            metrics["run_path"] = str(path)
            metrics["validation_mae"] = json.loads((path / "status.json").read_text())["validation_standardized_mae"]
            run_config = json.loads((path / "config.json").read_text())
            metrics["n_train"], metrics["n_test"] = run_config["n_train"], run_config["n_test"]
            metrics["memory_size"] = run_config["n_memory"]
            metrics["latency_ms"] = pd.read_csv(path / "timing.csv").inference_batch_ms.iloc[0]
            metrics["latency_scope"] = "Per-run observed batch timing during concurrent training; separate serial benchmark required for method efficiency comparison"
            records.append(metrics)
    raw = pd.concat(records, ignore_index=True)
    raw.to_csv(args.output / "sweep_metrics.csv", index=False)
    raw.groupby(["parameter", "parameter_value", "channel", "metric"]).agg(mean=("value", "mean"),
        std=("value", "std"), n_seeds=("seed", "nunique"), n_events=("n_events", "first")).reset_index().to_csv(
            args.output / "sweep_summary.csv", index=False)
    (args.output / "status.json").write_text(json.dumps({"status": "completed", "finished_unix": time.time(),
        "protocol": "Prespecified sensitivity comparisons; no test-based parameter selection. L uses a common L20-complete set and original fixed L10 reference labels."}, indent=2))


if __name__ == "__main__":
    main()
