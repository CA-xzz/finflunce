"""Train prespecified reference comparisons on a shared strict test target."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from run_suite import schedule, SEEDS


def common_target_metrics(dataset, run, targets):
    events = pd.read_csv(dataset / "events.csv", dtype={"event_id": str})
    values = np.load(run / "predictions.npz", allow_pickle=False)
    ids = values["event_indices"]
    np.testing.assert_array_equal(events.iloc[ids].event_id.to_numpy(str), targets["event_ids"])
    truth, prediction = targets["truth"], values["prediction"]
    if truth.shape != prediction.shape or truth.shape != values["truth"].shape:
        raise ValueError("Reference predictions do not share the common target shape")
    if any(not np.isfinite(array).all() for array in [truth, values["truth"], prediction]):
        raise ValueError("Nonfinite reference predictions or targets")
    normal = json.loads((dataset / "normalization.json").read_text())
    tolerance = 8 * np.finfo(np.float32).eps * (
        np.abs(truth) + np.asarray(normal["y_std"]) + np.abs(normal["y_mean"]))
    if (np.abs(truth - values["truth"]) > tolerance).any():
        raise ValueError("Run truth differs from canonical reference target beyond float32 rounding")
    metrics = pd.read_csv(run / "horizon_metrics.csv")
    channels = {"AR": 0, "turnover": 1, "log_volume": 2}
    for row in metrics.itertuples():
        error = prediction[:, :row.horizon, channels[row.channel]] - truth[:, :row.horizon, channels[row.channel]]
        if row.metric == "MAE":
            value = np.abs(error).mean()
        elif row.metric == "RMSE":
            value = np.sqrt(np.square(error).mean())
        else:
            raise ValueError("Unexpected reference metric: " + row.metric)
        metrics.at[row.Index, "value"] = value
    metrics["common_original_unit_target"] = True
    metrics.to_csv(run / "common_target_horizon_metrics.csv", index=False)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--suite", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g not in [0, 1, 2, 3] for g in args.gpus):
        raise ValueError("Use distinct allocated GPUs 0–3")
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    for line in usage.strip().splitlines():
        gpu, memory, utilization = map(int, line.split(","))
        if gpu in args.gpus and (memory > 512 or utilization > 5):
            raise RuntimeError("Selected GPU is busy")
    args.epochs, args.patience = 100, 15
    args.output = args.output or args.root / "results/references"
    args.output.mkdir(parents=True, exist_ok=True)
    suite = args.suite or args.root / "results/suite"
    config = json.loads((suite / "selected_configs.json").read_text())["RAIF"]["config"]
    plan = pd.read_csv(args.plan or args.root / "data/reference_comparisons/comparison_plan.csv")
    frames = []
    canonical = None
    for row in plan.itertuples():
        args.data = Path(row.dataset)
        targets = np.load(args.data / "evaluation_targets.npz", allow_pickle=False)
        if canonical is None:
            canonical = {name: targets[name].copy() for name in targets.files}
        else:
            for name in targets.files:
                np.testing.assert_array_equal(canonical[name], targets[name])
        jobs = [{"path": f"{row.group}/{row.scheme}/seed{seed}", "model": "RAIF", "seed": seed,
                 "config": config, "extra": ["--variant", row.variant]} for seed in SEEDS]
        schedule(jobs, args)
        for seed in SEEDS:
            path = args.output / f"{row.group}/{row.scheme}/seed{seed}"
            metrics = common_target_metrics(args.data, path, canonical)
            metrics["training_group"], metrics["reference_scheme"] = row.group, row.scheme
            metrics["n_train"] = row.n_train
            metrics["evaluation_label_version"] = row.evaluation_label_version
            frames.append(metrics)
        pd.concat(frames, ignore_index=True).to_csv(args.output / "reference_metrics_seeds.csv", index=False)
    raw = pd.concat(frames, ignore_index=True)
    raw.groupby(["training_group", "reference_scheme", "channel", "horizon", "metric", "evaluation_label_version"]).agg(
        mean=("value", "mean"), std=("value", "std"), n_seeds=("seed", "nunique"),
        n_train=("n_train", "first"), n_test=("n_events", "first"), n_dates=("n_dates", "first")).reset_index().to_csv(
            args.output / "reference_metrics.csv", index=False)
    (args.output / "status.json").write_text(json.dumps({"status": "completed", "finished_unix": time.time(),
        "datasets": len(plan), "seeds": SEEDS, "shared_test_reference": "strict",
        "hyperparameters": "Frozen main RAIF validation-selected configuration; no reference-test-based selection."}, indent=2))


if __name__ == "__main__":
    main()
