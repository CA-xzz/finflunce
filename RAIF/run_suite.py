"""Bounded GPU scheduling for validation selection and independent final seeds."""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

MODELS = ["DLinear", "PatchTST", "iTransformer", "ECT", "RAFT", "RATL", "RAIF"]
TRIALS = [{"hidden": 32, "lr": 0.001}, {"hidden": 64, "lr": 0.0003}]
SEEDS = [41, 42, 43]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def schedule(jobs, args):
    queue, active, failures = deque(jobs), {}, []
    args.output.mkdir(parents=True, exist_ok=True)
    while queue or active:
        for gpu in args.gpus:
            if gpu in active or not queue or failures:
                continue
            job = queue.popleft()
            output = args.output / job["path"]
            status_file = output / "status.json"
            if status_file.exists():
                status = json.loads(status_file.read_text())
                if status.get("status") == "completed":
                    config = json.loads((output / "config.json").read_text())
                    if config["data_hash"] != digest(args.data / "dataset.npz") or config["normalization_hash"] != digest(args.data / "normalization.json"):
                        raise RuntimeError("Completed run input changed; choose a new output directory: " + str(output))
                    if any(config["code_hashes"][name] != digest(Path(__file__).with_name(name)) for name in ["models.py", "train.py"]):
                        raise RuntimeError("Completed run implementation changed; choose a new output directory: " + str(output))
                    expected = dict(job.get("config", {}), model=job["model"], seed=job["seed"], epochs=args.epochs, patience=args.patience)
                    if any(config.get(name) != value for name, value in expected.items()):
                        raise RuntimeError("Completed run configuration changed: " + str(output))
                    extra = job.get("extra", [])
                    for index, flag in enumerate(extra):
                        if flag.startswith("--"):
                            key = flag[2:].replace("-", "_")
                            value = True if index + 1 == len(extra) or extra[index + 1].startswith("--") else extra[index + 1]
                            if str(config.get(key)) != str(value):
                                raise RuntimeError("Completed run extra option changed: " + str(output))
                    print(json.dumps({"event": "already_completed", "path": job["path"]}), flush=True)
                    continue
                raise RuntimeError(f"Existing unfinished/failed run is not retried automatically: {output}")
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).with_name("train.py")),
                "--data", str(args.data), "--output", str(output), "--model", job["model"],
                "--seed", str(job["seed"]), "--epochs", str(args.epochs), "--patience", str(args.patience),
                "--threads", "2", "--batch-size", "64", "--timeout-seconds", "7200"]
            for key, value in job.get("config", {}).items():
                command += ["--" + key.replace("_", "-"), str(value)]
            command += job.get("extra", [])
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            log = (output / "run.log").open("w")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            active[gpu] = (process, log, job, time.monotonic())
            write_json(output / "launch.json", {"command": command, "gpu": gpu, "pid": process.pid,
                       "started_unix": time.time(), "hard_timeout_seconds": 7200})
            print(json.dumps({"event": "started", "path": job["path"], "gpu": gpu, "pid": process.pid}), flush=True)
        for gpu, (process, log, job, started) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                status_path = args.output / job["path"] / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                if code != 0 or status.get("status") != "completed":
                    failures.append({"path": job["path"], "exit_code": code, "status": status})
                print(json.dumps({"event": "finished", "path": job["path"], "exit_code": code,
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
                del active[gpu]
            elif time.monotonic() - started > 7260:
                print(json.dumps({"event": "hard_timeout", "path": job["path"], "pid": process.pid}), flush=True)
                process.terminate()
        write_json(args.output / "suite_progress.json", {"queued": len(queue),
            "running": [{"gpu": gpu, "path": item[2]["path"], "pid": item[0].pid,
                         "elapsed_seconds": time.monotonic() - item[3]} for gpu, item in active.items()],
            "failures": failures, "updated_unix": time.time()})
        if failures and not active:
            raise RuntimeError("Run failed; remaining jobs were not launched: " + str(failures))
        if active:
            time.sleep(5)


def aggregate(args):
    for phase, prefix in [("main", "overall"), ("ablation", "ablation"), ("retrieval", "retrieval"), ("capacity", "capacity")]:
        files = sorted((args.output / phase).glob("**/horizon_metrics.csv"))
        if not files:
            continue
        frames = []
        for file in files:
            frame = pd.read_csv(file)
            frame["run_path"] = str(file.parent.relative_to(args.output))
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(args.output / f"{prefix}_metrics_seeds.csv", index=False)
        keys = ["model", "variant", "retrieval", "channel", "horizon", "metric", "split", "label_version", "memory_mode"]
        summary = raw.groupby(keys, dropna=False).agg(mean=("value", "mean"), std=("value", "std"),
            n_seeds=("seed", "nunique"), n_events=("n_events", "first"), n_dates=("n_dates", "first")).reset_index()
        summary.to_csv(args.output / f"{prefix}_metrics.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--phase", choices=["main", "extensions", "all"], default="all")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    normal = json.loads((args.data / "normalization.json").read_text())
    if normal.get("ready_for_formal_training") is not True:
        raise ValueError("Historical input checks have not passed")
    if len(set(args.gpus)) != len(args.gpus) or any(g not in [0, 1, 2, 3] for g in args.gpus):
        raise ValueError("This server allocation permits distinct GPUs 0–3 only")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    usage = {int(line.split(",")[0]): [int(v.strip()) for v in line.split(",")[1:]] for line in query.strip().splitlines()}
    if any(usage[g][0] > 512 or usage[g][1] > 5 for g in args.gpus):
        raise RuntimeError("One of the requested GPUs is busy; no jobs started")
    write_json(args.output / "suite_protocol.json", {"models": MODELS, "trials": TRIALS,
        "tuning_seed": 40, "final_seeds": SEEDS, "epochs": args.epochs, "patience": args.patience,
        "selection_metric": "validation standardized equal-channel MAE", "test_used_for_selection": False,
        "scope": "Main models, five RAIF ablations, retrieval orderings, RATL memory-size comparison.",
        "remaining_paper_experiments": ["three single-reference common-target comparisons",
            "K/L/training-quality parameter comparisons", "rolling memory", "frozen retrieval replacements",
            "serial same-device efficiency measurement"]})
    if args.phase in ["main", "all"]:
        tuning = [{"path": f"tuning/{model}/trial{trial}", "model": model, "seed": 40,
                   "config": config, "extra": ["--validation-only"]}
                  for model in MODELS for trial, config in enumerate(TRIALS)]
        schedule(tuning, args)
        chosen = {}
        for model in MODELS:
            candidates = []
            for trial, config in enumerate(TRIALS):
                metrics = json.loads((args.output / f"tuning/{model}/trial{trial}/validation_summary.json").read_text())
                candidates.append((metrics["validation_standardized_mae"], trial, config))
            value, trial, config = min(candidates, key=lambda x: (x[0], x[1]))
            chosen[model] = {"trial": trial, "config": config, "validation_mae": value}
        write_json(args.output / "selected_configs.json", chosen)
        jobs = [{"path": f"main/{model}/seed{seed}", "model": model, "seed": seed, "config": chosen[model]["config"]}
                for model in MODELS for seed in SEEDS]
        jobs.insert(0, {"path": "main/HistoricalMean/seed41", "model": "HistoricalMean", "seed": 41})
        schedule(jobs, args)
        aggregate(args)
    else:
        chosen = json.loads((args.output / "selected_configs.json").read_text())
    if args.phase in ["extensions", "all"]:
        config = chosen["RAIF"]["config"]
        jobs = [{"path": f"ablation/{variant}/seed{seed}", "model": "RAIF", "seed": seed, "config": config,
                 "extra": ["--variant", variant]}
                for variant in ["no_retrieval", "no_response", "no_events", "no_quality", "no_attention"] for seed in SEEDS]
        jobs += [{"path": f"retrieval/{retrieval}/seed{seed}", "model": "RAIF", "seed": seed, "config": config,
                  "extra": ["--retrieval", retrieval]}
                 for retrieval in ["state", "random", "reverse"] for seed in SEEDS]
        jobs += [{"path": f"capacity/RAIF/seed{seed}", "model": "RAIF", "seed": seed, "config": config,
                  "extra": ["--memory-subset", "ratl"]} for seed in SEEDS]
        schedule(jobs, args)
        aggregate(args)
    write_json(args.output / "suite_status.json", {"status": "completed_requested_phases", "phase": args.phase,
               "finished_unix": time.time(), "complete_paper_experiments": False})


if __name__ == "__main__":
    main()
