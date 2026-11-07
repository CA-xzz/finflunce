"""Serial checkpoint evaluations after the expanded training suite finishes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_suite import MODELS, SEEDS, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    data, suite = root / "data/prepared_expanded", root / "results/suite_expanded"
    status = json.loads((suite / "suite_status.json").read_text())
    if status.get("phase") != "all" or status.get("status") != "completed_requested_phases":
        raise ValueError("Complete the main suite and extensions first")
    raif = [suite / f"main/RAIF/seed{seed}" for seed in SEEDS]
    runs = [suite / "main/HistoricalMean/seed41"] + [suite / f"main/{model}/seed{seed}"
        for model in MODELS for seed in SEEDS]
    phases = [("rolling", "rolling_expanded", raif),
              ("frozen", "frozen_retrieval_expanded", raif),
              ("timing", "serial_timing_expanded", runs)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    progress = {"completed": [], "running": None, "status": "running"}
    progress_path = root / "results/extra_evaluations_expanded.json"
    for mode, name, checkpoints in phases:
        progress["running"] = mode
        write_json(progress_path, progress)
        # Allow utilization sampling to clear after the preceding CUDA process.
        time.sleep(2)
        command = [sys.executable, str(root / "evaluate_extras.py"), "--data", str(data),
            "--runs"] + list(map(str, checkpoints)) + ["--mode", mode, "--output",
            str(root / "results" / name), "--threads", "2", "--batch-size", "64",
            "--warmup", "10", "--repeats", "50"]
        with (root / "logs" / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            progress.update(status="failed", returncode=result.returncode)
            write_json(progress_path, progress)
            raise RuntimeError("Supplementary evaluation failed: " + mode)
        progress["completed"].append(mode)
        print(json.dumps({"completed": mode}), flush=True)
    progress.update(status="completed", running=None, finished_unix=time.time())
    write_json(progress_path, progress)


if __name__ == "__main__":
    main()
