"""Execute the nine new fits; keep all existing incremental models frozen."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", nargs=3, type=int, default=[0, 2, 3])
    args = parser.parse_args()
    root = args.root.resolve()
    if len(set(args.gpus)) != 3 or any(g not in [0, 1, 2, 3] for g in args.gpus):
        raise ValueError("Choose three distinct allocated GPUs 0–3")
    verification = json.loads((root / "data/trajectory_comparison/verification.json").read_text())
    assert verification["status"] == "passed"
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"], text=True)
    for line in usage.strip().splitlines():
        gpu, memory, utilization = map(int, line.split(","))
        if gpu in args.gpus and (memory > 512 or utilization > 5):
            raise RuntimeError("An allocated GPU is busy; no training started")
    output = root / "results/trajectory_comparison"
    logs = root / "logs/trajectory_comparison"
    output.mkdir(exist_ok=True, parents=True); logs.mkdir(exist_ok=True, parents=True)
    state = dict(status="running", started_unix=time.time(), pid=os.getpid(),
                 expected_new_component_fits=9, incremental_checkpoints_reused=9,
                 seeds=[41, 42, 43], gpus=args.gpus, completed_components=[])
    def save():
        path = output / "training_status.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n"); tmp.replace(path)
    save()
    def run(role, seed, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        command = [sys.executable, str(root / "train_trajectory_component.py"), "--root", str(root),
                   "--role", role, "--seed", str(seed), "--threads", "2"]
        with (logs / f"{role}_seed{seed}.log").open("a") as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=7500)
        return dict(role=role, seed=seed, gpu=gpu)
    try:
        for role in ["reference", "ect_direct", "raft_direct"]:
            state["phase"] = role; save()
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(run, role, seed, gpu) for seed, gpu in zip([41, 42, 43], args.gpus)]
                for job in as_completed(jobs):
                    item = job.result(); state["completed_components"].append(item); save()
                    print(json.dumps(item), flush=True)
        state.update(status="completed", phase=None, finished_unix=time.time()); save()
    except BaseException as error:
        state.update(status="failed", error=str(error), finished_unix=time.time()); save()
        raise


if __name__ == "__main__":
    main()
