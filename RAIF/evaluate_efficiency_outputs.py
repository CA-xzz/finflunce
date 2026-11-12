"""Evaluate and time the same trained common-memory efficiency checkpoints.

This supplements, and never replaces, the original main experiment. No model
or hyperparameter is fitted or selected here. Test outcomes are used only after
prediction to calculate errors, and never enter a forward-call dictionary.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch

import benchmark_serial_training as bench
import evaluate_extras as extras
import train as core


def bound_manifest(path, data, events, expected_split):
    frame = pd.read_csv(path, dtype={"event_id": str})
    if frame.event_id.duplicated().any() or frame.dataset_index.duplicated().any():
        raise ValueError("Manifest identities must be unique")
    values = frame.dataset_index.to_numpy()
    if not np.issubdtype(values.dtype, np.integer) or (values < 0).any() or (values >= len(events)).any():
        raise ValueError("Manifest dataset indices are invalid")
    actual = events.iloc[values]
    if not np.array_equal(actual.event_id.to_numpy(), frame.event_id.to_numpy()):
        raise ValueError("Manifest event IDs do not match the dataset indices")
    if not np.array_equal(actual.event_date.astype(str).to_numpy(), frame.event_date.astype(str).to_numpy()):
        raise ValueError("Manifest event dates do not match the dataset")
    ids = torch.as_tensor(values, device=data["x"].device, dtype=torch.long)
    if not bool((data["split"][ids] == expected_split).all()):
        raise ValueError("Manifest contains an event outside the required split")
    return ids


def discover(source, data_path, data, events, expected_memory):
    status = json.loads((source / "status.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    if status.get("status") != "completed" or status.get("training_runs") != 22:
        raise ValueError("The full serial training benchmark must complete before aligned evaluation")
    if protocol["benchmark_script_sha256"] != core.sha256(Path(bench.__file__)):
        raise ValueError("The source training benchmark script has changed")
    for key, file in [("dataset_sha256", "dataset.npz"), ("normalization_sha256", "normalization.json")]:
        if protocol[key] != core.sha256(data_path / file):
            raise ValueError("Source efficiency protocol uses different data")
    for file, digest in protocol["code_hashes"].items():
        if digest != core.sha256(Path(__file__).with_name(file)):
            raise ValueError("Frozen source differs: " + file)
    if protocol["common_memory_sha256"] != core.sha256(source / "common_memory_manifest.csv"):
        raise ValueError("Common memory manifest has changed")
    common = bound_manifest(source / "common_memory_manifest.csv", data, events, 0)
    queries = bound_manifest(source / "batch1_query_manifest.csv", data, events, 2)
    if len(common) != expected_memory:
        raise ValueError("Unexpected common memory size")
    options = protocol["options"]
    test = (data["split"] == 2).nonzero().flatten()
    expected_queries = test[np.linspace(0, len(test) - 1, min(options["query_count"], len(test)), dtype=int)]
    if not torch.equal(queries, expected_queries):
        raise ValueError("Batch-one query manifest differs from the frozen selection")
    costs = pd.read_csv(source / "training_timing.csv")
    if len(costs) != 22 or costs.duplicated(["model", "seed"]).any():
        raise ValueError("Incomplete or duplicate source training times")
    runs = []
    for name in bench.MODELS:
        for seed in ((41,) if name == "HistoricalMean" else bench.SEEDS):
            path = source / "training" / name / ("seed%d" % seed)
            config = extras.read_run(path, data_path)
            state = json.loads((path / "status.json").read_text())
            if config.get("efficiency_only") is not True or state.get("mode") != "efficiency_only":
                raise ValueError("Expected an efficiency-only training checkpoint")
            if config["model"] != name or config["seed"] != seed or config["n_memory"] != len(common):
                raise ValueError("Efficiency checkpoint identity or memory count mismatch")
            if config.get("benchmark_script_sha256") != protocol["benchmark_script_sha256"]:
                raise ValueError("Checkpoint benchmark hash differs from the source protocol")
            cost = costs.loc[(costs.model == name) & (costs.seed == seed)]
            if len(cost) != 1 or not np.isclose(cost.training_seconds.iloc[0], state["training_seconds"], rtol=1e-12):
                raise ValueError("Checkpoint and measured training cost differ")
            # This also verifies that the RATL residual subset is the common memory.
            bench.training_partitions(config, data, common)
            runs.append((path, config, cost.iloc[0].to_dict()))
    return runs, common, queries, protocol


@torch.no_grad()
def predict_test(model, data, memory, batch_size, guard, timeout_seconds):
    ids = (data["split"] == 2).nonzero().flatten()
    model.eval()
    cache = model.refresh_memory(memory) if core.retrieves(model) else None
    predictions = []
    started = time.monotonic()
    for batch in ids.split(batch_size):
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError("Aligned test prediction timeout")
        guard.maybe_check("aligned_test_prediction")
        query = extras.query_inputs(data, batch)
        if "y" in query:
            raise AssertionError("Query outcomes entered prediction")
        result = model(query, memory, cache)
        predictions.append(result["prediction"].cpu().numpy())
    return ids, np.concatenate(predictions)


def export_accuracy(output, ids, predictions, data, events, normal, config, checkpoint_hash):
    indices = ids.cpu().numpy()
    mean, scale = np.asarray(normal["y_mean"]), np.asarray(normal["y_std"])
    truth = data["y"][ids].cpu().numpy().astype(np.float64) * scale + mean
    prediction = predictions.astype(np.float64) * scale + mean
    if truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid efficiency prediction shape or values")
    selected = events.iloc[indices].reset_index(drop=True)
    metadata = dict(model=config["model"], seed=config["seed"], variant="full", split="test",
        label_version=str(selected.label_version.iloc[0]), memory_mode="fixed_common_training", n_memory=config["n_memory"],
        evaluation_kind="same_capacity_efficiency_checkpoint", checkpoint_sha256=checkpoint_hash)
    rows = pd.DataFrame(core.metric_rows(truth, prediction, selected, metadata))
    rows.to_csv(output / "horizon_metrics.csv", index=False)
    rows[rows.horizon == 11].to_csv(output / "overall_metrics.csv", index=False)
    np.savez_compressed(output / "predictions.npz", event_indices=indices, truth=truth, prediction=prediction)
    standardized_mae = float(np.abs(predictions - data["y"][ids].cpu().numpy()).mean())
    return rows, standardized_mae


def summarize(output, costs, metrics):
    frame = pd.DataFrame(costs)
    frame.to_csv(output / "aligned_timing.csv", index=False)
    accuracy = pd.concat(metrics, ignore_index=True)
    accuracy.to_csv(output / "horizon_metrics.csv", index=False)
    accuracy[accuracy.horizon == 11].to_csv(output / "overall_metrics.csv", index=False)
    summary = []
    for name, group in frame.groupby("model", sort=False):
        for field in ["training_seconds", "optimization_seconds", "validation_seconds", "training_cache_seconds",
                      "ratl_base_seconds", "ratl_residual_seconds", "ratl_gamma_seconds", "initialization_seconds",
                      "inference_event_ms", "retrieval_event_ms", "canonical_memory_mb", "parameters",
                      "test_standardized_mae"]:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            if len(values):
                summary.append(dict(model=name, metric=field, n_seeds=len(values), mean=float(values.mean()),
                    std=float(values.std(ddof=1)) if len(values) > 1 else None))
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    grouped = accuracy.groupby(["model", "channel", "horizon", "metric"], sort=False).value.agg(["count", "mean", "std"]).reset_index()
    grouped.rename(columns={"count": "n_seeds"}).to_csv(output / "accuracy_summary.csv", index=False)


def execute(options):
    if options.output.exists() and any(options.output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing evaluation")
    if min(options.expected_memory, options.batch_size, options.threads, options.timeout_seconds) < 1:
        raise ValueError("Counts and timeout must be positive")
    device = torch.device(options.device)
    if device.type == "cuda" and str(extras.physical_device(device)) != "0":
        raise ValueError("Aligned efficiency is allocated to physical GPU 0")
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(options.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cpu_data, events, normal = core.load_data(options.data, torch.device("cpu"))
    runs, common_cpu, query_cpu, source_protocol = discover(options.source, options.data, cpu_data, events, options.expected_memory)
    timing_options = source_protocol["options"]
    if options.threads != timing_options["threads"] or options.batch_size != timing_options["batch_size"]:
        raise ValueError("Evaluation threads/batch must match the efficiency training protocol")
    events.iloc[common_cpu.numpy()].assign(dataset_index=common_cpu.numpy()).to_csv(options.output / "common_memory_manifest.csv", index=False)
    events.iloc[query_cpu.numpy()].assign(dataset_index=query_cpu.numpy()).to_csv(options.output / "batch1_query_manifest.csv", index=False)
    protocol = dict(source_training_root=str(options.source.resolve()), source_protocol_sha256=core.sha256(options.source / "protocol.json"),
        source_training_timing_sha256=core.sha256(options.source / "training_timing.csv"),
        evaluator_sha256=core.sha256(Path(__file__)), dataset_sha256=source_protocol["dataset_sha256"],
        normalization_sha256=source_protocol["normalization_sha256"], common_memory_sha256=core.sha256(options.output / "common_memory_manifest.csv"),
        query_manifest_sha256=core.sha256(options.output / "batch1_query_manifest.csv"),
        source_training_protocol=source_protocol, n_memory=len(common_cpu), n_test=int((cpu_data["split"] == 2).sum()),
        parameters_updated=False, hyperparameters_selected=False, normalization_updated=False,
        supplement_only=True, replaces_main_results=False,
        alignment="Training cost, test errors, and batch-one latency refer to the same common-memory efficiency checkpoint.",
        precision="float32", batch1_queries=len(query_cpu), batch1_repeats=timing_options["repeats"],
        batch1_warmup=timing_options["warmup"], pid=os.getpid(), started_unix=time.time())
    core.write_json(options.output / "protocol.json", protocol)
    core.write_json(options.output / "status.json", dict(status="running", pid=os.getpid(), started_unix=time.time()))
    guard = bench.QuietGPU(device, options.idle_gpus, 30.)
    costs, metrics = [], []
    try:
        with extras.serial_gpu_lock(options.output, guard):
            guard.establish()
            data = {key: value.to(device) for key, value in cpu_data.items()}
            common, queries = common_cpu.to(device), query_cpu.to(device)
            memory = core.subset(data, common)
            for path, config, training_cost in runs:
                name = config["model"] + "/seed%d" % config["seed"]
                destination = options.output / name
                destination.mkdir(parents=True)
                bench.seed_all(config["seed"])
                guard.check("before_efficiency_checkpoint_" + name)
                checkpoint = path / "checkpoint.pt"
                digest = core.sha256(checkpoint) if checkpoint.exists() else None
                model = extras.load_checkpoint(path, config, device)
                ids, predictions = predict_test(model, data, memory, options.batch_size, guard, options.timeout_seconds)
                rows, standardized = export_accuracy(destination, ids, predictions, data, events, normal, config, digest)
                metrics.append(rows)
                measured, samples = bench.batch_one_benchmark(model, data, memory, queries,
                    timing_options["warmup"], timing_options["repeats"], guard, options.timeout_seconds)
                if checkpoint.exists() and core.sha256(checkpoint) != digest:
                    raise ValueError("Checkpoint changed during evaluation")
                measured = dict(training_cost, **measured)
                measured.update(checkpoint_sha256=digest, source_efficiency_run=str(path.resolve()),
                    checkpoint_role="same_capacity_efficiency_checkpoint", test_standardized_mae=standardized,
                    n_test=len(ids), inference_common_memory_events=len(common),
                    same_checkpoint_accuracy_training_and_inference=True, accuracy_recomputed=True,
                    parameters_updated=False, replaces_main_results=False)
                costs.append(measured)
                pd.DataFrame(samples).to_csv(destination / "batch1_samples.csv", index=False)
                core.write_json(destination / "evaluation_config.json", measured)
                summarize(options.output, costs, metrics)
                core.write_json(options.output / "status.json", dict(status="running", completed_runs=len(costs), last_run=name,
                    pid=os.getpid(), updated_unix=time.time()))
                print(json.dumps(dict(run=name, status="completed", test_standardized_mae=standardized,
                    training_seconds=measured["training_seconds"], inference_event_ms=measured["inference_event_ms"])), flush=True)
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            guard.check("aligned_finished")
        core.write_json(options.output / "status.json", dict(status="completed", completed_runs=len(costs),
            no_observed_contention=True if device.type == "cuda" else None, finished_unix=time.time(),
            gpu_pid_ownership_method=guard.ownership_method, same_checkpoint_accuracy_training_and_inference=True))
    except Exception as error:
        core.write_json(options.output / "status.json", dict(status="failed", error=str(error), traceback=traceback.format_exc(), timing_valid=False))
        raise
    finally:
        core.write_json(options.output / "gpu_checks.json", guard.records)
        core.write_json(options.output / "companion_gpu_checks.json", guard.idle_records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-memory", type=int, default=912)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=7200.)
    parser.add_argument("--idle-gpus", nargs="*", type=int, default=[1, 2, 3])
    return parser.parse_args(argv)


if __name__ == "__main__":
    execute(parse_args())
