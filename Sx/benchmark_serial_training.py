"""Separate same-capacity training and actual batch-one inference benchmarks.

Frozen train.py supplies the optimizer loop, objective, validation selection,
and RATL partition. Training here creates new efficiency-only checkpoints with
the common training memory. Inference loads the original main checkpoints.
Neither operation produces test errors or replaces a main experiment result.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import platform
import random
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch

import evaluate_extras as extras
import train as core
from models import RATL, build_model


MODELS = ("HistoricalMean", "DLinear", "PatchTST", "iTransformer", "ECT", "RAFT", "RATL", "RAIF")
SEEDS = (41, 42, 43)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QuietGPU(extras.ExclusiveGPU):
    """Reuse PID/context checks, also require the other allocated GPUs idle."""
    def __init__(self, device, idle_gpus=(1, 2, 3), interval=30.):
        super().__init__(device)
        self.idle_gpus = tuple(str(i) for i in idle_gpus if str(i) != str(self.index))
        self.interval = interval
        self.last_check = 0.
        self.monitor_seconds = 0.
        self.idle_records = []

    def check_companions(self, stage):
        if self.index is None:
            return
        for index in self.idle_gpus:
            record = extras.gpu_snapshot(index)
            self.idle_records.append(dict(stage=stage, **record))
            if record["pids"] or record["memory_used_mib"] > 64 or record["utilization"] > 0:
                raise RuntimeError("Other allocated GPU is busy; timing aborted: GPU " + index)

    def initial(self):
        super().initial()
        self.check_companions("before_cuda_allocation")

    def establish(self):
        original = extras.warm_cuda_libraries

        def warm_training(device):
            original(device)
            # Materialize backward and optimizer libraries before measuring any
            # model, without changing the frozen evaluator's calibration code.
            with torch.enable_grad(), torch.cuda.device(device):
                x = torch.ones((2, 8, 64), device=device, requires_grad=True)
                weight = torch.ones((8, 8, 3), device=device, requires_grad=True)
                matrix = torch.ones((32, 32), device=device, requires_grad=True)
                opt = torch.optim.AdamW([weight, matrix])
                (torch.nn.functional.conv1d(x, weight).square().mean() +
                 (matrix @ matrix).square().mean()).backward()
                opt.step()
                core.synchronized(device)
                del x, weight, matrix, opt
                torch.cuda.empty_cache()
                core.synchronized(device)

        with patch.object(extras, "warm_cuda_libraries", warm_training):
            super().establish()

    def check(self, stage):
        begin = time.perf_counter()
        try:
            super().check(stage)
            self.check_companions(stage)
        finally:
            self.monitor_seconds += time.perf_counter() - begin
            self.last_check = time.monotonic()

    def maybe_check(self, stage):
        if time.monotonic() - self.last_check >= self.interval:
            self.check(stage)


def discover_runs(suite, data_path):
    suite = Path(suite)
    selection = json.loads((suite / "selected_configs.json").read_text())
    runs = []
    for name in MODELS:
        for seed in ((41,) if name == "HistoricalMean" else SEEDS):
            directory = suite / "main" / name / ("seed%d" % seed)
            config = extras.read_run(directory, data_path)
            status = json.loads((directory / "status.json").read_text())
            if config["model"] != name or config["seed"] != seed or status.get("mode") != "full":
                raise ValueError("Expected completed main model/seed: " + str(directory))
            if config.get("variant") != "full" or config.get("retrieval") != "response":
                raise ValueError("Efficiency comparison requires the full main models")
            if config.get("code_hashes", {}).get("train.py") != core.sha256(Path(core.__file__)):
                raise ValueError("Frozen training source hash differs")
            if name != "HistoricalMean":
                for key, value in selection[name]["config"].items():
                    if config.get(key) != value:
                        raise ValueError("Main configuration differs from validation selection: " + key)
                checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
                if checkpoint["model_config"] != config["model_config"]:
                    raise ValueError("Checkpoint/config mismatch")
            runs.append((directory, config))
    return runs


def common_ids(runs, data, events, expected=912):
    source = [extras.memory_indices(path, config, data, events) for path, config in runs
              if config["model"] in {"HistoricalMean", "RAFT", "RATL", "RAIF"}]
    common = extras.common_memory(source)
    if len(common) != expected:
        raise ValueError("Common memory count is %d, expected %d" % (len(common), expected))
    if not bool((data["split"][common] == 0).all()):
        raise ValueError("Common memory must be training-only")
    return common


def training_partitions(config, data, common):
    train_ids = (data["split"] == 0).nonzero().flatten()
    if config["model"] != "RATL":
        return [("main", train_ids)], None
    early, late, cutoff = core.partition_ratl(data, train_ids, config["ratl_base_fraction"], 11)
    if set(late.cpu().tolist()) != set(common.cpu().tolist()):
        raise ValueError("RATL residual partition differs from the common training memory")
    return [("base", early), ("residual", late)], cutoff


def record_training_population(output, partitions, common, config, data, events):
    rows = []
    ends = np.sort(data["end_day"][common].cpu().numpy())
    for stage, ids in partitions:
        frame = events.iloc[ids.cpu().numpy()].copy()
        frame["dataset_index"] = ids.cpu().numpy()
        frame["stage"] = stage
        frame["common_memory_member"] = np.isin(ids.cpu().numpy(), common.cpu().numpy())
        gap = config["ratl_gap"] if stage == "residual" else 0
        frame["mature_common_candidates"] = np.searchsorted(
            ends, data["day"][ids].cpu().numpy() - gap, side="left")
        frame["retrieval_used"] = config["model"] in {"RAFT", "RAIF", "RATL"} and stage != "base"
        rows.append(frame)
    pd.concat(rows, ignore_index=True).to_csv(output / "training_population.csv", index=False)


@contextmanager
def instrument_stage(model, device, guard, name):
    """Observe the unchanged fit_stage loop; scoped patches restore on errors."""
    samples = dict(batches=[], validation=[], cache=[])
    state = dict(epoch=1, in_validation=False, query_indices=None, begin=None)
    original_evaluate, original_optimizer = core.evaluate, torch.optim.AdamW
    original_refresh = getattr(model, "refresh_memory", None)

    def capture_query(module, args, kwargs):
        if module.training and torch.is_grad_enabled():
            state["query_indices"] = args[0]["index"].detach()

    def timed_evaluate(*args, **kwargs):
        guard.maybe_check(name + "_before_validation")
        core.synchronized(device)
        begin = time.perf_counter()
        state["in_validation"] = True
        try:
            result = original_evaluate(*args, **kwargs)
            core.synchronized(device)
            samples["validation"].append(dict(epoch=state["epoch"], seconds=time.perf_counter() - begin))
            state["epoch"] += 1
            return result
        finally:
            state["in_validation"] = False

    def timed_refresh(*args, **kwargs):
        core.synchronized(device)
        begin = time.perf_counter()
        result = original_refresh(*args, **kwargs)
        core.synchronized(device)
        samples["cache"].append(dict(epoch=state["epoch"], phase="validation" if state["in_validation"] else "training",
                                     seconds=time.perf_counter() - begin))
        return result

    class TimedAdamW(original_optimizer):
        def zero_grad(self, *args, **kwargs):
            guard.maybe_check(name + "_training")
            core.synchronized(device)
            state["begin"] = time.perf_counter()
            return super().zero_grad(*args, **kwargs)

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            core.synchronized(device)
            elapsed = time.perf_counter() - state["begin"]
            ids = state["query_indices"].cpu().tolist()
            samples["batches"].append(dict(epoch=state["epoch"], batch=len(samples["batches"]),
                                           seconds=elapsed, n_events=len(ids), dataset_indices=ids))
            return result

    hook = model.register_forward_pre_hook(capture_query, with_kwargs=True)
    try:
        if original_refresh is not None:
            model.refresh_memory = timed_refresh
        with patch.object(core, "evaluate", timed_evaluate), patch.object(torch.optim, "AdamW", TimedAdamW):
            yield samples
    finally:
        hook.remove()
        if original_refresh is not None:
            # Remove the temporary instance override, restoring normal method binding.
            delattr(model, "refresh_memory")


def timed_stage(model, data, ids, val_ids, memory, args, model_config, started, guard, stage):
    guard.check(stage + "_start")
    core.synchronized(data["x"].device)
    monitor_before = guard.monitor_seconds
    begin = time.perf_counter()
    prefix = "base_" if stage == "base" else ""
    with instrument_stage(model, data["x"].device, guard, stage) as samples:
        best, best_epoch, core_seconds = core.fit_stage(
            model, data, ids, val_ids, memory, args, model_config, started, prefix)
    core.synchronized(data["x"].device)
    elapsed = time.perf_counter() - begin
    monitoring = guard.monitor_seconds - monitor_before
    guard.check(stage + "_finished")
    for kind, rows in samples.items():
        with (args.output / (stage + "_" + kind + "_samples.jsonl")).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(dict(stage=stage, **row)) + "\n")
    optimization = sum(row["seconds"] for row in samples["batches"])
    validation = sum(row["seconds"] for row in samples["validation"])
    cache = sum(row["seconds"] for row in samples["cache"] if row["phase"] == "training")
    return dict(stage=stage, n_train=len(ids), epochs=len(samples["validation"]), best_epoch=best_epoch,
                validation_standardized_mae=best, stage_wall_seconds=elapsed,
                monitor_seconds=monitoring, stage_seconds=elapsed - monitoring,
                optimization_seconds=optimization, validation_seconds=validation, training_cache_seconds=cache,
                bookkeeping_seconds=elapsed - monitoring - optimization - validation - cache,
                core_fit_stage_seconds=core_seconds, optimizer_steps=len(samples["batches"]),
                total_training_event_visits=sum(row["n_events"] for row in samples["batches"]))


def train_benchmark(directory, config, data, events, common, output, options, guard):
    output.mkdir(parents=True)
    seed_all(config["seed"])
    args = SimpleNamespace(**config)
    args.output, args.data = output, options.data
    args.device, args.threads, args.batch_size = options.device, options.threads, options.batch_size
    args.validation_only, args.timeout_seconds = True, options.timeout_seconds
    partitions, cutoff = training_partitions(config, data, common)
    record_training_population(output, partitions, common, config, data, events)
    memory = core.subset(data, common)
    val_ids = (data["split"] == 1).nonzero().flatten()
    device = data["x"].device
    guard.check("before_initialization_" + config["model"])
    core.synchronized(device)
    begin = time.perf_counter()
    model = build_model(config["model"], **config["model_config"]).to(device)
    core.synchronized(device)
    initialization_seconds = time.perf_counter() - begin
    metadata = dict(config, data=str(options.data), output=str(output), validation_only=True,
                    efficiency_only=True, accuracy_recomputed=False, checkpoint_role="same_capacity_efficiency_only",
                    source_main_run=str(directory.resolve()), source_config_hash=core.sha256(directory / "config.json"),
                    training_batch_size=options.batch_size, training_threads=options.threads,
                    original_memory_events=config["n_memory"], n_memory=len(common),
                    memory_policy="common intersection of original train-only memories; maturity rules unchanged",
                    original_training_n_train=config["n_train"], training_partition_sizes={name: len(ids) for name, ids in partitions},
                    ratl_cutoff=cutoff, test_metrics_generated=False,
                    benchmark_script_sha256=core.sha256(Path(__file__)),
                    training_time_scope="optimizer steps, epoch caches, validation, checkpoint I/O, bookkeeping, and RATL gamma selection; measured GPU-monitor call time subtracted",
                    optimization_time_scope="synchronized zero_grad through optimizer.step, including forward, loss, backward and clipping; excludes query gathering and loss logging",
                    stage_clock_note="Per-step synchronization is measurement instrumentation; stage wall and monitor time are also retained")
    core.write_json(output / "config.json", metadata)
    core.write_json(output / "status.json", dict(status="running", mode="efficiency_only", pid=os.getpid(), started_unix=time.time()))
    stages, gamma_seconds, gamma_cache_seconds = [], 0., 0.
    started = time.monotonic()
    if config["model"] != "HistoricalMean":
        for stage, ids in partitions:
            target = model.base if stage == "base" else model
            if stage == "base":
                for parameter in target.parameters():
                    parameter.requires_grad_(True)
            stages.append(timed_stage(target, data, ids, val_ids, memory, args, config["model_config"], started, guard, stage))
            if stage == "base":
                model.freeze_base()
        if isinstance(model, RATL):
            guard.check("gamma_start")
            model.eval()
            core.synchronized(device)
            begin = time.perf_counter()
            cache = model.refresh_memory(memory)
            core.synchronized(device)
            gamma_cache_seconds = time.perf_counter() - begin
            rows = []
            for gamma in np.linspace(0, 1, 11):
                if time.monotonic() - started > args.timeout_seconds:
                    raise TimeoutError("Training benchmark timeout during RATL gamma selection")
                model.gamma.fill_(float(gamma))
                core.synchronized(device)
                start_gamma = time.perf_counter()
                value, _, _ = core.evaluate(model, data, val_ids, memory, args.batch_size, cache)
                core.synchronized(device)
                rows.append(dict(gamma=float(gamma), validation_standardized_mae=value,
                                 seconds=time.perf_counter() - start_gamma))
            chosen = min(rows, key=lambda row: (row["validation_standardized_mae"], row["gamma"]))
            model.gamma.fill_(chosen["gamma"])
            pd.DataFrame(rows).to_csv(output / "gamma_selection.csv", index=False)
            checkpoint = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
            checkpoint.update(model_state=model.state_dict(), validation_mae=chosen["validation_standardized_mae"], gamma=chosen["gamma"])
            torch.save(checkpoint, output / "checkpoint.pt")
            core.synchronized(device)
            gamma_seconds = time.perf_counter() - begin
            guard.check("gamma_finished")
    core.write_json(output / "stage_timing.json", stages)
    record = dict(model=config["model"], seed=config["seed"], efficiency_only=True, accuracy_recomputed=False,
                  source_main_run=str(directory.resolve()), training_run=str(output.resolve()),
                  initialization_seconds=initialization_seconds, training_batch_size=options.batch_size,
                  training_common_memory_events=len(common), original_main_memory_events=config["n_memory"],
                  training_seconds=sum(row["stage_seconds"] for row in stages) + gamma_seconds,
                  training_stage_wall_seconds=sum(row["stage_wall_seconds"] for row in stages) + gamma_seconds,
                  training_monitor_seconds=sum(row["monitor_seconds"] for row in stages),
                  optimization_seconds=sum(row["optimization_seconds"] for row in stages),
                  validation_seconds=sum(row["validation_seconds"] for row in stages),
                  training_cache_seconds=sum(row["training_cache_seconds"] for row in stages),
                  ratl_base_seconds=next((row["stage_seconds"] for row in stages if row["stage"] == "base"), 0.),
                  ratl_residual_seconds=next((row["stage_seconds"] for row in stages if row["stage"] == "residual"), 0.),
                  ratl_gamma_seconds=gamma_seconds, ratl_gamma_cache_seconds=gamma_cache_seconds,
                  total_epochs=sum(row["epochs"] for row in stages),
                  optimizer_steps=sum(row["optimizer_steps"] for row in stages),
                  main_or_residual_train_events=len(partitions[-1][1]),
                  ratl_base_train_events=len(partitions[0][1]) if isinstance(model, RATL) else 0,
                  parameters=sum(p.numel() for p in model.parameters()),
                  precision="float32", threads=options.threads, gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU development")
    core.write_json(output / "status.json", dict(status="completed", mode="efficiency_only", finished_unix=time.time(), **record))
    del model, memory
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


@torch.no_grad()
def batch_one_benchmark(model, data, memory, event_ids, warmup, repeats, guard, timeout_seconds=7200):
    """Every measured call has exactly one event; timings are never divided by B."""
    device = data["x"].device
    queries = [extras.query_inputs(data, ids.reshape(1)) for ids in event_ids]
    if any(len(query["x"]) != 1 or "y" in query for query in queries):
        raise AssertionError("Batch-one queries must have one event and no outcome")
    model.eval()
    core.synchronized(device)
    begin = time.perf_counter()
    cache = model.refresh_memory(memory) if core.retrieves(model) else None
    core.synchronized(device)
    preparation_seconds = time.perf_counter() - begin
    for iteration in range(warmup):
        query = queries[iteration % len(queries)]
        model(query, memory, cache)
        if core.retrieves(model):
            extras.retrieval_only(model, query, memory, cache)
    core.synchronized(device)
    guard.check("batch1_after_warmup")
    started, samples = time.monotonic(), []
    for repeat in range(repeats):
        for position, query in enumerate(queries):
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError("Batch-one inference timeout")
            guard.maybe_check("batch1_measurement")
            core.synchronized(device)
            begin = time.perf_counter()
            model(query, memory, cache)
            core.synchronized(device)
            inference = (time.perf_counter() - begin) * 1000
            retrieval = None
            if core.retrieves(model):
                core.synchronized(device)
                begin = time.perf_counter()
                extras.retrieval_only(model, query, memory, cache)
                core.synchronized(device)
                retrieval = (time.perf_counter() - begin) * 1000
            samples.append(dict(repeat=repeat, dataset_index=int(event_ids[position].item()), batch_size=1,
                                inference_ms=inference, retrieval_ms=retrieval))
    guard.check("batch1_finished")
    values = np.array([row["inference_ms"] for row in samples])
    retrieval_values = np.array([row["retrieval_ms"] for row in samples if row["retrieval_ms"] is not None])
    value_bytes = memory["y"].numel() * memory["y"].element_size()
    canonical = core.cache_bytes(cache) if isinstance(model, RATL) else core.cache_bytes(cache) + value_bytes
    if not core.retrieves(model) and model.__class__.__name__ != "HistoricalMean":
        canonical = 0
    return dict(inference_batch_size=1, inference_event_ms=float(values.mean()),
                inference_sample_std_ms=float(values.std(ddof=1)) if len(values) > 1 else 0.,
                inference_median_ms=float(np.median(values)), inference_p95_ms=float(np.quantile(values, .95)),
                retrieval_event_ms=float(retrieval_values.mean()) if len(retrieval_values) else None,
                retrieval_sample_std_ms=float(retrieval_values.std(ddof=1)) if len(retrieval_values) > 1 else None,
                inference_queries=len(queries), inference_repeats=repeats, inference_samples=len(samples),
                inference_warmup_calls=warmup, memory_preparation_seconds=preparation_seconds,
                canonical_memory_mb=canonical / 1e6,
                memory_input_tensor_mb=core.cache_bytes(memory) / 1e6,
                memory_cache_tensor_mb=core.cache_bytes(cache) / 1e6,
                parameters=sum(p.numel() for p in model.parameters())), samples


def save_summary(output, training_rows, inference_rows):
    training = pd.DataFrame(training_rows)
    inference = pd.DataFrame(inference_rows)
    if len(training):
        training.to_csv(output / "training_timing.csv", index=False)
    if len(inference):
        inference.to_csv(output / "batch1_timing.csv", index=False)
    # Do not join rows into one apparent run: the two model provenances differ.
    summaries = []
    for kind, frame, columns in [
        ("same_capacity_retraining", training, ["training_seconds", "optimization_seconds", "validation_seconds",
            "training_cache_seconds", "ratl_base_seconds", "ratl_residual_seconds", "ratl_gamma_seconds", "initialization_seconds"]),
        ("main_checkpoint_batch1", inference, ["inference_event_ms", "retrieval_event_ms", "canonical_memory_mb", "parameters"])]:
        if not len(frame):
            continue
        for name, group in frame.groupby("model", sort=False):
            for column in columns:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                if len(values):
                    summaries.append(dict(measurement=kind, model=name, metric=column, n_seeds=len(values),
                        mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else None))
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)


def execute(options):
    if options.output.exists() and any(options.output.iterdir()):
        raise FileExistsError("Refusing to overwrite any previous efficiency result")
    if min(options.batch_size, options.expected_memory, options.query_count, options.repeats,
           options.warmup, options.threads, options.timeout_seconds, options.monitor_interval) <= 0:
        raise ValueError("Counts, timeout, and monitor interval must be positive")
    options.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(options.device)
    if device.type == "cuda" and str(extras.physical_device(device)) != "0":
        raise ValueError("This efficiency allocation is physical GPU 0 only")
    torch.set_num_threads(options.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runs = discover_runs(options.suite, options.data)
    cpu_data, events, normal = core.load_data(options.data, torch.device("cpu"))
    common_cpu = common_ids(runs, cpu_data, events, options.expected_memory)
    test_ids = (cpu_data["split"] == 2).nonzero().flatten()
    positions = np.linspace(0, len(test_ids) - 1, min(options.query_count, len(test_ids)), dtype=int)
    query_cpu = test_ids[positions]
    events.iloc[common_cpu.numpy()].assign(dataset_index=common_cpu.numpy()).to_csv(options.output / "common_memory_manifest.csv", index=False)
    events.iloc[query_cpu.numpy()].assign(dataset_index=query_cpu.numpy()).to_csv(options.output / "batch1_query_manifest.csv", index=False)
    protocol = dict(mode=options.mode, benchmark_script_sha256=core.sha256(Path(__file__)),
        code_hashes={name: core.sha256(Path(__file__).with_name(name)) for name in ["models.py", "train.py", "evaluate_extras.py"]},
        dataset_sha256=core.sha256(options.data / "dataset.npz"), normalization_sha256=core.sha256(options.data / "normalization.json"),
        selection_sha256=core.sha256(options.suite / "selected_configs.json"),
        common_memory_sha256=core.sha256(options.output / "common_memory_manifest.csv"),
        options={key: str(value) if isinstance(value, Path) else value for key, value in vars(options).items()},
        n_events=len(events), n_memory=len(common_cpu), n_train=int((cpu_data["split"] == 0).sum()),
        n_validation=int((cpu_data["split"] == 1).sum()), n_test=len(test_ids),
        precision="float32", python_version=platform.python_version(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, started_unix=time.time(), pid=os.getpid(),
        efficiency_only=True, test_metrics_generated=False,
        training_provenance="New from-scratch same-capacity runs, selected main hyperparameters and seeds. Do not associate their time with main-run accuracy.",
        inference_provenance="Original saved main checkpoints, common fixed training memory, actual batch size 1. No test targets passed to model.",
        training_population="Full training split for all single-stage models; RATL earlier base and later residual partitions with its original 11-session separation.",
        training_time_definition="Sum of measured fit stages plus RATL gamma validation/cache/checkpoint stage. Includes validation, training cache refresh, checkpoint I/O and bookkeeping; excludes model initialization and directly measured GPU-monitor call time.",
        retrieval_time_definition="Query-key encoding, mature-candidate mask, similarity and Top-K; excludes response fusion. HistoricalMean has no retrieval timer.",
        inference_time_definition="Synchronized wall time of one resident-on-GPU query through full forward, cache prebuilt; excludes host/device transfer, cache preparation and monitor calls.",
        preprocessing_note="No reference construction is rerun; model initialization and inference memory preparation are separately timed.",
        memory_size_note="canonical_memory_mb counts cached retrieval keys and target values (RATL: cached keys plus residuals); input and cache tensor bytes are also listed. This excludes model weights, workspace, full resident dataset, and allocator/context overhead.",
        exclusivity_limit=("GPU 0 idle before allocation; PID/memory checks and companion GPU idle checks throughout for "
                           + str(options.idle_gpus) + ". Other GPUs and host CPU activity are not controlled. This does not prove whole-host exclusivity."))
    core.write_json(options.output / "protocol.json", protocol)
    guard = QuietGPU(device, options.idle_gpus, options.monitor_interval)
    training_rows, inference_rows = [], []
    core.write_json(options.output / "status.json", dict(status="running", pid=os.getpid(), started_unix=time.time()))
    try:
        with extras.serial_gpu_lock(options.output, guard):
            guard.establish()
            data = {key: value.to(device) for key, value in cpu_data.items()}
            common, query_ids = common_cpu.to(device), query_cpu.to(device)
            memory = core.subset(data, common)
            for directory, config in runs:
                name = config["model"] + "/seed%d" % config["seed"]
                print(json.dumps(dict(run=name, stage="starting", unix=time.time())), flush=True)
                if options.mode in {"all", "train"}:
                    training_rows.append(train_benchmark(directory, config, data, events, common,
                        options.output / "training" / name, options, guard))
                    save_summary(options.output, training_rows, inference_rows)
                if options.mode in {"all", "batch1"}:
                    seed_all(config["seed"])
                    guard.check("before_main_checkpoint_" + name)
                    model = extras.load_checkpoint(directory, config, device)
                    measured, samples = batch_one_benchmark(model, data, memory, query_ids,
                        options.warmup, options.repeats, guard, options.timeout_seconds)
                    inference_rows.append(dict(model=config["model"], seed=config["seed"],
                        source_main_run=str(directory.resolve()), checkpoint_role="original_main_checkpoint",
                        common_memory_events=len(common), original_memory_events=config["n_memory"],
                        checkpoint_sha256=core.sha256(directory / "checkpoint.pt") if config["model"] != "HistoricalMean" else None,
                        efficiency_only=True, accuracy_recomputed=False, precision="float32", threads=options.threads,
                        **measured))
                    path = options.output / "batch1" / name
                    path.mkdir(parents=True)
                    pd.DataFrame(samples).to_csv(path / "samples.csv", index=False)
                    core.write_json(path / "summary.json", inference_rows[-1])
                    del model
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    save_summary(options.output, training_rows, inference_rows)
                core.write_json(options.output / "status.json", dict(status="running", pid=os.getpid(),
                    completed_training_runs=len(training_rows), completed_inference_runs=len(inference_rows), last_run=name))
            guard.check("all_runs_finished")
        core.write_json(options.output / "status.json", dict(status="completed", mode=options.mode,
            training_runs=len(training_rows), inference_runs=len(inference_rows), finished_unix=time.time(),
            no_observed_contention=True if device.type == "cuda" else None, gpu_pid_ownership_method=guard.ownership_method))
    except Exception as error:
        core.write_json(options.output / "status.json", dict(status="failed", error=str(error), traceback=traceback.format_exc(),
            timing_valid=False, pid=os.getpid(), finished_unix=time.time()))
        raise
    finally:
        core.write_json(options.output / "gpu_checks.json", guard.records)
        core.write_json(options.output / "companion_gpu_checks.json", guard.idle_records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["all", "train", "batch1"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expected-memory", type=int, default=912)
    parser.add_argument("--query-count", type=int, default=128, help="Evenly spaced test event positions, same for every model")
    parser.add_argument("--repeats", type=int, default=5, help="Measured passes over all batch-one queries")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--monitor-interval", type=float, default=30)
    parser.add_argument("--idle-gpus", nargs="*", type=int, default=[1, 2, 3])
    return parser.parse_args(argv)


if __name__ == "__main__":
    execute(parse_args())
