"""Evaluate frozen checkpoints without changing their training or source data.

Rolling and frozen-strategy results are supplementary evaluations. Timing uses
one GPU serially and a common intersection of original training memories; it
does not replace the main accuracy results or historical training timers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch

from models import RAIF, RAFT, RATL, build_model, legal_memory
from train import (CHANNELS, cache_bytes, load_data, metric_rows, partition_ratl,
                   retrieves, sha256, subset, synchronized, write_json)


def run_name(directory):
    directory = Path(directory).resolve()
    return directory.name + "_" + __import__("hashlib").sha256(str(directory).encode()).hexdigest()[:8]


def read_run(directory, data_path):
    directory, data_path = Path(directory), Path(data_path)
    config = json.loads((directory / "config.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    if status.get("status") != "completed":
        raise ValueError("Checkpoint run is not completed: " + str(directory))
    for field, filename in [("data_hash", "dataset.npz"), ("normalization_hash", "normalization.json")]:
        if config.get(field) != sha256(data_path / filename):
            raise ValueError("Saved run uses a different dataset/normalization: " + str(directory))
    expected = config.get("code_hashes", {}).get("models.py")
    if expected is not None and expected != sha256(Path(__file__).with_name("models.py")):
        raise ValueError("models.py differs from the source used for the checkpoint")
    return config


def load_checkpoint(directory, config, device):
    model = build_model(config["model"], **config["model_config"]).to(device)
    path = Path(directory) / "checkpoint.pt"
    if path.exists():
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("model_config") != config["model_config"]:
            raise ValueError("Checkpoint and run model configurations differ")
        model.load_state_dict(checkpoint["model_state"], strict=True)
    elif config["model"] != "HistoricalMean":
        raise FileNotFoundError(path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def memory_indices(directory, config, data, events):
    manifest = Path(directory) / "memory_manifest.csv"
    if manifest.exists():
        ids = pd.read_csv(manifest, usecols=["event_id"], dtype={"event_id": str}).event_id
        if ids.duplicated().any():
            raise ValueError("Duplicate saved memory identities")
        position = pd.Series(np.arange(len(events)), index=events.event_id)
        mapped = position.reindex(ids)
        if mapped.isna().any():
            raise ValueError("Saved memory refers to an absent event")
        indices = torch.as_tensor(mapped.to_numpy(np.int64), device=data["x"].device)
    else:
        indices = (data["split"] == 0).nonzero().flatten()
        if config["model"] == "RATL" or config.get("memory_subset") == "ratl":
            _, indices, _ = partition_ratl(data, indices, config.get("ratl_base_fraction", 0.6), 11)
    if not len(indices) or not (data["split"][indices] == 0).all():
        raise ValueError("Initial checkpoint memory must contain only training events")
    expected_count = config.get("n_memory")
    if expected_count is not None and expected_count != len(indices):
        raise ValueError("Reconstructed memory size differs from the saved run")
    return indices


def query_inputs(data, indices):
    # Query outcomes are intentionally absent from the forward-call dictionary.
    return {name: value[indices] for name, value in data.items() if name != "y"}


def promoted_memory(data, indices):
    memory = subset(data, indices)
    memory["origin_split"] = memory["split"].clone()
    memory["split"] = torch.zeros_like(memory["split"])
    return memory


def retrieval_records(result, batch, mem_ids, data, events):
    records = []
    if "selected" not in result:
        return records
    chosen = result["selected"].cpu().numpy()
    valid = result["selected_valid"].cpu().numpy()
    similarity = result["similarity"].cpu().numpy()
    batch, mem_ids = batch.cpu().numpy(), mem_ids.cpu().numpy()
    for row, query_index in enumerate(batch):
        for rank in range(chosen.shape[1]):
            if not valid[row, rank]:
                continue
            historical = int(mem_ids[chosen[row, rank]])
            end = int(data["end_day"][historical].item())
            day = int(data["day"][query_index].item())
            if historical == query_index or end >= day:
                raise AssertionError("Invalid self, same-day, or immature retrieval")
            records.append(dict(query_id=events.event_id.iloc[query_index], historical_id=events.event_id.iloc[historical],
                                query_day=day, historical_day=int(data["day"][historical].item()),
                                historical_end_day=end, origin_split=int(data["split"][historical].item()),
                                rank=rank + 1, similarity=float(similarity[row, rank]),
                                q=float(data["q"][historical].item())))
    return records


@torch.no_grad()
def predict_fixed(model, data, events, initial_ids, batch_size):
    ids = (data["split"] == 2).nonzero().flatten()
    memory = subset(data, initial_ids)
    cache = model.refresh_memory(memory) if retrieves(model) else None
    predictions, records = [], []
    for batch in ids.split(batch_size):
        result = model(query_inputs(data, batch), memory, cache)
        predictions.append(result["prediction"].cpu().numpy())
        records.extend(retrieval_records(result, batch, initial_ids, data, events))
    return ids, np.concatenate(predictions), records, [], []


@torch.no_grad()
def predict_rolling(model, data, events, initial_ids, batch_size):
    """Predict whole dates before admitting any response from that date."""
    if not isinstance(model, RAIF):
        raise ValueError("Rolling evaluation currently supports frozen RAIF checkpoints")
    if not (data["split"][initial_ids] == 0).all():
        raise ValueError("Rolling initial memory must contain only training events")
    device = data["x"].device
    external = (data["split"] != 0).nonzero().flatten()
    processed = torch.zeros(len(data["x"]), dtype=torch.bool, device=device)
    in_memory = torch.zeros_like(processed)
    in_memory[initial_ids] = True
    mem_ids = initial_ids.clone()
    test_predictions, test_ids, records, updates, days = [], [], [], [], []
    for day_tensor in torch.unique(data["day"][external], sorted=True):
        day = int(day_tensor.item())
        # processed excludes all current-day queries, even when a batch is split.
        added = (processed & ~in_memory & (data["end_day"] < day)).nonzero().flatten()
        if len(added):
            if not ((data["split"][added] != 0) & (data["end_day"][added] < day) &
                    (data["day"][added] < day)).all():
                raise AssertionError("Attempted to promote an unavailable external response")
            mem_ids = torch.cat((mem_ids, added))
            in_memory[added] = True
            for index in added.cpu().tolist():
                updates.append(dict(event_id=events.event_id.iloc[index], index=index,
                                    origin_split=int(data["split"][index].item()),
                                    event_day=int(data["day"][index].item()),
                                    end_day=int(data["end_day"][index].item()),
                                    added_before_query_day=day))
        memory = promoted_memory(data, mem_ids)
        cache = model.refresh_memory(memory) if retrieves(model) else None
        on_day = external[data["day"][external] == day]
        eligible = data["end_day"][mem_ids] < day
        days.append(dict(query_day=day, query_count=len(on_day), storage_count=len(mem_ids),
                         mature_count=int(eligible.sum().item()), appended=len(added),
                         mature_train=int(((data["split"][mem_ids] == 0) & eligible).sum().item()),
                         mature_validation=int(((data["split"][mem_ids] == 1) & eligible).sum().item()),
                         mature_test=int(((data["split"][mem_ids] == 2) & eligible).sum().item())))
        for batch in on_day.split(batch_size):
            result = model(query_inputs(data, batch), memory, cache)
            is_test = data["split"][batch] == 2
            if is_test.any():
                test_ids.append(batch[is_test])
                test_predictions.append(result["prediction"][is_test].cpu().numpy())
                if "selected" in result:
                    relevant = {key: value[is_test] for key, value in result.items() if torch.is_tensor(value)}
                    records.extend(retrieval_records(relevant, batch[is_test], mem_ids, data, events))
        processed[on_day] = True
    ids = torch.cat(test_ids)
    predictions = np.concatenate(test_predictions)
    order = torch.argsort(ids)
    return ids[order], predictions[order.cpu().numpy()], records, updates, days


def export_predictions(output, data, events, normal, config, result, mode, strategy, run_directory):
    output.mkdir(parents=True, exist_ok=False)
    ids, normalized, retrieved, updates, days = result
    indices = ids.cpu().numpy()
    chosen = events.iloc[indices].reset_index(drop=True)
    mean, scale = np.asarray(normal["y_mean"]), np.asarray(normal["y_std"])
    actual = data["y"][ids].cpu().numpy().astype(np.float64) * scale + mean
    predicted = normalized.astype(np.float64) * scale + mean
    memory_mode = "rolling" if mode == "rolling" else "fixed"
    kind = "frozen_encoder_rolling_memory" if mode == "rolling" else "frozen_retrieval_replacement"
    metadata = dict(model=config["model"], variant=config.get("variant", "full"), seed=config["seed"],
                    split="test", label_version=str(chosen.label_version.iloc[0]),
                    memory_mode=memory_mode, retrieval=strategy, evaluation_kind=kind,
                    source_run=str(Path(run_directory).resolve()))
    metrics = pd.DataFrame(metric_rows(actual, predicted, chosen, metadata))
    metrics.to_csv(output / "horizon_metrics.csv", index=False)
    metrics[metrics.horizon == 11].to_csv(output / "overall_metrics.csv", index=False)
    n = len(chosen)
    pd.DataFrame(dict(run_id=run_name(run_directory), model=config["model"], seed=config["seed"],
                      split="test", event_id=np.repeat(chosen.event_id.to_numpy(), 33),
                      event_date=np.repeat(chosen.event_date.to_numpy(), 33),
                      tau=np.tile(np.repeat(np.arange(11), 3), n), channel=np.tile(CHANNELS, n * 11),
                      delta_true=actual.reshape(-1), delta_pred=predicted.reshape(-1),
                      q=np.repeat(chosen.q.to_numpy(), 33), label_version=metadata["label_version"],
                      memory_mode=memory_mode, retrieval=strategy, evaluation_kind=kind)).to_csv(output / "predictions.csv", index=False)
    np.savez_compressed(output / "predictions.npz", event_indices=indices, truth=actual, prediction=predicted)
    pd.DataFrame(retrieved, columns=["query_id", "historical_id", "query_day", "historical_day", "historical_end_day",
                                   "origin_split", "rank", "similarity", "q"]).to_csv(output / "retrieved_events.csv", index=False)
    pd.DataFrame(updates, columns=["event_id", "index", "origin_split", "event_day", "end_day",
                                 "added_before_query_day"]).to_csv(output / "memory_updates.csv", index=False)
    pd.DataFrame(days, columns=["query_day", "query_count", "storage_count", "mature_count", "appended",
                              "mature_train", "mature_validation", "mature_test"]).to_csv(output / "memory_by_day.csv", index=False)
    checkpoint = Path(run_directory) / "checkpoint.pt"
    metadata.update(checkpoint_sha256=sha256(checkpoint), dataset_sha256=config["data_hash"],
                    parameters_updated=False, normalization_updated=False,
                    origin_split_preserved=True, n_test=n,
                    validation_memory_rule="Validation events are observed chronologically and promoted only when end_day < query_day",
                    timing_meaning="This mode reports prediction accuracy, not exclusive latency")
    write_json(output / "evaluation_config.json", metadata)
    return metrics


def common_memory(all_indices, maximum=0):
    common = set(all_indices[0].cpu().tolist())
    for indices in all_indices[1:]:
        common.intersection_update(indices.cpu().tolist())
    ids = sorted(common)
    if maximum:
        ids = ids[:maximum]
    if not ids:
        raise ValueError("The saved memories have no common training events")
    return torch.tensor(ids, dtype=torch.long, device=all_indices[0].device)


@torch.no_grad()
def retrieval_only(model, query, memory, cache):
    allowed = legal_memory(query, memory)
    count = min(model.k, len(memory["x"]))
    if isinstance(model, RAIF):
        _, keys = model.encode(query)
        scores = model._similarity(query, memory, keys, cache)
        if model.retrieval == "reverse":
            scores = -scores
        elif model.retrieval == "random":
            qi, mi = query["index"][:, None], memory["index"][None, :]
            scores = ((((qi + 1) * 1103515245 + (mi + 17) * 2654435761)
                       ^ ((qi + 19) * (mi + 31) * 48271)) % 2147483647).float()
        return scores.masked_fill(~allowed, -torch.inf).topk(count, -1).indices
    if isinstance(model, RAFT):
        return [(keys @ historical.T).masked_fill(~allowed, -torch.inf).topk(count, -1).indices
                for keys, historical in zip(model.multiscale(query["x"]), cache)]
    if isinstance(model, RATL):
        keys = model.base.native_keys(query)
        scores = -((keys[:, None] - cache["keys"][None, :]) ** 2).mean(-1).transpose(1, 2)
        allowed = allowed & (memory["end_day"][None, :] < query["day"][:, None] - model.gap)
        return scores.masked_fill(~allowed[:, None, :], -torch.inf).topk(count, -1).indices
    raise TypeError("This model does not retrieve historical candidates")


def gpu_snapshot(index):
    def lines(arguments):
        result = subprocess.run(["nvidia-smi", *arguments], capture_output=True, text=True, check=True, timeout=10)
        return list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
    rows = lines(["--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    match = [row for row in rows if row and (str(row[0]) == str(index) or row[1].startswith(str(index)))]
    if len(match) != 1:
        raise ValueError("Cannot identify the selected physical GPU")
    row = match[0]
    processes = lines(["--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"])
    pids = [int(p[1]) for p in processes if len(p) >= 2 and p[0] == row[1] and p[1].isdigit()]
    return dict(index=int(row[0]), uuid=row[1], name=row[2], memory_used_mib=float(row[3]),
                utilization=float(row[4]), pids=pids, observed_unix=time.time())


def physical_device(device):
    logical = device.index or 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [value.strip() for value in visible.split(",")]
        if logical >= len(entries) or not entries[logical]:
            raise ValueError("CUDA device is outside CUDA_VISIBLE_DEVICES")
        return entries[logical]
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one physical GPU for reproducible exclusive timing")
    return logical


def process_identity():
    """Read every visible PID namespace ID; /proc may expose only the innermost."""
    namespace_pids = []
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("NSpid:"):
                namespace_pids = [int(value) for value in line.split()[1:] if value.isdigit()]
                break
    except OSError:
        pass
    return dict(local_pid=os.getpid(), namespace_pids=namespace_pids,
                pid_candidates=sorted(set([os.getpid(), *namespace_pids])))


@torch.no_grad()
def warm_cuda_libraries(device):
    """Materialize CUDA, cuBLAS and convolution allocations before calibration."""
    torch.cuda.init()
    with torch.cuda.device(device):
        matrix = torch.ones((64, 64), device=device)
        product = matrix @ matrix
        signal = torch.ones((2, 8, 64), device=device)
        kernel = torch.ones((8, 8, 3), device=device)
        convolution = torch.nn.functional.conv1d(signal, kernel, padding=1)
        torch.cuda.synchronize(device)
        del matrix, product, signal, kernel, convolution
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


class ExclusiveGPU:
    """Check idle state and monitor contention; never modify other processes."""
    def __init__(self, device):
        self.device = device
        self.records = []
        self.index = physical_device(device) if device.type == "cuda" else None
        self.context_mib = None
        self.identity = process_identity() if self.index is not None else {}
        self.initial_snapshot = None
        self.owned_reported_pid = None
        self.ownership_method = "not_observed"

    def observed_record(self, stage, record, reserved=None):
        expected = self.context_mib + reserved if self.context_mib is not None and reserved is not None else None
        owned = ({self.owned_reported_pid} if self.owned_reported_pid is not None
                 else set(self.identity.get("pid_candidates", [])))
        return dict(stage=stage, **record, **self.identity,
                    actual_memory_mib=record["memory_used_mib"], torch_reserved_mib=reserved,
                    context_mib=self.context_mib, expected_memory_mib=expected,
                    memory_margin_mib=256., maximum_memory_mib=expected + 256. if expected is not None else None,
                    excess_memory_mib=record["memory_used_mib"] - expected if expected is not None else None,
                    owned_reported_pid=self.owned_reported_pid, pid_ownership_method=self.ownership_method,
                    additional_reported_pids=sorted(set(record["pids"]) - owned),
                    exclusivity_limitation="PID mapping or a unique new PID after an idle start supports process attribution; it does not prove absolute GPU exclusivity")

    def initial(self):
        if self.index is None:
            return
        record = gpu_snapshot(self.index)
        self.records.append(self.observed_record("before_cuda_allocation", record))
        if record["pids"] or record["memory_used_mib"] > 64 or record["utilization"] > 0:
            raise RuntimeError("Selected GPU is not idle; exclusive timing was not started")
        self.initial_snapshot = record

    def establish(self):
        if self.index is None:
            return
        if self.initial_snapshot is None:
            raise RuntimeError("GPU idle state must be checked before context calibration")
        warm_cuda_libraries(self.device)
        record = gpu_snapshot(self.index)
        reported = set(record["pids"])
        mapped = reported & set(self.identity["pid_candidates"])
        if mapped:
            self.owned_reported_pid = min(mapped)
            self.ownership_method = "local_pid" if self.owned_reported_pid == self.identity["local_pid"] else "proc_self_status_NSpid"
        elif len(reported) == 1:
            self.owned_reported_pid = next(iter(reported))
            self.ownership_method = "inferred_unique_new_pid_after_idle_start_and_cuda_warmup"
        elif not reported:
            self.ownership_method = "process_table_empty; own PID not exposed"
        reserved = torch.cuda.memory_reserved(self.device) / 2**20
        self.context_mib = max(0., record["memory_used_mib"] - reserved)
        observed = self.observed_record("owned_cuda_context_after_library_warmup", record, reserved)
        self.records.append(observed)
        if len(reported) > 1 or observed["additional_reported_pids"]:
            raise RuntimeError("Another GPU process appeared before timing")

    def check(self, stage):
        if self.index is None:
            return
        if self.context_mib is None:
            raise RuntimeError("CUDA library allocations must be calibrated before timing checks")
        record = gpu_snapshot(self.index)
        observed = self.observed_record(stage, record, torch.cuda.memory_reserved(self.device) / 2**20)
        self.records.append(observed)
        if observed["additional_reported_pids"] or observed["actual_memory_mib"] > observed["maximum_memory_mib"]:
            raise RuntimeError("GPU contention or unaccounted memory detected; discard this timing run")


@contextmanager
def serial_gpu_lock(output, guard):
    name = "cpu" if guard.index is None else str(guard.index).replace("/", "_")
    lock_path = Path(tempfile.gettempdir()) / ("Sx_timing_" + name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            guard.initial()
            yield
        finally:
            write_json(Path(output) / "gpu_checks.json", guard.records)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def logged_training_cost(directory):
    total, epochs = {}, {}
    for prefix, label in [("base_", "base"), ("", "main")]:
        path = Path(directory) / (prefix + "training_history.csv")
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        total[label] = float(frame.seconds.sum()) if "seconds" in frame else 0.0
        epochs[label] = len(frame)
    return dict(logged_base_epoch_seconds=total["base"], logged_main_epoch_seconds=total["main"],
                logged_training_epoch_seconds=sum(total.values()), base_epochs=epochs["base"], main_epochs=epochs["main"],
                training_timing_exclusive=False, training_timing_source="Original per-epoch logs; includes each epoch's validation and may include shared-GPU contention")


@torch.no_grad()
def benchmark(model, query, memory, batch_size, repeats, warmup, device, guard):
    cache = model.refresh_memory(memory) if retrieves(model) else None
    for _ in range(warmup):
        model(query, memory, cache)
        if retrieves(model):
            retrieval_only(model, query, memory, cache)
    synchronized(device)
    # CUDA library allocations made during warmup belong to this process. The
    # original idle check and process checks precede any measurement.
    guard.check("after_warmup")
    inference, retrieval = [], []
    for repeat in range(repeats):
        synchronized(device)
        begin = time.perf_counter()
        model(query, memory, cache)
        synchronized(device)
        inference.append((time.perf_counter() - begin) * 1000)
        if retrieves(model):
            synchronized(device)
            begin = time.perf_counter()
            retrieval_only(model, query, memory, cache)
            synchronized(device)
            retrieval.append((time.perf_counter() - begin) * 1000)
        if repeat == repeats // 2:
            guard.check("measurement_midpoint")
    guard.check("measurement_finished")
    value_bytes = memory["y"].numel() * memory["y"].element_size()
    canonical = cache_bytes(cache) if isinstance(model, RATL) else cache_bytes(cache) + value_bytes
    if not retrieves(model) and model.__class__.__name__ != "HistoricalMean":
        canonical = 0
    return dict(inference_batch_ms=float(np.mean(inference)), inference_batch_std_ms=float(np.std(inference, ddof=1)) if repeats > 1 else 0.0,
                inference_event_ms=float(np.mean(inference) / batch_size),
                retrieval_batch_ms=float(np.mean(retrieval)) if retrieval else None,
                retrieval_batch_std_ms=float(np.std(retrieval, ddof=1)) if len(retrieval) > 1 else (0.0 if retrieval else None),
                parameters=sum(p.numel() for p in model.parameters()),
                frozen_base_parameters=sum(p.numel() for p in model.base.parameters()) if isinstance(model, RATL) else 0,
                canonical_memory_mb=canonical / 1e6, inference_samples_ms=inference, retrieval_samples_ms=retrieval)


def timing(args, configs, data, events, guard):
    device = data["x"].device
    source_ids = [memory_indices(directory, config, data, events) for directory, config in zip(args.runs, configs)
                  if config["model"] in {"RAIF", "RAFT", "RATL", "HistoricalMean"}]
    if not source_ids:
        source_ids = [(data["split"] == 0).nonzero().flatten()]
    common = common_memory(source_ids, args.memory_size)
    memory = subset(data, common)
    test_ids = (data["split"] == 2).nonzero().flatten()
    actual_batch = min(args.batch_size, len(test_ids))
    query = query_inputs(data, test_ids[:actual_batch])
    events.iloc[common.cpu().numpy()].assign(origin_split=0).to_csv(args.output / "timing_memory_manifest.csv", index=False)
    rows, samples = [], []
    for directory, config in zip(args.runs, configs):
        guard.check("before_" + run_name(directory))
        model = load_checkpoint(directory, config, device)
        measured = benchmark(model, query, memory, actual_batch, args.repeats, args.warmup, device, guard)
        samples.append(dict(source_run=str(Path(directory).resolve()), inference_ms=measured.pop("inference_samples_ms"),
                            retrieval_ms=measured.pop("retrieval_samples_ms")))
        rows.append(dict(model=config["model"], variant=config.get("variant", "full"), seed=config["seed"],
                         source_run=str(Path(directory).resolve()), device=str(device),
                         gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU development run",
                         batch_size=actual_batch, warmup=args.warmup, repeats=args.repeats,
                         common_memory_events=len(common), memory_mode="fixed_common_training_intersection",
                         memory_comparison_policy="Intersection of original train-only checkpoint memories; this latency benchmark does not replace original-memory accuracy evaluation",
                         original_memory_events=len(memory_indices(directory, config, data, events)),
                         original_memory_changed_for_timing=len(memory_indices(directory, config, data, events)) != len(common),
                         accuracy_recomputed=False,
                         inference_exclusive_requested=device.type == "cuda",
                         inference_timing_protocol="serial selected GPU, idle check before allocation, process/memory checks around measurements" if device.type == "cuda" else "CPU development check",
                         gpu_process_visibility=guard.ownership_method if device.type == "cuda" else "not_applicable",
                         owned_reported_pid=guard.owned_reported_pid,
                         gpu_exclusivity_limitation="Idle start and PID/memory checks detect observed contention; an empty table or inferred process attribution does not prove absolute exclusivity" if device.type == "cuda" else "not_applicable",
                         no_contention_detected=True if device.type == "cuda" else None, precision="float32",
                         retrieval_timing_scope="Query-key encoding, temporal admissibility mask, similarity calculation and Top-K; excludes response fusion",
                         **logged_training_cost(directory), **measured))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(args.output / "timing.csv", index=False)
    write_json(args.output / "timing_samples.json", samples)
    write_json(args.output / "gpu_checks.json", guard.records)
    return rows


def execute(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Supplementary output must be a fresh directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.batch_size < 1 or args.repeats < 1 or args.warmup < 1 or args.memory_size < 0:
        raise ValueError("Invalid batch/timing configuration")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    configs = [read_run(directory, args.data) for directory in args.runs]
    guard = ExclusiveGPU(device) if args.mode == "timing" else None
    context = serial_gpu_lock(args.output, guard) if args.mode == "timing" else __import__("contextlib").nullcontext()
    with context:
        if args.mode == "timing":
            guard.establish()
        data, events, normal = load_data(args.data, device)
        if args.mode == "timing":
            timing(args, configs, data, events, guard)
        else:
            tables = []
            for directory, config in zip(args.runs, configs):
                if config["model"] != "RAIF" or config.get("variant") == "no_retrieval":
                    raise ValueError("Rolling/frozen comparisons require a RAIF checkpoint with retrieval")
                initial = memory_indices(directory, config, data, events)
                strategies = [config.get("retrieval", "response")] if args.mode == "rolling" else ["response", "state", "random", "reverse"]
                for strategy in strategies:
                    model = load_checkpoint(directory, config, device)
                    model.retrieval = strategy
                    if args.mode == "rolling":
                        result = predict_rolling(model, data, events, initial, args.batch_size)
                    else:
                        result = predict_fixed(model, data, events, initial, args.batch_size)
                    output = args.output / (run_name(directory) + "_" + strategy)
                    tables.append(export_predictions(output, data, events, normal, config, result,
                                                     args.mode, strategy, directory))
                    del model
            pd.concat(tables, ignore_index=True).to_csv(args.output / "metrics.csv", index=False)
    write_json(args.output / "status.json", dict(status="completed", mode=args.mode,
               runs=[str(Path(path).resolve()) for path in args.runs], finished_unix=time.time(),
               dataset_modified=False, checkpoint_modified=False,
               evaluation_code_sha256=sha256(Path(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", "--runs", dest="runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["rolling", "frozen", "timing"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--memory-size", type=int, default=0, help="0 uses the full common training-memory intersection")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preexisting_nonempty_output = args.output.exists() and any(args.output.iterdir())
    try:
        execute(args)
    except BaseException as error:
        # Never mark partial measurements successful; source runs are untouched.
        if not preexisting_nonempty_output and args.output.exists() and not (args.output / "status.json").exists():
            write_json(args.output / "status.json", dict(status="failed", error=str(error), error_type=type(error).__name__))
        raise
