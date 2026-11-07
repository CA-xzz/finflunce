"""Train one seed/configuration, select on validation, and evaluate once.

Formal execution requires the data builder's explicit readiness flag. This
module never substitutes static attributes for point-in-time data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from models import RAIF, RAFT, RATL, build_model, legal_memory


CHANNELS = ("AR", "turnover", "log_volume")


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronized(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def subset(data, indices):
    return {name: array[indices] for name, array in data.items()}


def retrieves(model):
    return (isinstance(model, RAIF) and model.variant != "no_retrieval") or isinstance(model, (RAFT, RATL))


def cache_bytes(cache):
    if cache is None:
        return 0
    if isinstance(cache, dict):
        return sum(cache_bytes(value) for value in cache.values())
    if isinstance(cache, (list, tuple)):
        return sum(cache_bytes(value) for value in cache)
    return cache.numel() * cache.element_size()


def partition_ratl(data, train_ids, fraction=0.6, gap=11):
    days = torch.unique(data["day"][train_ids], sorted=True)
    cutoff = days[max(1, min(len(days) - 1, int(len(days) * fraction)))]
    early = train_ids[data["end_day"][train_ids] < cutoff]
    late = train_ids[data["day"][train_ids] >= cutoff + gap]
    if min(len(early), len(late)) < 2:
        raise ValueError("Insufficient events for time-separated RATL base and residual stages")
    return early, late, int(cutoff.item())


def objective(result, labels, quality, model, lambda_ret=0.1, huber_delta=1.0):
    per_event = F.huber_loss(result["prediction"], labels, delta=huber_delta, reduction="none").mean((1, 2))
    weights = quality.clamp_min(1e-8) if isinstance(model, RAIF) and model.use_quality else torch.ones_like(quality)
    weights = weights / weights.sum()
    forecast = (weights * per_event).sum()
    retrieval = (weights * result["retrieval_loss"]).sum()
    return forecast + lambda_ret * retrieval, forecast, retrieval


def load_data(directory, device):
    directory = Path(directory)
    normal = json.loads((directory / "normalization.json").read_text())
    if normal.get("ready_for_formal_training") is not True:
        raise ValueError("Formal training blocked: normalization.json must confirm ready_for_formal_training=true after point-in-time data verification")
    raw = dict(np.load(directory / "dataset.npz", allow_pickle=False))
    required = {"x", "y", "q", "platform", "influencer", "day", "end_day", "split"}
    if not required.issubset(raw):
        raise ValueError("Missing arrays: " + str(required - set(raw)))
    n, length, features = raw["x"].shape
    if raw["y"].shape != (n, 11, 3):
        raise ValueError("Expected y[N,11,3]")
    for name in required:
        if len(raw[name]) != n or not np.isfinite(raw[name]).all():
            raise ValueError("Length mismatch or nonfinite input: " + name)
    if not np.isin(raw["split"], [0, 1, 2]).all() or any(np.sum(raw["split"] == s) == 0 for s in (0, 1, 2)):
        raise ValueError("Training, validation, and test partitions must all be nonempty")
    if not np.all(raw["end_day"] == raw["day"] + 10):
        raise ValueError("Response end must equal query trading-day index plus 10")
    if (raw["q"] <= 0).any() or (raw["q"] > 1).any():
        raise ValueError("Reference quality must be in (0,1]")
    for earlier, later in [(0, 1), (1, 2)]:
        if raw["end_day"][raw["split"] == earlier].max() >= raw["day"][raw["split"] == later].min():
            raise ValueError("Response windows cross partition boundaries")
    for name, count in [("platform", "n_platforms"), ("influencer", "n_influencers")]:
        if (raw[name] < 0).any() or (raw[name] >= normal[count]).any():
            raise ValueError("Category outside training vocabulary: " + name)
        training_categories = set(raw[name][raw["split"] == 0].tolist()) | {0}
        if not set(raw[name].tolist()).issubset(training_categories):
            raise ValueError("Evaluation category was not in training; use unknown index 0")
    events = pd.read_csv(directory / "events.csv", dtype={"event_id": str})
    if len(events) != n or events.event_id.duplicated().any():
        raise ValueError("events.csv must have one unique event ID per array row")
    if len(events.label_version.unique()) != 1:
        raise ValueError("A run must use one label version")
    if not np.allclose(events.q.to_numpy(), raw["q"]):
        raise ValueError("events.csv and dataset.npz quality mismatch")
    for name in ("y_mean", "y_std"):
        value = np.asarray(normal[name], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all() or (name == "y_std" and (value <= 0).any()):
            raise ValueError("Invalid channel normalization: " + name)
    raw["index"] = np.arange(n, dtype=np.int64)
    data = {name: torch.as_tensor(value, device=device,
                                 dtype=torch.float32 if name in {"x", "y", "q"} else torch.long)
            for name, value in raw.items() if name in required | {"index"}}
    return data, events, normal


@torch.no_grad()
def evaluate(model, data, indices, memory, batch_size, cache=None, collect=False):
    model.eval()
    if retrieves(model) and cache is None:
        cache = model.refresh_memory(memory)
    predictions, selections = [], []
    for batch in indices.split(batch_size):
        result = model(subset(data, batch), memory, cache)
        predictions.append(result["prediction"])
        if collect and "selected" in result:
            selections.append((batch.cpu().numpy(), result["selected"].cpu().numpy(),
                               result["selected_valid"].cpu().numpy(), result["similarity"].cpu().numpy(),
                               result.get("selection_group", torch.full_like(result["selected"], -1)).cpu().numpy()))
    predicted = torch.cat(predictions)
    error = (predicted - data["y"][indices]).abs().mean().item()
    return error, predicted.cpu().numpy(), selections


def metric_rows(truth, predictions, events, base, horizons=(1, 3, 5, 10, 11)):
    rows = []
    for horizon in horizons:
        error = predictions[:, :horizon] - truth[:, :horizon]
        for channel, name in enumerate(CHANNELS):
            for metric, value in [("MAE", np.abs(error[:, :, channel]).mean()),
                                  ("RMSE", np.sqrt(np.square(error[:, :, channel]).mean()))]:
                rows.append(dict(base, channel=name, horizon=horizon, metric=metric,
                                 value=float(value), n_events=len(events), n_dates=events.event_date.nunique()))
    return rows


def save_outputs(output, model, data, events, norm, ids, predictions, selections, memory_ids, args):
    event_indices = ids.cpu().numpy()
    selected_events = events.iloc[event_indices].reset_index(drop=True)
    mean, std = np.asarray(norm["y_mean"]), np.asarray(norm["y_std"])
    truth = data["y"][ids].cpu().numpy().astype(np.float64) * std + mean
    predictions = predictions.astype(np.float64) * std + mean
    label_version = str(events.label_version.iloc[0])
    base = dict(model=args.model, variant=args.variant, seed=args.seed, split="test",
                label_version=label_version, memory_mode="fixed", retrieval=args.retrieval)
    metrics = pd.DataFrame(metric_rows(truth, predictions, selected_events, base))
    metrics.to_csv(output / "horizon_metrics.csv", index=False)
    metrics[metrics.horizon == 11].to_csv(output / "overall_metrics.csv", index=False)
    steps = []
    for tau in range(11):
        for row in metric_rows(truth[:, tau:tau + 1], predictions[:, tau:tau + 1], selected_events, base, horizons=(1,)):
            row.pop("horizon")
            row["tau"] = tau
            steps.append(row)
    pd.DataFrame(steps).to_csv(output / "step_metrics.csv", index=False)
    n = len(selected_events)
    long = pd.DataFrame({
        "run_id": args.model + "_" + args.variant + "_seed" + str(args.seed),
        "model": args.model, "seed": args.seed, "split": "test",
        "event_id": np.repeat(selected_events.event_id.to_numpy(), 33),
        "event_date": np.repeat(selected_events.event_date.to_numpy(), 33),
        "tau": np.tile(np.repeat(np.arange(11), 3), n),
        "channel": np.tile(CHANNELS, n * 11),
        "delta_true": truth.reshape(-1), "delta_pred": predictions.reshape(-1),
        "q": np.repeat(selected_events.q.to_numpy(), 33),
        "label_version": label_version, "memory_mode": "fixed"})
    long.to_csv(output / "predictions.csv", index=False)
    np.savez_compressed(output / "predictions.npz", event_indices=event_indices, truth=truth, prediction=predictions)
    records = []
    mem_idx = memory_ids.cpu().numpy()
    for batch, selected, valid, similarities, groups in selections:
        for pos, qi in enumerate(batch):
            for rank in range(selected.shape[1]):
                if valid[pos, rank]:
                    hi = mem_idx[selected[pos, rank]]
                    records.append(dict(query_id=events.event_id.iloc[qi], historical_id=events.event_id.iloc[hi],
                                        rank=rank + 1, similarity=float(similarities[pos, rank]),
                                        q=float(events.q.iloc[hi]),
                                        selection_group=int(groups[pos, rank]),
                                        end_day=int(data["end_day"][hi].item())))
    pd.DataFrame(records, columns=["query_id", "historical_id", "rank", "similarity", "q", "selection_group", "end_day"]).to_csv(
        output / "retrieved_events.csv", index=False)
    memory_frame = events.iloc[mem_idx].copy()
    memory_frame["day"] = data["day"][memory_ids].cpu().numpy()
    memory_frame["end_day"] = data["end_day"][memory_ids].cpu().numpy()
    memory_frame.to_csv(output / "memory_manifest.csv", index=False)
    auxiliary = {
        "ar_direction_accuracy": float((np.sign(predictions[:, :, 0]) == np.sign(truth[:, :, 0])).mean()),
        "peak_time_mae": {}, "near_flat_true_events": {},
    }
    for c, name in enumerate(CHANNELS):
        amplitude = np.abs(truth[:, :, c]).max(1)
        nonflat = amplitude > max(1e-12, std[c] * 1e-4)
        auxiliary["near_flat_true_events"][name] = int((~nonflat).sum())
        auxiliary["peak_time_mae"][name] = (float(np.abs(np.abs(predictions[nonflat, :, c]).argmax(1)
                                                                      - np.abs(truth[nonflat, :, c]).argmax(1)).mean())
                                                if nonflat.any() else None)
    write_json(output / "auxiliary_metrics.json", auxiliary)
    quality_cuts = np.quantile(data["q"][data["split"] == 0].cpu().numpy(), [1 / 3, 2 / 3])
    group_columns = {
        "period": pd.to_datetime(selected_events.event_date).dt.to_period("Q").astype(str).to_numpy(),
        "platform": selected_events.platform.fillna("unknown").astype(str).to_numpy(),
        "quality": np.asarray(["low", "middle", "high"])[np.searchsorted(quality_cuts, selected_events.q.to_numpy())],
        "influencer_status": np.where(data["influencer"][ids].cpu().numpy() == 0, "unknown_or_unseen", "seen"),
    }
    group_rows, counts = [], []
    for group_type, assignments in group_columns.items():
        for group in np.unique(assignments):
            mask = assignments == group
            sub = selected_events[mask]
            sufficient = len(sub) >= 20 and sub.event_date.nunique() >= 5
            counts.append(dict(group_type=group_type, group=group, n_events=len(sub),
                               n_dates=sub.event_date.nunique(), report_metrics=sufficient))
            if sufficient:
                group_rows.extend(metric_rows(truth[mask], predictions[mask], sub,
                                               dict(base, group_type=group_type, group=group), horizons=(11,)))
    pd.DataFrame(counts).to_csv(output / "group_counts.csv", index=False)
    pd.DataFrame(group_rows, columns=list(base) + ["group_type", "group", "channel", "horizon", "metric", "value", "n_events", "n_dates"]).to_csv(output / "group_metrics.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["RAIF", "ECT", "HistoricalMean", "DLinear", "PatchTST", "iTransformer", "RAFT", "RATL"], default="RAIF")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--variant", choices=sorted(RAIF.VARIANTS), default="full")
    parser.add_argument("--retrieval", choices=["response", "state", "random", "reverse"], default="response")
    parser.add_argument("--t-delta", type=float, default=1.0)
    parser.add_argument("--t-z", type=float, default=0.1)
    parser.add_argument("--t-r", type=float, default=0.2)
    parser.add_argument("--lambda-q", type=float, default=1.0)
    parser.add_argument("--lambda-ret", type=float, default=0.1)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--validation-only", action="store_true", help="Select checkpoints using validation; never evaluate test data")
    parser.add_argument("--raft-temperature", type=float, default=0.1)
    parser.add_argument("--ratl-base-fraction", type=float, default=0.6)
    parser.add_argument("--ratl-block", type=int, default=8)
    parser.add_argument("--ratl-temperature", type=float, default=1.0)
    parser.add_argument("--ratl-teacher-temperature", type=float, default=1.0)
    parser.add_argument("--ratl-teacher-weight", type=float, default=0.4)
    parser.add_argument("--ratl-gap", type=int, default=11, help="Extra trading days after a residual becomes mature")
    parser.add_argument("--memory-subset", choices=["all", "ratl"], default="all", help="For RAIF: match RATL residual-memory event identities")
    return parser.parse_args()


def fit_stage(model, data, train_ids, val_ids, memory, args, model_config, started, prefix=""):
    """A single stage; both RATL stages share this validation-only selection."""
    device = data["x"].device
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        best, _, _ = evaluate(model, data, val_ids, memory, args.batch_size)
        return best, 0, 0.0
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    history, best, best_epoch = [], float("inf"), 0
    stage_start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        cache = model.refresh_memory(memory) if retrieves(model) else None
        order = train_ids[torch.randperm(len(train_ids), device=device)]
        summed = np.zeros(3)
        seen = 0
        for batch in order.split(args.batch_size):
            if time.monotonic() - started > args.timeout_seconds:
                print("Hard timeout reached; stopping this run.", flush=True)
                raise TimeoutError("Configured hard timeout reached")
            query = subset(data, batch)
            optimizer.zero_grad(set_to_none=True)
            result = model(query, memory, cache, labels=query["y"])
            coefficient = args.ratl_teacher_weight if isinstance(model, RATL) else args.lambda_ret
            loss, forecast, retrieval = objective(result, query["y"], query["q"], model, coefficient, args.huber_delta)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
            optimizer.step()
            summed += np.array([loss.item(), forecast.item(), retrieval.item()]) * len(batch)
            seen += len(batch)
        val_mae, _, _ = evaluate(model, data, val_ids, memory, args.batch_size)
        if not np.isfinite(val_mae):
            raise FloatingPointError("Nonfinite validation MAE")
        row = dict(stage=prefix or "main", epoch=epoch, loss=summed[0] / seen,
                   forecast_loss=summed[1] / seen, retrieval_loss=summed[2] / seen,
                   validation_mae=val_mae, seconds=time.monotonic() - epoch_start)
        history.append(row)
        pd.DataFrame(history).to_csv(args.output / (prefix + "training_history.csv"), index=False)
        print(json.dumps(row), flush=True)
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            temporary = args.output / (prefix + "checkpoint.tmp")
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_mae": best,
                        "model_config": model_config}, temporary)
            temporary.replace(args.output / (prefix + "checkpoint.pt"))
        if epoch - best_epoch >= args.patience:
            break
    checkpoint = torch.load(args.output / (prefix + "checkpoint.pt"), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    synchronized(device)
    return best, best_epoch, time.monotonic() - stage_start


def run(args):
    # Defaults also keep direct programmatic callers compatible with the CLI.
    for name, value in dict(validation_only=False, raft_temperature=0.1, ratl_base_fraction=0.6,
                            ratl_block=8, ratl_temperature=1.0, ratl_teacher_temperature=1.0,
                            ratl_teacher_weight=0.4, ratl_gap=11, memory_subset="all").items():
        if not hasattr(args, name):
            setattr(args, name, value)
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Epochs, batch size, and patience must be positive")
    if args.timing_repeats < 1 or not 0 < args.ratl_base_fraction < 1:
        raise ValueError("Timing repeats must be positive; RATL base fraction must be in (0,1)")
    if min(args.t_delta, args.t_z, args.t_r, args.raft_temperature, args.ratl_temperature,
           args.ratl_teacher_temperature, args.huber_delta) <= 0 or args.ratl_block < 1:
        raise ValueError("Temperatures, Huber delta, and block size must be positive")
    if min(args.lambda_q, args.lambda_ret, args.ratl_teacher_weight, args.ratl_gap) < 0:
        raise ValueError("Loss weights and temporal gap must be nonnegative")
    if args.model != "RAIF" and (args.variant != "full" or args.retrieval != "response"):
        raise ValueError("Ablations and retrieval strategy apply only to RAIF")
    if (args.output / "checkpoint.pt").exists() or (args.output / "status.json").exists():
        raise FileExistsError("Output already contains a run; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    data, events, normal = load_data(args.data, device)
    train_ids, val_ids, test_ids = [(data["split"] == value).nonzero().flatten() for value in (0, 1, 2)]
    original_train_ids = train_ids
    memory_ids = train_ids
    base_ids, cutoff = None, None
    if args.model == "RATL" or args.memory_subset == "ratl":
        base_ids, residual_ids, cutoff = partition_ratl(data, train_ids, args.ratl_base_fraction, 11)
        memory_ids = residual_ids
        if args.model == "RATL":
            train_ids = residual_ids
        elif args.model != "RAIF":
            raise ValueError("RATL memory capacity matching is supported for RAIF only")
    memory = subset(data, memory_ids)
    model_config = dict(length=data["x"].shape[1], features=data["x"].shape[2], horizon=11, channels=3,
                        n_platforms=normal["n_platforms"], n_influencers=normal["n_influencers"],
                        hidden=args.hidden, heads=args.heads, layers=args.layers, key_dim=args.key_dim,
                        k=args.k, kernel=args.kernel, dropout=args.dropout, variant=args.variant,
                        retrieval=args.retrieval, t_delta=args.t_delta, t_z=args.t_z, t_r=args.t_r, lambda_q=args.lambda_q,
                        raft_temperature=args.raft_temperature, ratl_block=args.ratl_block,
                        ratl_temperature=args.ratl_temperature, ratl_teacher_temperature=args.ratl_teacher_temperature,
                        ratl_gap=args.ratl_gap)
    model = build_model(args.model, **model_config).to(device)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(model_config=model_config, torch_version=torch.__version__, numpy_version=np.__version__,
                    python_version=platform.python_version(), cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    n_train=len(train_ids), n_training_partition=len(original_train_ids),
                    n_memory=len(memory_ids), n_validation=len(val_ids), n_test=len(test_ids),
                    quality_used_in_training=isinstance(model, RAIF) and model.use_quality,
                    ret_candidate_scope="all mature training events", precision="float32",
                    data_hash=sha256(args.data / "dataset.npz"),
                    normalization_hash=sha256(args.data / "normalization.json"),
                    code_hashes={name: sha256(Path(__file__).with_name(name)) for name in ["models.py", "train.py"]},
                    baseline_implementation="local task adaptation; increment targets and event-conditioned output head",
                    baseline_adaptations={
                        "DLinear": "Shared temporal seasonal/trend projections, replication-padded moving average, learned feature-to-target projection and categorical output head",
                        "PatchTST": "Channel-independent patch encoder, patch length min(4,L), stride max(1,patch/2), learned feature-to-target projection; training-only global scaling rather than RevIN",
                        "iTransformer": "Variate tokens, temporal embedding and temporal projection, learned feature-to-target projection; training-only global scaling rather than instance output denormalization",
                        "ECT": "Temporal transformer with event-condition projection added to each input token, mean pooling, HxC forecast head and categorical output head",
                        "RAFT": "Official multiscale Pearson/Top-K/linear-fusion structure; event-window memory, three-channel increment values retain their level, no market-level output restoration; Huber objective; periods 1,2,4 use floor-length average pooling",
                        "RATL": "Public-equation adaptation, not exact unpublished J5: explicit query/residual-block/Direct-reference/position tokens, omitted unspecified scalar feature channels; target-channel keys project frozen variate tokens through learned output weights; Huber prediction plus teacher CE; standardized validation MAE selects gamma; chronological out-of-sample residuals",
                    }.get(args.model),
                    source_urls={
                        "RAFT": ["https://arxiv.org/html/2505.04163v1", "https://github.com/archon159/RAFT/blob/main/models/RAFT.py", "https://github.com/archon159/RAFT/blob/main/layers/Retrieval.py"],
                        "RATL": ["https://arxiv.org/html/2609.03937v1", "https://arxiv.org/src/2609.03937v1"],
                    }.get(args.model, []),
                    ratl_internal_cutoff=cutoff,
                    ratl_base_count=len(base_ids) if base_ids is not None else None,
                    group_min_events=20, group_min_dates=5)
    write_json(args.output / "config.json", metadata)
    write_json(args.output / "status.json", {"status": "running", "pid": os.getpid(), "started_unix": time.time()})
    started = time.monotonic()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    base_seconds, base_best = 0.0, None
    if isinstance(model, RATL):
        for parameter in model.base.parameters():
            parameter.requires_grad_(True)
        base_best, base_epoch, base_seconds = fit_stage(model.base, data, base_ids, val_ids,
                                                       subset(data, base_ids), args, model_config, started, "base_")
        model.freeze_base()
        phase = events.iloc[original_train_ids.cpu().numpy()][["event_id", "event_date"]].copy()
        phase["phase"] = "boundary_excluded"
        phase.loc[base_ids.cpu().numpy(), "phase"] = "base_fit"
        phase.loc[train_ids.cpu().numpy(), "phase"] = "residual_fit"
        phase.to_csv(args.output / "training_stages.csv", index=False)
        write_json(args.output / "base_status.json", {"best_epoch": base_epoch,
                   "validation_standardized_mae": base_best, "training_seconds": base_seconds,
                   "n_base_train": len(base_ids), "n_residual_train": len(train_ids),
                   "boundary_day": cutoff, "boundary_gap": 11})
    best, best_epoch, correction_seconds = fit_stage(model, data, train_ids, val_ids, memory, args, model_config, started)
    synchronized(device)
    training_seconds = time.monotonic() - started
    model.eval()
    cache = model.refresh_memory(memory) if retrieves(model) else None
    if isinstance(model, RATL):
        gamma_rows = []
        for gamma in np.linspace(0, 1, 11):
            model.gamma.fill_(float(gamma))
            value, _, _ = evaluate(model, data, val_ids, memory, args.batch_size, cache)
            gamma_rows.append({"gamma": float(gamma), "validation_standardized_mae": value})
        chosen = min(gamma_rows, key=lambda row: (row["validation_standardized_mae"], row["gamma"]))
        model.gamma.fill_(chosen["gamma"])
        best = chosen["validation_standardized_mae"]
        pd.DataFrame(gamma_rows).to_csv(args.output / "gamma_selection.csv", index=False)
        checkpoint = torch.load(args.output / "checkpoint.pt", map_location=device, weights_only=True)
        checkpoint.update(model_state=model.state_dict(), validation_mae=best, gamma=chosen["gamma"])
        torch.save(checkpoint, args.output / "checkpoint.pt")
    validation_summary = dict(status="completed", mode="validation_only" if args.validation_only else "full",
                              best_epoch=best_epoch, validation_standardized_mae=best,
                              base_validation_standardized_mae=base_best, base_training_seconds=base_seconds,
                              correction_training_seconds=correction_seconds, training_seconds=training_seconds,
                              gamma=float(model.gamma.item()) if isinstance(model, RATL) else None,
                              n_memory=len(memory_ids), total_seconds=time.monotonic() - started,
                              finished_unix=time.time())
    write_json(args.output / "validation_summary.json", validation_summary)
    if args.validation_only:
        write_json(args.output / "status.json", validation_summary)
        print(json.dumps(validation_summary), flush=True)
        return
    test_mae, predictions, selections = evaluate(model, data, test_ids, memory, args.batch_size, cache, collect=True)
    save_outputs(args.output, model, data, events, normal, test_ids, predictions, selections, memory_ids, args)
    benchmark_ids = test_ids[:args.batch_size]
    query = subset(data, benchmark_ids)
    samples, retrieval_times = [], []
    with torch.no_grad():
        for iteration in range(args.timing_repeats + 3):
            synchronized(device)
            start = time.perf_counter()
            model(query, memory, cache)
            synchronized(device)
            elapsed = time.perf_counter() - start
            if iteration >= 3:
                samples.append(elapsed)
            if retrieves(model):
                synchronized(device)
                start = time.perf_counter()
                allowed = legal_memory(query, memory)
                count = min(args.k, len(memory_ids))
                if isinstance(model, RAIF):
                    _, keys = model.encode(query)
                    similarities = model._similarity(query, memory, keys, cache)
                    if model.retrieval == "reverse":
                        similarities = -similarities
                    elif model.retrieval == "random":
                        qi, mi = query["index"][:, None], memory["index"][None, :]
                        similarities = ((((qi + 1) * 1103515245 + (mi + 17) * 2654435761)
                                         ^ ((qi + 19) * (mi + 31) * 48271)) % 2147483647).float()
                    similarities.masked_fill(~allowed, -torch.inf).topk(count, -1)
                elif isinstance(model, RAFT):
                    for query_keys, historical_keys in zip(model.multiscale(query["x"]), cache):
                        (query_keys @ historical_keys.T).masked_fill(~allowed, -torch.inf).topk(count, -1)
                else:
                    keys = model.base.native_keys(query)
                    scores = -((keys[:, None] - cache["keys"][None, :]) ** 2).mean(-1).transpose(1, 2)
                    allowed = allowed & (memory["end_day"][None, :] < query["day"][:, None] - model.gap)
                    scores.masked_fill(~allowed[:, None, :], -torch.inf).topk(count, -1)
                synchronized(device)
                if iteration >= 3:
                    retrieval_times.append(time.perf_counter() - start)
    memory_bytes = sum(value.numel() * value.element_size() for value in memory.values()) if retrieves(model) or args.model == "HistoricalMean" else 0
    key_bytes = cache_bytes(cache)
    canonical_memory_bytes = (key_bytes if isinstance(model, RATL)
                              else key_bytes + memory["y"].numel() * memory["y"].element_size())
    if not retrieves(model) and args.model != "HistoricalMean":
        canonical_memory_bytes = 0
    pd.DataFrame([dict(model=args.model, variant=args.variant, seed=args.seed, parameters=parameters,
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       frozen_base_parameters=sum(p.numel() for p in model.base.parameters()) if isinstance(model, RATL) else 0,
                       checkpoint_bytes=(args.output / "checkpoint.pt").stat().st_size if (args.output / "checkpoint.pt").exists() else 0)]).to_csv(args.output / "model_size.csv", index=False)
    pd.DataFrame([dict(model=args.model, variant=args.variant, seed=args.seed,
                       parameters=parameters, training_seconds=training_seconds,
                       base_training_seconds=base_seconds, correction_training_seconds=correction_seconds,
                       frozen_base_parameters=sum(p.numel() for p in model.base.parameters()) if isinstance(model, RATL) else 0,
                       inference_batch_ms=float(np.mean(samples) * 1000),
                       inference_event_ms=float(np.mean(samples) * 1000 / len(benchmark_ids)),
                       retrieval_batch_ms=float(np.mean(retrieval_times) * 1000) if retrieval_times else None,
                       timing_batch_size=len(benchmark_ids), repetitions=args.timing_repeats,
                       memory_total_mb=(memory_bytes + key_bytes) / 1e6,
                       key_and_response_memory_mb=canonical_memory_bytes / 1e6)]).to_csv(args.output / "timing.csv", index=False)
    final = dict(validation_summary, status="completed", best_epoch=best_epoch, validation_standardized_mae=best,
                 test_standardized_mae=test_mae, n_test=len(test_ids),
                 total_seconds=time.monotonic() - started, finished_unix=time.time())
    write_json(args.output / "status.json", final)
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    args = parse_args()
    existing_run = (args.output / "status.json").exists() or (args.output / "checkpoint.pt").exists()
    try:
        run(args)
    except BaseException as error:
        # Preserve a completed run if the caller accidentally reused its path.
        status_path = args.output / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if not existing_run:
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(status_path, {"status": "failed", "error": str(error),
                                     "error_type": type(error).__name__, "finished_unix": time.time()})
            (args.output / "failure.txt").write_text(traceback.format_exc())
        raise
