"""Train a full-trajectory component without changing the primary experiment.

The reference ECT is selected first. Both Direct methods use its frozen
predictions; their own checkpoints are selected on validation Delta error.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

import train as core
from models import DirectForecaster, RAFT

QUERY_FIELDS = ("x", "platform", "influencer", "day", "index")


class StateOnlyECT(DirectForecaster):
    def forward(self, query, memory=None, cached_keys=None, labels=None):
        inputs = {key: query[key] for key in QUERY_FIELDS if key in query}
        return super().forward(inputs, memory=None, cached_keys=None, labels=None)


class StateOnlyRAFT(RAFT):
    def forward(self, query, memory=None, cached_keys=None, labels=None):
        inputs = {key: query[key] for key in QUERY_FIELDS if key in query}
        return super().forward(inputs, memory, cached_keys, labels=None)


def build_component(model_name, configuration):
    if model_name == "ECT":
        return StateOnlyECT("ECT", **configuration)
    if model_name == "RAFT":
        return StateOnlyRAFT(**configuration)
    raise ValueError(model_name)


def incremental_validation(raw_prediction, reference_prediction, canonical_delta, delta_std):
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (raw_prediction, reference_prediction, canonical_delta)]
    if arrays[0].shape != arrays[1].shape or arrays[0].shape != arrays[2].shape:
        raise ValueError("Unaligned validation arrays")
    scale = np.asarray(delta_std, dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not (scale > 0).all() or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Invalid validation arrays/scales")
    return float(np.abs((arrays[0] - arrays[1] - arrays[2]) / scale).mean())


@torch.no_grad()
def predict(model, data, indices, memory, batch_size, cache=None, collect=False):
    model.eval()
    if core.retrieves(model) and cache is None:
        cache = model.refresh_memory(memory)
    predictions, selected_rows = [], []
    for batch in indices.split(batch_size):
        # Do not pass query outcomes even though they exist in the dataset.
        query = {key: data[key][batch] for key in QUERY_FIELDS if key in data}
        result = model(query, memory, cache, labels=None)
        predictions.append(result["prediction"].cpu().numpy())
        if collect and "selected" in result:
            selected_rows.append((batch.cpu().numpy(), result["selected"].cpu().numpy(),
                result["selected_valid"].cpu().numpy(), result["similarity"].cpu().numpy(),
                result["selection_group"].cpu().numpy()))
    return np.concatenate(predictions), selected_rows


@contextmanager
def validation_evaluator(data, normal, val_ids, reference_prediction, canonical_delta, delta_std):
    original = core.evaluate
    def evaluate(model, current_data, indices, memory, batch_size, cache=None, collect=False):
        if current_data is not data or not torch.equal(indices, val_ids) or collect:
            raise ValueError("Fit-stage evaluation must use validation only")
        prediction, selections = predict(model, current_data, indices, memory, batch_size, cache)
        if reference_prediction is None:
            # Same standardized MAE definition as the existing training loop.
            p = torch.as_tensor(prediction, device=indices.device)
            value = float((p - current_data["y"][indices]).abs().mean().item())
        else:
            raw = prediction.astype(np.float64) * np.asarray(normal["y_std"]) + np.asarray(normal["y_mean"])
            value = incremental_validation(raw, reference_prediction, canonical_delta, delta_std)
        return value, prediction, selections
    core.evaluate = evaluate
    try:
        yield
    finally:
        core.evaluate = original


def read_json(path):
    return json.loads(Path(path).read_text())


def execute(options):
    root = options.root.resolve()
    model_name = "RAFT" if options.role == "raft_direct" else "ECT"
    primary = root / "data/prepared_expanded"
    source_run = root / "results/suite_expanded/main" / model_name / f"seed{options.seed}"
    source = read_json(source_run / "config.json")
    assert read_json(source_run / "status.json")["status"] == "completed"
    assert source["seed"] == options.seed and source["model"] == model_name
    assert source["data_hash"] == core.sha256(primary / "dataset.npz")
    assert source["normalization_hash"] == core.sha256(primary / "normalization.json")
    for name in ("models.py", "train.py"):
        assert source["code_hashes"][name] == core.sha256(root / name), "Frozen implementation changed"
    directory = root / "data/trajectory_comparison" / ("reference_full" if options.role == "reference" else "recommended_full")
    output = root / "results/trajectory_comparison/components" / options.role / f"seed{options.seed}"
    code_hashes = {name: core.sha256(root / name) for name in
                   ("models.py", "train.py", "train_trajectory_component.py")}
    expected = dict(role=options.role, model=model_name, seed=options.seed,
        data_hash=core.sha256(directory / "dataset.npz"),
        normalization_hash=core.sha256(directory / "normalization.json"),
        primary_data_hash=source["data_hash"], primary_normalization_hash=source["normalization_hash"],
        source_main_config_sha256=core.sha256(source_run / "config.json"), code_hashes=code_hashes)
    reference_run = root / "results/trajectory_comparison/components/reference" / f"seed{options.seed}"
    if options.role != "reference":
        reference_status = read_json(reference_run / "status.json")
        assert reference_status["status"] == "completed"
        reference_config = read_json(reference_run / "config.json")
        assert reference_config["primary_data_hash"] == source["data_hash"]
        assert reference_config["seed"] == options.seed and reference_config["role"] == "reference"
        assert reference_config["code_hashes"] == code_hashes
        expected["reference_checkpoint_sha256"] = core.sha256(reference_run / "checkpoint.pt")
        expected["reference_validation_predictions_sha256"] = core.sha256(reference_run / "validation_predictions.npz")
        expected["reference_test_predictions_sha256"] = core.sha256(reference_run / "test_predictions.npz")
        for name in ("checkpoint", "validation_predictions", "test_predictions"):
            assert expected["reference_" + name + "_sha256"] == reference_status[name + "_sha256"]
    if (output / "status.json").exists():
        status, old = read_json(output / "status.json"), read_json(output / "config.json")
        assert status["status"] == "completed", "Partial run requires inspection before retry"
        assert all(old.get(key) == value for key, value in expected.items()), "Resume provenance differs"
        assert status["checkpoint_sha256"] == core.sha256(output / "checkpoint.pt")
        for split in ("validation", "test"):
            assert status[split + "_predictions_sha256"] == core.sha256(output / f"{split}_predictions.npz")
        print(json.dumps(dict(status="already_completed", role=options.role, seed=options.seed)), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(**source)
    args.data, args.output, args.seed = directory, output, options.seed
    args.device, args.threads = options.device, options.threads
    args.timeout_seconds = options.timeout_seconds
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    data, events, normal = core.load_data(directory, device)
    train_ids, val_ids, test_ids = [(data["split"] == i).nonzero().flatten() for i in (0, 1, 2)]
    assert [len(train_ids), len(val_ids), len(test_ids)] == [2170, 620, 2315]
    primary_normal = read_json(primary / "normalization.json")
    with np.load(primary / "dataset.npz", allow_pickle=False) as original:
        canonical = original["y"].astype(np.float64) * np.asarray(primary_normal["y_std"]) + np.asarray(primary_normal["y_mean"])
        np.testing.assert_array_equal(original["split"], data["split"].cpu().numpy())
        if options.role != "reference":
            np.testing.assert_array_equal(original["x"], data["x"].cpu().numpy())
            np.testing.assert_array_equal(normal["y_std"], primary_normal["y_std"])
    if options.role == "reference":
        assert not bool(data["platform"].any()) and not bool(data["influencer"].any())
    b_validation = None
    if options.role != "reference":
        with np.load(reference_run / "validation_predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["event_indices"], val_ids.cpu().numpy())
            b_validation = archive["prediction"].copy()
    model_config = source["model_config"]
    assert (model_config["length"], model_config["horizon"], model_config["channels"], model_config["k"]) == (10, 11, 3, 5)
    model = build_component(model_name, model_config).to(device)
    memory = core.subset(data, train_ids)
    metadata = dict(expected, data=str(directory), output=str(output), source_main_run=str(source_run),
        model_config=model_config, reference_run=str(reference_run) if options.role != "reference" else None,
        target="B" if options.role == "reference" else "Y", n_train=2170, n_validation=620, n_test=2315,
        n_memory=2170 if model_name == "RAFT" else 0,
        optimizer_settings={name: getattr(args, name) for name in
            ("epochs", "patience", "batch_size", "lr", "weight_decay", "huber_delta", "lambda_ret")},
        validation_objective="standardized_B_MAE" if options.role == "reference" else "standardized_incremental_MAE_with_frozen_predicted_B",
        query_fields=list(QUERY_FIELDS), query_future_inputs=False, true_reference_future_used_for_prediction=False,
        training_target_scale="train_control_B_std" if options.role == "reference" else "primary_train_Delta_std",
        retrieval="fixed multiscale Pearson, Top-5 per scale 1/2/4, mature training events only" if model_name == "RAFT" else "none",
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"), threads=options.threads,
        parameters=sum(p.numel() for p in model.parameters()), newly_added_experiment=True)
    core.write_json(output / "config.json", metadata)
    started = time.monotonic()
    core.write_json(output / "status.json", dict(status="running", pid=os.getpid(), started_unix=time.time()))
    try:
        with validation_evaluator(data, normal, val_ids, b_validation,
                                  canonical[val_ids.cpu().numpy()], primary_normal["y_std"]):
            best, epoch, seconds = core.fit_stage(model, data, train_ids, val_ids, memory, args, model_config, started)
        core.synchronized(device)
        for split, ids in (("validation", val_ids), ("test", test_ids)):
            p, selections = predict(model, data, ids, memory, args.batch_size, collect=True)
            raw = p.astype(np.float64) * np.asarray(normal["y_std"]) + np.asarray(normal["y_mean"])
            truth = data["y"][ids].cpu().numpy().astype(np.float64) * np.asarray(normal["y_std"]) + np.asarray(normal["y_mean"])
            np.savez_compressed(output / f"{split}_predictions.npz", event_indices=ids.cpu().numpy(), prediction=raw, truth=truth)
            if selections:
                rows = []
                history_ids = train_ids.cpu().numpy()
                for queries, selected, valid, scores, groups in selections:
                    for qi, selection, mask, score, group in zip(queries, selected, valid, scores, groups):
                        for rank in np.flatnonzero(mask):
                            hi = int(history_ids[selection[rank]])
                            rows.append(dict(query_id=events.event_id.iloc[qi], historical_id=events.event_id.iloc[hi],
                                rank=int(rank + 1), selection_group=int(group[rank]), similarity=float(score[rank]),
                                query_day=int(data["day"][qi]), end_day=int(data["end_day"][hi])))
                pd.DataFrame(rows).to_csv(output / f"{split}_retrieved_events.csv", index=False)
        if options.role != "reference":
            assert expected["reference_checkpoint_sha256"] == core.sha256(reference_run / "checkpoint.pt")
            assert expected["reference_validation_predictions_sha256"] == core.sha256(reference_run / "validation_predictions.npz")
            assert expected["reference_test_predictions_sha256"] == core.sha256(reference_run / "test_predictions.npz")
        assert expected["data_hash"] == core.sha256(directory / "dataset.npz")
        assert expected["primary_data_hash"] == core.sha256(primary / "dataset.npz")
        core.write_json(output / "status.json", dict(status="completed", role=options.role, model=model_name, seed=options.seed,
            best_epoch=epoch, validation_standardized_mae=best, validation_objective=metadata["validation_objective"],
            training_seconds=seconds, total_seconds=time.monotonic() - started, finished_unix=time.time(),
            checkpoint_sha256=core.sha256(output / "checkpoint.pt"), query_future_inputs=False,
            validation_predictions_sha256=core.sha256(output / "validation_predictions.npz"),
            test_predictions_sha256=core.sha256(output / "test_predictions.npz")))
    except BaseException as error:
        core.write_json(output / "status.json", dict(status="failed", error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=["reference", "ect_direct", "raft_direct"], required=True)
    parser.add_argument("--seed", type=int, choices=[41, 42, 43], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    execute(parser.parse_args())
