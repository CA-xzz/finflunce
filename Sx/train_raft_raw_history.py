"""Auxiliary RAFT: raw historical Y values, unchanged Delta prediction target.

Uses the main RAFT validation-selected configuration, without another search.
All changes to train.py's in-memory callables are scoped and restored. Neither
models.py nor train.py is edited. Saved checkpoints require this wrapper when
reconstructing their historical response memory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

import train as core
from models import RAFT

EXPERIMENT = "RAFT_raw_Y_history"
STD_FLOOR = 1e-8
QUERY_FIELDS = ("x", "platform", "influencer", "day", "index")
MEMORY_FIELDS = ("x", "platform", "influencer", "day", "end_day", "index", "split")
RUNTIME_KEYS = {"data", "output", "seed", "device", "threads", "timeout_seconds", "timing_repeats", "validation_only"}


class RawHistoryRAFT(RAFT):
    """Keep RAFT parameter/state structure; replace only historical values."""

    def __init__(self, history_indices, history_values, **config):
        super().__init__(**config)
        indices = torch.as_tensor(history_indices, dtype=torch.long)
        values = torch.as_tensor(history_values, dtype=torch.float32, device=indices.device)
        if len(indices) < 1 or values.shape != (len(indices), self.horizon, self.channels):
            raise ValueError("Raw history must be a nonempty aligned [M,H,C] bank")
        if not bool(torch.isfinite(values).all()) or not bool((indices[1:] > indices[:-1]).all()):
            raise ValueError("Raw history IDs must be sorted and unique; values must be finite")
        # These are data, not learned parameters or checkpoint state. The saved
        # source hash and normalization rebuild the bank when restoring a model.
        self.register_buffer("_history_indices", indices.detach().clone(), persistent=False)
        self.register_buffer("_history_values", values.detach().clone(), persistent=False)

    def response_memory(self, memory):
        if memory is None or len(memory["x"]) == 0:
            return memory
        if "index" not in memory or "split" not in memory:
            raise ValueError("Raw history requires explicit memory indices and split labels")
        if not bool((memory["split"] == 0).all()):
            raise ValueError("Raw response memory must contain only training events")
        positions = torch.searchsorted(self._history_indices, memory["index"])
        if bool((positions >= len(self._history_indices)).any()):
            raise ValueError("A memory event is absent from the training-only raw bank")
        if not bool((self._history_indices[positions] == memory["index"]).all()):
            raise ValueError("A memory event is absent from the training-only raw bank")
        result = {key: memory[key] for key in MEMORY_FIELDS if key in memory}
        result["y"] = self._history_values[positions]
        return result

    def forward(self, query, memory=None, cached_keys=None, labels=None):
        # Delta labels stay in core.fit_stage's loss calculation. Any query
        # outcome fields, including raw Y, are explicitly omitted here.
        inputs = {key: query[key] for key in QUERY_FIELDS if key in query}
        return super().forward(inputs, self.response_memory(memory), cached_keys, labels=None)


def load_training_raw_history(labels_path, data, events, normal):
    """Align only training-event raw Y and fit channel scales on that bank."""
    labels_path = Path(labels_path)
    train_indices = (data["split"] == 0).nonzero().flatten()
    positions = train_indices.cpu().numpy()
    train_events = events.iloc[positions].copy()
    event_ids = train_events.event_id.astype(str).tolist()
    selected_ids = set(event_ids)
    required = ["event_id", "tau", "channel", "y", "baseline", "delta", "label_version", "target_available_at"]
    pieces = []
    for chunk in pd.read_csv(labels_path, usecols=required, dtype=str, keep_default_na=False, chunksize=100000):
        # Validation/test raw outcomes never enter numeric tensors or scales.
        piece = chunk[chunk.event_id.isin(selected_ids)].copy()
        if len(piece):
            pieces.append(piece)
    if not pieces:
        raise ValueError("labels.csv contains no selected training-memory events")
    labels = pd.concat(pieces, ignore_index=True)
    numeric_tau = pd.to_numeric(labels.tau, errors="raise")
    if not numeric_tau.isin(range(11)).all():
        raise ValueError("Raw-history tau must be an integer in 0..10")
    labels["tau"] = numeric_tau.astype(int)
    if not labels.channel.isin(core.CHANNELS).all():
        raise ValueError("Unknown raw-history response channel")
    if labels.duplicated(["event_id", "tau", "channel"]).any():
        raise ValueError("Duplicate training raw-history labels")
    grid = pd.MultiIndex.from_product([event_ids, range(11), core.CHANNELS], names=["event_id", "tau", "channel"])
    labels = labels.set_index(["event_id", "tau", "channel"]).reindex(grid)
    if len(labels) != len(event_ids) * 33 or labels.isna().any().any():
        raise ValueError("Incomplete training raw-history response grid")
    if set(labels.label_version) != {str(events.label_version.iloc[0])}:
        raise ValueError("Raw-history label version differs from the Delta dataset")
    values = labels[["y", "baseline", "delta"]].apply(pd.to_numeric, errors="raise").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite training raw-history values")
    if not np.allclose(values[:, 0] - values[:, 1], values[:, 2], rtol=2e-5, atol=2e-6):
        raise ValueError("Raw Y minus baseline does not match the supplied Delta")
    delta = values[:, 2].reshape(len(event_ids), 11, 3)
    expected_delta = data["y"][train_indices].cpu().numpy().astype(np.float64) * np.asarray(normal["y_std"]) + np.asarray(normal["y_mean"])
    if not np.allclose(delta, expected_delta, rtol=2e-5, atol=2e-6):
        raise ValueError("labels.csv Delta does not match the unchanged training target")
    if "end_date" not in train_events:
        raise ValueError("events.csv end_date is required to verify raw-response availability")
    available = pd.to_datetime(labels.target_available_at, errors="raise").to_numpy()
    expected_available = np.repeat((pd.to_datetime(train_events.end_date) + pd.Timedelta(days=1)).to_numpy(), 33)
    if not np.array_equal(available, expected_available):
        raise ValueError("Raw-response target_available_at must equal event end_date plus one day")
    raw = values[:, 0].reshape(len(event_ids), 11, 3)
    mean, empirical_std = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1), ddof=0)
    std = np.maximum(empirical_std, STD_FLOOR)
    standardized = ((raw - mean) / std).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("Nonfinite standardized raw-history values")
    metadata = dict(channels=list(core.CHANNELS), raw_y_mean=mean.tolist(), raw_y_std=std.tolist(),
                    raw_y_empirical_std=empirical_std.tolist(), std_floor=STD_FLOOR, std_ddof=0,
                    fit_scope="All and only fixed training-memory events, across events and horizons, per channel",
                    n_memory=len(event_ids), n_observations_per_channel=len(event_ids) * 11,
                    fit_event_ids=event_ids, fit_dataset_indices=positions.tolist(),
                    fit_event_ids_sha256=hashlib.sha256(json.dumps(event_ids, ensure_ascii=False).encode()).hexdigest(),
                    labels_sha256=core.sha256(labels_path), labels_path=str(labels_path.resolve()),
                    target_channels=list(core.CHANNELS), target_y_mean=normal["y_mean"], target_y_std=normal["y_std"],
                    target="Unchanged standardized reference-aligned Delta from dataset.npz",
                    label_version=str(events.label_version.iloc[0]))
    return train_indices, torch.as_tensor(standardized, device=data["x"].device), metadata


@contextmanager
def argv_context(arguments):
    previous = sys.argv
    sys.argv = list(arguments)
    try:
        yield
    finally:
        sys.argv = previous


def frozen_arguments(options):
    """Bind to the already-selected main run, never use test metrics."""
    main_run = options.main_run.resolve()
    config_path, status_path = main_run / "config.json", main_run / "status.json"
    config = json.loads(config_path.read_text())
    status = json.loads(status_path.read_text())
    if config.get("model") != "RAFT" or config.get("variant") != "full" or config.get("memory_subset", "all") != "all":
        raise ValueError("Configuration source must be the full main RAFT with all training memory")
    if status.get("status") != "completed" or status.get("mode") != "full":
        raise ValueError("Wait for the selected main RAFT run to complete before auxiliary training")
    selection_path = options.selection or (main_run.parent.parent.parent / "selected_configs.json")
    selection_path = selection_path.resolve()
    selection = json.loads(selection_path.read_text())["RAFT"]
    if not selection.get("config") or any(config.get(k) != v for k, v in selection["config"].items()):
        raise ValueError("Main RAFT hyperparameters differ from the frozen validation selection")
    tuning_run = selection_path.parent / "tuning" / "RAFT" / ("trial" + str(selection["trial"]))
    tuning_config = json.loads((tuning_run / "config.json").read_text())
    tuning_status = json.loads((tuning_run / "status.json").read_text())
    if tuning_status.get("status") != "completed" or tuning_status.get("mode") != "validation_only":
        raise ValueError("Selected trial must have completed without test evaluation")
    if tuning_config.get("model") != "RAFT" or any(tuning_config.get(k) != v for k, v in selection["config"].items()):
        raise ValueError("Validation trial does not match the frozen RAFT selection")
    for field, filename in [("data_hash", "dataset.npz"), ("normalization_hash", "normalization.json")]:
        if config.get(field) != core.sha256(options.data / filename) or config.get(field) != tuning_config.get(field):
            raise ValueError("Frozen main run and auxiliary data/normalization differ")
    for filename in ("models.py", "train.py"):
        expected_hash = core.sha256(Path(__file__).with_name(filename))
        if config.get("code_hashes", {}).get(filename) != expected_hash or tuning_config.get("code_hashes", {}).get(filename) != expected_hash:
            raise ValueError("Frozen main implementation has changed: " + filename)
    with argv_context(["train.py", "--data", str(options.data), "--output", str(options.output), "--model", "RAFT"]):
        args = core.parse_args()
    for key in vars(args):
        if key not in RUNTIME_KEYS:
            if key not in config:
                raise ValueError("Frozen main config lacks training option " + key)
            setattr(args, key, config[key])
            if tuning_config.get(key) != config[key]:
                raise ValueError("Main and selected trial training settings differ: " + key)
    # Epoch budget and early stopping are also fixed, not runtime overrides.
    args.epochs, args.patience, args.batch_size = config["epochs"], config["patience"], config["batch_size"]
    args.data, args.output, args.seed = options.data, options.output, options.seed
    args.device = options.device
    args.threads = options.threads if options.threads is not None else config["threads"]
    args.timeout_seconds, args.timing_repeats = config["timeout_seconds"], config["timing_repeats"]
    args.validation_only = False
    provenance = dict(experiment=EXPERIMENT, main_run=str(main_run), main_config_sha256=core.sha256(config_path),
                      selection_file=str(selection_path), selection_sha256=core.sha256(selection_path),
                      selected_trial=int(selection["trial"]), selected_trial_config_sha256=core.sha256(tuning_run / "config.json"),
                      hyperparameters_retuned=False, test_used_for_hyperparameter_selection=False,
                      frozen_training_options={k: getattr(args, k) for k in vars(args) if k not in RUNTIME_KEYS},
                      wrapper_file=str(Path(__file__).resolve()), wrapper_sha256=core.sha256(__file__),
                      class_name="RawHistoryRAFT", checkpoint_requires_wrapper=True,
                      query_raw_y_provided=False, memory_mode="fixed_train_only_end_day_strictly_before_query_day",
                      architecture_and_trainable_parameter_structure="Identical to main RAFT; raw bank is nonpersistent data buffers",
                      limitations=["Auxiliary local RAFT adaptation, not an exact original-paper reproduction.",
                                   "Raw Y is standardized with fixed training-memory channel statistics; Delta output scaling is unchanged.",
                                   "Query labels are removed before the base forward call; raw Y never enters the shared data/query dictionary.",
                                   "Inherited timing and memory totals omit the extra wrapper raw-bank storage; these are not the main exclusive efficiency benchmark."])
    return args, provenance


@contextmanager
def training_wrapper(labels_path, output, provenance):
    original_load, original_build, original_write = core.load_data, core.build_model, core.write_json
    state = {}

    def load_data(directory, device):
        data, events, normal = original_load(directory, device)
        indices, values, scaling = load_training_raw_history(labels_path, data, events, normal)
        state.update(indices=indices, values=values, scaling=scaling)
        original_write(output / "raw_history_normalization.json", scaling)
        details = dict(provenance, labels_sha256=scaling["labels_sha256"], raw_bank_bytes=values.numel() * values.element_size(),
                       events_sha256=core.sha256(Path(directory) / "events.csv"),
                       raw_bank_index_bytes=indices.numel() * indices.element_size(), raw_y_mean=scaling["raw_y_mean"], raw_y_std=scaling["raw_y_std"])
        state["details"] = details
        original_write(output / "raw_history_metadata.json", details)
        frame = events.iloc[indices.cpu().numpy()].copy()
        frame["dataset_index"] = indices.cpu().numpy()
        frame["day"] = data["day"][indices].cpu().numpy()
        frame["end_day"] = data["end_day"][indices].cpu().numpy()
        frame["origin_split"] = 0
        frame.to_csv(output / "raw_history_memory_manifest.csv", index=False)
        return data, events, normal

    def build_model(name, **config):
        if name != "RAFT" or "indices" not in state:
            raise ValueError("Raw-history wrapper is restricted to loaded RAFT data")
        return RawHistoryRAFT(state["indices"], state["values"], **config)

    def write_json(path, value):
        if Path(path).name in {"config.json", "status.json", "validation_summary.json"}:
            value = dict(value, auxiliary_experiment=EXPERIMENT, history_value="training_standardized_raw_Y",
                         checkpoint_requires_wrapper=True)
            if Path(path).name == "config.json":
                # Generic evaluators must not silently reconstruct ordinary
                # Delta-memory RAFT from this otherwise compatible state dict.
                value["model"], value["base_model"] = EXPERIMENT, "RAFT"
                value["raw_history"] = state["details"]
                value["baseline_adaptations"] = "Main RAFT architecture and frozen hyperparameters; only historical response values changed to training-standardized raw Y; prediction target remains Delta"
        return original_write(path, value)

    core.load_data, core.build_model, core.write_json = load_data, build_model, write_json
    try:
        yield state
    finally:
        core.load_data, core.build_model, core.write_json = original_load, original_build, original_write


def annotate_outputs(output):
    for path in Path(output).glob("*.csv"):
        if path.stat().st_size < 2:
            continue
        frame = pd.read_csv(path, dtype={"event_id": str, "query_id": str, "historical_id": str})
        if "model" in frame:
            frame["model"] = EXPERIMENT
            frame["auxiliary_experiment"] = EXPERIMENT
            frame["history_value"] = "training_standardized_raw_Y"
            if "run_id" in frame:
                frame["run_id"] = frame.run_id.str.replace("RAFT_", EXPERIMENT + "_", regex=False)
            frame.to_csv(path, index=False)


def restore_checkpoint(run, data_path, device="cpu", labels_path=None):
    run, data_path, device = Path(run), Path(data_path), torch.device(device)
    config = json.loads((run / "config.json").read_text())
    details = config.get("raw_history", {})
    if config.get("auxiliary_experiment") != EXPERIMENT or details.get("wrapper_sha256") != core.sha256(__file__):
        raise ValueError("Checkpoint requires its matching raw-history wrapper")
    for filename in ("models.py", "train.py"):
        if config.get("code_hashes", {}).get(filename) != core.sha256(Path(__file__).with_name(filename)):
            raise ValueError("Checkpoint base implementation changed: " + filename)
    for field, filename in [("data_hash", "dataset.npz"), ("normalization_hash", "normalization.json")]:
        if config[field] != core.sha256(data_path / filename):
            raise ValueError("Checkpoint dataset/normalization changed")
    labels_path = Path(labels_path) if labels_path else data_path / "labels.csv"
    if details["labels_sha256"] != core.sha256(labels_path):
        raise ValueError("Raw-history labels source changed")
    if details["events_sha256"] != core.sha256(data_path / "events.csv"):
        raise ValueError("Raw-history event identities/order changed")
    data, events, normal = core.load_data(data_path, device)
    indices, values, scaling = load_training_raw_history(labels_path, data, events, normal)
    saved = json.loads((run / "raw_history_normalization.json").read_text())
    for field in ("raw_y_mean", "raw_y_std", "fit_event_ids_sha256"):
        if scaling[field] != saved[field]:
            raise ValueError("Raw-history training scales/identities changed")
    model = RawHistoryRAFT(indices, values, **config["model_config"]).to(device)
    checkpoint = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, data, events, normal


def execute(options):
    options.data, options.output = options.data.resolve(), options.output.resolve()
    if options.output.exists() and any(options.output.iterdir()):
        raise FileExistsError("Raw-history output must be a fresh directory")
    args, provenance = frozen_arguments(options)
    labels = options.labels or options.data / "labels.csv"
    try:
        with training_wrapper(labels, options.output, provenance):
            core.run(args)
        annotate_outputs(options.output)
    except BaseException as error:
        options.output.mkdir(parents=True, exist_ok=True)
        core.write_json(options.output / "status.json", dict(status="failed", auxiliary_experiment=EXPERIMENT,
            error=str(error), error_type=type(error).__name__, finished_unix=time.time()))
        (options.output / "failure.txt").write_text(traceback.format_exc())
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-run", type=Path, required=True, help="Completed selected main/RAFT/seedXX run; only its configuration is read")
    parser.add_argument("--selection", type=Path, help="Defaults to the parent suite's selected_configs.json")
    parser.add_argument("--labels", type=Path, help="Defaults to DATA/labels.csv")
    parser.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    execute(parse_args())
