"""Validate and summarize the user-adopted single-reference experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (41, 42, 43)
MODELS = ("DLinear", "PatchTST", "iTransformer", "ECT", "RAFT", "RATL", "RAIF")
CHANNELS = ("AR", "turnover", "log_volume")
HORIZONS = (1, 3, 5, 10, 11)
SCHEMES = ("strict", "nearest", "expanded_nearest")
EXPECTED = {"suite": 63, "parameters": 30, "references": 18, "rolling": 3,
            "frozen_retrieval": 12, "serial_timing": 22, "raft_raw_history": 3}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def metric_rows(truth, predicted, dates):
    require(truth.shape == predicted.shape and truth.ndim == 3 and truth.shape[1:] == (11, 3), "Invalid prediction shape")
    require(len(truth) > 0 and np.isfinite(truth).all() and np.isfinite(predicted).all(), "Empty/nonfinite prediction tensors")
    rows = []
    for horizon in HORIZONS:
        error = predicted[:, :horizon] - truth[:, :horizon]
        for index, channel in enumerate(CHANNELS):
            for metric, value in [("MAE", np.abs(error[:, :, index]).mean()),
                                  ("RMSE", np.sqrt(np.square(error[:, :, index]).mean()))]:
                rows.append(dict(channel=channel, horizon=horizon, metric=metric, value=float(value),
                                 n_events=len(truth), n_dates=len(set(dates))))
    return pd.DataFrame(rows)


def verify_frozen_response(predicted, baseline, normal):
    require(predicted.shape == baseline.shape and np.isfinite(predicted).all() and np.isfinite(baseline).all(), "Invalid frozen response prediction arrays")
    tolerance = 1e-6 * (np.abs(baseline) + np.asarray(normal["y_std"]) + np.abs(normal["y_mean"]))
    require(np.all(np.abs(predicted - baseline) <= tolerance), "Frozen response predictions differ from the main RAIF checkpoint")


def verify_metric_file(path, computed, model, seed):
    saved = pd.read_csv(path)
    keys = ["channel", "horizon", "metric"]
    require(len(saved) == 30 and not saved.duplicated(keys).any(), "Incomplete/duplicate horizon metrics")
    require(saved.model.eq(model).all() and saved.seed.eq(seed).all(), "Metric model/seed identity differs")
    actual = saved.set_index(keys).sort_index()
    expected = computed.set_index(keys).sort_index()
    require(actual.index.equals(expected.index), "Unexpected horizon/channel/metric keys")
    require(np.isfinite(actual.value).all(), "Nonfinite saved metrics")
    np.testing.assert_allclose(actual.value, expected.value, rtol=1e-10, atol=1e-12)
    for field in ["n_events", "n_dates"]:
        np.testing.assert_array_equal(actual[field], expected[field])


def aggregate_metrics(frame):
    if frame.empty:
        return pd.DataFrame()
    keys = ["experiment", "comparison", "model", "variant", "retrieval", "channel", "horizon", "metric", "target_set"]
    require(not frame.duplicated(keys + ["seed"]).any(), "Duplicate seed metric in a comparison")
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        expected = {41} if key[2] == "HistoricalMean" else set(SEEDS)
        require(group.n_events.nunique() == group.n_dates.nunique() == 1, "Seeds have different test coverage")
        rows.append(dict(zip(keys, key), mean=float(group.value.mean()),
            std=float(group.value.std(ddof=1)) if len(group) > 1 else np.nan,
            n_seeds=len(group), expected_seeds=len(expected),
            status="completed" if set(group.seed) == expected else "incomplete",
            n_events=int(group.n_events.iloc[0]), n_dates=int(group.n_dates.iloc[0])))
    return pd.DataFrame(rows)


def validate_reference_plan(plan):
    require(not plan.duplicated(["group", "scheme"]).any(), "Duplicate reference plan")
    require(set(zip(plan.group, plan.scheme)) == {(group, scheme) for group in ["full_training", "intersect_training"] for scheme in SCHEMES}, "Reference plan has missing/unexpected schemes")
    require(len(plan) == 6 and plan.variant.eq("full").all(), "Expected six single-reference datasets")
    return plan


class Report:
    def __init__(self, root, output):
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.results = self.root / "results"
        self.output.mkdir(parents=True, exist_ok=True)
        self.issues, self.inventory, self.frames = [], [], []
        self.datasets, self.predictions = {}, {}
        self.target_sets = {}
        self.timing = pd.DataFrame()
        self.reference_scope = {}

    def resolve(self, value):
        path = Path(value)
        if path.exists():
            return path.resolve()
        if not path.is_absolute():
            return self.root / path
        if "Sx" in path.parts:
            return self.root.joinpath(*path.parts[path.parts.index("Sx") + 1:])
        return path

    def checked(self, group, name, function):
        try:
            return function()
        except (OSError, ValueError, KeyError, IndexError, TypeError, AssertionError) as error:
            self.issues.append(dict(experiment=group, item=name, reason=str(error)))
            return None

    def top_status(self, group, directory, mode=None, suite=False):
        def verify():
            value = read_json(directory / ("suite_status.json" if suite else "status.json"))
            require(value.get("status") == ("completed_requested_phases" if suite else "completed"), "Experiment status is not completed")
            if suite:
                require(value.get("phase") == "all", "The suite did not complete all phases")
            if mode:
                require(value.get("mode") == mode, "Supplementary evaluation mode differs")
                require(value.get("dataset_modified") is False and value.get("checkpoint_modified") is False, "Frozen-evaluation modification flags differ")
            return value
        return self.checked(group, str(directory), verify)

    def dataset(self, path):
        path = self.resolve(path)
        if path in self.datasets:
            return self.datasets[path]
        with np.load(path / "dataset.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        events = pd.read_csv(path / "events.csv", dtype={"event_id": str})
        normal = read_json(path / "normalization.json")
        require(normal.get("q_rule") == "exp_only" and normal.get("quality_neff_factor_enabled") is False,
                "Dataset does not use the current exponential-only quality rule")
        require(normal.get("ready_for_formal_training") is True, "Dataset source checks are incomplete")
        require(not events.event_id.duplicated().any() and len(events) > 0, "Duplicate/empty dataset event IDs")
        require(all(len(value) == len(events) and np.isfinite(value).all() for value in arrays.values()), "Dataset arrays are unaligned or nonfinite")
        require(arrays["y"].shape == (len(events), 11, 3), "Dataset target shape differs")
        require(np.isin(arrays["split"], [0, 1, 2]).all(), "Invalid dataset partitions")
        require(all(np.any(arrays["split"] == part) for part in [0, 1, 2]), "Empty dataset partition")
        require(events.label_version.nunique() == 1 and events.label_version.iloc[0] == normal["label_version"], "Dataset label contract differs")
        mean, scale = np.asarray(normal["y_mean"]), np.asarray(normal["y_std"])
        require(mean.shape == scale.shape == (3,) and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), "Invalid target scales")
        ids = np.flatnonzero(arrays["split"] == 2)
        value = dict(path=path, arrays=arrays, events=events, normal=normal, test_indices=ids,
            test_ids=events.event_id.to_numpy(str)[ids], truth=arrays["y"][ids].astype(np.float64) * scale + mean,
            dataset_sha256=digest(path / "dataset.npz"), normalization_sha256=digest(path / "normalization.json"))
        self.datasets[path] = value
        return value

    def assert_truth(self, actual, expected, data):
        require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), "Invalid/nonfinite target array")
        tolerance = 8 * np.finfo(np.float32).eps * (np.abs(expected) + np.asarray(data["normal"]["y_std"]) + np.abs(data["normal"]["y_mean"]))
        require(np.all(np.abs(actual - expected) <= tolerance), "Fixed test labels differ beyond float32 normalization rounding")

    def target(self, name, ids, truth):
        require(len(ids) > 0 and len(ids) == len(set(ids)) and np.isfinite(truth).all(), "Invalid canonical target")
        if name in self.target_sets:
            old_ids, old_truth = self.target_sets[name]
            np.testing.assert_array_equal(ids, old_ids)
            np.testing.assert_array_equal(truth, old_truth)
        else:
            self.target_sets[name] = (np.asarray(ids).copy(), np.asarray(truth).copy())
        return self.target_sets[name]

    def check_config(self, config, data, model, seed):
        require(config.get("model") == model and config.get("seed") == seed, "Run model/seed differs")
        require(config.get("data_hash") == data["dataset_sha256"] and config.get("normalization_hash") == data["normalization_sha256"], "Run dataset or normalization hashes differ")

    def prediction(self, run, data):
        with np.load(run / "predictions.npz", allow_pickle=False) as archive:
            ids, truth, predicted = archive["event_indices"], archive["truth"], archive["prediction"]
        require(ids.dtype.kind in "iu" and ids.ndim == 1, "Prediction indices must be integer positions")
        np.testing.assert_array_equal(ids, data["test_indices"])
        require(predicted.shape == truth.shape == data["truth"].shape and np.isfinite(predicted).all(), "Invalid/nonfinite predictions")
        np.testing.assert_allclose(truth, data["truth"], rtol=1e-10, atol=1e-12)
        require(np.isfinite(truth).all(), "Nonfinite saved truth")
        self.predictions[run.resolve()] = predicted
        return truth, predicted

    def add_metrics(self, frame, group, comparison, model, seed, config, target, run):
        frame = frame.copy()
        frame["experiment"], frame["comparison"], frame["model"], frame["seed"] = group, comparison, model, seed
        frame["variant"], frame["retrieval"] = config.get("variant", "full"), config.get("retrieval", "response")
        frame["target_set"], frame["run_path"] = target, str(run)
        frame["n_train"] = config.get("n_training_partition", config.get("n_train", np.nan))
        self.frames.append(frame)

    def training_run(self, group, comparison, run, model, seed, expected_data=None, validation=False, target_name="main_test", canonical=None, required=None):
        entry = dict(experiment=group, comparison=comparison, model=model, seed=seed,
                     run_path=str(run), observed=(run / "status.json").exists(), status="incomplete")
        self.inventory.append(entry)
        def verify():
            config, status = read_json(run / "config.json"), read_json(run / "status.json")
            require(status.get("status") == "completed", "Training run is not completed")
            require(status.get("mode") == ("validation_only" if validation else "full"), "Training run mode differs")
            require(np.isfinite(status["validation_standardized_mae"]), "Nonfinite validation score")
            data = self.dataset(config["data"])
            if expected_data is not None:
                require(data["dataset_sha256"] == expected_data["dataset_sha256"] and data["normalization_sha256"] == expected_data["normalization_sha256"], "Unexpected data source for comparison")
            self.check_config(config, data, model, seed)
            for field, value in (required or {}).items():
                require(config.get(field) == value, "Unexpected training option " + field)
            if validation:
                require(not (run / "predictions.npz").exists(), "Validation-only trial contains test predictions")
            else:
                truth, predicted = self.prediction(run, data)
                dates = data["events"].event_date.to_numpy()[data["test_indices"]]
                source_metrics = metric_rows(truth, predicted, dates)
                verify_metric_file(run / "horizon_metrics.csv", source_metrics, model, seed)
                if canonical is None:
                    canonical_ids, canonical_truth = self.target(target_name, data["test_ids"], truth)
                else:
                    canonical_ids, canonical_truth = canonical
                    np.testing.assert_array_equal(data["test_ids"], canonical_ids)
                    self.assert_truth(truth, canonical_truth, data)
                    self.target(target_name, canonical_ids, canonical_truth)
                metrics = metric_rows(canonical_truth, predicted, dates)
                if group == "references":
                    verify_metric_file(run / "common_target_horizon_metrics.csv", metrics, model, seed)
                    saved = pd.read_csv(run / "common_target_horizon_metrics.csv")
                    require(saved.common_original_unit_target.eq(True).all(), "Reference metrics did not use canonical targets")
                if group == "raft_raw_history":
                    self.verify_raw_history(run, data, config)
                self.add_metrics(metrics, group, comparison, model, seed, config, target_name, run)
            entry.update(status="completed", n_train=int((data["arrays"]["split"] == 0).sum()),
                         n_test=len(data["test_indices"]), dataset=str(data["path"]))
            return data
        return self.checked(group, str(run), verify)

    def verify_raw_history(self, run, data, config):
        require(config.get("auxiliary_experiment") == "RAFT_raw_Y_history", "Missing raw-history experiment identity")
        detail = read_json(run / "raw_history_metadata.json")
        scales = read_json(run / "raw_history_normalization.json")
        require(detail.get("hyperparameters_retuned") is False and detail.get("test_used_for_hyperparameter_selection") is False and detail.get("query_raw_y_provided") is False, "Raw-history protocol flags differ")
        require(detail["labels_sha256"] == digest(data["path"] / "labels.csv") and detail["events_sha256"] == digest(data["path"] / "events.csv"), "Raw-history source hashes differ")
        ids = np.flatnonzero(data["arrays"]["split"] == 0)
        require(scales["fit_event_ids"] == data["events"].event_id.iloc[ids].tolist(), "Raw-history scaling did not use the fixed training set")
        np.testing.assert_array_equal(scales["fit_dataset_indices"], ids)
        require(np.isfinite(scales["raw_y_mean"]).all() and np.isfinite(scales["raw_y_std"]).all() and (np.asarray(scales["raw_y_std"]) > 0).all(), "Invalid raw-history training scales")

    def suite(self, main):
        suite = self.results / "suite_expanded"
        self.top_status("suite", suite, suite=True)
        for model in MODELS:
            for trial in [0, 1]:
                self.training_run("suite", "tuning/" + model, suite / f"tuning/{model}/trial{trial}", model, 40, main, validation=True)
        for model in ("HistoricalMean", *MODELS):
            for seed in ([41] if model == "HistoricalMean" else SEEDS):
                self.training_run("suite", "main", suite / f"main/{model}/seed{seed}", model, seed, main)
        for variant in ["no_retrieval", "no_response", "no_events", "no_quality", "no_attention"]:
            for seed in SEEDS:
                self.training_run("suite", "ablation/" + variant, suite / f"ablation/{variant}/seed{seed}", "RAIF", seed, main, required={"variant": variant})
        for retrieval in ["state", "random", "reverse"]:
            for seed in SEEDS:
                self.training_run("suite", "retrieval/" + retrieval, suite / f"retrieval/{retrieval}/seed{seed}", "RAIF", seed, main, required={"retrieval": retrieval})
        for seed in SEEDS:
            self.training_run("suite", "capacity/ratl_memory", suite / f"capacity/RAIF/seed{seed}", "RAIF", seed, main, required={"memory_subset": "ratl"})
            self.training_run("raft_raw_history", "raw_Y_history", suite / f"raft_raw_history/seed{seed}", "RAFT_raw_Y_history", seed, main)

    def parameters(self, main):
        directory = self.results / "parameters_expanded"
        self.top_status("parameters", directory)
        if main is None:
            return
        canonical = self.target("main_test", main["test_ids"], main["truth"])
        specs = [("K", value, str(value)) for value in [1, 3, 10, 20]]
        specs += [("L", value, str(value)) for value in [5, 10, 20]]
        specs += [("quality_quantile", value, str(value)) for value in [.25, .5, .75]]
        for parameter, value, path_value in specs:
            target = canonical
            target_name = "main_test"
            expected_data = main if parameter == "K" else None
            if parameter == "L":
                def load_l():
                    config = read_json(directory / f"L/{path_value}/seed41/config.json")
                    data = self.dataset(config["data"])
                    require(data["arrays"]["x"].shape[1] == value and data["normal"]["reference_length"] == 10, "L comparison changed reference window")
                    with np.load(data["path"] / "evaluation_targets.npz", allow_pickle=False) as archive:
                        return data, self.target("common_L20_test", archive["event_ids"], archive["truth"])
                loaded = self.checked("parameters", "L" + path_value + " targets", load_l)
                if loaded is None:
                    continue
                expected_data, target = loaded
                target_name = "common_L20_test"
            for seed in SEEDS:
                self.training_run("parameters", f"{parameter}={value}", directory / f"{parameter}/{path_value}/seed{seed}",
                    "RAIF", seed, expected_data=expected_data,
                    target_name=target_name, canonical=target, required={"k": value} if parameter == "K" else None)
        # Baseline points reuse main runs and never count as additional training.
        for parameter, value in [("K", 5), ("quality_quantile", 0)]:
            for seed in SEEDS:
                selected = [frame for frame in self.frames if frame.experiment.iloc[0] == "suite" and frame.comparison.iloc[0] == "main" and frame.model.iloc[0] == "RAIF" and frame.seed.iloc[0] == seed]
                if selected:
                    frame = selected[0].copy()
                    frame["experiment"], frame["comparison"] = "parameters", f"{parameter}={value}"
                    frame["reused_main_run"] = True
                    self.frames.append(frame)

    def references(self):
        directory = self.results / "references_expanded"
        self.top_status("references", directory)
        def load_plan():
            plan = pd.read_csv(self.root / "data/reference_comparisons_expanded/comparison_plan.csv")
            return validate_reference_plan(plan)
        plan = self.checked("references", "comparison_plan", load_plan)
        if plan is None:
            return
        intersection = None
        for row in plan.itertuples():
            def load_source():
                nonlocal intersection
                data = self.dataset(row.dataset)
                require(len(data["test_indices"]) == row.n_test and int((data["arrays"]["split"] == 0).sum()) == row.n_train, "Reference plan coverage differs from dataset")
                contract = data["normal"]["reference_comparison"]
                require(contract["evaluation_label_version"] == row.evaluation_label_version and contract["training_label_version"] == row.training_label_version, "Reference label versions differ from plan")
                if row.group == "intersect_training":
                    ids = data["events"].event_id[data["arrays"]["split"] == 0].to_numpy(str)
                    if intersection is None:
                        intersection = ids
                    else:
                        np.testing.assert_array_equal(ids, intersection)
                with np.load(data["path"] / "evaluation_targets.npz", allow_pickle=False) as archive:
                    target = self.target("strict_common_test", archive["event_ids"], archive["truth"])
                self.reference_scope = dict(common_test_events=len(target[0]), common_training_intersection=len(intersection) if intersection is not None else None,
                    scope="Strict-reference common subset diagnostic; main expanded sample is unchanged", training_labels="Each scheme retains its own training and validation labels")
                return data, target
            result = self.checked("references", f"{row.group}/{row.scheme} source", load_source)
            if result is None:
                continue
            data, target = result
            for seed in SEEDS:
                self.training_run("references", f"{row.group}/{row.scheme}", directory / f"{row.group}/{row.scheme}/seed{seed}", "RAIF", seed,
                    expected_data=data, target_name="strict_common_test", canonical=target, required={"variant": row.variant})

    def extra(self, main, mode, name):
        group = "rolling" if mode == "rolling" else "frozen_retrieval"
        directory = self.results / name
        status = self.top_status(group, directory, mode)
        if main is None:
            return
        expected = {(seed, strategy) for seed in SEEDS for strategy in (["response"] if mode == "rolling" else ["response", "state", "random", "reverse"])}
        observed = set()
        for path in sorted(directory.glob("*/evaluation_config.json")):
            entry = dict(experiment=group, run_path=str(path.parent), observed=True, status="incomplete")
            self.inventory.append(entry)
            def verify():
                config = read_json(path)
                seed, strategy = config["seed"], config["retrieval"]
                require((seed, strategy) in expected and (seed, strategy) not in observed, "Unexpected/duplicate frozen evaluation")
                observed.add((seed, strategy))
                entry.update(model="RAIF", seed=seed, comparison=strategy)
                require(config["dataset_sha256"] == main["dataset_sha256"], "Evaluation dataset differs")
                require(config.get("parameters_updated") is False and config.get("normalization_updated") is False, "Frozen checkpoint/scales were changed")
                require(config["memory_mode"] == ("rolling" if mode == "rolling" else "fixed"), "Memory mode differs")
                source = self.resolve(config["source_run"])
                expected_source = self.results / f"suite_expanded/main/RAIF/seed{seed}"
                require(source == expected_source, "Unexpected source checkpoint")
                self.check_config(read_json(source / "config.json"), main, "RAIF", seed)
                require(config["checkpoint_sha256"] == digest(source / "checkpoint.pt"), "Frozen source checkpoint hash differs")
                truth, predicted = self.prediction(path.parent, main)
                dates = main["events"].event_date.to_numpy()[main["test_indices"]]
                metrics = metric_rows(truth, predicted, dates)
                verify_metric_file(path.parent / "horizon_metrics.csv", metrics, "RAIF", seed)
                if mode == "frozen" and strategy == "response":
                    baseline = self.predictions.get(source)
                    if baseline is None:
                        _, baseline = self.prediction(source, main)
                    verify_frozen_response(predicted, baseline, main["normal"])
                self.verify_retrieval(path.parent, main, mode)
                self.add_metrics(metrics, group, strategy, "RAIF", seed, config, "main_test", path.parent)
                entry.update(status="completed", n_test=len(truth), dataset=str(main["path"]))
            self.checked(group, str(path.parent), verify)
        if observed != expected:
            self.issues.append(dict(experiment=group, item="seed/strategy coverage", reason=f"Missing {sorted(expected - observed)}"))
        if status is not None:
            actual_runs = {self.resolve(path) for path in status["runs"]}
            expected_runs = {self.results / f"suite_expanded/main/RAIF/seed{seed}" for seed in SEEDS}
            self.checked(group, "source checkpoint list", lambda: require(actual_runs == expected_runs, "Unexpected source run set"))

    def verify_retrieval(self, directory, data, mode):
        retrieval = pd.read_csv(directory / "retrieved_events.csv", dtype={"query_id": str, "historical_id": str})
        positions = pd.Series(np.arange(len(data["events"])), index=data["events"].event_id)
        if len(retrieval):
            query = positions.reindex(retrieval.query_id)
            history = positions.reindex(retrieval.historical_id)
            require(query.notna().all() and history.notna().all(), "Unknown retrieval event identity")
            q, h = query.to_numpy(int), history.to_numpy(int)
            a = data["arrays"]
            require(np.all(a["end_day"][h] < a["day"][q]) and np.all(q != h), "Retrieved response was not mature")
            np.testing.assert_array_equal(retrieval.origin_split, a["split"][h])
            np.testing.assert_array_equal(retrieval.historical_end_day, a["end_day"][h])
            np.testing.assert_array_equal(retrieval.query_day, a["day"][q])
            require(np.isfinite(retrieval.similarity).all(), "Nonfinite retrieval scores")
            if mode == "frozen":
                require(np.all(a["split"][h] == 0), "Frozen evaluation retrieved a nontraining event")
        updates = pd.read_csv(directory / "memory_updates.csv", dtype={"event_id": str})
        if mode == "frozen":
            require(updates.empty, "Fixed-memory evaluation has memory updates")
        elif len(updates):
            require(not updates.event_id.duplicated().any(), "Duplicate rolling memory promotions")
            ids = positions.reindex(updates.event_id)
            require(ids.notna().all(), "Unknown promoted event identity")
            ids = ids.to_numpy(int)
            np.testing.assert_array_equal(updates.origin_split, data["arrays"]["split"][ids])
            np.testing.assert_array_equal(updates.end_day, data["arrays"]["end_day"][ids])
            require(updates.origin_split.isin([1, 2]).all() and (updates.end_day < updates.added_before_query_day).all(), "Rolling memory promoted an unavailable response")

    def serial_timing(self, main):
        directory = self.results / "serial_timing_expanded"
        status = self.top_status("serial_timing", directory, "timing")
        if status is None or main is None:
            return
        def verify():
            frame = pd.read_csv(directory / "timing.csv")
            expected = {(model, seed) for model in MODELS for seed in SEEDS} | {("HistoricalMean", 41)}
            require(len(frame) == 22 and set(zip(frame.model, frame.seed)) == expected, "Serial timing model/seed coverage differs")
            require(frame.batch_size.eq(64).all() and frame.warmup.eq(10).all() and frame.repeats.eq(50).all(), "Serial timing batch/warmup/repeat protocol differs")
            require(frame.no_contention_detected.eq(True).all() and frame.training_timing_exclusive.eq(False).all(), "Timing scope or contention flag differs")
            require(frame.memory_mode.eq("fixed_common_training_intersection").all(), "Timing memory protocol differs")
            memory = pd.read_csv(directory / "timing_memory_manifest.csv", dtype={"event_id": str})
            require(not memory.event_id.duplicated().any() and memory.origin_split.eq(0).all() and frame.common_memory_events.eq(len(memory)).all(), "Timing memory identities differ")
            train = set(main["events"].event_id[main["arrays"]["split"] == 0])
            require(set(memory.event_id) <= train, "Timing memory contains nontraining events")
            sample_rows = read_json(directory / "timing_samples.json")
            samples = {self.resolve(row["source_run"]): row for row in sample_rows}
            require(len(samples) == len(sample_rows) == 22, "Timing sample source coverage differs")
            expected_sources = {self.results / f"suite_expanded/main/{model}/seed{seed}" for model, seed in expected}
            require({self.resolve(value) for value in status["runs"]} == expected_sources == set(samples), "Timing source checkpoints differ")
            memory_sets = []
            for row in frame.itertuples():
                source = self.resolve(row.source_run)
                require(source == self.results / f"suite_expanded/main/{row.model}/seed{row.seed}", "Timing row source identity differs")
                self.check_config(read_json(source / "config.json"), main, row.model, row.seed)
                source_memory = pd.read_csv(source / "memory_manifest.csv", dtype={"event_id": str})
                if row.model in {"HistoricalMean", "RAIF", "RAFT", "RATL"}:
                    memory_sets.append(set(source_memory.event_id))
                require(row.original_memory_events == len(source_memory), "Original timing memory count differs")
                inference = np.asarray(samples[source]["inference_ms"], dtype=float)
                require(len(inference) == 50 and np.isfinite(inference).all() and (inference > 0).all(), "Invalid inference timing samples")
                np.testing.assert_allclose([row.inference_batch_ms, row.inference_batch_std_ms, row.inference_event_ms],
                    [inference.mean(), inference.std(ddof=1), inference.mean() / 64], rtol=1e-10, atol=1e-12)
                retrieval = np.asarray(samples[source]["retrieval_ms"], dtype=float)
                if row.model in {"RAIF", "RAFT", "RATL"}:
                    require(len(retrieval) == 50 and np.isfinite(retrieval).all() and (retrieval > 0).all(), "Invalid retrieval timing samples")
                    np.testing.assert_allclose([row.retrieval_batch_ms, row.retrieval_batch_std_ms], [retrieval.mean(), retrieval.std(ddof=1)], rtol=1e-10, atol=1e-12)
                else:
                    require(len(retrieval) == 0 and pd.isna(row.retrieval_batch_ms), "Nonretrieval model has retrieval latency")
            require(set(memory.event_id) == set.intersection(*memory_sets), "Timing memory is not the original training-memory intersection")
            checks = read_json(directory / "gpu_checks.json")
            require(len(checks) > 2 and checks[0]["stage"] == "before_cuda_allocation", "Missing initial GPU state")
            require(not checks[0]["pids"] and checks[0]["utilization"] == 0 and checks[0]["memory_used_mib"] <= 64, "Timing did not start on an observed idle GPU")
            require(all(row["index"] == 0 and not row["additional_reported_pids"] for row in checks), "GPU identity/contention checks differ")
            require(all(row["maximum_memory_mib"] is None or row["actual_memory_mib"] <= row["maximum_memory_mib"] for row in checks), "Timing memory contention was detected")
            self.timing = frame
            for row in frame.itertuples():
                self.inventory.append(dict(experiment="serial_timing", comparison="serial_common_memory", model=row.model, seed=row.seed, status="completed", observed=True, run_path=str(directory), n_memory=len(memory)))
        self.checked("serial_timing", "timing measurements", verify)

    def finish(self, main):
        raw = pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame()
        aggregate = aggregate_metrics(raw)
        if len(aggregate):
            for row in aggregate[aggregate.status != "completed"].drop_duplicates(["experiment", "comparison", "model"]).itertuples():
                self.issues.append(dict(experiment=row.experiment, item=row.comparison, reason="Incomplete seed set in accepted metrics"))
        inventory = pd.DataFrame(self.inventory)
        counts = {}
        status_patterns = {
            "suite": [f"suite_expanded/{part}/**/status.json" for part in ["tuning", "main", "ablation", "retrieval", "capacity"]],
            "parameters": [f"parameters_expanded/{part}/**/status.json" for part in ["K", "L", "quality_quantile"]],
            "references": [f"references_expanded/{part}/**/status.json" for part in ["full_training", "intersect_training"]],
            "rolling": ["rolling_expanded/*/evaluation_config.json"],
            "frozen_retrieval": ["frozen_retrieval_expanded/*/evaluation_config.json"],
            "raft_raw_history": ["suite_expanded/raft_raw_history/seed*/status.json"],
        }
        for group, expected in EXPECTED.items():
            selected = inventory[inventory.experiment == group] if len(inventory) else inventory
            valid = int(selected.status.eq("completed").sum()) if len(selected) else 0
            observed = sum(len(list(self.results.glob(pattern))) for pattern in status_patterns.get(group, []))
            if group == "serial_timing" and (self.results / "serial_timing_expanded/timing.csv").exists():
                try:
                    observed = len(pd.read_csv(self.results / "serial_timing_expanded/timing.csv"))
                except (OSError, ValueError):
                    observed = 0
            counts[group] = dict(expected=expected, observed=observed, validated=valid)
            if observed > expected:
                self.issues.append(dict(experiment=group, item="run count", reason=f"Unexpected additional runs: {observed} observed, {expected} expected"))
            if valid != expected and not any(issue["experiment"] == group for issue in self.issues):
                self.issues.append(dict(experiment=group, item="run count", reason=f"Expected {expected}, validated {valid}"))
        raw.to_csv(self.output / "supplementary_metrics_seeds.csv", index=False)
        aggregate.to_csv(self.output / "supplementary_metrics.csv", index=False)
        inventory.to_csv(self.output / "supplementary_run_index.csv", index=False)
        for name, selected in [("parameter_metrics", "parameters"), ("reference_metrics", "references"),
                               ("rolling_metrics", "rolling"), ("frozen_retrieval_metrics", "frozen_retrieval"), ("raft_raw_history_metrics", "raft_raw_history")]:
            (aggregate[aggregate.experiment == selected] if len(aggregate) else aggregate).to_csv(self.output / (name + ".csv"), index=False)
        timing_summary = pd.DataFrame()
        if len(self.timing):
            self.timing.to_csv(self.output / "serial_timing_seeds.csv", index=False)
            timing_summary = self.timing.groupby(["model", "variant"], dropna=False).agg(
                inference_batch_ms_mean=("inference_batch_ms", "mean"), inference_batch_ms_std=("inference_batch_ms", "std"),
                inference_event_ms_mean=("inference_event_ms", "mean"), inference_event_ms_std=("inference_event_ms", "std"),
                retrieval_batch_ms_mean=("retrieval_batch_ms", "mean"), retrieval_batch_ms_std=("retrieval_batch_ms", "std"),
                memory_mb_mean=("canonical_memory_mb", "mean"), n_seeds=("seed", "nunique"),
                memory_events=("common_memory_events", "first"), batch_size=("batch_size", "first")).reset_index()
        timing_summary.to_csv(self.output / "serial_timing_summary.csv", index=False)
        figures = self.parameter_figures(aggregate)
        remaining = sorted(set(issue["experiment"] for issue in self.issues))
        summary = dict(status="completed" if not remaining else "incomplete", remaining_experiments=remaining,
            run_counts=counts, actual_training_runs=sum(counts[group]["validated"] for group in ["suite", "parameters", "references", "raft_raw_history"]),
            actual_validation_only_runs=sum(row.get("status") == "completed" and row.get("comparison", "").startswith("tuning/") for row in self.inventory),
            actual_supplementary_prediction_evaluations=counts["rolling"]["validated"] + counts["frozen_retrieval"]["validated"],
            actual_serial_timing_evaluations=counts["serial_timing"]["validated"],
            main_events=len(main["events"]) if main else None, main_test_events=len(main["test_indices"]) if main else None,
            reference_scope=self.reference_scope, reference_schemes=list(SCHEMES),
            quality_rule="exp_only", quality_neff_factor_enabled=False, seed_values=list(SEEDS),
            aggregation="Per-seed original-unit MAE and RMSE, then arithmetic seed mean and sample SD (ddof=1); a single-seed SD is unavailable",
            parameter_target_policy="K and quality retention use the main fixed test target. L uses its separate common L20-complete event set and original L10 reference labels.",
            timing_scope="Main report and parameter run timers are concurrent-run observations; serial_timing is a separate single-GPU inference benchmark on a common training-memory intersection. Logged training duration is not exclusive timing.",
            timing_exclusivity_limit="Idle-start and process/memory checks show no observed contention; inferred PID attribution does not prove absolute exclusivity.",
            reference_target_policy="Every reference scheme uses identical canonical strict original-unit test targets; each scheme retains its own train/validation labels.",
            source_status_files_modified=False, figures=figures, issues=self.issues)
        write_json(self.output / "supplementary_summary.json", summary)
        self.markdown(summary)
        index_rows = []
        for name in ["supplementary_run_index.csv", "supplementary_metrics_seeds.csv", "supplementary_metrics.csv", "parameter_metrics.csv", "reference_metrics.csv", "rolling_metrics.csv", "frozen_retrieval_metrics.csv", "raft_raw_history_metrics.csv", "serial_timing_seeds.csv", "serial_timing_summary.csv"]:
            path = self.output / name
            if path.exists():
                index_rows.append(dict(file=name, sha256=digest(path), bytes=path.stat().st_size))
        pd.DataFrame(index_rows).to_csv(self.output / "supplementary_csv_index.csv", index=False)
        return summary

    def parameter_figures(self, frame):
        if frame.empty:
            return []
        selected = frame[(frame.experiment == "parameters") & (frame.metric == "MAE") & (frame.horizon == 11)]
        if selected.empty:
            return []
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        files = []
        for parameter, title in [("K", "K: fixed main test events"), ("L", "L: separate common L20-complete test set; fixed L10 reference labels"),
                                 ("quality_quantile", "Training quality cutoff: fixed main test events")]:
            rows = selected[selected.comparison.str.startswith(parameter + "=")].copy()
            if rows.empty:
                continue
            require(rows.target_set.nunique() == 1 and rows.n_events.nunique() == 1, "Parameter plot mixes test populations")
            rows["parameter_value"] = rows.comparison.str.split("=").str[1].astype(float)
            fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
            for axis, channel, unit in zip(axes, CHANNELS, ["Abnormal return", "Turnover (percentage points)", "Log volume"]):
                values = rows[rows.channel == channel].sort_values("parameter_value")
                axis.errorbar(values.parameter_value, values["mean"], yerr=values["std"].fillna(0), fmt="o-", capsize=4)
                axis.set(xlabel=parameter, ylabel="MAE (original units)", title=unit, xticks=values.parameter_value)
                axis.grid(alpha=.2)
            expected_values = {"K": {1, 3, 5, 10, 20}, "L": {5, 10, 20}, "quality_quantile": {0, .25, .5, .75}}[parameter]
            incomplete = "; partial parameter grid" if set(rows.parameter_value) != expected_values else ""
            if (rows.status != "completed").any():
                incomplete += "; incomplete seeds"
            fig.suptitle(f"{title}\nH=11; test n={int(rows.n_events.iloc[0])}; seed mean ± sample SD{incomplete}", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, .9))
            filename = "parameter_" + parameter + "_mae.png"
            fig.savefig(self.output / filename, dpi=180)
            plt.close(fig)
            files.append(filename)
        return files

    def markdown(self, summary):
        lines = [f"补充实验基于 {summary['main_events']} 个事件，主测试集包含 {summary['main_test_events']} 个事件。", "",
            "各指标先按种子计算原单位 MAE、RMSE，再报告种子均值和样本标准差。单个种子不报告标准差。", "",
            "| 实验 | 运行数 |", "|---|---:|"]
        names = {"suite": "主实验及扩展（含验证调参）", "parameters": "K、L、质量筛选参数", "references": "参考构造对照", "rolling": "滚动记忆", "frozen_retrieval": "冻结检索替换", "serial_timing": "串行推断计时", "raft_raw_history": "RAFT 原始响应记忆"}
        for group, row in summary["run_counts"].items():
            lines.append(f"| {names[group]} | {row['validated']} |")
        scope = summary["reference_scope"]
        if scope:
            lines += ["", f"参考构造对照包含严格匹配、受约束单近邻和单近邻，每个事件固定一个参考股票。三个方案使用 {scope['common_test_events']} 个严格公共测试事件，训练交集版本包含 {scope['common_training_intersection']} 个事件。主实验使用完整的 {summary['main_events']} 个事件。各方案保留自身训练与验证标签，测试目标统一为严格参考标签。"]
        lines += ["", "K 与质量筛选使用固定主测试集。L 使用独立的 L20 完整共同事件集，参考标签仍由原 L10 构造。图中为 H=11 的 MAE 均值和样本标准差。", "",
            "主报告和参数运行中的时间字段来自并行运行记录。serial_timing_summary.csv 单列 GPU 0 上的串行推断计时，使用相同训练记忆交集。训练日志耗时不作为独占运行时间。", "",
            "冻结检索替换沿用同一种子的主 RAIF 参数，仅改变检索排序。滚动记忆按照事件响应完成时间纳入历史事件。", "",
            "误差与计时数据的文件索引见 supplementary_csv_index.csv，实验设置见 supplementary_run_index.csv。"]
        if summary["remaining_experiments"]:
            lines += ["", "未完成部分：" + "、".join(summary["remaining_experiments"]) + "。具体缺失或校验失败记录见 supplementary_summary.json。"]
        (self.output / "SUPPLEMENTARY.md").write_text("\n".join(lines) + "\n")


def run(root, output):
    report = Report(root, output)
    main = report.checked("suite", "prepared_expanded", lambda: report.dataset(report.root / "data/prepared_expanded"))
    if main is not None:
        report.target("main_test", main["test_ids"], main["truth"])
    report.suite(main)
    report.parameters(main)
    report.references()
    report.extra(main, "rolling", "rolling_expanded")
    report.extra(main, "frozen", "frozen_retrieval_expanded")
    report.serial_timing(main)
    return report.finish(main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = run(options.root, options.output)
    print(json.dumps({key: result[key] for key in ["status", "run_counts", "remaining_experiments"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "completed" else 2)
