"""Prepare reference-comparison datasets; this command never trains a model.

Each scheme keeps its own training and validation targets. Every scheme uses
the same complete test events and the same strict-reference test target.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from prepare import CHANNELS, EPS, digest, encode_categories, write_json


@dataclass
class Prepared:
    path: Path
    arrays: dict
    events: pd.DataFrame
    normal: dict
    quality: pd.DataFrame
    mode: str

    @classmethod
    def load(cls, path, mode):
        path = Path(path)
        with np.load(path / "dataset.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        events = pd.read_csv(path / "events.csv", dtype={"event_id": str})
        normal = json.loads((path / "normalization.json").read_text())
        config = json.loads((path / "config.json").read_text())
        quality = pd.read_csv(path / "reference_quality.csv", dtype={"event_id": str}).set_index("event_id")
        if mode is None:
            mode = config.get("reference_mode")
        if mode not in {"soft", "strict", "nearest", "expanded_nearest"} or config.get("reference_mode") != mode:
            raise ValueError(f"{path} is not a {mode} reference dataset")
        n = len(events)
        if events.event_id.duplicated().any() or any(len(a) != n for a in arrays.values()):
            raise ValueError(f"Invalid event/array alignment: {path}")
        if quality.index.has_duplicates or set(events.event_id) != set(quality.index):
            raise ValueError(f"Invalid quality/event alignment: {path}")
        if events.label_version.nunique() != 1 or str(events.label_version.iloc[0]) != normal["label_version"]:
            raise ValueError(f"Inconsistent source label versions: {path}")
        if arrays["y"].shape != (n, 11, 3) or not np.isin(arrays["split"], [0, 1, 2]).all():
            raise ValueError(f"Invalid shape or splits: {path}")
        for key in ["x", "y", "q"]:
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"Nonfinite {key}: {path}")
        return cls(path, arrays, events, normal, quality, mode)

    def ids(self, split):
        return set(self.events.loc[self.arrays["split"] == split, "event_id"])

    def raw(self, name):
        return self.arrays[name].astype(np.float64) * np.asarray(self.normal[name + "_std"]) + np.asarray(self.normal[name + "_mean"])


def common_ids(sources, split):
    common = set.intersection(*(data.ids(split) for data in sources.values()))
    if not common:
        raise ValueError(f"No common complete {'training' if split == 0 else 'test'} events; thresholds were not relaxed")
    strict = sources["strict"]
    ordered = strict.events.loc[(strict.arrays["split"] == split) & strict.events.event_id.isin(common), "event_id"].tolist()
    strict_metadata = strict.events.assign(day_array=strict.arrays["day"], end_array=strict.arrays["end_day"]).set_index("event_id")
    for source in sources.values():
        candidate = source.events.assign(day_array=source.arrays["day"], end_array=source.arrays["end_day"]).set_index("event_id")
        for field in ["event_date", "day_array", "end_array"]:
            if not np.array_equal(candidate.loc[ordered, field].to_numpy(), strict_metadata.loc[ordered, field].to_numpy()):
                raise ValueError(f"Shared events have inconsistent {field}")
    return ordered


def quality_on_intersection(source, events, training):
    quality = source.quality.loc[events.event_id].copy()
    s_b = max(float(np.median(quality.balance.to_numpy()[training])), EPS)
    s_d = max(float(np.median(quality.mean_distance.to_numpy()[training])), EPS)
    neff = quality.effective_n.to_numpy()
    q = np.exp(-quality.balance.to_numpy() / s_b - quality.mean_distance.to_numpy() / s_d)
    if source.normal.get("quality_effective_n_factor", True):
        q = q * neff / (neff + 1)
    quality["q"] = np.maximum(q, EPS).astype(np.float32)
    return quality, s_b, s_d


def prepare_variant(source, strict, output, test_ids, test_truth, train_ids, group, scheme,
                    variant, all_verified, comparison_hash):
    lookup = {event_id: i for i, event_id in enumerate(source.events.event_id)}
    train_keep = source.arrays["split"] == 0
    if train_ids is not None:
        train_keep &= source.events.event_id.isin(train_ids).to_numpy()
    train_positions = np.flatnonzero(train_keep)
    validation_positions = np.flatnonzero(source.arrays["split"] == 1)
    test_positions = np.array([lookup[event_id] for event_id in test_ids], dtype=int)
    positions = np.r_[train_positions, validation_positions, test_positions]
    if len(train_positions) == 0 or len(validation_positions) == 0:
        raise ValueError(f"{scheme}: empty training or validation partition")
    arrays = {name: values[positions].copy() for name, values in source.arrays.items()}
    events = source.events.iloc[positions].copy().reset_index(drop=True)
    training, testing = arrays["split"] == 0, arrays["split"] == 2
    normal = dict(source.normal)
    original_y = source.raw("y")[positions]
    raw_y = original_y.copy()
    # This operation touches test rows only; validation retains its own label.
    raw_y[testing] = test_truth
    np.testing.assert_array_equal(raw_y[~testing], original_y[~testing])
    if train_ids is None:
        # Keep original training bytes and scales, avoiding a needless roundtrip.
        arrays["y"][testing] = ((test_truth - np.asarray(normal["y_mean"])) / np.asarray(normal["y_std"])).astype(np.float32)
        np.testing.assert_array_equal(arrays["y"][training], source.arrays["y"][train_positions])
        quality = source.quality.loc[events.event_id].copy()
    else:
        raw_x = source.raw("x")[positions]
        for name, values in [("x", raw_x), ("y", raw_y)]:
            mean = values[training].mean(axis=(0, 1))
            std = np.maximum(values[training].std(axis=(0, 1)), EPS)
            normal[name + "_mean"], normal[name + "_std"] = mean.tolist(), std.tolist()
            arrays[name] = ((values - mean) / std).astype(np.float32)
        for category in ["platform", "influencer"]:
            arrays[category], vocabulary = encode_categories(events[category].to_numpy(), training)
            normal[category + "_vocab"] = vocabulary
            normal["n_" + ("platforms" if category == "platform" else "influencers")] = len(vocabulary)
        quality, s_b, s_d = quality_on_intersection(source, events, training)
        arrays["q"] = quality.q.to_numpy(np.float32)
        normal["quality_balance_scale"], normal["quality_distance_scale"] = s_b, s_d
    events["q"] = arrays["q"]
    training_version = str(source.normal["label_version"])
    evaluation_version = str(strict.normal["label_version"])
    contract = {"group": group, "scheme": scheme, "reference_mode": source.mode,
        "required_model": "RAIF", "required_variant": variant,
        "training_label_version": training_version, "validation_label_version": training_version,
        "evaluation_label_version": evaluation_version, "comparison_hash": comparison_hash,
        "evaluation_scope": "Diagnostic comparison on the common complete strict-reference test subset; source main datasets remain unchanged",
        "training_event_ids": events.loc[training, "event_id"].tolist(), "test_event_ids": test_ids,
        "validation_targets": "Own reference labels; no test-based checkpoint or hyperparameter selection",
        "intersection_refit": train_ids is not None,
        "reference_selection_refit": False,
        "normalization_and_quality_fit": "Selected training events only",
        "quality_policy": "Observed reference quality retained; no_quality model variant removes its three model/loss effects" if variant == "no_quality" else "Quality-weighted RAIF",
        "floating_point_policy": "Canonical original-unit targets are exactly identical; dataset float32 roundtrip is checked with a scale-dependent rounding tolerance."}
    if source.normal.get("quality_effective_n_factor") is False:
        contract["quality_effective_n_factor"] = False
        contract["quality_neff_factor_enabled"] = False
        contract["q_rule"] = "exp_only"
        contract["quality_formula"] = "max(exp(-balance/s_b - mean_distance/s_d), 1e-8)"
    version = "comparison_" + hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
    # train.py requires a single dataset contract ID; per-partition target
    # identities remain separately explicit for downstream reporting.
    events["label_version"] = version
    events["training_label_version"] = training_version
    events["evaluation_label_version"] = evaluation_version
    events["row_target_label_version"] = np.where(testing, evaluation_version, training_version)
    normal.update(label_version=version, training_label_version=training_version,
                  evaluation_label_version=evaluation_version, reference_comparison=contract,
                  normalization_fit_event_ids=events.loc[training, "event_id"].tolist(),
                  ready_for_formal_training=bool(all_verified))
    output.mkdir(parents=True)
    np.savez_compressed(output / "dataset.npz", **arrays)
    events.to_csv(output / "events.csv", index=False)
    write_json(output / "normalization.json", normal)
    write_json(output / "config.json", contract)
    quality.reset_index().to_csv(output / "reference_quality.csv", index=False)
    canonical = test_truth.copy()
    np.testing.assert_array_equal(canonical, test_truth)
    np.savez_compressed(output / "evaluation_targets.npz", event_ids=np.asarray(test_ids, dtype=str), truth=canonical)
    recovered = arrays["y"][testing].astype(np.float64) * np.asarray(normal["y_std"]) + np.asarray(normal["y_mean"])
    tolerance = 8 * np.finfo(np.float32).eps * (np.abs(canonical) + np.asarray(normal["y_std"]) + np.abs(normal["y_mean"]))
    if not np.all(np.abs(recovered - canonical) <= tolerance):
        raise AssertionError("Test targets differ beyond float32 normalization rounding")
    manifest = events[["event_id", "event_date", "split", "label_version", "training_label_version",
                       "evaluation_label_version", "row_target_label_version"]].copy()
    manifest["source_array_position"] = positions
    manifest.to_csv(output / "sample_manifest.csv", index=False)
    manifest.loc[testing].to_csv(output / "evaluation_manifest.csv", index=False)
    events.groupby("split").agg(n_events=("event_id", "size"), n_dates=("event_date", "nunique")).to_csv(output / "coverage.csv")
    if (source.path / "coverage.csv").exists():
        shutil.copy2(source.path / "coverage.csv", output / "original_coverage.csv")
    flow = []
    for split, name in [(0, "train"), (1, "validation"), (2, "test")]:
        flow.append({"split": name, "original_generated_events": int((source.arrays["split"] == split).sum()),
                     "comparison_events": int((arrays["split"] == split).sum())})
    pd.DataFrame(flow).to_csv(output / "sample_flow.csv", index=False)
    source_files = [source.path / "dataset.npz", source.path / "normalization.json", source.path / "reference_quality.csv",
                    strict.path / "dataset.npz", strict.path / "normalization.json"]
    pd.DataFrame([{"path": str(path), "sha256": digest(path)} for path in source_files]).to_csv(output / "source_manifest.csv", index=False)
    return {"group": group, "scheme": scheme, "dataset": str(output), "model": "RAIF", "variant": variant,
            "n_train": int(training.sum()), "n_validation": int((arrays["split"] == 1).sum()),
            "n_test": int(testing.sum()), "training_label_version": training_version,
            "evaluation_label_version": evaluation_version, "dataset_contract": version}


def run(soft, strict, nearest, output, variants=None, single_reference=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Comparison output is not empty; choose a fresh directory")
    source_paths = [("strict", strict), ("nearest", nearest)]
    if not single_reference:
        if soft is None:
            raise ValueError("Legacy reference comparisons require a soft dataset")
        source_paths.insert(0, ("soft", soft))
    elif soft is not None:
        raise ValueError("Single-reference comparisons do not accept a soft dataset")
    sources = {name: Prepared.load(path, name) for name, path in source_paths}
    extra = variants or {}
    if single_reference and set(extra) != {"expanded_nearest"}:
        raise ValueError("Single-reference comparisons require exactly expanded_nearest as the third scheme")
    for name, path in extra.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in {"soft", "strict", "nearest", "soft_quality", "soft_uniform"}:
            raise ValueError("Invalid or reserved extra reference variant name: " + name)
        sources[name] = Prepared.load(path, None)
    if single_reference:
        for name, source in sources.items():
            config = json.loads((source.path / "config.json").read_text())
            if source.mode != name:
                raise ValueError(f"{name}: reference mode differs from the declared single-reference scheme")
            if config.get("m") != 1 or source.normal.get("quality_effective_n_factor") is not False:
                raise ValueError(f"{name}: single-reference comparisons require M=1 and the quality formula without effective-N factor")
            if not source.quality.effective_n.eq(1).all():
                raise ValueError(f"{name}: a multi-reference quality record is present")
            expected_q = np.maximum(np.exp(-source.quality.loc[source.events.event_id, "balance"].to_numpy() /
                source.normal["quality_balance_scale"] - source.quality.loc[source.events.event_id, "mean_distance"].to_numpy() /
                source.normal["quality_distance_scale"]), EPS).astype(np.float32)
            np.testing.assert_allclose(source.arrays["q"], expected_q, rtol=1e-6, atol=0)
    if len({source.arrays["x"].shape[1:] for source in sources.values()}) != 1:
        raise ValueError("Reference comparisons require identical input windows/features")
    test_ids, train_ids = common_ids(sources, 2), common_ids(sources, 0)
    strict_data = sources["strict"]
    strict_lookup = {event_id: i for i, event_id in enumerate(strict_data.events.event_id)}
    test_truth = strict_data.raw("y")[[strict_lookup[event_id] for event_id in test_ids]]
    all_verified = all(source.normal.get("ready_for_formal_training") is True for source in sources.values())
    comparison_hash = hashlib.sha256(test_truth.tobytes() + json.dumps(test_ids).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"event_id": test_ids}).to_csv(output / "common_test_events.csv", index=False)
    pd.DataFrame({"event_id": train_ids}).to_csv(output / "common_training_events.csv", index=False)
    plans = []
    schemes = [("strict", "strict", "full"), ("nearest", "nearest", "full")]
    if not single_reference:
        schemes = [("soft_quality", "soft", "full"), ("soft_uniform", "soft", "no_quality")] + schemes
    schemes.extend((name, name, "full") for name in extra)
    for group, intersection in [("full_training", None), ("intersect_training", train_ids)]:
        for scheme, mode, variant in schemes:
            plans.append(prepare_variant(sources[mode], strict_data, output / group / scheme,
                test_ids, test_truth, intersection, group, scheme, variant, all_verified, comparison_hash))
    for plan in plans:
        with np.load(Path(plan["dataset"]) / "evaluation_targets.npz", allow_pickle=False) as targets:
            np.testing.assert_array_equal(targets["event_ids"], np.asarray(test_ids))
            np.testing.assert_array_equal(targets["truth"], test_truth)
    pd.DataFrame(plans).to_csv(output / "comparison_plan.csv", index=False)
    protocol = {"test_reference": "strict", "common_test_count": len(test_ids),
        "common_training_count": len(train_ids), "ready_for_formal_training": all_verified,
        "canonical_test_hash": comparison_hash, "models_started": False,
        "evaluation_scope": "Diagnostic comparison on the common complete strict-reference test subset; does not restrict any source main sample",
        "training_labels": "Each scheme's original reference construction",
        "validation_labels": "Each scheme's original reference construction",
        "intersection_behavior": "Restrict training IDs, refit x/y scales, category vocabulary, and q median scales; original selected controls and their labels remain fixed",
        "original_coverage_preserved": True, "required_variants": {a: c for a, _, c in schemes}}
    if single_reference:
        protocol.update(single_reference=True, quality_effective_n_factor=False,
            quality_neff_factor_enabled=False, q_rule="exp_only",
            quality_formula="max(exp(-balance/s_b - mean_distance/s_d), 1e-8)",
            reference_schemes={"strict": "Strict Matching", "nearest": "Constrained Nearest Neighbor",
                               "expanded_nearest": "Nearest Neighbor"}, reference_count=1)
    write_json(output / "comparison_protocol.json", protocol)
    print(json.dumps({"status": "prepared", "datasets": len(plans), "n_test": len(test_ids),
                      "n_training_intersection": len(train_ids), "ready_for_formal_training": all_verified}), flush=True)
    return plans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["strict", "nearest", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--soft", type=Path)
    parser.add_argument("--single-reference", action="store_true",
                        help="Compare exactly Strict Matching, Constrained Nearest Neighbor, and Nearest Neighbor")
    parser.add_argument("--expanded-nearest", type=Path, help="Third dataset in --single-reference mode")
    parser.add_argument("--variants", nargs="*", default=[], metavar="NAME=PREPARED_DIR",
                        help="Extra reference datasets included in the common test and training intersections")
    args = parser.parse_args()
    variants = {}
    for entry in args.variants:
        name, separator, path = entry.partition("=")
        if not separator or not path or name in variants:
            parser.error("Each extra variant needs a unique NAME=PREPARED_DIR")
        variants[name] = Path(path)
    if args.expanded_nearest is not None:
        if "expanded_nearest" in variants:
            parser.error("Supply expanded_nearest only once")
        variants["expanded_nearest"] = args.expanded_nearest
    run(args.soft, args.strict, args.nearest, args.output, variants, single_reference=args.single_reference)
