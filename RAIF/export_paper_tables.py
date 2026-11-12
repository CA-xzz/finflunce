"""Export measured paper tables without changing the manuscript or frozen runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["AR", "turnover", "log_volume"]
MODELS = ["HistoricalMean", "DLinear", "PatchTST", "iTransformer", "ECT", "RAFT", "RATL", "RAIF"]
SCHEMES = ["strict", "nearest", "expanded_nearest"]


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def wide(frame, keys):
    """Keep seed SD separate from mean; never pool seed RMSEs."""
    rows = []
    for names, group in frame.groupby(keys, sort=False, dropna=False):
        names = names if isinstance(names, tuple) else (names,)
        row = dict(zip(keys, names))
        assert not group.duplicated(["channel", "metric"]).any()
        for metric in group.itertuples():
            label = f"{metric.channel}_{metric.metric}"
            row[label + "_mean"] = metric.mean
            row[label + "_std"] = metric.std
        for field in ["n_events", "n_dates", "n_seeds", "status"]:
            if field in group:
                assert group[field].nunique(dropna=False) == 1
                row[field] = group[field].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def reference_paths(root):
    return (root / "data/reference_comparisons_expanded",
            root / "results/references_expanded",
            root / "data/reference_variants_expanded/strict")


def verify_quality(directory):
    normal = read_json(directory / "normalization.json")
    assert normal["q_rule"] == "exp_only" and normal["quality_neff_factor_enabled"] is False
    quality = pd.read_csv(directory / "reference_quality.csv")
    events = pd.read_csv(directory / "events.csv", dtype={"event_id": str})
    quality = quality.set_index("event_id").loc[events.event_id]
    expected = np.maximum(np.exp(-quality.balance.to_numpy() / normal["quality_balance_scale"]
                                 -quality.mean_distance.to_numpy() / normal["quality_distance_scale"]), 1e-8).astype(np.float32)
    arrays = dict(np.load(directory / "dataset.npz", allow_pickle=False))
    np.testing.assert_allclose(arrays["q"], expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(quality.q, expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(events.q, expected, rtol=1e-6, atol=1e-12)
    if "n_references" in quality:
        assert quality.n_references.eq(1).all()
    return dict(q_rule="exp_only", lower_floor=1e-8, quality_neff_factor_enabled=False,
                events_verified=len(events), dataset_sha256=digest(directory / "dataset.npz"))


def verify_references(root):
    """Recompute every revised reference error from the canonical test target."""
    comparison, results, strict = reference_paths(root)
    protocol = read_json(comparison / "comparison_protocol.json")
    status = read_json(results / "status.json")
    assert status["status"] == "completed" and status["datasets"] == 6
    assert status["seeds"] == [41, 42, 43]
    plan = pd.read_csv(comparison / "comparison_plan.csv")
    assert len(plan) == 6 and not plan.duplicated(["group", "scheme"]).any()
    assert set(zip(plan.group, plan.scheme)) == {(group, scheme) for group in ["full_training", "intersect_training"] for scheme in SCHEMES}
    selected = read_json(root / "results/suite_expanded/selected_configs.json")["RAIF"]["config"]
    canonical, records, validated = None, [], 0
    expected_runs = {f"{r.group}/{r.scheme}/seed{s}" for r in plan.itertuples() for s in [41, 42, 43]}
    assert {str(p.parent.relative_to(results)) for p in results.glob("**/status.json") if p.parent != results} == expected_runs
    for entry in plan.itertuples():
        directory = comparison / entry.group / entry.scheme
        normal = read_json(directory / "normalization.json")
        verify_quality(directory)
        events = pd.read_csv(directory / "events.csv", dtype={"event_id": str})
        target = dict(np.load(directory / "evaluation_targets.npz", allow_pickle=False))
        if canonical is None:
            canonical = target
        else:
            for key in ["event_ids", "truth"]:
                assert np.array_equal(canonical[key], target[key])
        assert len(target["event_ids"]) == entry.n_test == protocol["common_test_count"]
        assert events.split.eq("train").sum() == entry.n_train
        assert events.split.eq("validation").sum() == entry.n_validation
        if entry.group == "intersect_training":
            assert entry.n_train == protocol["common_training_count"]
        for seed in [41, 42, 43]:
            run = results / entry.group / entry.scheme / f"seed{seed}"
            assert read_json(run / "status.json")["status"] == "completed"
            config = read_json(run / "config.json")
            assert config["data_hash"] == digest(directory / "dataset.npz")
            assert config["normalization_hash"] == digest(directory / "normalization.json")
            assert config["model"] == "RAIF" and config["variant"] == entry.variant and config["seed"] == seed
            assert all(config[key] == value for key, value in selected.items())
            for code in ["train.py", "models.py"]:
                assert config["code_hashes"][code] == digest(root / code)
            pred = dict(np.load(run / "predictions.npz", allow_pickle=False))
            assert np.array_equal(events.iloc[pred["event_indices"]].event_id.to_numpy(str), target["event_ids"])
            assert pred["truth"].shape == pred["prediction"].shape == target["truth"].shape
            assert np.isfinite(pred["prediction"]).all()
            tolerance = 8 * np.finfo(np.float32).eps * (np.abs(target["truth"]) + np.asarray(normal["y_std"]) + np.abs(normal["y_mean"]))
            assert np.all(np.abs(pred["truth"] - target["truth"]) <= tolerance)
            metrics = pd.read_csv(run / "common_target_horizon_metrics.csv")
            assert len(metrics) == 30 and not metrics.duplicated(["channel", "horizon", "metric"]).any()
            for row in metrics.itertuples():
                error = (pred["prediction"] - target["truth"])[:, :row.horizon, CHANNELS.index(row.channel)]
                value = np.abs(error).mean() if row.metric == "MAE" else np.sqrt(np.square(error).mean())
                assert np.isclose(value, row.value, rtol=1e-10, atol=1e-12)
                records.append(dict(training_group=entry.group, reference_scheme=entry.scheme, seed=seed,
                                    channel=row.channel, horizon=row.horizon, metric=row.metric, value=value))
            validated += 1
    measured = pd.DataFrame(records)
    keys = ["training_group", "reference_scheme", "channel", "horizon", "metric"]
    recomputed = measured.groupby(keys).value.agg(["mean", "std"]).sort_index()
    aggregate = pd.read_csv(results / "reference_metrics.csv").set_index(keys).sort_index()
    assert recomputed.index.equals(aggregate.index)
    assert np.allclose(recomputed, aggregate[["mean", "std"]], rtol=1e-10, atol=1e-12)
    assert aggregate.n_seeds.eq(3).all() and aggregate.n_test.eq(protocol["common_test_count"]).all()
    assert aggregate.n_dates.nunique() == 1
    coverage = pd.read_csv(strict / "coverage.csv").set_index("split")
    config = read_json(strict / "config.json")
    assert coverage.n_events.to_dict() == {"train": 278, "validation": 82, "test": 280}
    assert config["strict_criteria"]["ar_correlation_minimum"] == .7
    assert config["industry_policy"] == "historical_effective_interval"
    assert config["industry_publication_times_verified"] is False
    return dict(status="passed", revision="single_reference", schemes=SCHEMES, validated_runs=validated,
                q_rule="exp_only", quality_neff_factor_enabled=False,
                canonical_test_events=protocol["common_test_count"],
                canonical_test_dates=int(aggregate.n_dates.iloc[0]),
                common_training_events=protocol["common_training_count"],
                strict_complete_events=int(coverage.n_events.sum()),
                strict_split_counts=coverage.n_events.to_dict(),
                strict_ar_correlation_minimum=config["strict_criteria"]["ar_correlation_minimum"],
                strict_dataset_sha256=digest(strict / "dataset.npz"),
                comparison_protocol_sha256=digest(comparison / "comparison_protocol.json"),
                reference_metrics_sha256=digest(results / "reference_metrics.csv"),
                independent_prediction_metric_recomputation=True)


def export(root, output):
    comparison, reference_results, strict_source = reference_paths(root)
    reference_check = verify_references(root)
    main_quality_check = verify_quality(root / "data/prepared_expanded")
    output.mkdir(parents=True, exist_ok=True)
    report, suite = root / "results/report_expanded", root / "results/suite_expanded"
    source_hashes, outputs = {}, []

    def read(path):
        source_hashes[str(path.relative_to(root))] = digest(path)
        return pd.read_csv(path)

    def save(frame, name):
        path = output / name
        frame.to_csv(path, index=False)
        outputs.append(dict(file=name, sha256=digest(path), rows=len(frame)))
        return frame

    events = read(root / "data/prepared_expanded/events.csv")
    manifest = read(root / "results/preflight_expanded/events_manifest.csv")
    flow = read(root / "data/prepared_expanded/sample_flow.csv")
    raw = read(root / "data/source_expanded/original_data/events.csv")
    decisions = read(root / "data/supplemental_market/all_event_decisions.csv")
    stat = [dict(item="original_recommendation_rows", value=len(decisions), unit="rows"),
            dict(item="identity_accepted_recommendation_rows", value=len(raw), unit="rows"),
            dict(item="identity_excluded_recommendation_rows", value=len(decisions)-len(raw), unit="rows"),
            dict(item="original_influencer_identifiers", value=decisions.original_influencer.nunique(), unit="identifiers"),
            dict(item="identity_accepted_influencer_identifiers", value=raw.bigv_name.nunique(), unit="identifiers"),
            dict(item="merged_stock_trading_date_events", value=len(manifest), unit="events"),
            dict(item="retained_model_events", value=len(events), unit="events"),
            dict(item="retained_over_all_merged_events", value=100*len(events)/len(manifest), unit="percent"),
            dict(item="retained_over_temporally_eligible_events", value=100*len(events)/manifest.split.ne("excluded").sum(), unit="percent")]
    for reason, n in manifest.loc[manifest.split.eq("excluded"), "exclusion_reason"].value_counts().items():
        stat.append(dict(item="temporal_exclusion_"+reason, value=n, unit="events"))
    save(pd.DataFrame(stat), "table1_data.csv")
    save(flow, "table1_sample_flow.csv")
    save(read(root / "data/prepared_expanded/coverage.csv"), "table1_splits.csv")

    overall = read(report / "overall_metrics.csv")
    horizons = read(report / "horizon_metrics.csv")
    checked = overall.set_index(["model", "channel", "metric"])[["mean", "std"]].sort_index()
    other = horizons[horizons.horizon.eq(11)].set_index(["model", "channel", "metric"])[["mean", "std"]].sort_index()
    assert checked.index.equals(other.index) and np.allclose(checked, other, equal_nan=True)
    save(wide(overall, ["model"]), "table2_overall.csv")
    save(wide(horizons, ["model", "horizon"]), "table3_horizons.csv")
    save(read(report / "step_metrics_seeds.csv"), "table3_step_metrics_seeds.csv")

    full = overall[overall.model.eq("RAIF")].assign(variant="full", retrieval="response")
    ablations = pd.concat([full, read(suite / "ablation_metrics.csv")], ignore_index=True)
    ablations = ablations[ablations.horizon.eq(11)]
    sizes = []
    for variant in ["full", "no_retrieval", "no_response", "no_events", "no_quality", "no_attention"]:
        directory = suite / ("main/RAIF" if variant == "full" else "ablation/"+variant)
        frames = [read(directory / f"seed{seed}/model_size.csv") for seed in [41, 42, 43]]
        frame = pd.concat(frames)
        assert frame.parameters.nunique() == frame.trainable_parameters.nunique() == 1
        sizes.append(dict(variant=variant, parameters=frame.parameters.iloc[0], trainable_parameters=frame.trainable_parameters.iloc[0]))
    save(wide(ablations, ["variant"]).merge(pd.DataFrame(sizes), on="variant", validate="one_to_one"), "table4_ablation.csv")

    retrieval = pd.concat([full, read(suite / "retrieval_metrics.csv")], ignore_index=True)
    retrieval = retrieval[retrieval.horizon.eq(11)].assign(evaluation="retrained")
    frozen = read(report / "frozen_retrieval_metrics.csv")
    frozen = frozen[frozen.horizon.eq(11)].assign(evaluation="frozen_checkpoint")
    save(wide(pd.concat([retrieval, frozen], ignore_index=True), ["evaluation", "retrieval"]), "table5a_retrieval.csv")

    (output / "reference_verification.json").write_text(json.dumps(reference_check, indent=2)+"\n")
    plan = read(comparison / "comparison_plan.csv")
    refs = read(reference_results / "reference_metrics.csv").rename(columns={"n_test": "n_events"})
    refs["comparison"] = refs.training_group + "/" + refs.reference_scheme
    refs = wide(refs[refs.horizon.eq(11)], ["comparison"])
    refs[["training_scope", "scheme"]] = refs.comparison.str.split("/", expand=True)
    sources = {"expanded_nearest": root / "data/prepared_expanded",
               "nearest": root / "data/reference_variants_expanded/nearest"}
    sources["strict"] = strict_source
    records = []
    for scheme, directory in sources.items():
        verify_quality(directory)
        quality = read(directory / "reference_quality.csv")
        coverage = read(directory / "coverage.csv")
        config_path = directory / "config.json"
        source_hashes[str(config_path.relative_to(root))] = digest(config_path)
        config = read_json(config_path)
        record = dict(scheme=scheme, original_complete_events=int(coverage.n_events.sum()),
                      original_coverage_percent=100*coverage.n_events.sum()/len(manifest), coverage_denominator=len(manifest),
                      reference_mode=config["reference_mode"])
        for split in ["train", "validation", "test"]:
            record["original_"+split+"_events"] = int(coverage.set_index("split").loc[split, "n_events"])
        for field in ["balance", "mean_distance", "q"]:
            record["original_"+field+"_median"] = quality[field].median()
            record["original_"+field+"_mean"] = quality[field].mean()
        record["original_quality_summary_scope"] = "original_complete_events_all_splits"
        records.append(record)
    metadata = pd.DataFrame(records)
    assert refs.n_events.eq(reference_check["canonical_test_events"]).all() and refs.n_seeds.eq(3).all()
    refs = refs.merge(plan.rename(columns={"group": "training_scope"})[["training_scope", "scheme", "n_train", "n_validation", "training_label_version", "evaluation_label_version"]], on=["training_scope", "scheme"], validate="one_to_one")
    refs = refs.merge(metadata, on="scheme", validate="many_to_one")
    training_quality = []
    for row in plan.itertuples():
        directory = comparison / row.group / row.scheme
        selected_events = read(directory / "events.csv")
        actual_quality = read(directory / "reference_quality.csv").set_index("event_id")
        selected_ids = selected_events.loc[selected_events.split.eq("train"), "event_id"]
        assert len(selected_ids) == row.n_train
        actual_quality = actual_quality.loc[selected_ids]
        item = dict(training_scope=row.group, scheme=row.scheme)
        for field in ["balance", "mean_distance", "q"]:
            item["training_"+field+"_median"] = actual_quality[field].median()
            item["training_"+field+"_mean"] = actual_quality[field].mean()
        # The saved array contains the quality actually used by the model.
        selected_q = selected_events.loc[selected_events.split.eq("train"), "q"].to_numpy()
        assert np.allclose(actual_quality.q, selected_q, rtol=1e-6, atol=1e-12)
        training_quality.append(item)
    refs = refs.merge(pd.DataFrame(training_quality), on=["training_scope", "scheme"], validate="one_to_one")
    save(refs, "table5b_reference.csv")
    sweep = read(root / "results/parameters_expanded/sweep_metrics.csv")
    retention = read(root / "data/quality_comparisons_expanded/retention.csv")
    retention = pd.concat([pd.DataFrame([dict(quantile=0, cutoff=0, n_train=int(events.split.eq("train").sum()), retained_fraction=1.0, n_test=int(events.split.eq("test").sum()))]), retention], ignore_index=True)
    q = sweep[sweep.parameter.eq("quality_quantile")].merge(retention, left_on="parameter_value", right_on="quantile", suffixes=("", "_declared"), validate="many_to_one")
    assert len(q) == 4 * 3 * 3 * 2 and q.seed.nunique() == 3
    assert np.array_equal(q.n_train, q.n_train_declared) and np.array_equal(q.n_test, q.n_test_declared)
    save(q.drop(columns=["n_train_declared", "n_test_declared"]), "quality_retention_metrics_seeds.csv")
    save(sweep, "parameter_metrics_seeds.csv")
    save(read(report / "group_metrics.csv"), "table7_original_condition_groups.csv")
    completion = root / "results/paper_completion"
    if (completion / "summary.json").exists():
        auxiliary = read(completion / "auxiliary_metrics.csv")
        save(auxiliary, "auxiliary_metrics.csv")
        groups = read(completion / "memory_group_metrics.csv")
        fixed = overall[overall.model.isin(["RAIF", "ECT"])].copy()
        fixed["setting"] = fixed.model.map({"RAIF": "RAIF_fixed", "ECT": "ECT_direct"})
        rolling = read(report / "rolling_metrics.csv")
        rolling = rolling[rolling.horizon.eq(11)].assign(setting="RAIF_rolling")
        global_metrics = pd.concat([fixed, rolling], ignore_index=True).assign(group_type="all_test", group="all")
        save(wide(pd.concat([groups, global_metrics], ignore_index=True), ["setting", "group_type", "group"]), "table7_generalization.csv")
        save(read(completion / "group_counts.csv"), "table7_group_counts.csv")

    efficiency = root / "results/serial_training_expanded"
    if (efficiency / "status.json").exists() and read_json(efficiency / "status.json")["status"] == "completed":
        train_cost = read(efficiency / "training_timing.csv")
        inference_cost = read(efficiency / "batch1_timing.csv")
        summaries = read(efficiency / "summary.csv")
        expected_pairs = {(model, seed) for model in MODELS for seed in ([41] if model == "HistoricalMean" else [41,42,43])}
        for frame in [train_cost, inference_cost]:
            assert len(frame) == 22 and set(zip(frame.model, frame.seed)) == expected_pairs
            assert frame.efficiency_only.all() and not frame.accuracy_recomputed.any()
        assert train_cost.training_common_memory_events.eq(912).all()
        assert train_cost.training_batch_size.eq(64).all()
        assert inference_cost.common_memory_events.eq(912).all()
        assert inference_cost.inference_batch_size.eq(1).all()
        assert inference_cost.inference_queries.eq(128).all() and inference_cost.inference_repeats.eq(5).all()
        assert inference_cost.inference_samples.eq(640).all()
        assert np.isfinite(train_cost.training_seconds).all() and train_cost.training_seconds.ge(0).all()
        assert np.isfinite(inference_cost.inference_event_ms).all() and inference_cost.inference_event_ms.gt(0).all()
        save(train_cost, "table6_training_seeds.csv")
        save(inference_cost, "table6_batch1_seeds.csv")
        save(summaries, "table6_measurements.csv")
        cost_rows = []
        for model in MODELS:
            row = dict(model=model, common_memory_events=912, training_batch_size=64, inference_batch_size=1,
                       retraining_runs=0 if model == "HistoricalMean" else 3,
                       inference_seed_count=1 if model == "HistoricalMean" else 3,
                       preprocessing_seconds_shared=read_json(root / "data/prepared_expanded/preparation_summary.json")["seconds"],
                       preprocessing_measurement_scope="original_reference_construction_measurement",
                       preprocessing_includes_quality_revision=False,
                       reference_selection_recomputed_for_quality_revision=False,
                       training_model_source="separate_common_capacity_retraining", inference_model_source="original_main_checkpoint_common_memory")
            for kind, prefix in [("same_capacity_retraining", "training"), ("main_checkpoint_batch1", "inference")]:
                group = summaries[(summaries.model == model) & (summaries.measurement == kind)]
                assert not group.metric.duplicated().any()
                for metric in group.itertuples():
                    row[prefix+"_"+metric.metric+"_mean"] = metric.mean
                    row[prefix+"_"+metric.metric+"_std"] = metric.std
            size = read(suite / "main" / model / "seed41/model_size.csv")
            row["parameters"] = size.parameters.iloc[0]
            row["trainable_parameters"] = size.trainable_parameters.iloc[0]
            row["frozen_base_parameters"] = size.frozen_base_parameters.iloc[0]
            cost_rows.append(row)
        save(pd.DataFrame(cost_rows), "table6_separate_training_and_original_checkpoint_cost.csv")

    aligned = root / "results/efficiency_aligned_expanded"
    if (aligned / "status.json").exists() and read_json(aligned / "status.json")["status"] == "completed":
        cost = read(aligned / "aligned_timing.csv")
        summary = read(aligned / "summary.csv")
        accuracy = read(aligned / "accuracy_summary.csv")
        assert len(cost) == 22 and set(zip(cost.model, cost.seed)) == expected_pairs
        assert cost.same_checkpoint_accuracy_training_and_inference.all() and cost.accuracy_recomputed.all()
        assert not cost.parameters_updated.any() and not cost.replaces_main_results.any()
        assert cost.training_common_memory_events.eq(912).all() and cost.inference_common_memory_events.eq(912).all()
        assert cost.inference_batch_size.eq(1).all() and cost.n_test.eq(2315).all()
        assert cost.inference_queries.eq(128).all() and cost.inference_samples.eq(640).all()
        checked_times = cost.merge(train_cost[["model", "seed", "training_seconds"]], on=["model", "seed"], suffixes=("", "_source"), validate="one_to_one")
        assert np.allclose(checked_times.training_seconds, checked_times.training_seconds_source, rtol=1e-12, atol=1e-12)
        arrays = dict(np.load(root / "data/prepared_expanded/dataset.npz", allow_pickle=False))
        normal = read_json(root / "data/prepared_expanded/normalization.json")
        ids = np.flatnonzero(arrays["split"] == 2)
        truth = arrays["y"][ids].astype(float)*np.asarray(normal["y_std"])+np.asarray(normal["y_mean"])
        metrics = read(aligned / "horizon_metrics.csv")
        for run in cost.itertuples():
            directory = aligned / run.model / f"seed{run.seed}"
            prediction = dict(np.load(directory / "predictions.npz", allow_pickle=False))
            assert np.array_equal(prediction["event_indices"], ids) and np.array_equal(prediction["truth"], truth)
            assert prediction["prediction"].shape == truth.shape and np.isfinite(prediction["prediction"]).all()
            if run.model != "HistoricalMean":
                checkpoint = efficiency / "training" / run.model / f"seed{run.seed}/checkpoint.pt"
                assert digest(checkpoint) == run.checkpoint_sha256
            observed = metrics[(metrics.model == run.model) & (metrics.seed == run.seed)]
            assert len(observed) == 30
            for point in observed.itertuples():
                error = (prediction["prediction"]-truth)[:, :point.horizon, CHANNELS.index(point.channel)]
                value = np.abs(error).mean() if point.metric == "MAE" else np.sqrt(np.square(error).mean())
                assert np.isclose(point.value, value, rtol=1e-7, atol=1e-10)
            samples = read(directory / "batch1_samples.csv")
            assert len(samples) == 640 and samples.batch_size.eq(1).all()
            assert np.isclose(samples.inference_ms.mean(), run.inference_event_ms, rtol=1e-12)
        aggregated = metrics.groupby(["model", "channel", "horizon", "metric"]).value.agg(["mean", "std"]).sort_index()
        supplied = accuracy.set_index(["model", "channel", "horizon", "metric"])[["mean", "std"]].sort_index()
        assert aggregated.index.equals(supplied.index) and np.allclose(aggregated, supplied, rtol=1e-12, atol=1e-14, equal_nan=True)
        save(cost, "table6_aligned_cost_seeds.csv")
        save(accuracy, "table6_same_capacity_accuracy.csv")
        rows = []
        for model in MODELS:
            row = dict(model=model, training_batch_size=64, inference_batch_size=1, common_memory_events=912,
                       training_and_inference_model="same_common_capacity_checkpoint", n_test=2315,
                       n_seeds=1 if model == "HistoricalMean" else 3,
                       preprocessing_seconds_shared=read_json(root / "data/prepared_expanded/preparation_summary.json")["seconds"])
            row.update(preprocessing_measurement_scope="original_reference_construction_measurement",
                       preprocessing_includes_quality_revision=False,
                       reference_selection_recomputed_for_quality_revision=False)
            for point in summary[summary.model.eq(model)].itertuples():
                row[point.metric+"_mean"], row[point.metric+"_std"] = point.mean, point.std
            for point in accuracy[accuracy.model.eq(model) & accuracy.horizon.eq(11)].itertuples():
                key = point.channel+"_"+point.metric
                row[key+"_mean"], row[key+"_std"] = point.mean, point.std
            size = read(suite / "main" / model / "seed41/model_size.csv")
            assert row["parameters_mean"] == size.parameters.iloc[0]
            row["parameters"] = size.parameters.iloc[0]
            row["trainable_parameters"] = size.trainable_parameters.iloc[0]
            row["frozen_base_parameters"] = size.frozen_base_parameters.iloc[0]
            rows.append(row)
        save(pd.DataFrame(rows), "table6_cost.csv")

    (output / "table_export_manifest.json").write_text(json.dumps(dict(
        primary_protocol="expanded_nearest", dataset_sha256=digest(root / "data/prepared_expanded/dataset.npz"),
        source_sha256=source_hashes, files=outputs,
        reference_revision="single_reference", reference_verification=reference_check,
        quality_rule=main_quality_check,
        reference_coverage_definition=f"Complete labelled events retained across all temporal partitions divided by 5686 merged stock/trading-date events; test error is evaluated on the same {reference_check['canonical_test_events']} canonical strict-reference events.",
        reference_attribute_scope="Strict Matching uses historical effective industry intervals and AR correlation >=0.7; first-publication dates remain unverified. Constrained Nearest Neighbor retains its source-update availability filter. Nearest Neighbor does not restrict industry or capitalization. Every event uses one fixed control.",
        manuscript_modified=False), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return outputs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    files = export(args.root, args.output)
    print(json.dumps(dict(status="exported", n_files=len(files), output=str(args.output))))
