"""Export the five-method comparison from completed, source-bound predictions.

This script never trains models or changes the original main experiment.
All reported errors use the primary dataset's inverse-standardized Delta.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("AR", "turnover", "log_volume")
SEEDS = (41, 42, 43)
METHODS = ("ECT-Direct", "ECT-Incremental", "RAFT-Direct", "RAFT-Incremental", "RAIF")
CAPTION = "Comparison of Full-Trajectory and Incremental-Response Forecasting"
LABEL = "tab:incremental_comparison"
OPTIMIZER_FIELDS = ("epochs", "patience", "batch_size", "lr", "weight_decay", "huber_delta", "lambda_ret")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def exact(actual, expected, name):
    require(np.array_equal(actual, expected), f"{name}: array differs")


def inverse_target(arrays, normal):
    return arrays["y"].astype(np.float64) * np.asarray(normal["y_std"]) + np.asarray(normal["y_mean"])


def check_predictions(arrays, indices, truth, name):
    require({"prediction", "truth", "event_indices"}.issubset(arrays), f"{name}: missing NPZ fields")
    ids = arrays["event_indices"]
    require(np.issubdtype(ids.dtype, np.integer), f"{name}: indices must be integers")
    exact(ids, indices, f"{name}/indices")
    for key in ("prediction", "truth"):
        require(arrays[key].dtype == np.float64, f"{name}/{key}: expected physical-unit float64")
        require(arrays[key].shape == truth.shape, f"{name}/{key}: invalid shape")
        require(np.isfinite(arrays[key]).all(), f"{name}/{key}: nonfinite values")
    exact(arrays["truth"], truth, f"{name}/truth")
    return arrays["prediction"]


def metric_records(method, seed, prediction, truth, n_dates):
    require(prediction.shape == truth.shape and prediction.ndim == 3 and prediction.shape[2] == 3,
            "Metric arrays must have shape [events, horizon, 3]")
    require(np.isfinite(prediction).all() and np.isfinite(truth).all(), "Metric arrays must be finite")
    error = prediction - truth
    values = {"MAE": np.abs(error).mean(axis=(0, 1)), "RMSE": np.sqrt(np.square(error).mean(axis=(0, 1)))}
    return [dict(method=method, seed=seed, channel=channel, metric=metric, value=float(values[metric][i]),
                 n_events=len(truth), n_dates=n_dates, horizon=truth.shape[1])
            for i, channel in enumerate(CHANNELS) for metric in ("MAE", "RMSE")]


def aggregate_metrics(records):
    frame = pd.DataFrame(records)
    require(len(frame) == len(METHODS) * len(SEEDS) * 6, "Expected exactly 90 seed/metric rows")
    require(not frame.duplicated(["method", "seed", "channel", "metric"]).any(), "Duplicate metric keys")
    rows = []
    for method in METHODS:
        row = {"method": method}
        for channel in CHANNELS:
            for metric in ("MAE", "RMSE"):
                group = frame[(frame.method == method) & (frame.channel == channel) & (frame.metric == metric)]
                require(set(group.seed) == set(SEEDS), f"Missing seeds: {method}/{channel}/{metric}")
                values = group.sort_values("seed").value.to_numpy(np.float64)
                row[f"{channel}_{metric}_mean"] = float(values.mean())
                row[f"{channel}_{metric}_std"] = float(values.std(ddof=1))
        for field in ("n_events", "n_dates", "horizon"):
            require(frame[field].nunique() == 1, f"Inconsistent {field}")
            row[field] = int(frame[field].iloc[0])
        row["n_seeds"] = len(SEEDS)
        rows.append(row)
    return frame, pd.DataFrame(rows)


class Inputs:
    """Bind every consumed file and detect changes during export."""

    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path)
        name = path.relative_to(self.root).as_posix()
        value = digest(path)
        require(expected is None or value == expected, f"Hash mismatch: {name}")
        require(name not in self.hashes or self.hashes[name] == value, f"Input changed: {name}")
        self.hashes[name] = value
        return value

    def json(self, path):
        self.bind(path)
        return json.loads(path.read_text())

    def npz(self, path):
        self.bind(path)
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def csv(self, path, **kwargs):
        self.bind(path)
        return pd.read_csv(path, **kwargs)

    def unchanged(self):
        for name, value in self.hashes.items():
            require(digest(self.root / name) == value, f"Input changed during export: {name}")


def load_data(inputs):
    root = inputs.root
    primary = root / "data/prepared_expanded"
    directory = root / "data/trajectory_comparison"
    normal = inputs.json(primary / "normalization.json")
    arrays = inputs.npz(primary / "dataset.npz")
    events = inputs.csv(primary / "events.csv", dtype={"event_id": str})
    protocol = inputs.json(directory / "protocol.json")
    verification = inputs.json(directory / "verification.json")
    require(verification["status"] == "passed", "Trajectory data verification has not passed")
    primary_hash = inputs.bind(primary / "dataset.npz", protocol["source_dataset_sha256"])
    normal_hash = inputs.bind(primary / "normalization.json")
    inputs.bind(directory / "paired_arrays.npz", verification["paired_arrays_sha256"])
    inputs.bind(directory / "event_alignment.csv", verification["event_alignment_sha256"])
    paired = inputs.npz(directory / "paired_arrays.npz")
    require(normal["q_rule"] == "exp_only" and normal["quality_neff_factor_enabled"] is False,
            "Primary quality rule differs from the retained experiment")
    require(len(events) == 5105 and not events.event_id.duplicated().any(), "Expected 5105 unique primary events")
    indices = {name: np.flatnonzero(arrays["split"] == i) for i, name in enumerate(("train", "validation", "test"))}
    require({key: len(value) for key, value in indices.items()} == {"train": 2170, "validation": 620, "test": 2315},
            "Primary partitions changed")
    require(arrays["x"].shape == (5105, 10, 3) and arrays["y"].shape == (5105, 11, 3), "Primary tensor dimensions changed")
    canonical = inverse_target(arrays, normal)
    exact(paired["canonical_delta"], canonical, "paired/canonical_delta")
    exact(paired["event_id"], events.event_id.to_numpy(), "paired/event_id")
    for key in ("split", "day", "end_day"):
        exact(paired[key], arrays[key], f"paired/{key}")
    prepared = {}
    for role in ("recommended_full", "reference_full"):
        source = directory / role
        hashes = verification["datasets"][role]
        for name in ("dataset", "normalization", "events"):
            suffix = {"dataset": "npz", "normalization": "json", "events": "csv"}[name]
            inputs.bind(source / f"{name}.{suffix}", hashes[f"{name}_sha256"])
        data, norm = inputs.npz(source / "dataset.npz"), inputs.json(source / "normalization.json")
        config = inputs.json(source / "config.json")
        frame = inputs.csv(source / "events.csv", dtype={"event_id": str})
        require(config["source_dataset_sha256"] == primary_hash, f"{role}: primary source hash differs")
        require(norm["source_primary_normalization_sha256"] == normal_hash, f"{role}: primary normal hash differs")
        exact(frame.event_id.to_numpy(), events.event_id.to_numpy(), f"{role}/event_ids")
        for key in ("split", "day", "end_day", "q"):
            exact(data[key], arrays[key], f"{role}/{key}")
        if role == "recommended_full":
            for key in ("x", "platform", "influencer"):
                exact(data[key], arrays[key], f"{role}/{key}")
            exact(norm["y_std"], normal["y_std"], f"{role}/Delta target scale")
        else:
            require(not data["platform"].any() and not data["influencer"].any(), "Reference categorical inputs are not zero")
        prepared[role] = (data, norm, hashes)
    return dict(primary_hash=primary_hash, normal_hash=normal_hash, arrays=arrays, normal=normal,
                events=events, indices=indices, canonical=canonical, prepared=prepared)


def load_main(inputs, data):
    root = inputs.root
    evidence = inputs.json(root / "results/single_reference_revision/independent_predictions_validation.json")
    require(evidence["status"] == "passed", "Original main predictions have not passed independent verification")
    original_runs = evidence["original_device_replay"]["runs"]
    runs = {entry["run"]: entry for entry in original_runs}
    require(len(runs) == len(original_runs), "Duplicate original replay entries")
    output = {}
    for model in ("ECT", "RAFT", "RAIF"):
        for seed in SEEDS:
            directory = root / "results/suite_expanded/main" / model / f"seed{seed}"
            config, status = inputs.json(directory / "config.json"), inputs.json(directory / "status.json")
            name = directory.relative_to(root).as_posix()
            require(status["status"] == "completed" and status["mode"] == "full", f"Incomplete source: {name}")
            require(config["model"] == model and config["seed"] == seed, f"Source model/seed differs: {name}")
            require(config["data_hash"] == data["primary_hash"] and config["normalization_hash"] == data["normal_hash"],
                    f"Source primary hashes differ: {name}")
            require((config["n_train"], config["n_validation"], config["n_test"], config["epochs"], config["patience"])
                    == (2170, 620, 2315, 100, 15), f"Source partition/training settings changed: {name}")
            for code in ("models.py", "train.py"):
                inputs.bind(root / code, config["code_hashes"][code])
            replay = runs[name]
            require(replay["exact_equal"] is True and replay["n_test"] == 2315, f"Source replay is incomplete: {name}")
            require(replay["data_sha256"] == data["primary_hash"] and replay["normalization_sha256"] == data["normal_hash"],
                    f"Source replay belongs to different data: {name}")
            inputs.bind(directory / "checkpoint.pt", replay["checkpoint_sha256"])
            inputs.bind(directory / "predictions.npz", replay["prediction_sha256"])
            archive = inputs.npz(directory / "predictions.npz")
            prediction = check_predictions(archive, data["indices"]["test"], data["canonical"][data["indices"]["test"]], name)
            output[(model, seed)] = dict(directory=directory, config=config, status=status, prediction=prediction)
    return output


def check_retrieval(inputs, path, data, indices):
    frame = inputs.csv(path, dtype={"query_id": str, "historical_id": str})
    events = data["events"].set_index("event_id")
    expected_queries = data["events"].iloc[indices].event_id
    require(set(frame.query_id) == set(expected_queries), f"{path.name}: query coverage differs")
    require(len(frame) == len(indices) * 15, f"{path.name}: expected 15 retrieved items per query")
    require(set(frame.selection_group) == {1, 2, 4}, f"{path.name}: scale groups differ")
    require(frame.groupby(["query_id", "selection_group"]).size().eq(5).all(), f"{path.name}: Top-5 per scale differs")
    require(not frame.duplicated(["query_id", "selection_group", "historical_id"]).any(), f"{path.name}: duplicate history within scale")
    require(not frame.duplicated(["query_id", "rank"]).any(), f"{path.name}: duplicate retrieval rank")
    require(set(frame.historical_id).issubset(set(events.index)), f"{path.name}: unknown history")
    history, queries = events.loc[frame.historical_id], events.loc[frame.query_id]
    require(history.split.eq("train").all(), f"{path.name}: non-training history")
    require((history.end_day.to_numpy() < queries.day.to_numpy()).all(), f"{path.name}: immature history")
    exact(frame.end_day.to_numpy(), history.end_day.to_numpy(), f"{path.name}/history day")
    exact(frame.query_day.to_numpy(), queries.day.to_numpy(), f"{path.name}/query day")
    return len(frame)


def load_components(inputs, data, main):
    root = inputs.root
    output, checks = {}, []
    for seed in SEEDS:
        for role, model in (("reference", "ECT"), ("ect_direct", "ECT"), ("raft_direct", "RAFT")):
            directory = root / "results/trajectory_comparison/components" / role / f"seed{seed}"
            config, status = inputs.json(directory / "config.json"), inputs.json(directory / "status.json")
            name = directory.relative_to(root).as_posix()
            require(status["status"] == "completed", f"Incomplete component: {name}")
            require((config["role"], config["model"], config["seed"]) == (role, model, seed), f"Component identity differs: {name}")
            require((status["role"], status["model"], status["seed"]) == (role, model, seed), f"Component status identity differs: {name}")
            require(config["primary_data_hash"] == data["primary_hash"] and config["primary_normalization_hash"] == data["normal_hash"],
                    f"Component primary source differs: {name}")
            source = main[(model, seed)]
            inputs.bind(source["directory"] / "config.json", config["source_main_config_sha256"])
            require(config["model_config"] == source["config"]["model_config"], f"Component architecture changed: {name}")
            require(config["optimizer_settings"] == {key: source["config"][key] for key in OPTIMIZER_FIELDS},
                    f"Component optimizer settings changed: {name}")
            require((config["n_train"], config["n_validation"], config["n_test"]) == (2170, 620, 2315), f"Component partitions changed: {name}")
            require(config["query_future_inputs"] is False and config["true_reference_future_used_for_prediction"] is False,
                    f"Component uses query/reference future inputs: {name}")
            require(set(config["query_fields"]) == {"x", "platform", "influencer", "day", "index"}, f"Query input contract changed: {name}")
            for code in ("models.py", "train.py", "train_trajectory_component.py"):
                inputs.bind(root / code, config["code_hashes"][code])
            inputs.bind(directory / "checkpoint.pt", status["checkpoint_sha256"])
            prepared_role = "reference_full" if role == "reference" else "recommended_full"
            arrays, normal, hashes = data["prepared"][prepared_role]
            require(config["data_hash"] == hashes["dataset_sha256"] and config["normalization_hash"] == hashes["normalization_sha256"],
                    f"Component target data hashes differ: {name}")
            expected_objective = "standardized_B_MAE" if role == "reference" else "standardized_incremental_MAE_with_frozen_predicted_B"
            require(config["validation_objective"] == status["validation_objective"] == expected_objective,
                    f"Component validation objective changed: {name}")
            if role != "reference":
                reference = output[("reference", seed)]
                for filename, field in (("checkpoint.pt", "checkpoint_sha256"),
                                        ("validation_predictions.npz", "validation_predictions_sha256"),
                                        ("test_predictions.npz", "test_predictions_sha256")):
                    require(config["reference_" + field] == reference["status"][field],
                            f"Shared reference binding differs: {name}/{field}")
                    inputs.bind(reference["directory"] / filename, config["reference_" + field])
            targets = inverse_target(arrays, normal)
            predictions = {}
            retrieval_count = 0
            for split in ("validation", "test"):
                indices = data["indices"][split]
                inputs.bind(directory / f"{split}_predictions.npz", status[f"{split}_predictions_sha256"])
                archive = inputs.npz(directory / f"{split}_predictions.npz")
                predictions[split] = check_predictions(archive, indices, targets[indices], f"{name}/{split}")
                if role == "raft_direct":
                    retrieval_count += check_retrieval(inputs, directory / f"{split}_retrieved_events.csv", data, indices)
            indices = data["indices"]["validation"]
            if role == "reference":
                validation_mae = float(np.abs((predictions["validation"] - targets[indices]) / np.asarray(normal["y_std"])).mean())
                # Training computes this reduction in float32; export uses float64.
                np.testing.assert_allclose(validation_mae, status["validation_standardized_mae"], rtol=2e-6, atol=2e-7)
            else:
                baseline = output[("reference", seed)]["predictions"]["validation"]
                validation_mae = float(np.abs((predictions["validation"] - baseline - data["canonical"][indices])
                                            / np.asarray(data["normal"]["y_std"])).mean())
                np.testing.assert_allclose(validation_mae, status["validation_standardized_mae"], rtol=1e-12, atol=1e-12)
            output[(role, seed)] = dict(directory=directory, predictions=predictions, config=config, status=status)
            checks.append(dict(role=role, seed=seed, best_epoch=int(status["best_epoch"]),
                               validation_objective=expected_objective, recomputed_validation_mae=validation_mae,
                               saved_validation_mae=status["validation_standardized_mae"], retrieved_rows_checked=retrieval_count,
                               checkpoint_sha256=status["checkpoint_sha256"]))
    return output, checks


def format_cell(row, channel, metric, latex=False):
    separator = r" $\pm$ " if latex else " ± "
    return f"{row[f'{channel}_{metric}_mean']:.6f}{separator}{row[f'{channel}_{metric}_std']:.6f}"


def prediction_form(method):
    return "Full" if method.endswith("-Direct") else "Incremental"


def latex_table(table):
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{CAPTION}}}", rf"\label{{{LABEL}}}",
             r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Method & Prediction & \multicolumn{2}{c}{AR} & \multicolumn{2}{c}{Turnover} & \multicolumn{2}{c}{Log Volume} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r" & & MAE & RMSE & MAE & RMSE & MAE & RMSE \\", r"\midrule"]
    for row in table.to_dict("records"):
        cells = [row["method"], prediction_form(row["method"])] + [format_cell(row, channel, metric, latex=True) for channel in CHANNELS for metric in ("MAE", "RMSE")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", r"\begin{minipage}{\linewidth}\footnotesize",
              r"All methods are evaluated on incremental responses. Full-trajectory methods separately predict recommended-stock and reference-stock trajectories and subtract the predictions. Values are mean $\pm$ sample standard deviation across seeds 41, 42, and 43 on 2,315 test events.",
              r"\end{minipage}", r"\end{table}", ""]
    return "\n".join(lines)


def markdown(table, root):
    lines = ["# 完整轨迹与增量响应预测对照", "", CAPTION, "",
             "| 方法 | 预测形式 | AR MAE | AR RMSE | 换手率 MAE | 换手率 RMSE | 对数成交量 MAE | 对数成交量 RMSE |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in table.to_dict("records"):
        cells = [row["method"], prediction_form(row["method"])] + [format_cell(row, channel, metric) for channel in CHANNELS for metric in ("MAE", "RMSE")]
        lines.append("| " + " | ".join(cells) + " |")
    dates = int(table.n_dates.iloc[0])
    lines += ["", "表中所有方法均评价增量响应。Direct 方法先分别预测推荐股票完整轨迹 $\\widehat{Y}$ 和参考股票完整轨迹 $\\widehat{B}$，再以 $\\widehat{Y}-\\widehat{B}$ 作为增量预测。Incremental 方法及 RAIF 直接预测增量响应。",
              "", "| 方法 | 预测形式公式 | 历史辅助信息 |", "|---|---|---|",
              r"| ECT-Direct | $\widehat{\Delta}=\widehat{Y}_{\mathrm{ECT}}-\widehat{B}_{\mathrm{ECT}}$ | 无 |",
              r"| ECT-Incremental | $\widehat{\Delta}=f_{\mathrm{ECT}}(X)$ | 无 |",
              r"| RAFT-Direct | $\widehat{\Delta}=\widehat{Y}_{\mathrm{RAFT}}-\widehat{B}_{\mathrm{ECT}}$ | 原始市场轨迹 |",
              r"| RAFT-Incremental | $\widehat{\Delta}=f_{\mathrm{RAFT}}(X,\mathcal{H}_{\Delta})$ | 历史增量轨迹 |",
              r"| RAIF | $\widehat{\Delta}=f_{\mathrm{RAIF}}(X,\mathcal{H}_{\Delta})$ | 响应相关历史增量 |",
              "", r"其中，$X$ 表示模型的事件前输入，$\mathcal{H}_{\Delta}$ 表示检索到的历史增量轨迹；完整轨迹方法中的参考预测由同种子的共享 ECT 参考模型提供。历史辅助信息列指用于检索的历史轨迹。",
              "", f"主样本共 5,105 个事件，训练、验证、测试分别为 2,170、620、2,315 个事件；测试集覆盖 {dates} 个交易日。各方法使用相同的随机种子 41、42、43、事件顺序和测试标签。输入为事件前 10 个交易日，预测长度为 11 个交易日。MAE 和 RMSE 先在每个种子的全部测试事件及预测时点上计算，再报告三个种子的均值与样本标准差（分母为 2）。AR、换手率及对数成交量均使用原数据单位。",
              "", "所有方法的评价标签均由原主数据中的标准化增量标签反标准化获得。完整轨迹训练标签采用同一批事件及已固定的单近邻参考股票，参考股票不重新选择。",
              "", "每个种子先训练一个 ECT 参考模型，以参考股票的事件前状态预测完整参考轨迹，类别输入均为 0，并按标准化参考轨迹验证集 MAE 选择模型。该参考模型固定后，同种子的 ECT-Direct 和 RAFT-Direct 共享其预测结果。两种 Direct 方法训练推荐股票完整轨迹，并按最终增量预测的标准化验证集 MAE 选择模型；预测时不使用真实参考股票未来轨迹。",
              "", "推荐股票完整轨迹以训练集均值中心化，除以原主任务增量标签的训练集标准差；参考股票完整轨迹使用参考轨迹自身的训练集均值和标准差。标准化参数仅由训练集拟合。",
              "", "ECT-Incremental、RAFT-Incremental 和 RAIF 复用原主实验对应种子的模型与预测。新增 ECT、RAFT 组件沿用各自原主实验的模型结构、优化器和固定超参数，最多训练 100 轮，早停耐心值为 15；本项实验不再次调参。两个 Direct 方案额外使用参考股票的事件前状态和共享参考模型，因此整体方案的输入与总参数量包含这部分成本。",
              "", "RAFT-Direct 与 RAFT-Incremental 均在尺度 1、2、4 上按 Pearson 相似度各检索 Top-5 历史事件。候选记忆仅含 2,170 个训练事件，且历史响应结束日须早于查询日；各方法分别使用其训练目标对应的历史轨迹。RAIF 保留原主实验的检索与修正配置，质量权重由匹配状态偏差和距离确定，不含有效参考数项。",
              "", " · ".join(f"[{label}](<{Path(root).resolve() / 'results/trajectory_comparison' / filename}>)"
                                 for label, filename in (("逐种子指标", "metrics_seeds.csv"),
                                                         ("均值与标准差 CSV", "table2_trajectory_comparison.csv"),
                                                         ("LaTeX 表格", "table2_trajectory_comparison.tex"))), ""]
    return "\n".join(lines)


def execute(root, validate_only=False):
    root = Path(root).resolve()
    inputs = Inputs(root)
    data = load_data(inputs)
    main = load_main(inputs, data)
    components, component_checks = load_components(inputs, data, main)
    indices = data["indices"]["test"]
    truth = data["canonical"][indices]
    n_dates = int(data["events"].iloc[indices].event_date.nunique())
    predictions, records = {}, []
    for method in METHODS:
        for seed in SEEDS:
            arrays = dict(event_indices=indices.copy(), truth=truth.copy())
            if method.endswith("-Direct"):
                role = "ect_direct" if method == "ECT-Direct" else "raft_direct"
                predicted_y = components[(role, seed)]["predictions"]["test"]
                predicted_b = components[("reference", seed)]["predictions"]["test"]
                arrays.update(prediction=predicted_y - predicted_b, predicted_y=predicted_y, predicted_b=predicted_b)
            else:
                model = method.removesuffix("-Incremental")
                arrays["prediction"] = main[(model, seed)]["prediction"].copy()
            check_predictions(arrays, indices, truth, f"final/{method}/{seed}")
            predictions[(method, seed)] = arrays
            records.extend(metric_records(method, seed, arrays["prediction"], truth, n_dates))
    for seed in SEEDS:
        exact(predictions[("ECT-Direct", seed)]["predicted_b"], predictions[("RAFT-Direct", seed)]["predicted_b"],
              f"shared predicted B/seed{seed}")
        for field in ("reference_checkpoint_sha256", "reference_validation_predictions_sha256", "reference_test_predictions_sha256"):
            require(components[("ect_direct", seed)]["config"][field] == components[("raft_direct", seed)]["config"][field],
                    f"Direct branches used different reference artifacts: seed{seed}/{field}")
    metrics, table = aggregate_metrics(records)
    inputs.unchanged()
    verification = dict(status="passed", generated_at_utc=datetime.now(timezone.utc).isoformat(),
                        methods=list(METHODS), seeds=list(SEEDS), runs=15, new_component_runs=9, reused_main_runs=9,
                        n_train=2170, n_validation=620, n_test=2315, n_test_dates=n_dates,
                        metrics_rows=len(metrics), table_rows=len(table), channels=list(CHANNELS),
                        canonical_truth_source="data/prepared_expanded/dataset.npz:y inverse standardized using primary normalization",
                        canonical_truth_exact_for_all_methods=True, same_seed_reference_predictions_exactly_shared=True,
                        direct_prediction_rule="predicted_Y - predicted_B", true_B_used_for_prediction=False,
                        aggregation="arithmetic mean and sample standard deviation (ddof=1) across seed metrics",
                        existing_main_predictions_and_checkpoint_hashes_match_original_device_replay=True,
                        source_files_unchanged=True, component_checks=component_checks,
                        input_sha256=inputs.hashes, exporter_sha256=digest(Path(__file__)))
    if validate_only:
        return dict(status="validated_only", runs=15, metrics_rows=len(metrics), table_rows=len(table))
    output = root / "results/trajectory_comparison"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for (method, seed), arrays in predictions.items():
        path = output / "predictions" / method / f"seed{seed}" / "predictions.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".npz.tmp").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        path.with_suffix(".npz.tmp").replace(path)
        written.append(path)
    for filename, frame in (("metrics_seeds.csv", metrics), ("table2_trajectory_comparison.csv", table)):
        path = output / filename
        frame.to_csv(path.with_suffix(".csv.tmp"), index=False, float_format="%.17g")
        path.with_suffix(".csv.tmp").replace(path)
        written.append(path)
    for path, text in ((output / "table2_trajectory_comparison.tex", latex_table(table)),
                       (root / "trajectory_comparison.md", markdown(table, root))):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)
        written.append(path)
    inputs.unchanged()
    verification["output_sha256"] = {path.relative_to(root).as_posix(): digest(path) for path in written}
    report = output / "export_verification.json"
    report.with_suffix(".json.tmp").write_text(json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    report.with_suffix(".json.tmp").replace(report)
    return dict(status="exported", output=str(output), markdown=str(root / "trajectory_comparison.md"),
                runs=15, metrics_rows=len(metrics), table_rows=len(table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="Verify complete inputs without writing any output")
    arguments = parser.parse_args()
    print(json.dumps(execute(arguments.root, arguments.validate_only), ensure_ascii=False))
