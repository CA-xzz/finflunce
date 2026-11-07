"""Summarize completed Sx runs without training or changing the manuscript.

Every accepted main-run prediction is checked against the prepared test tensor.
Missing runs remain missing. Seed-level RMSE is averaged only after each seed's
RMSE has been computed. Case selection is independent of predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["HistoricalMean", "DLinear", "PatchTST", "iTransformer", "ECT", "RAFT", "RATL", "RAIF"]
DIRECT_MODELS = ["DLinear", "PatchTST", "iTransformer", "ECT"]
CHANNELS = ["AR", "turnover", "log_volume"]
CHANNEL_LABELS = ["Abnormal return", "Turnover (percentage points)", "Log volume"]
HORIZONS = [1, 3, 5, 10, 11]
CASE_SEED = 20261105
KEYS = ["event_id", "tau", "channel"]


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def load_prepared(directory):
    needed = [directory / name for name in ["dataset.npz", "events.csv", "normalization.json"]]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        return None, {"status": "incomplete", "missing": missing}
    normal = read_json(needed[2])
    with np.load(needed[0], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ["y", "q", "split", "platform", "influencer", "day", "end_day"]}
    events = pd.read_csv(needed[1], dtype={"event_id": str, "stock": str})
    require(not events.event_id.duplicated().any(), "Duplicate prepared event IDs")
    require(arrays["y"].shape == (len(events), 11, 3), "Prepared y must have shape [N,11,3]")
    require(all(len(array) == len(events) for array in arrays.values()), "Prepared array/event length mismatch")
    require(all(np.isfinite(array).all() for array in arrays.values()), "Nonfinite prepared array values")
    require(np.isin(arrays["split"], [0, 1, 2]).all(), "Invalid prepared split")
    mean, std = np.asarray(normal["y_mean"], dtype=float), np.asarray(normal["y_std"], dtype=float)
    require(mean.shape == std.shape == (3,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(), "Invalid target normalization")
    truth_all = arrays["y"].astype(np.float64) * std + mean
    train_ids, test_ids = np.flatnonzero(arrays["split"] == 0), np.flatnonzero(arrays["split"] == 2)
    require(len(train_ids) > 0 and len(test_ids) > 0, "Training/test set is empty")
    require(np.allclose(events.q.to_numpy(float), arrays["q"], rtol=1e-6, atol=1e-12), "Prepared q mismatch")
    require(events.label_version.nunique() == 1, "Prepared label version is ambiguous")
    events = events.copy()
    events["event_date"] = pd.to_datetime(events.event_date).dt.strftime("%Y-%m-%d")
    test = events.iloc[test_ids].reset_index(drop=True)
    expected = pd.DataFrame({"event_id": np.repeat(test.event_id.to_numpy(), 33),
        "tau": np.tile(np.repeat(np.arange(11), 3), len(test)),
        "channel": np.tile(CHANNELS, len(test) * 11), "delta_true": truth_all[test_ids].reshape(-1)})
    expected_index = pd.MultiIndex.from_frame(expected[KEYS])
    split_rows = []
    for code, name in [(0, "train"), (1, "validation"), (2, "test")]:
        subset = events.loc[arrays["split"] == code]
        split_rows.append({"split": name, "n_events": len(subset), "n_dates": subset.event_date.nunique(),
                           "first_date": subset.event_date.min() if len(subset) else None,
                           "last_date": subset.event_date.max() if len(subset) else None})
    exclusions_path = directory / "exclusions.csv"
    exclusions = pd.read_csv(exclusions_path) if exclusions_path.exists() and exclusions_path.stat().st_size > 1 else pd.DataFrame()
    exclusion_counts = (exclusions.reason.fillna("unspecified").value_counts().astype(int).to_dict()
                        if "reason" in exclusions else {})
    info = {"status": "available", "splits": split_rows, "label_version": str(events.label_version.iloc[0]),
            "dataset_sha256": digest(needed[0]), "normalization_sha256": digest(needed[2]),
            "normalization_ready_flag": normal.get("ready_for_formal_training"),
            "preparation": read_json(directory / "preparation_summary.json", {}),
            "exclusion_counts": exclusion_counts,
            "limitations": normal.get("limitations", []),
            "attribute_verification": normal.get("attribute_verification", {}),
            "reference_mode": read_json(directory / "config.json", {}).get("reference_mode"),
            "reference_configuration": read_json(directory / "reference_tuning_protocol.json", {}),
            "source_manifest": records(pd.read_csv(directory / "source_manifest.csv")) if (directory / "source_manifest.csv").exists() else []}
    return {"arrays": arrays, "events": events, "test": test, "test_ids": test_ids, "train_ids": train_ids,
            "truth": truth_all[test_ids], "truth_all": truth_all, "normal": normal,
            "expected": expected, "expected_index": expected_index}, info


def validate_predictions(frame, prepared, model, seed):
    required = set(KEYS + ["model", "seed", "split", "delta_true", "delta_pred", "event_date", "label_version", "memory_mode"])
    require(required.issubset(frame.columns), "Missing prediction fields in %s/%s" % (model, seed))
    require(len(frame) == len(prepared["expected"]), "Prediction row count differs from prepared test set")
    require(not frame.duplicated(KEYS).any(), "Duplicate prediction event/tau/channel keys")
    require(frame.model.eq(model).all() and frame.seed.eq(seed).all(), "Prediction model/seed does not match run")
    require(frame.split.eq("test").all() and frame.memory_mode.eq("fixed").all(), "Main predictions must use test/fixed memory")
    require(frame.label_version.eq(str(prepared["events"].label_version.iloc[0])).all(), "Prediction label version mismatch")
    require(np.isfinite(frame[["delta_true", "delta_pred"]].to_numpy(float)).all(), "Nonfinite predictions or truth")
    current = pd.MultiIndex.from_frame(frame[KEYS])
    expected = prepared["expected_index"]
    require(len(current.difference(expected)) == 0 and len(expected.difference(current)) == 0,
            "Prediction event/tau/channel set differs from prepared test set")
    ordered = frame.set_index(KEYS).reindex(expected)
    require(np.allclose(ordered.delta_true.to_numpy(float), prepared["expected"].delta_true.to_numpy(float), rtol=1e-10, atol=1e-12),
            "Prediction truth differs from prepared test labels")
    dates = pd.to_datetime(ordered.event_date).dt.strftime("%Y-%m-%d").to_numpy()
    require(np.array_equal(dates, np.repeat(prepared["test"].event_date.to_numpy(), 33)), "Prediction event date mismatch")
    return ordered.delta_pred.to_numpy(float).reshape(-1, 11, 3)


def per_seed_metrics(truth, predicted, model, seed, dates, horizons=HORIZONS):
    rows = []
    for horizon in horizons:
        error = predicted[:, :horizon] - truth[:, :horizon]
        for channel, name in enumerate(CHANNELS):
            for metric, value in [("MAE", np.abs(error[:, :, channel]).mean()),
                                  ("RMSE", np.sqrt(np.square(error[:, :, channel]).mean()))]:
                rows.append({"model": model, "seed": seed, "channel": name, "horizon": horizon,
                             "metric": metric, "value": float(value), "n_events": len(truth),
                             "n_dates": len(np.unique(dates))})
    return rows


def check_metric_file(path, computed, keys):
    if not path.exists():
        return "missing"
    observed = pd.read_csv(path)
    require(not observed.duplicated(keys).any(), "Duplicate metrics: " + str(path))
    expected = computed.set_index(keys)
    actual = observed.set_index(keys)
    require(set(actual.index) == set(expected.index), "Metric keys differ: " + str(path))
    require(np.allclose(actual.reindex(expected.index).value.to_numpy(float), expected.value.to_numpy(float), rtol=1e-7, atol=1e-10),
            "Saved metrics disagree with verified predictions: " + str(path))
    return "verified"


def choose_cases(prepared, maximum=6):
    train_q = prepared["arrays"]["q"][prepared["train_ids"]]
    cuts = np.quantile(train_q, [1 / 3, 2 / 3])
    test = prepared["test"].copy()
    test["platform"] = test.platform.fillna("unknown").astype(str)
    test["quality_group"] = np.asarray(["low", "middle", "high"])[np.searchsorted(cuts, test.q.to_numpy(float), side="left")]
    rng = np.random.default_rng(CASE_SEED)
    strata = sorted(test.groupby(["platform", "quality_group"]).groups)
    order = rng.permutation(len(strata))
    pools = []
    for position in order:
        platform, quality = strata[position]
        ids = test.loc[test.platform.eq(platform) & test.quality_group.eq(quality), "event_id"].sort_values().to_numpy()
        pools.append(list(rng.permutation(ids)))
    selected = []
    while len(selected) < min(maximum, len(test)):
        for pool in pools:
            if pool and len(selected) < min(maximum, len(test)):
                selected.append(pool.pop())
    cases = test.set_index("event_id").loc[selected].reset_index()
    cases.insert(0, "case_id", np.arange(1, len(cases) + 1))
    cases["selection_seed"] = CASE_SEED
    cases["selection_rule"] = "random stratum order, shuffled event IDs, round-robin; no prediction values used"
    columns = ["case_id", "event_id", "event_date", "platform", "quality_group", "q", "selection_seed", "selection_rule"]
    return cases[columns], [float(value) for value in cuts]


def aggregate_metrics(raw, seeds):
    rows = []
    for model in MODELS:
        expected_seeds = {41} if model == "HistoricalMean" else set(seeds)
        for horizon in HORIZONS:
            for channel in CHANNELS:
                for metric in ["MAE", "RMSE"]:
                    selected = raw.loc[(raw.model == model) & (raw.horizon == horizon) & (raw.channel == channel) & (raw.metric == metric)]
                    require(not selected.seed.duplicated().any(), "Duplicate model/seed metric")
                    observed = set(selected.seed.tolist())
                    rows.append({"model": model, "channel": channel, "horizon": horizon, "metric": metric,
                                 "mean": float(selected.value.mean()) if len(selected) else np.nan,
                                 "std": float(selected.value.std(ddof=1)) if len(selected) > 1 else np.nan,
                                 "n_seeds": len(selected), "expected_seeds": len(expected_seeds),
                                 "status": "completed" if observed == expected_seeds else "incomplete",
                                 "n_events": int(selected.n_events.iloc[0]) if len(selected) else 0,
                                 "n_dates": int(selected.n_dates.iloc[0]) if len(selected) else 0})
    return pd.DataFrame(rows)


def make_group_rows(prepared, predicted, model, seed, cuts):
    test, arrays, ids = prepared["test"], prepared["arrays"], prepared["test_ids"]
    assignments = {
        "period": pd.to_datetime(test.event_date).dt.to_period("Q").astype(str).to_numpy(),
        "platform": test.platform.fillna("unknown").astype(str).to_numpy(),
        "quality": np.asarray(["low", "middle", "high"])[np.searchsorted(cuts, test.q.to_numpy(float), side="left")],
        "influencer_status": np.where(arrays["influencer"][ids] == 0, "unknown_or_unseen", "seen"),
    }
    rows, counts = [], []
    for group_type, values in assignments.items():
        for group in np.unique(values):
            mask = values == group
            dates = test.event_date.to_numpy()[mask]
            sufficient = int(mask.sum()) >= 20 and len(np.unique(dates)) >= 5
            counts.append({"group_type": group_type, "group": str(group), "n_events": int(mask.sum()),
                           "n_dates": len(np.unique(dates)), "report_metrics": sufficient,
                           "status": "available" if sufficient else "insufficient_group_size"})
            if sufficient and predicted is not None:
                for row in per_seed_metrics(prepared["truth"][mask], predicted[mask], model, seed, dates, [11]):
                    rows.append(dict(row, group_type=group_type, group=str(group)))
    return rows, counts


def select_direct(validation_rows, seeds):
    means, incomplete = [], []
    for model in DIRECT_MODELS:
        rows = [row for row in validation_rows if row["model"] == model]
        if set(row["seed"] for row in rows) != set(seeds):
            incomplete.append(model)
        else:
            means.append({"model": model, "mean_validation_standardized_mae": float(np.mean([row["validation_standardized_mae"] for row in rows])), "n_seeds": len(rows)})
    if incomplete:
        return {"status": "incomplete", "missing_models_or_seeds": incomplete, "candidate_scores": means,
                "selection_uses_test": False}
    chosen = min(means, key=lambda row: (row["mean_validation_standardized_mae"], row["model"]))
    return dict(chosen, status="completed", candidate_scores=means, selection_uses_test=False,
                selection_rule="Lowest mean validation standardized MAE across main runs; ties resolved by model name")


def make_figures(output, prepared, predictions, steps, cases, best, raif_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    status = {"step_errors": "incomplete", "cases": "incomplete", "retrieval": "incomplete"}
    if len(steps):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for channel, ax, label in zip(CHANNELS, axes, CHANNEL_LABELS):
            for model in MODELS:
                subset = steps.loc[(steps.channel == channel) & (steps.model == model) & (steps.metric == "MAE")]
                if not len(subset):
                    continue
                summary = subset.groupby("tau").value.agg(["mean", "std", "count"])
                x = summary.index.to_numpy()
                line, = ax.plot(x, summary["mean"], marker="o", markersize=2.5, label=model)
                if (summary["count"] > 1).all():
                    ax.fill_between(x, summary["mean"] - summary["std"], summary["mean"] + summary["std"], alpha=.10, color=line.get_color())
            ax.set(title=label, xlabel="Forecast step", ylabel="MAE", xticks=[0, 2, 4, 6, 8, 10])
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        fig.savefig(output / "step_errors.png", bbox_inches="tight")
        plt.close(fig)
        status["step_errors"] = "available; mean and sample SD over observed seeds"
    if best.get("status") != "completed" or ("RAIF", 41) not in predictions or (best.get("model"), 41) not in predictions:
        return status
    test_lookup = dict(zip(prepared["test"].event_id, range(len(prepared["test"]))))
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 2.8 * len(cases)), squeeze=False, constrained_layout=True)
    for row, case in enumerate(cases.itertuples()):
        index = test_lookup[case.event_id]
        for channel, ax in enumerate(axes[row]):
            ax.plot(prepared["truth"][index, :, channel], color="black", linewidth=1.7, label="Observed")
            ax.plot(predictions[("RAIF", 41)][index, :, channel], color="#c44e52", label="RAIF (seed 41)")
            ax.plot(predictions[(best["model"], 41)][index, :, channel], color="#4c72b0", linestyle="--", label=best["model"] + " (seed 41)")
            ax.set(title="Case %d | %s | %s" % (case.case_id, case.event_date, CHANNEL_LABELS[channel]), xlabel="Forecast step", ylabel="Increment")
            ax.axhline(0, color="gray", linewidth=.5, alpha=.5)
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "cases.png", bbox_inches="tight")
    plt.close(fig)
    status["cases"] = "available; fixed stratified sample, validation-selected direct comparator"
    retrieval_path = raif_dir / "retrieved_events.csv"
    if not retrieval_path.exists():
        return status
    retrieved = pd.read_csv(retrieval_path, dtype={"query_id": str, "historical_id": str})
    needed = {"query_id", "historical_id", "rank", "similarity", "q", "end_day"}
    require(needed.issubset(retrieved), "Missing RAIF retrieval fields")
    selected = retrieved.loc[retrieved.query_id.isin(cases.event_id)].copy()
    require(not selected.duplicated(["query_id", "rank"]).any()
            and not selected.duplicated(["query_id", "historical_id"]).any(), "Duplicate RAIF retrieval case ranks/events")
    require((selected["rank"] >= 1).all() and (selected["rank"] % 1 == 0).all(), "Invalid retrieval rank")
    lookup = dict(zip(prepared["events"].event_id, range(len(prepared["events"]))))
    historical_rows = []
    for row in selected.itertuples():
        require(row.query_id in lookup and row.historical_id in lookup, "Unknown retrieval event ID")
        qi, hi = lookup[row.query_id], lookup[row.historical_id]
        a = prepared["arrays"]
        require(a["split"][hi] == 0 and a["end_day"][hi] < a["day"][qi] and hi != qi,
                "Retrieval case violates fixed mature-training memory")
        require(int(row.end_day) == int(a["end_day"][hi]), "Retrieval end_day mismatch")
        require(np.isfinite([row.similarity, row.q]).all(), "Nonfinite retrieval case fields")
        require(np.isclose(float(row.q), float(a["q"][hi]), rtol=1e-6, atol=1e-12), "Retrieval quality mismatch")
        for tau in range(11):
            for c, channel in enumerate(CHANNELS):
                historical_rows.append({"query_id": row.query_id, "historical_id": row.historical_id,
                    "rank": row.rank, "tau": tau, "channel": channel, "historical_delta": float(prepared["truth_all"][hi, tau, c])})
    selected.to_csv(output / "retrieval_case_manifest.csv", index=False)
    pd.DataFrame(historical_rows, columns=["query_id", "historical_id", "rank", "tau", "channel", "historical_delta"]).to_csv(output / "retrieval_case_paths.csv", index=False)
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 2.8 * len(cases)), squeeze=False, constrained_layout=True)
    for row_index, case in enumerate(cases.itertuples()):
        query_index = lookup[case.event_id]
        candidates = selected.loc[selected.query_id == case.event_id].sort_values("rank")
        for c, ax in enumerate(axes[row_index]):
            for number, history in enumerate(candidates.itertuples()):
                hi = lookup[history.historical_id]
                ax.plot(prepared["truth_all"][hi, :, c], color="#4c72b0", alpha=.28, linewidth=.8,
                        label="Retrieved histories" if number == 0 else None)
            ax.plot(prepared["truth_all"][query_index, :, c], color="black", linewidth=1.7, label="Query observed (display only)")
            ax.plot(predictions[("RAIF", 41)][test_lookup[case.event_id], :, c], color="#c44e52", label="RAIF prediction")
            ax.set(title="Case %d | %s | %d histories" % (case.case_id, CHANNEL_LABELS[c], len(candidates)), xlabel="Response step", ylabel="Increment")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "retrieval_cases.png", bbox_inches="tight")
    plt.close(fig)
    status["retrieval"] = "available; historical paths inverse-transformed from prepared y; query truth used only for plotting"
    return status


def markdown_report(summary, overall):
    lines = ["# 实验结果", ""]
    if summary["data"].get("status") != "available":
        return "# 实验结果\n\n准备数据尚未完整，当前没有可核对的正式预测结果。\n"
    split_text = "，".join("%s %d 个事件、%d 个交易日" % (row["split"], row["n_events"], row["n_dates"]) for row in summary["data"]["splits"])
    lines += [split_text + "。预测窗口为 11 个交易日，三个通道分别评价。", ""]
    if summary.get("protocol_role") == "secondary_reference_sensitivity":
        lines += ["本组为独立的参考构造敏感性实验，使用本组参考标签及共同测试事件。主方案采用扩大样本后的单最近邻参考；两组指标分别列示。消融、检索排序及同容量对照见同一 suite 的独立结果表。", ""]
    if summary["data"].get("reference_mode") == "expanded_nearest":
        lines += ["每个事件按完整的事前市场状态匹配唯一最近邻股票。行业、板块和市值比例不限制候选范围。历史行业按有效区间描述，逐股首次公告时间未知；来源更新时间未作为首次公告时间。行业与市值仅用于匹配结果的描述，不参与最近邻距离或模型输入。", ""]
    else:
        lines += ["历史行业采用固定 Git 提交保存的申万历史明细副本。该组参考构造使用计入日期与更新日期中较晚者再加一天的保守筛选规则；更新时间不代表首次公告时间。", ""]
    lines += ["历史行业副本尚未与官方 XLS 原文件逐行比对。推荐记录只有日期，无法核实盘中发布时间和 15:00 切点。历史记忆仅包含响应窗口在查询日前已结束的训练事件。样本范围及来源限制另见数据说明。", "",
              "| 模型 | AR MAE / RMSE | 换手率 MAE / RMSE | 对数成交量 MAE / RMSE | 种子数 |", "|---|---:|---:|---:|---:|"]
    for model in MODELS:
        cells = []
        for channel in CHANNELS:
            values = []
            for metric in ["MAE", "RMSE"]:
                row = overall.loc[(overall.model == model) & (overall.channel == channel) & (overall.metric == metric)].iloc[0]
                if not row.n_seeds:
                    values.append("未完成")
                else:
                    value = "%.5g" % row["mean"]
                    if pd.notna(row["std"]):
                        value += " ± %.3g" % row["std"]
                    if row.status != "completed":
                        value += "（部分）"
                    values.append(value)
            cells.append(" / ".join(values))
        row = overall.loc[overall.model == model].iloc[0]
        lines.append("| %s | %s | %s | %s | %d/%d |" % (model, *cells, row.n_seeds, row.expected_seeds))
    lines += ["", "表中均值和样本标准差由各随机种子的独立指标汇总；RMSE 在每个种子内计算后取均值。单个种子不提供种子间标准差。", ""]
    best = summary["best_direct"]
    if best.get("status") == "completed":
        lines += ["案例中的直接预测对照为 %s，依据主运行的平均验证集标准化 MAE 选定。案例按平台与训练集参考质量三分位分层抽取，随机种子为 %d，选样不使用预测误差。" % (best["model"], CASE_SEED), ""]
    else:
        lines += ["直接预测对照的验证指标尚未齐全，案例比较未完成。", ""]
    lines += ["分组至少包含 20 个事件和 5 个独立交易日才报告误差，其余分组仅列规模。计时表记录已有运行的实测值；并行训练期间的计时不能替代同设备串行效率实验。", "",
              "基线为统一输入、增量标签和事件条件下的任务适配，具体差异保存在各运行 config.json。RATL 为公开公式适配，不视为未公开实现的逐项复现。", ""]
    incomplete = summary.get("remaining_paper_experiments", [])
    if incomplete:
        lines += ["尚未完成或未纳入本汇总的论文实验：" + "；".join(str(value) for value in incomplete) + "。", ""]
    if summary.get("supplementary_summary"):
        lines += ["补充实验的范围、结果和文件索引见 supplementary_summary.json。主表与严格参考子集对照分别评价。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supplement-summary", type=Path)
    parser.add_argument("--secondary-protocol", action="store_true", help="Label this report as a separate reference sensitivity suite, rather than a paper-wide completion report")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # These are this report's generated figures; prevent stale images being
    # presented as current when a later invocation has incomplete inputs.
    for name in ["step_errors.png", "cases.png", "retrieval_cases.png", "retrieval_case_manifest.csv", "retrieval_case_paths.csv"]:
        (args.output / name).unlink(missing_ok=True)
    protocol = read_json(args.suite / "suite_protocol.json", {})
    seeds = [int(seed) for seed in protocol.get("final_seeds", [41, 42, 43])]
    require(len(seeds) == len(set(seeds)) and len(seeds) > 0, "Invalid expected seed list")
    prepared, data_info = load_prepared(args.data)
    predictions, availability, validation, metric_rows, step_rows, group_rows, timing_rows = {}, [], [], [], [], [], []
    group_counts = []
    cases = pd.DataFrame(columns=["case_id", "event_id", "event_date", "platform", "quality_group", "q", "selection_seed", "selection_rule"])
    cuts = []
    if prepared is not None:
        cases, cuts = choose_cases(prepared)
        _, group_counts = make_group_rows(prepared, None, "", 0, cuts)
    for model in MODELS:
        for seed in ([41] if model == "HistoricalMean" else seeds):
            directory = args.suite / "main" / model / ("seed" + str(seed))
            status = read_json(directory / "status.json", {})
            row = {"model": model, "seed": seed, "run_path": str(directory), "status": "incomplete",
                   "reported_status": status.get("status", "missing")}
            if status.get("status") != "completed" or not (directory / "predictions.csv").exists():
                availability.append(row)
                continue
            require(prepared is not None, "Completed predictions cannot be validated without prepared data")
            config = read_json(directory / "config.json", {})
            require(config.get("model") == model and config.get("seed") == seed, "Main run config identity mismatch")
            require(config.get("variant") == "full" and config.get("retrieval") == "response"
                    and config.get("memory_subset", "all") == "all", "A variant cannot be aggregated as a main model")
            require(config.get("data_hash") == data_info["dataset_sha256"], "Run dataset hash differs: " + str(directory))
            require(config.get("normalization_hash") == data_info["normalization_sha256"], "Run normalization hash differs: " + str(directory))
            frame = pd.read_csv(directory / "predictions.csv", dtype={"event_id": str})
            predicted = validate_predictions(frame, prepared, model, seed)
            predictions[(model, seed)] = predicted
            metrics = pd.DataFrame(per_seed_metrics(prepared["truth"], predicted, model, seed, prepared["test"].event_date.to_numpy()))
            row["horizon_metrics_check"] = check_metric_file(directory / "horizon_metrics.csv", metrics, ["model", "seed", "channel", "horizon", "metric"])
            row["overall_metrics_check"] = check_metric_file(directory / "overall_metrics.csv", metrics.loc[metrics.horizon == 11], ["model", "seed", "channel", "horizon", "metric"])
            row["status"] = "completed"
            row["prediction_sha256"] = digest(directory / "predictions.csv")
            availability.append(row)
            metric_rows.extend(metrics.to_dict(orient="records"))
            for tau in range(11):
                for metric in per_seed_metrics(prepared["truth"][:, tau:tau + 1], predicted[:, tau:tau + 1], model, seed, prepared["test"].event_date.to_numpy(), [1]):
                    metric.pop("horizon")
                    step_rows.append(dict(metric, tau=tau))
            grouped, group_counts = make_group_rows(prepared, predicted, model, seed, cuts)
            group_rows.extend(grouped)
            valid = read_json(directory / "validation_summary.json", status)
            value = valid.get("validation_standardized_mae")
            if isinstance(value, (int, float)) and np.isfinite(value):
                validation.append({"model": model, "seed": seed, "validation_standardized_mae": float(value)})
            timing_path = directory / "timing.csv"
            if timing_path.exists():
                timing = pd.read_csv(timing_path)
                require(len(timing) == 1 and timing.model.eq(model).all() and timing.seed.eq(seed).all(), "Timing model/seed mismatch")
                timing = timing.assign(gpu=config.get("gpu"), gpu_visibility=config.get("gpu_visibility"), precision=config.get("precision"), n_memory=config.get("n_memory"))
                timing_rows.extend(records(timing))
    metric_columns = ["model", "seed", "channel", "horizon", "metric", "value", "n_events", "n_dates"]
    raw_metrics = pd.DataFrame(metric_rows, columns=metric_columns)
    aggregate = aggregate_metrics(raw_metrics, seeds)
    overall = aggregate.loc[aggregate.horizon == 11].copy()
    raw_metrics.to_csv(args.output / "horizon_metrics_seeds.csv", index=False)
    aggregate.to_csv(args.output / "horizon_metrics.csv", index=False)
    overall.to_csv(args.output / "overall_metrics.csv", index=False)
    pd.DataFrame(availability).to_csv(args.output / "run_availability.csv", index=False)
    pd.DataFrame(validation, columns=["model", "seed", "validation_standardized_mae"]).to_csv(args.output / "validation_scores.csv", index=False)
    steps = pd.DataFrame(step_rows, columns=["model", "seed", "channel", "metric", "value", "n_events", "n_dates", "tau"])
    steps.to_csv(args.output / "step_metrics_seeds.csv", index=False)
    groups = pd.DataFrame(group_rows, columns=metric_columns + ["group_type", "group"])
    groups.to_csv(args.output / "group_metrics_seeds.csv", index=False)
    group_summary = groups.groupby(["model", "group_type", "group", "channel", "horizon", "metric"], dropna=False).agg(
        mean=("value", "mean"), std=("value", "std"), n_seeds=("seed", "nunique"), n_events=("n_events", "first"), n_dates=("n_dates", "first")).reset_index() if len(groups) else pd.DataFrame(columns=["model", "group_type", "group", "channel", "horizon", "metric", "mean", "std", "n_seeds", "n_events", "n_dates"])
    group_grid = pd.DataFrame([
        {"model": model, "group_type": group["group_type"], "group": group["group"], "channel": channel,
         "horizon": 11, "metric": metric, "expected_events": group["n_events"], "expected_dates": group["n_dates"],
         "group_sufficient": group["report_metrics"]}
        for model in MODELS for group in group_counts for channel in CHANNELS for metric in ["MAE", "RMSE"]
    ])
    if len(group_grid):
        keys = ["model", "group_type", "group", "channel", "horizon", "metric"]
        group_summary = group_grid.merge(group_summary, on=keys, how="left", validate="one_to_one")
        group_summary["n_seeds"] = group_summary.n_seeds.fillna(0).astype(int)
        group_summary["n_events"], group_summary["n_dates"] = group_summary.expected_events, group_summary.expected_dates
        group_summary["status"] = ["insufficient_group_size" if not sufficient else
            ("completed" if n == (1 if model == "HistoricalMean" else len(seeds)) else "incomplete")
            for model, n, sufficient in zip(group_summary.model, group_summary.n_seeds, group_summary.group_sufficient)]
        group_summary = group_summary.drop(columns=["expected_events", "expected_dates", "group_sufficient"])
    group_summary.to_csv(args.output / "group_metrics.csv", index=False)
    pd.DataFrame(group_counts, columns=["group_type", "group", "n_events", "n_dates", "report_metrics", "status"]).to_csv(args.output / "group_counts.csv", index=False)
    timing = pd.DataFrame(timing_rows)
    if not len(timing):
        timing = pd.DataFrame(columns=["model", "seed", "parameters", "training_seconds", "inference_batch_ms",
                                      "inference_event_ms", "retrieval_batch_ms", "memory_total_mb"])
    timing.to_csv(args.output / "timing_seeds.csv", index=False)
    timing_summary = []
    for model in MODELS:
        selected = timing.loc[timing.model == model] if "model" in timing else pd.DataFrame()
        for metric in ["parameters", "training_seconds", "base_training_seconds", "correction_training_seconds", "inference_batch_ms", "inference_event_ms", "retrieval_batch_ms", "memory_total_mb", "key_and_response_memory_mb"]:
            values = pd.to_numeric(selected[metric], errors="coerce").dropna() if metric in selected else pd.Series(dtype=float)
            timing_status = "observed" if len(values) else "incomplete"
            if not len(values) and metric == "retrieval_batch_ms" and len(selected) and model not in ["RAFT", "RATL", "RAIF"]:
                timing_status = "not_applicable"
            timing_summary.append({"model": model, "metric": metric, "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan, "n_seeds": len(values),
                "status": timing_status, "serial_same_device_comparison": False})
    pd.DataFrame(timing_summary).to_csv(args.output / "timing.csv", index=False)
    cases.to_csv(args.output / "case_manifest.csv", index=False)
    best = select_direct(validation, seeds)
    figure_status = {"step_errors": "incomplete", "cases": "incomplete", "retrieval": "incomplete"}
    if prepared is not None:
        figure_status = make_figures(args.output, prepared, predictions, steps, cases, best, args.suite / "main" / "RAIF" / "seed41")
    main_complete = all(row["status"] == "completed" for row in availability)
    summary = {"status": "main_completed" if main_complete else "incomplete", "data": data_info,
        "protocol_role": "secondary_reference_sensitivity" if args.secondary_protocol else "primary",
        "expected_main_models": MODELS, "expected_trainable_seeds": seeds, "historical_mean_seed": 41,
        "main_run_availability": availability, "overall_h11": records(overall),
        "metric_aggregation": "Compute per-seed MAE/RMSE on original units, then arithmetic seed mean and sample SD (ddof=1). A single-seed SD is unavailable.",
        "prediction_checks": "Every accepted main model/seed has exactly the prepared test event/tau/channel keys and matching inverse-transformed truth; dataset and normalization hashes also match.",
        "best_direct": best, "case_selection": {"seed": CASE_SEED, "max_cases": 6, "n_cases": len(cases), "training_quality_terciles": cuts, "uses_predictions": False},
        "figures": figure_status, "group_min_events": 20, "group_min_dates": 5,
        "timing_scope": "Observed per-run timings; concurrent GPU training and variable memory sizes do not constitute serial same-device efficiency comparison.",
        "preprocessing_seconds": data_info.get("preparation", {}).get("seconds"),
        "remaining_paper_experiments": protocol.get("remaining_paper_experiments", ["Suite protocol unavailable; paper-wide completion has not been established"]),
        "paper_modified": False}
    if args.secondary_protocol:
        summary["outside_this_secondary_report"] = summary["remaining_paper_experiments"]
        summary["remaining_paper_experiments"] = []
    if args.supplement_summary:
        supplementary = read_json(args.supplement_summary)
        require(supplementary and supplementary.get("status") == "completed", "Supplementary summary is incomplete")
        summary["supplementary_summary"] = {"source": str(args.supplement_summary), "sha256": digest(args.supplement_summary)}
        summary["remaining_paper_experiments"] = supplementary["remaining_experiments"]
        write_json(args.output / "supplementary_summary.json", supplementary)
    write_json(args.output / "summary.json", summary)
    (args.output / "RESULTS.md").write_text(markdown_report(summary, overall), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "accepted_main_runs": len(predictions), "expected_main_runs": len(availability),
                      "output": str(args.output), "figures": figure_status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
