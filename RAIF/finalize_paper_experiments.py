"""Assemble the final experiment index from completed, verified components."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def finish(root):
    root = root.resolve()
    table = root / "results/paper_tables"
    exports = read(table / "table_export_manifest.json")
    assert exports["reference_revision"] == "single_reference"
    assert exports["quality_rule"]["q_rule"] == "exp_only"
    assert exports["quality_rule"]["quality_neff_factor_enabled"] is False
    normal = read(root / "data/prepared_expanded/normalization.json")
    assert normal["q_rule"] == "exp_only" and normal["quality_neff_factor_enabled"] is False
    reference = read(table / "reference_verification.json")
    assert reference["status"] == "passed" and reference["validated_runs"] == 18
    assert reference["revision"] == "single_reference" and exports["reference_verification"] == reference
    assert set(reference["schemes"]) == {"strict", "nearest", "expanded_nearest"}
    comparison_directory = root / "data/reference_comparisons_expanded"
    reference_directory = root / "results/references_expanded"
    strict_directory = root / "data/reference_variants_expanded/strict"
    assert read(reference_directory / "status.json")["status"] == "completed"
    assert digest(reference_directory / "reference_metrics.csv") == reference["reference_metrics_sha256"]
    assert digest(strict_directory / "dataset.npz") == reference["strict_dataset_sha256"]
    assert digest(comparison_directory / "comparison_protocol.json") == reference["comparison_protocol_sha256"]
    n_reference_test, n_reference_train = reference["canonical_test_events"], reference["common_training_events"]
    revision_provenance = {}
    for name in ["data_migration_verification.json", "independent_predictions_validation.json"]:
        path = root / "results/single_reference_revision" / name
        if name == "data_migration_verification.json":
            assert path.is_file(), f"Missing single-reference data verification: {path}"
        if path.is_file():
            assert read(path)["status"] == "passed"
            revision_provenance[name.removesuffix(".json")] = dict(path=str(path.relative_to(root)), sha256=digest(path))
    primary = read(root / "results/report_expanded/supplementary_summary.json")
    auxiliary = read(root / "results/paper_completion/summary.json")
    efficiency = read(root / "results/serial_training_expanded/status.json")
    efficiency_protocol = read(root / "results/serial_training_expanded/protocol.json")
    aligned = read(root / "results/efficiency_aligned_expanded/status.json")
    assert all(x["status"] == "completed" for x in [primary, auxiliary, efficiency, aligned])
    assert primary["remaining_experiments"] == [] and primary["issues"] == []
    assert primary["actual_training_runs"] == 114 and primary["actual_validation_only_runs"] == 14
    assert efficiency["training_runs"] == efficiency["inference_runs"] == 22
    assert efficiency["no_observed_contention"] is True
    assert aligned["completed_runs"] == 22 and aligned["no_observed_contention"] is True
    assert aligned["same_checkpoint_accuracy_training_and_inference"] is True
    for directory in ["serial_training_expanded", "efficiency_aligned_expanded"]:
        assert read(root / "results" / directory / "independent_validation.json")["status"] == "passed"
    assert efficiency_protocol["n_memory"] == 912 and efficiency_protocol["n_test"] == 2315
    expected_data_hash = digest(root / "data/prepared_expanded/dataset.npz")
    assert expected_data_hash == auxiliary["dataset_sha256"] == efficiency_protocol["dataset_sha256"]
    assert auxiliary["fixed_response_replay_exactly_matches_main"] is True

    mapping = [
        ("Table 1", "样本、覆盖率与时间划分", ["results/paper_tables/table1_data.csv", "results/paper_tables/table1_sample_flow.csv", "results/paper_tables/table1_splits.csv"]),
        ("Table 2", "八种方法的 11 步 MAE、RMSE", ["results/paper_tables/table2_overall.csv"]),
        ("Table 3", "1、3、5、10、11 步与逐步误差", ["results/paper_tables/table3_horizons.csv", "results/paper_tables/table3_step_metrics_seeds.csv", "results/report_expanded/step_errors.png"]),
        ("Table 4", "五组重新训练的结构消融及参数量", ["results/paper_tables/table4_ablation.csv"]),
        ("Table 5(a)", "四种检索排序的重训与冻结替换", ["results/paper_tables/table5a_retrieval.csv"]),
        ("Table 5(b)", "参考构造与训练交集对照", ["results/paper_tables/table5b_reference.csv", "results/paper_tables/reference_verification.json", "data/reference_comparisons_expanded/comparison_protocol.json"]),
        ("Table 6", "同一检查点的训练成本、单事件延迟与误差", ["results/paper_tables/table6_cost.csv", "results/paper_tables/table6_aligned_cost_seeds.csv", "results/paper_tables/table6_same_capacity_accuracy.csv", "results/paper_tables/table6_training_seeds.csv"]),
        ("Table 7", "时期、平台、质量、博主及记忆更新分组", ["results/paper_tables/table7_generalization.csv", "results/paper_tables/table7_group_counts.csv"]),
        ("Figure 3", "固定案例的查询前状态与历史检索轨迹", ["results/paper_completion/figure3_query_prestate.png", "results/paper_completion/figure3_query_prestate.pdf", "results/report_expanded/retrieval_cases.png", "results/report_expanded/retrieval_case_manifest.csv", "results/report_expanded/retrieval_case_paths.csv"]),
        ("Figure 4", "六个固定案例的真实与预测轨迹", ["results/report_expanded/cases.png", "results/report_expanded/case_manifest.csv"]),
        ("参数分析", "K、L 及质量保留比例", ["results/paper_tables/parameter_metrics_seeds.csv", "results/paper_tables/quality_retention_metrics_seeds.csv", "results/report_expanded/parameter_K_mae.png", "results/report_expanded/parameter_L_mae.png", "results/report_expanded/parameter_quality_quantile_mae.png"]),
        ("辅助指标", "AR 方向准确率、峰值时间误差与平坦轨迹", ["results/paper_tables/auxiliary_metrics.csv", "results/paper_completion/peak_time_diagnostics_seeds.csv"]),
        ("RATL 容量对照", "相同的 912 条历史记忆", ["results/suite_expanded/capacity_metrics.csv"]),
        ("RAFT 辅助对照", "原始响应记忆、相同增量预测目标", ["results/report_expanded/raft_raw_history_metrics.csv"]),
    ]
    if revision_provenance:
        mapping.append(("参考版本说明", "规则、来源与独立复核", [entry["path"] for entry in revision_provenance.values()]))
    rows = []
    for item, description, files in mapping:
        for file in files:
            path = root / file
            assert path.is_file() and path.stat().st_size > 0, file
            rows.append(dict(paper_item=item, description=description, file=file, bytes=path.stat().st_size, sha256=digest(path)))
    pd.DataFrame(rows).to_csv(root / "results/paper_experiment_manifest.csv", index=False)

    for record in exports["files"]:
        assert digest(table / record["file"]) == record["sha256"]
    table2, table3 = pd.read_csv(table / "table2_overall.csv"), pd.read_csv(table / "table3_horizons.csv")
    assert len(table2) == 8 and len(table3) == 40
    group = pd.read_csv(table / "table7_generalization.csv")
    assert set(group.setting) == {"RAIF_fixed", "RAIF_rolling", "ECT_direct"}
    for kind in ["source_platform", "source_influencer"]:
        assert group[(group.setting == "RAIF_fixed") & (group.group_type == kind)].n_events.sum() == 2315

    completion = dict(status="completed", updated_utc=datetime.now(timezone.utc).isoformat(),
        primary_protocol="expanded_nearest", n_events=5105, n_train=2170, n_validation=620, n_test=2315,
        dataset_sha256=expected_data_hash, all_requested_experiment_categories_covered=True,
        active_reference_revision="single_reference", reference_comparison=reference,
        reference_revision_provenance=revision_provenance,
        q_rule="exp_only", quality_neff_factor_enabled=False,
        run_counts=dict(training_selection_and_deterministic_baseline=114,
                        validation_only_total=14, reference_training=18,
                        efficiency_retraining=21, deterministic_training_cost_records=1,
                        rolling_prediction_evaluations=3, frozen_retrieval_evaluations=12,
                        batch64_timing_model_seed_records=22,
                        batch1_original_checkpoint_timing_records=22,
                        efficiency_aligned_prediction_and_batch1_records=22),
        expected_trainable_seeds=[41,42,43], historical_mean_seed_count=1,
        efficiency=dict(gpu="NVIDIA GeForce RTX 4090, physical GPU 0", train_batch=64,
                        inference_batch=1, memory_events=912, inference_queries=128,
                        repeats_per_query=5, no_observed_contention=True,
                        whole_host_exclusivity_claimed=False,
                        training_source="separate same-capacity retraining",
                        table6_inference_and_accuracy_source="the same common-capacity checkpoints",
                        additional_inference_source="original main checkpoints with common memory",
                        preprocessing_measurement_scope="original_reference_construction_measurement",
                        preprocessing_includes_quality_revision=False,
                        reference_selection_recomputed_for_quality_revision=False),
        remaining_execution_tasks=[], manuscript_modified=False,
        scope_notes=["Primary Nearest Neighbor uses one fixed control for each of 5105 events.",
                     "The three reference schemes are Strict Matching, Constrained Nearest Neighbor, and Nearest Neighbor.",
                     f"Reference comparisons use {n_reference_test} canonical strict-reference test targets and {n_reference_train} common training events.",
                     exports["reference_attribute_scope"],
                     "The correlation minimum 0.7 was user-adopted after coverage review; it is not a preregistered threshold.",
                     "Reference quality is max(exp(-balance/scale_balance-distance/scale_distance),1e-8), with training-fitted scales.",
                     "Historical industry effective dates do not establish first-publication availability. Intraday recommendation timestamps remain unavailable.",
                     "RATL and other baselines are documented task adaptations.",
                     "The manuscript is maintained separately from the experiment artifacts."],
        evidence=dict(primary_summary="results/report_expanded/supplementary_summary.json",
                      auxiliary_summary="results/paper_completion/summary.json",
                      efficiency_status="results/serial_training_expanded/status.json",
                      aligned_efficiency_status="results/efficiency_aligned_expanded/status.json",
                      training_validation="results/serial_training_expanded/independent_validation.json",
                      aligned_validation="results/efficiency_aligned_expanded/independent_validation.json",
                      reference_status="results/references_expanded/status.json",
                      reference_verification="results/paper_tables/reference_verification.json",
                      file_manifest="results/paper_experiment_manifest.csv"))
    (root / "results/paper_experiment_completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2)+"\n")

    def link(label, file):
        return f"[{label}]({root / file})"

    text = ["# 论文实验结果", "",
        "主方案包含 5,105 个事件：训练 2,170 个、验证 620 个、测试 2,315 个。每个事件固定一个按事前市场状态选出的最近邻股票，预测未来 11 个交易日的 AR、换手率和对数成交量增量。七个可训练模型使用随机种子 41、42、43；Historical Mean 为确定性单次结果。", "",
        "当前设置包含 114 次训练、验证选择或确定性基线运行，其中三种参考方案在两种训练范围下各使用三个种子，共 18 次参考对照运行。另有 21 次同容量计时重训、15 次滚动或冻结预测重评。批量 64 的计时包含 22 组；原主检查点和同容量检查点的单事件计时分别包含 22 组。", "",
        "| 论文项目 | 内容 | 结果文件 |", "|---|---|---|"]
    for item, description, files in mapping:
        links = " · ".join(link(Path(file).name, file) for file in files)
        text.append(f"| {item} | {description} | {links} |")
    text += ["", "主测试集的完整窗口 MAE：", "",
             "| 模型 | AR | 换手率（百分点） | 对数成交量 |", "|---|---:|---:|---:|"]
    for row in table2.itertuples():
        values = []
        for channel in ["AR", "turnover", "log_volume"]:
            mean, std = getattr(row, channel+"_MAE_mean"), getattr(row, channel+"_MAE_std")
            values.append(f"{mean:.6f}" + (f" ± {std:.6f}" if np.isfinite(std) else ""))
        text.append(f"| {row.model} | {' | '.join(values)} |")
    winners = {channel: str(table2.loc[table2[channel+"_MAE_mean"].idxmin(), "model"])
               for channel in ["AR", "turnover", "log_volume"]}
    text += ["", f"AR、换手率和对数成交量的 MAE 均值最低模型分别为 {winners['AR']}、{winners['turnover']} 和 {winners['log_volume']}。", "",
        f"参考构造仅包含严格匹配（Strict Matching）、受约束单近邻（Constrained Nearest Neighbor）和单近邻（Nearest Neighbor）。每个事件固定一个参考股票；共同测试集包含 {n_reference_test} 个事件，训练交集包含 {n_reference_train} 个事件。严格匹配采用历史行业生效区间及 AR 路径相关系数不低于 0.7，完整事件为 640 个，训练、验证和自身测试分别为 278、82 和 280 个。0.7 阈值在覆盖率复核后由作者确定。", "",
        "参考质量为 exp(-平衡偏差/训练尺度-参考距离/训练尺度)，下限为 1e-8。主方案为 5,105 个事件的单近邻，行业、板块和市值不限制候选股票范围。", "",
        "效率表统一使用 912 条历史记忆。训练时间、测试误差和单事件延迟对应同一组重训检查点，原主检查点的单事件计时另存。训练时间含验证、缓存刷新和检查点写入；RATL 的基础、残差及 gamma 选择分别列示。参考构造耗时沿用原先固定参考股票及标签时的一次实测；此次 q 更新保留参考股票与标签，q 重算、行情及属性下载耗时未计入该项。", "",
        "历史行业记录具有有效区间，逐股首次公告时间仍未知。推荐记录只有日期，盘中发布时间无法核实。主方案中的行业与市值用于样本描述。严格匹配采用历史生效区间，受约束单近邻保留保守日期筛选规则。", "",
        "Table 1–7 的数据与 Figure 3–4 的实测图位于 RAIF，论文正文另行维护。", "",
        link("完整文件清单", "results/paper_experiment_manifest.csv") + " · " + link("运行汇总", "results/paper_experiment_completion.json") + " · " + link("数据说明", "results/industry_recovery/DATA_SCOPE.md")]
    if revision_provenance:
        labels = {"data_migration_verification": "数据版本核验", "independent_predictions_validation": "预测结果核验"}
        text += ["", " · ".join(link(labels[key], value["path"]) for key, value in revision_provenance.items())]
    (root / "EXPERIMENTS.md").write_text("\n".join(text)+"\n")

    state = read(root / "RUN_STATE.json")
    for key in ["active_remote_pid", "secondary_soft_suite", "secondary_soft_runs_completed", "old_results", "reference_revision"]:
        state.pop(key, None)
    state.update(status="completed_single_reference_experiments",
                 updated_utc=completion["updated_utc"], remaining_execution_tasks=[],
                 full_paper_completion="results/paper_experiment_completion.json", experiment_index="EXPERIMENTS.md",
                 serial_training="results/serial_training_expanded", aligned_efficiency="results/efficiency_aligned_expanded",
                 auxiliary_analysis="results/paper_completion", paper_tables="results/paper_tables", manuscript_modified=False,
                 active_reference_revision="single_reference", reference_comparison=reference,
                 reference_results="results/references_expanded", active_remote_processes=[],
                 training_and_validation_selection_runs=114, validation_only_runs=14,
                 experiment_run_counts=primary["run_counts"], full_run_counts=completion["run_counts"],
                 q_rule="exp_only", quality_neff_factor_enabled=False)
    state["primary_data"] = dict(directory="data/prepared_expanded", events=5105,
                                 split_counts={"train":2170,"validation":620,"test":2315},
                                 reference_mode="expanded_nearest", one_control_per_event=True,
                                 dataset_sha256=expected_data_hash,
                                 normalization_sha256=digest(root / "data/prepared_expanded/normalization.json"),
                                 q_rule="exp_only", quality_neff_factor_enabled=False)
    state["reference_revision"] = dict(status="completed", completed_utc=completion["updated_utc"],
                                      schemes=reference["schemes"],
                                      strict_complete_events=reference["strict_complete_events"],
                                      strict_split_counts=reference["strict_split_counts"],
                                      correlation_minimum=reference["strict_ar_correlation_minimum"],
                                      industry_policy="historical_effective_interval",
                                      common_test_events=n_reference_test, common_training_events=n_reference_train,
                                      training_runs_completed=reference["validated_runs"], results="results/references_expanded",
                                      q_rule="exp_only", quality_neff_factor_enabled=False, provenance=revision_provenance)
    state["checks"] = ["All 114 active training/selection/baseline runs completed and outputs validated.",
                       "The three reference schemes use one control per event and share canonical test targets.",
                       "Reference quality is exponential-only with a 1e-8 floor and training-fitted scales.",
                       "Fixed-response frozen evaluation agrees with the corresponding main predictions.",
                       "Efficiency training, accuracy and latency use aligned common-capacity checkpoints."]
    (root / "RUN_STATE.json").write_text(json.dumps(state, ensure_ascii=False, indent=2)+"\n")
    return completion


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    result = finish(args.root)
    print(json.dumps({"status":result["status"], "n_events":result["n_events"], "remaining":result["remaining_execution_tasks"]}))
