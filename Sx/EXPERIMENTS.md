# 论文实验结果

主方案包含 5,105 个事件：训练 2,170 个、验证 620 个、测试 2,315 个。每个事件固定一个按事前市场状态选出的最近邻股票，预测未来 11 个交易日的 AR、换手率和对数成交量增量。七个可训练模型使用随机种子 41、42、43；Historical Mean 为确定性单次结果。

当前设置包含 114 次训练、验证选择或确定性基线运行，其中三种参考方案在两种训练范围下各使用三个种子，共 18 次参考对照运行。另有 21 次同容量计时重训、15 次滚动或冻结预测重评。批量 64 的计时包含 22 组；原主检查点和同容量检查点的单事件计时分别包含 22 组。

| 论文项目 | 内容 | 结果文件 |
|---|---|---|
| Table 1 | 样本、覆盖率与时间划分 | [table1_data.csv](published_results/paper_tables/table1_data.csv) · [table1_sample_flow.csv](published_results/paper_tables/table1_sample_flow.csv) · [table1_splits.csv](published_results/paper_tables/table1_splits.csv) |
| Table 2 | 八种方法的 11 步 MAE、RMSE | [table2_overall.csv](published_results/paper_tables/table2_overall.csv) |
| Table 3 | 1、3、5、10、11 步与逐步误差 | [table3_horizons.csv](published_results/paper_tables/table3_horizons.csv) · [table3_step_metrics_seeds.csv](published_results/paper_tables/table3_step_metrics_seeds.csv) · [step_errors.png](published_results/report_expanded/step_errors.png) |
| Table 4 | 五组重新训练的结构消融及参数量 | [table4_ablation.csv](published_results/paper_tables/table4_ablation.csv) |
| Table 5(a) | 四种检索排序的重训与冻结替换 | [table5a_retrieval.csv](published_results/paper_tables/table5a_retrieval.csv) |
| Table 5(b) | 参考构造与训练交集对照 | [table5b_reference.csv](published_results/paper_tables/table5b_reference.csv) · [reference_verification.json](published_results/paper_tables/reference_verification.json) · [comparison_protocol.json](data/reference_comparisons_expanded/comparison_protocol.json) |
| Table 6 | 同一检查点的训练成本、单事件延迟与误差 | [table6_cost.csv](published_results/paper_tables/table6_cost.csv) · [table6_aligned_cost_seeds.csv](published_results/paper_tables/table6_aligned_cost_seeds.csv) · [table6_same_capacity_accuracy.csv](published_results/paper_tables/table6_same_capacity_accuracy.csv) · [table6_training_seeds.csv](published_results/paper_tables/table6_training_seeds.csv) |
| Table 7 | 时期、平台、质量、博主及记忆更新分组 | [table7_generalization.csv](published_results/paper_tables/table7_generalization.csv) · [table7_group_counts.csv](published_results/paper_tables/table7_group_counts.csv) |
| Figure 3 | 固定案例的查询前状态与历史检索轨迹 | [figure3_query_prestate.png](published_results/paper_completion/figure3_query_prestate.png) · [figure3_query_prestate.pdf](published_results/paper_completion/figure3_query_prestate.pdf) · [retrieval_cases.png](published_results/report_expanded/retrieval_cases.png) · [retrieval_case_manifest.csv](published_results/report_expanded/retrieval_case_manifest.csv) · [retrieval_case_paths.csv](published_results/report_expanded/retrieval_case_paths.csv) |
| Figure 4 | 六个固定案例的真实与预测轨迹 | [cases.png](published_results/report_expanded/cases.png) · [case_manifest.csv](published_results/report_expanded/case_manifest.csv) |
| 参数分析 | K、L 及质量保留比例 | [parameter_metrics_seeds.csv](published_results/paper_tables/parameter_metrics_seeds.csv) · [quality_retention_metrics_seeds.csv](published_results/paper_tables/quality_retention_metrics_seeds.csv) · [parameter_K_mae.png](published_results/report_expanded/parameter_K_mae.png) · [parameter_L_mae.png](published_results/report_expanded/parameter_L_mae.png) · [parameter_quality_quantile_mae.png](published_results/report_expanded/parameter_quality_quantile_mae.png) |
| 辅助指标 | AR 方向准确率、峰值时间误差与平坦轨迹 | [auxiliary_metrics.csv](published_results/paper_tables/auxiliary_metrics.csv) · [peak_time_diagnostics_seeds.csv](published_results/paper_completion/peak_time_diagnostics_seeds.csv) |
| RATL 容量对照 | 相同的 912 条历史记忆 | [capacity_metrics.csv](published_results/suite_expanded/capacity_metrics.csv) |
| RAFT 辅助对照 | 原始响应记忆、相同增量预测目标 | [raft_raw_history_metrics.csv](published_results/report_expanded/raft_raw_history_metrics.csv) |
| 参考版本说明 | 规则、来源与独立复核 | [data_migration_verification.json](published_results/single_reference_revision/data_migration_verification.json) · [independent_predictions_validation.json](published_results/single_reference_revision/independent_predictions_validation.json) |

主测试集的完整窗口 MAE：

| 模型 | AR | 换手率（百分点） | 对数成交量 |
|---|---:|---:|---:|
| HistoricalMean | 0.030551 | 4.029097 | 0.764036 |
| DLinear | 0.030589 ± 0.000012 | 3.395059 ± 0.020808 | 0.730116 ± 0.004888 |
| PatchTST | 0.030711 ± 0.000100 | 3.455844 ± 0.012776 | 0.730387 ± 0.005682 |
| iTransformer | 0.030627 ± 0.000045 | 3.552158 ± 0.055440 | 0.726422 ± 0.004242 |
| ECT | 0.030577 ± 0.000021 | 3.436803 ± 0.045170 | 0.719782 ± 0.005619 |
| RAFT | 0.030707 ± 0.000047 | 3.409538 ± 0.030569 | 0.735662 ± 0.000887 |
| RATL | 0.030853 ± 0.000172 | 3.580220 ± 0.042245 | 0.735770 ± 0.015359 |
| RAIF | 0.030544 ± 0.000026 | 3.608403 ± 0.014148 | 0.730515 ± 0.002526 |

AR、换手率和对数成交量的 MAE 均值最低模型分别为 RAIF、DLinear 和 ECT。

参考构造仅包含严格匹配（Strict Matching）、受约束单近邻（Constrained Nearest Neighbor）和单近邻（Nearest Neighbor）。每个事件固定一个参考股票；共同测试集包含 235 个事件，训练交集包含 157 个事件。严格匹配采用历史行业生效区间及 AR 路径相关系数不低于 0.7，完整事件为 640 个，训练、验证和自身测试分别为 278、82 和 280 个。0.7 阈值在覆盖率复核后由作者确定。

参考质量为 exp(-平衡偏差/训练尺度-参考距离/训练尺度)，下限为 1e-8。主方案为 5,105 个事件的单近邻，行业、板块和市值不限制候选股票范围。

效率表统一使用 912 条历史记忆。训练时间、测试误差和单事件延迟对应同一组重训检查点，原主检查点的单事件计时另存。训练时间含验证、缓存刷新和检查点写入；RATL 的基础、残差及 gamma 选择分别列示。参考构造耗时沿用原先固定参考股票及标签时的一次实测；此次 q 更新保留参考股票与标签，q 重算、行情及属性下载耗时未计入该项。

历史行业记录具有有效区间，逐股首次公告时间仍未知。推荐记录只有日期，盘中发布时间无法核实。主方案中的行业与市值用于样本描述。严格匹配采用历史生效区间，受约束单近邻保留保守日期筛选规则。

Table 1–7 的数据与 Figure 3–4 的实测图位于 Sx，论文正文另行维护。

[完整文件清单](published_results/paper_experiment_manifest.csv) · [运行汇总](published_results/paper_experiment_completion.json) · [数据说明](published_results/industry_recovery/DATA_SCOPE.md)

[数据版本核验](published_results/single_reference_revision/data_migration_verification.json) · [预测结果核验](published_results/single_reference_revision/independent_predictions_validation.json)

## 完整轨迹与增量响应预测对照

Comparison of Full-Trajectory and Incremental-Response Forecasting。五种方法统一在2,315个测试事件上评价增量响应的三通道MAE、RMSE。新增9个整体轨迹组件训练，复用原增量模型的9个种子结果。两个Direct方法共用同种子的参考预测器。

[结果与设置](trajectory_comparison.md) · [LaTeX表格](published_results/trajectory_comparison/table2_trajectory_comparison.tex) · [训练入口](run_trajectory_comparison.py) · [独立核验](published_results/trajectory_comparison/independent_validation.json)

该表供论文5.3节整体预测性能之后使用，论文正文另行维护。
