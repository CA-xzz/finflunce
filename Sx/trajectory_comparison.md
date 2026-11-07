# 完整轨迹与增量响应预测对照

Comparison of Full-Trajectory and Incremental-Response Forecasting

| 方法 | 预测形式 | AR MAE | AR RMSE | 换手率 MAE | 换手率 RMSE | 对数成交量 MAE | 对数成交量 RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| ECT-Direct | Full | 0.030789 ± 0.000080 | 0.045656 ± 0.000121 | 3.108760 ± 0.022613 | 5.661019 ± 0.056123 | 0.507059 ± 0.001847 | 0.686292 ± 0.003548 |
| ECT-Incremental | Incremental | 0.030577 ± 0.000021 | 0.045515 ± 0.000028 | 3.436803 ± 0.045170 | 6.146669 ± 0.036472 | 0.719782 ± 0.005619 | 0.940169 ± 0.007241 |
| RAFT-Direct | Full | 0.030990 ± 0.000047 | 0.045895 ± 0.000060 | 3.215418 ± 0.057743 | 5.833760 ± 0.061711 | 0.517970 ± 0.001663 | 0.701749 ± 0.002688 |
| RAFT-Incremental | Incremental | 0.030707 ± 0.000047 | 0.045636 ± 0.000038 | 3.409538 ± 0.030569 | 6.093510 ± 0.025600 | 0.735662 ± 0.000887 | 0.957480 ± 0.002223 |
| RAIF | Incremental | 0.030544 ± 0.000026 | 0.045481 ± 0.000040 | 3.608403 ± 0.014148 | 6.556614 ± 0.028840 | 0.730515 ± 0.002526 | 0.959032 ± 0.002274 |

表中所有方法均评价增量响应。Direct 方法先分别预测推荐股票完整轨迹 $\widehat{Y}$ 和参考股票完整轨迹 $\widehat{B}$，再以 $\widehat{Y}-\widehat{B}$ 作为增量预测。Incremental 方法及 RAIF 直接预测增量响应。

| 方法 | 预测形式公式 | 历史辅助信息 |
|---|---|---|
| ECT-Direct | $\widehat{\Delta}=\widehat{Y}_{\mathrm{ECT}}-\widehat{B}_{\mathrm{ECT}}$ | 无 |
| ECT-Incremental | $\widehat{\Delta}=f_{\mathrm{ECT}}(X)$ | 无 |
| RAFT-Direct | $\widehat{\Delta}=\widehat{Y}_{\mathrm{RAFT}}-\widehat{B}_{\mathrm{ECT}}$ | 原始市场轨迹 |
| RAFT-Incremental | $\widehat{\Delta}=f_{\mathrm{RAFT}}(X,\mathcal{H}_{\Delta})$ | 历史增量轨迹 |
| RAIF | $\widehat{\Delta}=f_{\mathrm{RAIF}}(X,\mathcal{H}_{\Delta})$ | 响应相关历史增量 |

其中，$X$ 表示模型的事件前输入，$\mathcal{H}_{\Delta}$ 表示检索到的历史增量轨迹；完整轨迹方法中的参考预测由同种子的共享 ECT 参考模型提供。历史辅助信息列指用于检索的历史轨迹。

主样本共 5,105 个事件，训练、验证、测试分别为 2,170、620、2,315 个事件；测试集覆盖 141 个交易日。各方法使用相同的随机种子 41、42、43、事件顺序和测试标签。输入为事件前 10 个交易日，预测长度为 11 个交易日。MAE 和 RMSE 先在每个种子的全部测试事件及预测时点上计算，再报告三个种子的均值与样本标准差（分母为 2）。AR、换手率及对数成交量均使用原数据单位。

所有方法的评价标签均由原主数据中的标准化增量标签反标准化获得。完整轨迹训练标签采用同一批事件及已固定的单近邻参考股票，参考股票不重新选择。

每个种子先训练一个 ECT 参考模型，以参考股票的事件前状态预测完整参考轨迹，类别输入均为 0，并按标准化参考轨迹验证集 MAE 选择模型。该参考模型固定后，同种子的 ECT-Direct 和 RAFT-Direct 共享其预测结果。两种 Direct 方法训练推荐股票完整轨迹，并按最终增量预测的标准化验证集 MAE 选择模型；预测时不使用真实参考股票未来轨迹。

推荐股票完整轨迹以训练集均值中心化，除以原主任务增量标签的训练集标准差；参考股票完整轨迹使用参考轨迹自身的训练集均值和标准差。标准化参数仅由训练集拟合。

ECT-Incremental、RAFT-Incremental 和 RAIF 复用原主实验对应种子的模型与预测。新增 ECT、RAFT 组件沿用各自原主实验的模型结构、优化器和固定超参数，最多训练 100 轮，早停耐心值为 15；本项实验不再次调参。两个 Direct 方案额外使用参考股票的事件前状态和共享参考模型，因此整体方案的输入与总参数量包含这部分成本。

RAFT-Direct 与 RAFT-Incremental 均在尺度 1、2、4 上按 Pearson 相似度各检索 Top-5 历史事件。候选记忆仅含 2,170 个训练事件，且历史响应结束日须早于查询日；各方法分别使用其训练目标对应的历史轨迹。RAIF 保留原主实验的检索与修正配置，质量权重由匹配状态偏差和距离确定，不含有效参考数项。

[逐种子指标](<published_results/trajectory_comparison/metrics_seeds.csv>) · [均值与标准差 CSV](<published_results/trajectory_comparison/table2_trajectory_comparison.csv>) · [LaTeX 表格](<published_results/trajectory_comparison/table2_trajectory_comparison.tex>)
