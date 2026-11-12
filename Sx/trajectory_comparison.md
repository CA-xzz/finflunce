# Comparison of Full-Trajectory and Incremental-Response Forecasting

以下采用当前论文报告的最新数值。所有方法均评价增量响应的三通道 MAE 与 RMSE。

| 方法 | 预测形式 | 异常收益 MAE | 异常收益 RMSE | 换手率 MAE | 换手率 RMSE | 对数成交量 MAE | 对数成交量 RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ECT-整体 | 整体轨迹 | 0.03005 ± 0.00013 | 0.04505 ± 0.00016 | 3.670 ± 0.050 | 6.480 ± 0.070 | 0.751 ± 0.006 | 0.990 ± 0.008 |
| ECT-增量 | 增量响应 | 0.02878 ± 0.00011 | 0.04295 ± 0.00013 | 3.490 ± 0.041 | 6.190 ± 0.058 | 0.722 ± 0.005 | 0.950 ± 0.007 |
| RAFT-整体 | 整体轨迹 | 0.02955 ± 0.00012 | 0.04415 ± 0.00015 | 3.580 ± 0.047 | 6.330 ± 0.066 | 0.737 ± 0.006 | 0.970 ± 0.008 |
| RAFT-增量 | 增量响应 | 0.02842 ± 0.00010 | 0.04253 ± 0.00012 | 3.430 ± 0.044 | 6.100 ± 0.061 | 0.712 ± 0.005 | 0.937 ± 0.006 |
| RAIF | 增量响应 | 0.02702 ± 0.00009 | 0.04061 ± 0.00012 | 3.220 ± 0.030 | 5.800 ± 0.048 | 0.686 ± 0.004 | 0.903 ± 0.006 |

两个 Direct 方法按同一种子共用基于参考股票事前状态的参考预测器。RAFT 两种形式采用相同的检索设置，分别使用整体轨迹和增量轨迹记忆。

[CSV](published_results/trajectory_comparison/table2_trajectory_comparison.csv) · [LaTeX](published_results/trajectory_comparison/table2_trajectory_comparison.tex) · [来源](published_results/source/incremental_comparison.json)
