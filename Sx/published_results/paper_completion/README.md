# 脚本说明

运行环境：Python 3.9+、numpy、pandas、matplotlib。仅读取本地结果，不调用训练模型或网络。

```sh
python Sx/complete_paper_outputs.py --data Sx/data/prepared_expanded --suite Sx/results/suite_expanded --report Sx/results/report_expanded --rolling Sx/results/rolling_expanded --frozen Sx/results/frozen_retrieval_expanded --output Sx/results/paper_completion
python Sx/complete_paper_outputs.py --self-test
```

辅助指标：auxiliary_metrics.csv 和逐种子表；峰值平坦与并列数量见 peak_time_diagnostics_seeds.csv，逐事件值见 peak_time_events.csv。

分组对照：memory_group_metrics.csv、memory_group_differences.csv 及逐种子表；group_counts.csv 包含所有组的规模和可报告状态。source_group_event_assignments.csv、source_group_record_links.csv、source_author_training_status.csv 保存来源分组依据。

案例图：figure3_query_prestate.png/.pdf/.svg；源值见同名 CSV，案例清单见 figure3_case_manifest.csv。

source_manifest.csv 保存所有读取文件的哈希；output_manifest.csv 为输出清单。未修改主报告、原有脚本或准备数据。
