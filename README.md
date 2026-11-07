# finflunce · RAIF 实验代码与数据

RAIF 金融推荐事件预测实验。全部代码和数据位于 `Sx/`，主任务预测推荐股票相对匹配参考股票的未来增量响应，输出 AR、换手率和对数成交量三个通道。

## 内容

| 路径 | 内容 |
|---|---|
| [`Sx/models.py`](Sx/models.py)、[`Sx/train.py`](Sx/train.py) | RAIF、ECT、RAFT、RATL、DLinear、PatchTST、iTransformer、Historical Mean 及训练与评价 |
| [`Sx/reproduce.py`](Sx/reproduce.py) | 主实验、消融、检索、参数、参考方案、轨迹对照、泛化及效率实验入口 |
| [`Sx/data/`](Sx/data/) | 可直接训练的数据集与事件筛选记录 |
| [`Sx/DATA.md`](Sx/DATA.md) | 数据字段、时间划分和参考方案 |
| [`Sx/论文主表数据.md`](Sx/论文主表数据.md) | 原论文主表的实测数据 |
| [`Sx/trajectory_comparison.md`](Sx/trajectory_comparison.md) | Full-Trajectory vs. Incremental-Response Forecasting 的设置与实测结果 |
| [`Sx/published_results/paper_tables/`](Sx/published_results/paper_tables/) | 主表 CSV |
| [`Sx/published_results/trajectory_comparison/table2_trajectory_comparison.tex`](Sx/published_results/trajectory_comparison/table2_trajectory_comparison.tex) | 新增对照实验的 LaTeX 表格 |

`published_results/` 保存已完成实验的表格、图、配置和汇总指标。重新训练的检查点、逐事件预测及日志写入 `results/` 和 `logs/`。仓库不包含原实验的模型权重和逐事件预测文件；这些文件由训练命令生成。

## 安装与数据检查

原实验环境为 Linux、Python 3.9.5、PyTorch 2.5.1 + CUDA 12.1。数据读取和单元测试也支持 Python 3.11。

```bash
git clone https://github.com/CA-xzz/finflunce.git
cd finflunce
python3 -m venv Sx/envs/Sx
source Sx/envs/Sx/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Sx/requirements.txt
python Sx/check_dataset.py
python -m unittest discover -s Sx -p 'test_*.py'
```

仅运行 CPU 测试时，可使用 PyTorch CPU 安装源 `https://download.pytorch.org/whl/cpu`。完整原环境包列表保存在 `Sx/requirements.lock.txt`，其中 CUDA 依赖针对 Linux。

可直接用仓库内的 `dataset.npz` 训练，17 组处理后数据覆盖主实验和各项对照。数据文件的校验值见 `Sx/data/manifest.json`。原始行情、历史市值、历史行业来源文件和下载缓存不上传到本仓库；原始数据处理脚本保留，需另行提供对应输入。

## 运行实验

以下命令在一张空闲 GPU 上依次运行全部保留实验，包含三个随机种子 41、42、43。训练和选择参数与已完成实验一致：最多 100 个 epoch，早停耐心值 15，batch size 64，验证集选模。

```bash
python Sx/reproduce.py --gpu 0 --stage all
```

也可分阶段执行。后续阶段读取本次 `suite` 生成的参数或检查点：

```bash
python Sx/reproduce.py --gpu 0 --stage suite
python Sx/reproduce.py --gpu 0 --stage parameters
python Sx/reproduce.py --gpu 0 --stage references
python Sx/reproduce.py --gpu 0 --stage raw-history
python Sx/reproduce.py --gpu 0 --stage trajectory
python Sx/reproduce.py --gpu 0 --stage extras
python Sx/reproduce.py --gpu 0 --stage efficiency
python Sx/reproduce.py --gpu 0 --stage reports
```

`--dry-run` 只列出命令。现有调度器支持 GPU 编号 0–3。效率实验检查所选 GPU 的占用情况，新的硬件与负载会影响计时结果。单个模型可直接运行：

```bash
python Sx/train.py \
  --data Sx/data/prepared_expanded \
  --output Sx/results/example_RAIF \
  --model RAIF --seed 41 --device cuda
```

多卡入口为 `run_suite.py`、`run_parameters.py`、`run_references.py` 和 `run_trajectory_comparison.py`。单卡入口自动重建参考对照计划中的本机路径；数据中原有路径字段记录原实验来源。

## 新增轨迹对照

包含 ECT-Direct、ECT-Incremental、RAFT-Direct、RAFT-Incremental 和 RAIF。所有方法在同一组 2,315 个测试事件、11 步预测窗口上计算增量响应的 MAE 和 RMSE。

Direct 方法分别预测推荐股票轨迹与参考股票轨迹，再计算差值。参考预测器仅使用参考股票的推荐前状态，同一种子下由 ECT-Direct 和 RAFT-Direct 共用。RAFT 两种形式使用相同的历史候选、Pearson 距离，以及尺度 1、2、4 各 Top-5 的检索预算。

`trajectory` 阶段复核本次训练的九个增量模型预测，训练九个完整轨迹组件，并导出三种子 mean ± sample std。运行结果写入 `Sx/results/trajectory_comparison/`。

## 实验范围

三个参考方案均为单参考，参考权重为 1。主方案为单近邻的 5,105 个事件。严格匹配采用 AR 路径相关系数不低于 0.70，对应 640 个事件。参考对照在共同的 235 个测试事件上使用相同目标，训练交集对照包含 157 个事件。

行业数据按历史生效区间整理，逐股首次公告日期未核实；受约束单近邻沿用来源更新日期筛选。三组事件不保证嵌套。基线为本项目的统一实现，具体适配见训练配置。数据来源与处理记录保存在各数据目录的 metadata、source manifest 和配置文件中。
