"""Render all measured Table 1–7 data into one standalone Markdown file."""
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/paper_tables"
OUTPUT = ROOT / "论文主表数据.md"
EXPORT_MANIFEST = json.loads((SOURCE / "table_export_manifest.json").read_text())
assert EXPORT_MANIFEST["reference_revision"] == "single_reference"
assert EXPORT_MANIFEST["quality_rule"]["q_rule"] == "exp_only"
MODELS = ["HistoricalMean", "DLinear", "PatchTST", "iTransformer", "ECT", "RAFT", "RATL", "RAIF"]
MODEL_NAMES = {"HistoricalMean": "Historical Mean", "RAFT": "RAFT（适配）", "RATL": "RATL（适配）"}
CHANNELS = [("AR", "AR"), ("turnover", "换手率"), ("log_volume", "对数成交量")]
VARIANTS = {"full": "完整 RAIF", "no_retrieval": "移除历史检索", "no_response": "移除响应监督检索", "no_quality": "移除参考质量权重", "no_attention": "移除时序交叉注意力", "no_events": "移除事件条件编码"}
SCHEMES = {"strict": "严格匹配", "nearest": "受约束单近邻", "expanded_nearest": "单近邻"}
SCHEME_ORDER = list(SCHEMES)
STRATEGIES = {"response": "响应相关性", "state": "输入状态", "random": "随机", "reverse": "逆序"}
SETTINGS = {"RAIF_fixed": "RAIF 固定记忆", "RAIF_rolling": "RAIF 滚动记忆", "ECT_direct": "ECT"}
parts = []
rendered_rows = {}


def read(name):
    return pd.read_csv(SOURCE / name)


def add(text=""):
    parts.append(text)


def value(x):
    return "不适用" if pd.isna(x) else format(float(x), ".8g")


def integer(x):
    return f"{int(x):,}"


def estimate(row, key):
    mean, std = row[key + "_mean"], row[key + "_std"]
    if pd.isna(mean):
        return "样本不足" if row.get("status") == "insufficient_group_size" else "不适用"
    assert math.isfinite(float(mean))
    if pd.isna(std):
        return value(mean)
    assert math.isfinite(float(std)) and std >= 0
    return value(mean) + " ± " + value(std)


def table(headers, rows, tag=None):
    rows = list(rows)
    assert all(len(row) == len(headers) for row in rows)
    add("| " + " | ".join(headers) + " |")
    add("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        add("| " + " | ".join(cells) + " |")
    add()
    if tag:
        rendered_rows[tag] = rendered_rows.get(tag, 0) + len(rows)


def errors(frame, label_column, labels, order=None, extra=None, tag=None):
    if order is not None:
        frame = frame.set_index(label_column).loc[order].reset_index()
    headers = ["模型／设置"]
    if extra:
        headers += [title for _, title in extra]
    headers += [name + " " + metric for _, name in CHANNELS for metric in ["MAE", "RMSE"]]
    rows = []
    for _, row in frame.iterrows():
        cells = [labels.get(row[label_column], row[label_column])]
        if extra:
            cells += [integer(row[key]) for key, _ in extra]
        cells += [estimate(row, channel + "_" + metric) for channel, _ in CHANNELS for metric in ["MAE", "RMSE"]]
        rows.append(cells)
    table(headers, rows, tag)


add("# RAIF 论文主表数据")
add()
add("主方案采用 5,105 个事件，每个事件按推荐前市场状态匹配一个最近邻股票。行业、板块和市值不限制主方案候选范围。预测窗口为 11 个交易日。Table 5(b) 使用公共严格参考测试子集；Table 6 使用统一历史记忆容量的效率实验。")
add()
add("预测误差均为原单位：AR 使用小数收益，换手率使用百分点，成交量以股为单位取自然对数。均值与样本标准差按随机种子汇总，记为均值 ± 标准差。七个可训练模型使用种子 41、42、43；Historical Mean 为确定性单次结果，不报告种子间标准差。数值保留八位有效数字。")
add()
add("## Table 1　数据规模、覆盖率与时间划分")
add()
names = {
"original_recommendation_rows": "原始推荐记录数", "identity_accepted_recommendation_rows": "证券身份筛选后记录数",
"identity_excluded_recommendation_rows": "证券身份筛选排除记录数", "original_influencer_identifiers": "原始博主标识数",
"identity_accepted_influencer_identifiers": "证券身份筛选后博主标识数", "merged_stock_trading_date_events": "去重并合并后的股票交易日事件数",
"retained_model_events": "主实验完整事件数", "retained_over_all_merged_events": "主实验事件占合并事件比例",
"retained_over_temporally_eligible_events": "主实验事件占时间划分合格事件比例", "temporal_exclusion_test_embargo": "测试边界隔离期排除事件数",
"temporal_exclusion_response_crosses_training_boundary": "响应窗口跨训练边界排除事件数", "temporal_exclusion_response_crosses_validation_boundary": "响应窗口跨验证边界排除事件数",
"temporal_exclusion_validation_embargo": "验证边界隔离期排除事件数"}
data = read("table1_data.csv")
table(["项目", "数值"], [[names[r.item], value(r.value) + "%" if r.unit == "percent" else integer(r.value)] for r in data.itertuples()], "table1_data")
flow = read("table1_sample_flow.csv").pivot(index="group", columns="stage", values="n_events")
group_names = {"all": "全部合并事件", "eligible": "时间划分合格事件", "train": "训练期", "validation": "验证期", "test": "测试期", "excluded": "时间规则排除事件"}
stages = ["input_events", "complete_prestate", "fixed_nearest_reference", "complete_fixed_reference_label", "retained_for_model"]
table(["范围", "输入事件", "前状态完整", "已固定最近邻", "固定参考标签完整", "模型保留事件"],
      [[label] + [integer(flow.loc[key, stage]) for stage in stages] for key, label in group_names.items()], "table1_flow")
add("全部合并事件中，标签完整数包含时间规则已排除的事件；主实验仅使用模型保留事件。时间划分合格的 5,300 个事件中，130 个缺少完整前状态，65 个在固定参考后缺少完整响应窗口。")
add()
splits = read("table1_splits.csv").set_index("split")
table(["分区", "事件数", "独立交易日数", "首个事件日", "最后事件日"],
      [[group_names[key], integer(splits.loc[key,"n_events"]), integer(splits.loc[key,"n_dates"]), splits.loc[key,"first_date"], splits.loc[key,"last_date"]] for key in ["train","validation","test"]], "table1_splits")
assert splits.n_events.sum() == 5105
reference = read("table5b_reference.csv")
base = reference[reference.training_scope.eq("full_training")].set_index("scheme")
assert reference.n_events.nunique() == reference.n_dates.nunique() == 1
COMMON_TEST_EVENTS = int(reference.n_events.iloc[0])
COMMON_TEST_DATES = int(reference.n_dates.iloc[0])
intersection = reference[reference.training_scope.eq("intersect_training")]
assert intersection.n_train.nunique() == 1
COMMON_TRAIN_EVENTS = int(intersection.n_train.iloc[0])
table(["参考方案", "完整事件", "覆盖率", "训练", "验证", "测试"],
      [[SCHEMES[key], integer(base.loc[key,"original_complete_events"]), f"{base.loc[key,'original_coverage_percent']:.2f}%", integer(base.loc[key,"n_train"]), integer(base.loc[key,"n_validation"]), integer(base.loc[key,"original_test_events"])] for key in SCHEME_ORDER])
add(f"覆盖率分母为 5,686 个合并事件；训练、验证和测试列为各方案自身保留的事件数。Table 5(b) 的共同评价子集包含 {integer(COMMON_TEST_EVENTS)} 个事件、{integer(COMMON_TEST_DATES)} 个独立交易日。")
add()

overall = read("table2_overall.csv")
add("## Table 2　整体预测结果")
add()
add("完整 11 步预测；共同测试集为 2,315 个事件、141 个独立交易日。")
add()
errors(overall, "model", MODEL_NAMES, MODELS, [("n_seeds", "种子数")], "table2")

add("## Table 3　不同预测长度的误差")
add()
add("@h 表示前 h 个交易日的累计 MAE 或 RMSE；使用同一组 11 步预测，@11 与 Table 2 一致。")
add()
horizons = read("table3_horizons.csv")
for channel, title in CHANNELS:
    for metric in ["MAE", "RMSE"]:
        add(f"### {title} {metric}")
        add()
        rows = []
        for model in MODELS:
            f = horizons[horizons.model.eq(model)].set_index("horizon")
            rows.append([MODEL_NAMES.get(model,model)] + [estimate(f.loc[h], channel+"_"+metric) for h in [1,3,5,10,11]])
            for stat in ["mean","std"]:
                a = f.loc[11,channel+"_"+metric+"_"+stat]
                b = overall.set_index("model").loc[model,channel+"_"+metric+"_"+stat]
                assert (pd.isna(a) and pd.isna(b)) or a == b
        table(["模型", "@1", "@3", "@5", "@10", "@11"], rows, "table3_panels")

add("## Table 4　RAIF 消融实验")
add()
add("每个变体重新训练，均使用三个随机种子和相同的 2,315 个测试事件。参数量包括完整模型中的所有参数。")
add()
errors(read("table4_ablation.csv"), "variant", VARIANTS, list(VARIANTS), [("parameters","参数量")], "table4")

add("## Table 5　历史检索与参考构造")
add()
retrieval = read("table5a_retrieval.csv")
for key, title in [("retrained", "(a1) 检索排序重训"), ("frozen_checkpoint", "(a2) 固定检查点的检索替换")]:
    add("### " + title)
    add()
    add("Top-K=5，训练历史记忆 2,170 条，测试事件 2,315 个，三个随机种子。各策略覆盖率均为 89.781921%（5,105 / 5,686）。" + ("各排序策略分别重新训练。" if key == "retrained" else "沿用相同种子的主 RAIF 检查点，仅改变检索排序。"))
    add()
    errors(retrieval[retrieval.evaluation.eq(key)], "retrieval", STRATEGIES, list(STRATEGIES), tag="table5a")
add("### (b) 参考构造对照的规模与规则")
add()
add(f"三种方案均为每个事件固定一个参考股票，统一在 {integer(COMMON_TEST_EVENTS)} 个严格参考事件、{integer(COMMON_TEST_DATES)} 个独立交易日上评价，测试目标完全相同。训练与验证使用各方案自己的标签。训练交集版本的训练事件均为 {integer(COMMON_TRAIN_EVENTS)} 个。单近邻为主方案。")
add("三种方案英文名称依次为 Strict Matching、Constrained Nearest Neighbor 和 Nearest Neighbor。公共评价仅保留三种方案均有完整标签的测试事件。")
add()
table(["方案", "候选股票筛选条件", "参考选择方式"], [
    ["严格匹配", "当日未被推荐且前状态完整；同行业、同板块，前一交易日总市值比在 [0.5, 2]；前 10 日 AR 路径相关系数 ≥0.7，累计 AR 绝对差 ≤0.2，收益波动率比在 [0.5, 2]，平均换手率比在 [0.7, 1.5]", "在通过全部筛选的候选中，选择 Mahalanobis 距离最小的一只股票"],
    ["受约束单近邻", "当日未被推荐且前状态完整；同行业、同板块，前一交易日总市值比在 [0.5, 2]", "在通过筛选的候选中，选择 Mahalanobis 距离最小的一只股票"],
    ["单近邻", "当日未被推荐且前状态完整", "在完整候选范围内，选择 Mahalanobis 距离最小的一只股票"],
])
add("表中比值均为参考股票相对于被推荐股票的比值。严格匹配按历史行业生效区间确定行业，受约束单近邻还要求行业数据源的更新时间不晚于事件日；行业首次公开时间仍未知。两种行业可用性口径不同，保留事件集合不预设逐级包含。0.7 的相关性下限在覆盖率复核后由作者确定。")
add()
add(r"参考质量定义为 $q_e=\max\{\exp(-b_e/s_b-d_e/s_d),10^{-8}\}$，其中 $b_e$ 为匹配状态偏差，$d_e$ 为参考距离；两个尺度由训练事件确定。")
add()
for scope, title in [("full_training", "(b1) 各方案自身训练集"), ("intersect_training", "(b2) 公共训练事件交集")]:
    add("### " + title)
    add()
    errors(reference[reference.training_scope.eq(scope)], "scheme", SCHEMES, SCHEME_ORDER, [("n_train","训练事件")], "table5b")
add("### (b3) 参考属性统计")
add()
add("下列各格为样本分布的均值 / 中位数，不是种子均值与标准差。原完整事件统计和实际训练集统计分开列示；训练交集版本重新确定质量标尺。")
add()
quality_fields = [("balance","平衡偏差"), ("mean_distance","参考距离"), ("q","质量 q")]
for scope, title, prefix in [("full_training","原完整事件（全部分区）","original_"), ("full_training","各方案自身训练集","training_"), ("intersect_training","公共训练交集","training_")]:
    add("**"+title+"**")
    add()
    q = reference[reference.training_scope.eq(scope)].set_index("scheme")
    table(["方案"]+[title for _,title in quality_fields],
          [[SCHEMES[key]]+[value(q.loc[key,prefix+field+"_mean"])+" / "+value(q.loc[key,prefix+field+"_median"]) for field,_ in quality_fields] for key in SCHEME_ORDER])

add("## Table 6　计算成本")
add()
add("同容量实验统一采用 912 条训练历史记忆。训练成本、推断延迟及本表所列误差来自同一组重训检查点。设备为 GPU 0 的 NVIDIA GeForce RTX 4090，精度为 float32，CPU 线程数为 2；训练批量为 64，推断批量为 1。RATL 按原时间分段使用 1,145 个基础模型训练事件和 912 个残差训练事件；其余可训练模型使用 2,170 个训练事件。")
add()
cost = read("table6_cost.csv").set_index("model").loc[MODELS].reset_index()
add("### (a) 参数与推断开销")
add()
table(["模型", "总参数", "可训练参数", "冻结基础参数", "单事件推断（ms）", "单事件检索（ms）", "记忆（MB）"],
      [[MODEL_NAMES.get(r.model,r.model),integer(r.parameters),integer(r.trainable_parameters),integer(r.frozen_base_parameters),estimate(r,"inference_event_ms"),estimate(r,"retrieval_event_ms"),estimate(r,"canonical_memory_mb")] for _,r in cost.iterrows()], "table6_cost")
add("每个模型／种子在同样的 128 个查询事件上重复 5 遍，共 640 次单事件测量；预热 20 次。表中标准差为种子间样本标准差。推断时间包括编码、检索与预测，输入和记忆已驻留 GPU；参考构造、主机与设备传输、记忆缓存准备时间另计。记忆项为缓存键及响应值；RATL 使用缓存键及残差值，单位为十进制 MB。")
add()
add("### (b) 训练开销（秒）")
add()
fields = [("training_seconds","训练总计"),("optimization_seconds","优化步骤"),("validation_seconds","逐轮验证"),("training_cache_seconds","训练缓存刷新"),("initialization_seconds","模型初始化")]
table(["模型"]+[title for _,title in fields]+["原参考构造实测"],
      [[MODEL_NAMES.get(r.model,r.model)]+[estimate(r,key) for key,_ in fields]+[value(r.preprocessing_seconds_shared)] for _,r in cost.iterrows()])
add("训练总计包括拟合阶段、验证、缓存刷新、检查点写入及其他训练操作，并加入 RATL 的 gamma 选择阶段；模型初始化单列，直接测得的 GPU 监测调用耗时已扣除。原参考构造实测列沿用固定参考股票及标签时的一次测量，不报告种子标准差。此次 q 更新保留参考股票与标签，该列未计入 q 重算、行情及属性下载耗时。Historical Mean 不进行参数优化。")
add()
add("### (c) RATL 训练阶段（秒）")
add()
r = cost.set_index("model").loc["RATL"]
table(["模型", "基础模型", "残差修正", "gamma 选择", "训练总计"], [["RATL（适配）"]+[estimate(r,key) for key in ["ratl_base_seconds","ratl_residual_seconds","ratl_gamma_seconds","training_seconds"]]])
add("### (d) 同容量检查点的预测误差")
add()
add("测试集仍为 2,315 个事件。这里的检查点使用 912 条共同记忆，与 Table 2 的主实验设置分别列示。")
add()
errors(cost,"model",MODEL_NAMES,MODELS,tag="table6_accuracy")
table(["模型","三个通道标准化 MAE 等权平均"], [[MODEL_NAMES.get(r.model,r.model),estimate(r,"test_standardized_mae")] for _,r in cost.iterrows()])

add("## Table 7　泛化、分组与记忆更新")
add()
add("各组比较 RAIF 固定记忆、同一检查点的滚动记忆和验证集选择的直接基线 ECT，均为三个随机种子。滚动记忆仅追加响应窗口已在当前查询日前结束的事件。质量分组使用训练集三分位阈值。博主已见状态按训练词表确定，平台和博主来源分组仅用于结果描述。")
add()
add("每组至少包含 20 个事件和 5 个独立交易日才报告误差；样本不足的组仅列规模。")
add()
groups = read("table7_generalization.csv")
group_types = [
("period_half","(a) 测试前后期",{"earlier":"2025-10-01 前","later":"2025-10-01 起"}),
("source_platform","(b) 原始来源平台",{"weibo_only":"仅微博","wechat_only":"仅微信","multiple_platforms":"多个平台","missing":"来源缺失"}),
("quality","(c) 参考质量",{"low":"低质量","middle":"中质量","high":"高质量"}),
("source_influencer","(d) 来源博主是否已见",{"all_seen":"全部已见","all_unseen":"全部未见","mixed":"已见与未见混合","missing":"来源缺失"}),
("all_test","(e) 全测试集与记忆策略",{"all":"全部测试事件"}),
("period","(f) 季度分组",{"2025Q3":"2025 年第三季度","2025Q4":"2025 年第四季度","2026Q1":"2026 年第一季度"}),
("platform","(g) 模型使用的平台条件",{"__UNKNOWN__":"未知平台条件","微信":"微信","微博":"微博"}),
("influencer_status","(h) 模型使用的博主条件",{"seen":"已知博主嵌入","unknown_or_unseen":"未知或未见博主嵌入"}),
("condition_known","(i) 事件条件是否明确",{"known":"明确","unknown":"未知"})]
for kind, title, labels in group_types:
    add("### "+title)
    add()
    subset = groups[groups.group_type.eq(kind)].set_index(["group","setting"])
    rows=[]
    for group, label in labels.items():
        for setting, setting_label in SETTINGS.items():
            r=subset.loc[(group,setting)]
            rows.append([label,setting_label,integer(r.n_events),integer(r.n_dates)]+[estimate(r,c+"_"+m) for c,_ in CHANNELS for m in ["MAE","RMSE"]])
    table(["分组","模型／记忆","事件数","交易日数"]+[label+" "+m for _,label in CHANNELS for m in ["MAE","RMSE"]], rows, "table7")
add("未知模型条件与原始来源缺失分别统计。多来源记录的先后顺序不能从日频数据确定，因此模型可使用未知条件；来源平台及作者类别仍可用于事后分组描述。")
add()
add("## 数据范围与来源")
add()
add("推荐记录只有日期，盘中发布时间及 15:00 切点无法核实。历史行业记录具有有效区间，但逐股首次公告时间未知；行业与市值用于主样本描述。严格匹配按历史生效区间确定行业，受约束单近邻保留保守日期筛选规则。所有数值来自已保存实验输出。")
add()
files = ["table1_data.csv","table1_sample_flow.csv","table1_splits.csv","table2_overall.csv","table3_horizons.csv","table4_ablation.csv","table5a_retrieval.csv","table5b_reference.csv","table6_cost.csv","table7_generalization.csv"]
table(["数据文件","位置"], [[name, f"[{name}]({SOURCE/name})"] for name in files])

assert rendered_rows == {"table1_data":13,"table1_flow":6,"table1_splits":3,"table2":8,"table3_panels":48,"table4":6,"table5a":8,"table5b":6,"table6_cost":8,"table6_accuracy":8,"table7":72}, rendered_rows
content = "\n".join(parts)+"\n"
assert "nan" not in content.lower() and "inf ±" not in content
OUTPUT.write_text(content,encoding="utf-8")
print(f"{OUTPUT}\n{len(content.splitlines())} lines; {OUTPUT.stat().st_size} bytes")
