"""Export current manuscript tables and figures without changing training runs."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import argparse
import contextlib
import csv
import hashlib
import io
import json
import re
import runpy
import shutil
import tempfile

INPUTS = [
    'main.tex', 'main.aux', 'sections/Experiments.tex',
    'results/supplied_tables.json', 'results/incremental_comparison.json',
    'scripts/export_experiment_results.py', 'scripts/plot_supplied_results.py',
    'tables/overall.tex', 'tables/incremental_comparison.tex', 'tables/reference.tex',
    'worklog/data_figures_20261106/plot_values.json',
    'worklog/experiment_style_20261106/plot_values.json',
]
FIGURES = ['forecast_horizons', 'group_performance', 'ablation_comparison', 'hyperparameters']
CHANNEL_METRICS = ['AR_MAE', 'AR_RMSE', 'turnover_MAE', 'turnover_RMSE',
                   'log_volume_MAE', 'log_volume_RMSE']
NUMBER = re.compile(r'^([+-]?\d+(?:\.\d+)?)(?:\s*±\s*([+-]?\d+(?:\.\d+)?))?\s*(%)?$')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def numeric(value):
    match = NUMBER.fullmatch(str(value).strip().replace(',', ''))
    if not match:
        return None
    mean, std, percent = match.groups()
    if std is not None and Decimal(std) < 0:
        raise ValueError('Negative reported standard deviation')
    return mean, std or '', 'percent' if percent else ''


def metric_table(output, filename, rows, prefix_header=('model',), value_offset=1):
    header = list(prefix_header) + [metric + suffix for metric in CHANNEL_METRICS for suffix in ('_mean', '_std')]
    result = []
    for row in rows:
        values = []
        for value in row[value_offset:]:
            parsed = numeric(value)
            if parsed is None:
                raise ValueError(f'Expected reported metric: {value}')
            values.extend(parsed[:2])
        if len(values) != 12:
            raise ValueError(f'Expected six channels/metrics: {row}')
        result.append([*row[:value_offset], *values])
    write_csv(output / filename, header, result)


def verify(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    for entry in manifest['files']:
        path = output / entry['path']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Bundle file differs: {path}')
    tables = json.loads((output / 'paper_results.json').read_text())['tables']
    cells = 0
    for table in tables:
        with (output / 'tables' / (table['id'] + '.csv')).open(encoding='utf-8', newline='') as stream:
            rows = list(csv.reader(stream))
        expected = [[str(value) for value in row] for row in [table['header'], *table['rows']]]
        if rows != expected:
            raise ValueError(f'CSV cells differ: {table["id"]}')
        cells += sum(map(len, expected))
    result = dict(status='passed', tables=len(tables), table_cells=cells,
                  files_checked=len(manifest['files']), validation_scope='manuscript source and exported files')
    print(json.dumps(result, ensure_ascii=False))
    return result


def export(paper, output, confirmed_on=None):
    paper, output = paper.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    inputs = INPUTS + [f'figures/{name}.{ext}' for name in FIGURES for ext in ('pdf', 'png')]
    source_hashes = {name: digest(paper / name) for name in inputs}
    with tempfile.TemporaryDirectory(prefix='Sx-paper-export-') as directory:
        copy = Path(directory)
        for name in INPUTS:
            path = copy / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paper / name, path)
        (copy / 'worklog/experiment_trim_20261109').mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            module = runpy.run_path(str(copy / 'scripts/export_experiment_results.py'))
        tables = module['EXPORTED']
        validation = json.loads((copy / 'worklog/experiment_trim_20261109/data_export_validation.json').read_text())
        if not validation['all_passed']:
            raise ValueError('Manuscript table/chart source checks failed')
        output.mkdir(parents=True, exist_ok=True)
        document = (copy / '实验结果完整数据.md').read_text()
        document = document.replace('(results/supplied_tables.json)', '(source/supplied_tables_active.json)')
        document = document.replace('(results/incremental_comparison.json)', '(source/incremental_comparison.json)')
        document = document.replace('[当前实验章节](sections/Experiments.tex)', '当前实验章节：`paper_new/sections/Experiments.tex`')
        document = document.replace('[数据图绘制脚本](scripts/plot_supplied_results.py)', '数据图绘制脚本：`paper_new/scripts/plot_supplied_results.py`')
        (output / '论文主表数据.md').write_text(document, encoding='utf-8')

    supplied = json.loads((paper / 'results/supplied_tables.json').read_text())
    active = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 20]
    write_json(output / 'source/supplied_tables_active.json', {
        'status': supplied['status'],
        'panels': [dict(source_panel=i, **supplied['tables'][i]) for i in active],
    })
    for source, target in [
        ('results/incremental_comparison.json', 'source/incremental_comparison.json'),
        ('worklog/data_figures_20261106/plot_values.json', 'source/chart_values.json'),
        ('worklog/experiment_style_20261106/plot_values.json', 'source/parameter_chart_values.json'),
    ]:
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paper / source, destination)
    write_json(output / 'paper_results.json', {'source_kind': 'current_manuscript_reported_results', 'tables': tables})
    long_rows = []
    for table in tables:
        write_csv(output / 'tables' / (table['id'] + '.csv'), table['header'], table['rows'])
        for index, row in enumerate(table['rows']):
            for column, value in enumerate(row):
                parsed = numeric(value)
                if parsed is not None:
                    long_rows.append([table['id'], index, column, table['header'][column], *parsed, value])
    write_csv(output / 'reported_values.csv', ['table', 'row', 'column', 'heading', 'mean', 'std', 'unit', 'reported_text'], long_rows)
    by_id = {table['id']: table for table in tables}
    paper_tables = output / 'paper_tables'
    for identifier, filename in [('overall', 'table2_overall.csv'), ('ablation_raw', 'table4_ablation.csv'),
                                 ('retrieval_raw', 'table5a_retrieval.csv')]:
        metric_table(paper_tables, filename, by_id[identifier]['rows'])
    metric_table(output / 'trajectory_comparison', 'table2_trajectory_comparison.csv',
                 by_id['incremental_comparison']['rows'], ('method', 'prediction'), 2)
    metric_table(paper_tables, 'table5b_reference.csv', by_id['reference_own']['rows'], ('scheme', 'n_train'), 2)
    for identifier, filename in [('direction_peak', 'auxiliary_metrics.csv'), ('setup', 'table1_setup.csv'),
                                 ('reference_design', 'table1_reference_splits.csv'),
                                 ('parameter_15_raw', 'parameter_K.csv'), ('parameter_16_raw', 'parameter_L.csv')]:
        table = by_id[identifier]
        write_csv(paper_tables / filename, table['header'], table['rows'])
    for name in ['overall', 'incremental_comparison', 'reference']:
        path = output / 'latex' / (name + '.tex')
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paper / f'tables/{name}.tex', path)
    shutil.copy2(output / 'latex/incremental_comparison.tex', output / 'trajectory_comparison/table2_trajectory_comparison.tex')
    for name in FIGURES:
        for ext in ('pdf', 'png'):
            path = output / f'figures/{name}.{ext}'
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paper / f'figures/{name}.{ext}', path)
    write_json(output / 'validation.json', dict(
        status='passed', source_table_and_chart_checks=validation['checks'],
        reported_tables=len(tables), reported_numeric_cells=len(long_rows),
        validation_scope='Exact source/table/chart consistency; no training rerun',
    ))
    (output / 'README.md').write_text(
        '# 当前论文结果\n\n'
        '数值与图表来源于 `paper_new` 当前版本，由作者确认为最新结果。\n\n'
        '- `论文主表数据.md`：论文采用的全部表格与图形数值。\n'
        '- `tables/`：29 张表格的 CSV，保留原报告精度。\n'
        '- `paper_tables/`：主表指标，其中误差均值和标准差分别保存。\n'
        '- `trajectory_comparison/`：整体轨迹与增量响应对照。\n'
        '- `latex/`、`figures/`：当前论文使用的表格与四张实验数据图。\n'
        '- `source/`、`manifest.json`：有效数据来源、绘图数值与文件校验值。\n\n'
        '这里保存论文报告的汇总值。各随机种子结果、检查点和逐事件预测由实际训练运行生成，'
        '本目录不将旧运行输出绑定到更新后的汇总值。\n', encoding='utf-8')
    if any(digest(paper / name) != value for name, value in source_hashes.items()):
        raise RuntimeError('Manuscript source changed during export')
    entries = [dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size, sha256=digest(path))
               for path in sorted(output.rglob('*')) if path.is_file()]
    write_json(output / 'manifest.json', dict(
        schema_version=1, source_kind='current_manuscript_reported_results',
        author_confirmed_current_on=confirmed_on, created_utc=datetime.now(timezone.utc).isoformat(),
        exporter_sha256=digest(Path(__file__)),
        source_files_sha256=source_hashes, training_executed_for_this_export=False,
        seed_level_predictions_provided=False, files=entries,
    ))
    return verify(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--confirmed-on', help='Date when the author identified this source version as current')
    options = parser.parse_args()
    if options.verify:
        verify(options.output.resolve())
    elif options.paper is None:
        parser.error('--paper is required for export')
    else:
        export(options.paper, options.output, options.confirmed_on)
