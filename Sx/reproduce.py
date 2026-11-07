"""Run the retained experiments sequentially on one GPU, using bundled data."""
from pathlib import Path
import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys

STAGES = ('suite', 'parameters', 'references', 'raw-history', 'trajectory', 'extras', 'efficiency', 'reports')
MODELS = ('DLinear', 'PatchTST', 'iTransformer', 'ECT', 'RAFT', 'RATL', 'RAIF')
SEEDS = (41, 42, 43)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    parser.add_argument('--gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / 'data/prepared_expanded'
    results = root / 'results'
    suite = results / 'suite_expanded'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')

    def run(script, *flags):
        command = [sys.executable, str(root / script), *map(str, flags)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, env=env, check=True)

    if not args.dry_run:
        (root / 'logs').mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        # Fixed preprocessing inputs, used by table and figure exporters.
        if not (results / 'preflight_expanded').exists():
            shutil.copytree(root / 'published_results/preflight_expanded', results / 'preflight_expanded')
    for stage in STAGES if args.stage == 'all' else (args.stage,):
        if stage == 'suite':
            run('run_suite.py', '--data', data, '--output', suite, '--gpus', args.gpu, '--phase', 'all')
        elif stage == 'parameters':
            run('run_parameters.py', '--root', root, '--suite', suite, '--data', data,
                '--windows', root / 'data/window_comparisons_expanded',
                '--quality', root / 'data/quality_comparisons_expanded',
                '--output', results / 'parameters_expanded', '--gpus', args.gpu)
        elif stage == 'references':
            source = root / 'data/reference_comparisons_expanded/comparison_plan.csv'
            plan = results / 'reference_plan.csv'
            if not args.dry_run:
                with source.open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields, rows = reader.fieldnames, list(reader)
                for row in rows:
                    row['dataset'] = str(source.parent / row['group'] / row['scheme'])
                with plan.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fields)
                    writer.writeheader(); writer.writerows(rows)
            run('run_references.py', '--root', root, '--suite', suite, '--plan', plan,
                '--output', results / 'references_expanded', '--gpus', args.gpu)
        elif stage == 'raw-history':
            for seed in SEEDS:
                run('train_raft_raw_history.py', '--data', data, '--seed', seed,
                    '--main-run', suite / f'main/RAFT/seed{seed}',
                    '--output', suite / f'raft_raw_history/seed{seed}', '--threads', 2)
        elif stage == 'trajectory':
            run('replay_main_predictions.py', '--root', root)
            for role in ('reference', 'ect_direct', 'raft_direct'):
                for seed in SEEDS:
                    run('train_trajectory_component.py', '--root', root, '--role', role, '--seed', seed)
            run('export_trajectory_comparison.py', '--root', root)
        elif stage == 'extras':
            raif = [suite / f'main/RAIF/seed{seed}' for seed in SEEDS]
            all_runs = [suite / 'main/HistoricalMean/seed41'] + [suite / f'main/{model}/seed{seed}' for model in MODELS for seed in SEEDS]
            for mode, destination, runs in [('rolling', 'rolling_expanded', raif),
                ('frozen', 'frozen_retrieval_expanded', raif), ('timing', 'serial_timing_expanded', all_runs)]:
                run('evaluate_extras.py', '--data', data, '--runs', *runs, '--mode', mode,
                    '--output', results / destination, '--threads', 2, '--batch-size', 64, '--warmup', 10, '--repeats', 50)
        elif stage == 'efficiency':
            run('benchmark_serial_training.py', '--data', data, '--suite', suite,
                '--output', results / 'serial_training_expanded', '--idle-gpus')
            run('evaluate_efficiency_outputs.py', '--data', data, '--source', results / 'serial_training_expanded',
                '--output', results / 'efficiency_aligned_expanded', '--idle-gpus')
        elif stage == 'reports':
            report = results / 'report_expanded'
            run('summarize_results.py', '--data', data, '--suite', suite, '--output', report)
            run('summarize_supplementary.py', '--root', root, '--output', report)
            run('summarize_results.py', '--data', data, '--suite', suite, '--output', report,
                '--supplement-summary', report / 'supplementary_summary.json')
            run('export_case_inputs.py', '--data', data, '--report', report)
            run('complete_paper_outputs.py', '--data', data, '--suite', suite, '--report', report,
                '--rolling', results / 'rolling_expanded', '--frozen', results / 'frozen_retrieval_expanded',
                '--source-events', root / 'data/source_expanded/original_data/events.csv',
                '--calendar', results / 'preflight_expanded/calendar.csv', '--output', results / 'paper_completion')
            run('export_paper_tables.py', '--root', root, '--output', results / 'paper_tables')


if __name__ == '__main__':
    main()
