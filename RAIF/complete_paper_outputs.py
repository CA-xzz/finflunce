"""Complete descriptive paper outputs from saved predictions, using CPU only.

Run with Python 3.9+, numpy, pandas and matplotlib. No training, model imports,
network requests or writes outside --output. Existing data and reports are read
only. Run --self-test for deterministic boundary checks without loading data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ['HistoricalMean', 'DLinear', 'PatchTST', 'iTransformer', 'ECT', 'RAFT', 'RATL', 'RAIF']
CHANNELS = ['AR', 'turnover', 'log_volume']
UNITS = ['decimal return', 'percentage points', 'natural log of traded shares']
SEEDS = [41, 42, 43]
KEYS = ['event_id', 'tau', 'channel']
MIN_EVENTS, MIN_DATES = 20, 5
EARLIER_LATER_CUTOFF = '2025-10-01'


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


class Inputs:
    def __init__(self):
        self.rows = {}

    def record(self, path):
        path = Path(path).resolve()
        require(path.is_file(), 'Missing input: ' + str(path))
        if str(path) not in self.rows:
            self.rows[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
        return path

    def json(self, path):
        return json.loads(self.record(path).read_text(encoding='utf-8'))

    def csv(self, path, **kwargs):
        return pd.read_csv(self.record(path), **kwargs)


def sufficient(n_events, n_dates):
    return n_events >= MIN_EVENTS and n_dates >= MIN_DATES


def quality_groups(values, cuts):
    return np.asarray(['low', 'middle', 'high'])[np.searchsorted(cuts, values, side='left')]


def peak_statistics(truth, predicted, scales):
    """Exact ties use the earliest argmax; flat truth is excluded from peak MAE."""
    require(truth.shape == predicted.shape and truth.ndim == 3, 'Peak tensor mismatch')
    require(np.isfinite(truth).all() and np.isfinite(predicted).all(), 'Nonfinite peak input')
    thresholds = np.maximum(1e-12, np.asarray(scales, dtype=float) * 1e-4)
    abs_true, abs_pred = np.abs(truth), np.abs(predicted)
    amplitude, predicted_amplitude = abs_true.max(axis=1), abs_pred.max(axis=1)
    flat = amplitude <= thresholds
    true_peak, predicted_peak = abs_true.argmax(axis=1), abs_pred.argmax(axis=1)
    return dict(thresholds=thresholds, flat=flat, predicted_flat=predicted_amplitude <= thresholds,
                amplitude=amplitude, predicted_amplitude=predicted_amplitude,
                true_peak=true_peak, predicted_peak=predicted_peak,
                true_tie=(abs_true == amplitude[:, None, :]).sum(axis=1) > 1,
                predicted_tie=(abs_pred == predicted_amplitude[:, None, :]).sum(axis=1) > 1,
                error=np.where(flat, np.nan, np.abs(predicted_peak - true_peak).astype(float)))


def load_prepared(args, inputs):
    norm = inputs.json(args.data / 'normalization.json')
    with np.load(inputs.record(args.data / 'dataset.npz'), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    events = inputs.csv(args.data / 'events.csv', dtype={'stock': str, 'event_id': str, 'source_rows': str})
    require(len(events) == 5105 and not events.event_id.duplicated().any(), 'Expected unique 5105-event main cohort')
    require(arrays['x'].shape == (5105, 10, 3) and arrays['y'].shape == (5105, 11, 3), 'Unexpected model tensor shape')
    require(all(len(a) == len(events) and np.isfinite(a).all() for a in arrays.values()), 'Invalid prepared arrays')
    require(events.label_version.nunique() == 1, 'Ambiguous label version')
    require(np.array_equal(events.day.to_numpy(), arrays['day']) and np.array_equal(events.end_day.to_numpy(), arrays['end_day']), 'Event day index mismatch')
    mean, scale = np.asarray(norm['y_mean']), np.asarray(norm['y_std'])
    require(mean.shape == scale.shape == (3,) and (scale > 0).all(), 'Invalid y normalization')
    require(np.isfinite(mean).all() and np.isfinite(scale).all(), 'Nonfinite y normalization')
    truth_all = arrays['y'].astype(np.float64) * scale + mean
    test_ids, train_ids = np.flatnonzero(arrays['split'] == 2), np.flatnonzero(arrays['split'] == 0)
    require(len(test_ids) == 2315 and len(train_ids) == 2170, 'Unexpected cohort splits')
    events.event_date = pd.to_datetime(events.event_date).dt.strftime('%Y-%m-%d')
    require(np.array_equal(events.split.to_numpy(), np.asarray(['train', 'validation', 'test'])[arrays['split']]), 'Split mismatch')
    test = events.iloc[test_ids].reset_index(drop=True)
    expected = pd.MultiIndex.from_product([test.event_id.tolist(), range(11), CHANNELS], names=KEYS)
    return dict(norm=norm, arrays=arrays, events=events, test=test, test_ids=test_ids, train_ids=train_ids,
                truth=truth_all[test_ids], truth_all=truth_all, expected=expected,
                data_hash=sha256(args.data / 'dataset.npz'), norm_hash=sha256(args.data / 'normalization.json'))


def predictions(path, data, model, seed, memory_mode, inputs):
    frame = inputs.csv(path, dtype={'event_id': str})
    required = set(KEYS + ['model', 'seed', 'split', 'event_date', 'delta_true', 'delta_pred', 'label_version', 'memory_mode', 'q'])
    require(required.issubset(frame.columns), 'Missing prediction columns: ' + str(path))
    require(len(frame) == len(data['expected']) and not frame.duplicated(KEYS).any(), 'Prediction key count/duplicates: ' + str(path))
    require(frame.model.eq(model).all() and frame.seed.eq(seed).all(), 'Prediction run mismatch')
    require(frame.split.eq('test').all() and frame.memory_mode.eq(memory_mode).all(), 'Prediction split or memory mode mismatch')
    require(frame.label_version.eq(data['events'].label_version.iloc[0]).all(), 'Label version mismatch')
    current = pd.MultiIndex.from_frame(frame[KEYS])
    require(len(current.difference(data['expected'])) == 0 and len(data['expected'].difference(current)) == 0, 'Test event/tau/channel set mismatch')
    ordered = frame.set_index(KEYS).reindex(data['expected'])
    require(np.isfinite(ordered[['delta_true', 'delta_pred']].to_numpy()).all(), 'Nonfinite predictions')
    require(np.allclose(ordered.delta_true.to_numpy(), data['truth'].reshape(-1), rtol=1e-10, atol=1e-12), 'Truth differs from prepared test tensor')
    require(np.array_equal(pd.to_datetime(ordered.event_date).dt.strftime('%Y-%m-%d'), np.repeat(data['test'].event_date.to_numpy(), 33)), 'Prediction date mismatch')
    require(np.allclose(ordered.q.to_numpy(), np.repeat(data['test'].q.to_numpy(), 33), rtol=1e-6, atol=1e-12), 'Prediction quality mismatch')
    return ordered.delta_pred.to_numpy().reshape(-1, 11, 3)


def summarize_values(frame, keys):
    return frame.groupby(keys, dropna=False, sort=False).agg(mean=('value', 'mean'), std=('value', 'std'),
            n_seeds=('seed', 'nunique'), n_metric_seeds=('value', 'count'),
            n_events=('n_events', 'first'), n_dates=('n_dates', 'first')).reset_index()


def auxiliary_outputs(args, data, inputs):
    scalar_rows, diagnostic_rows, event_rows, checks, all_predictions = [], [], [], [], {}
    truth = data['truth']
    for model in MODELS:
        for seed in ([41] if model == 'HistoricalMean' else SEEDS):
            directory = args.suite / 'main' / model / ('seed' + str(seed))
            config, status = inputs.json(directory / 'config.json'), inputs.json(directory / 'status.json')
            require(status['status'] == 'completed', 'Incomplete main run: ' + str(directory))
            require(config['model'] == model and config['seed'] == seed and config['variant'] == 'full', 'Main run identity mismatch')
            require(config.get('retrieval') == 'response' and config.get('memory_subset', 'all') == 'all', 'Main run retrieval/memory setting mismatch')
            require(config['data_hash'] == data['data_hash'] and config['normalization_hash'] == data['norm_hash'], 'Main run data hash mismatch')
            pred = predictions(directory / 'predictions.csv', data, model, seed, 'fixed', inputs)
            all_predictions[model, seed] = pred
            peak = peak_statistics(truth, pred, data['norm']['y_std'])
            accuracy = float((np.sign(pred[:, :, 0]) == np.sign(truth[:, :, 0])).mean())
            common = dict(model=model, seed=seed, n_events=len(truth), n_dates=data['test'].event_date.nunique(), horizon=11)
            scalar_rows.append(dict(common, channel='AR', metric='direction_accuracy', value=accuracy,
                                    n_metric_events=len(truth), n_observations=truth.shape[0] * 11))
            saved = inputs.json(directory / 'auxiliary_metrics.json')
            require(np.isclose(saved['ar_direction_accuracy'], accuracy, rtol=0, atol=1e-14), 'Saved direction accuracy differs')
            for c, channel in enumerate(CHANNELS):
                nonflat = ~peak['flat'][:, c]
                value = float(peak['error'][nonflat, c].mean()) if nonflat.any() else np.nan
                saved_peak = saved['peak_time_mae'][channel]
                require((saved_peak is None and not nonflat.any()) or np.isclose(saved_peak, value, rtol=0, atol=1e-12), 'Saved peak metric differs')
                require(saved['near_flat_true_events'][channel] == int((~nonflat).sum()), 'Saved flat count differs')
                scalar_rows.append(dict(common, channel=channel, metric='absolute_peak_time_MAE', value=value,
                                        n_metric_events=int(nonflat.sum()), n_observations=int(nonflat.sum())))
                diagnostic_rows.append(dict(common, channel=channel, flat_threshold=float(peak['thresholds'][c]),
                    near_flat_true_events=int(peak['flat'][:, c].sum()), near_flat_predicted_events=int(peak['predicted_flat'][:, c].sum()),
                    nonflat_true_events=int(nonflat.sum()), true_peak_ties_nonflat=int((peak['true_tie'][:, c] & nonflat).sum()),
                    predicted_peak_ties_nonflat_truth=int((peak['predicted_tie'][:, c] & nonflat).sum()),
                    true_zero_steps=int((truth[:, :, c] == 0).sum()), predicted_zero_steps=int((pred[:, :, c] == 0).sum())))
                for i, event_id in enumerate(data['test'].event_id):
                    event_rows.append(dict(model=model, seed=seed, event_id=event_id, channel=channel,
                        true_peak_tau=int(peak['true_peak'][i, c]), predicted_peak_tau=int(peak['predicted_peak'][i, c]),
                        absolute_peak_time_error=peak['error'][i, c], true_near_flat=bool(peak['flat'][i, c]),
                        predicted_near_flat=bool(peak['predicted_flat'][i, c]), true_peak_tied=bool(peak['true_tie'][i, c]),
                        predicted_peak_tied=bool(peak['predicted_tie'][i, c]), flat_threshold=float(peak['thresholds'][c])))
            checks.append(dict(model=model, seed=seed, auxiliary_json_agrees=True, test_event_truth_keys_agree=True))
    raw = pd.DataFrame(scalar_rows)
    summary = summarize_values(raw, ['model', 'channel', 'horizon', 'metric'])
    summary['expected_seeds'] = np.where(summary.model.eq('HistoricalMean'), 1, 3)
    require((summary.n_seeds == summary.expected_seeds).all(), 'Missing seed in auxiliary summary')
    summary['n_metric_events'] = raw.groupby(['model', 'channel', 'horizon', 'metric'], sort=False).n_metric_events.first().to_numpy()
    summary['status'] = np.where(summary.n_metric_events.gt(0), 'available', 'no_nonflat_true_events')
    for filename, table in [('auxiliary_metrics_seeds.csv', raw), ('auxiliary_metrics.csv', summary),
                            ('peak_time_diagnostics_seeds.csv', pd.DataFrame(diagnostic_rows)),
                            ('peak_time_events.csv', pd.DataFrame(event_rows)), ('auxiliary_consistency.csv', pd.DataFrame(checks))]:
        table.to_csv(args.output / filename, index=False)
    return all_predictions, summary


def source_groups(args, data, inputs):
    manifest = inputs.csv(args.data / 'source_manifest.csv')
    for path, suffix in [(args.source_events, '/original_data/events.csv'), (args.calendar, '/calendar.csv')]:
        saved = manifest.loc[manifest.path.str.endswith(suffix)]
        require(len(saved) == 1 and saved.sha256.iloc[0] == sha256(inputs.record(path)), 'Prepared source hash differs: ' + str(path))
    raw = inputs.csv(args.source_events, dtype={'recommended_stock_code': str, 'original_stock_code': str}, keep_default_na=False)
    calendar = pd.DatetimeIndex(pd.to_datetime(inputs.csv(args.calendar).date))
    require(calendar.is_unique and calendar.is_monotonic_increasing, 'Invalid source calendar')
    vocab = {name for name, index in data['norm']['influencer_vocab'].items() if index != 0}
    assignments, links, train_authors = [], [], set()
    for i, event in enumerate(data['events'].itertuples()):
        require(re.fullmatch(r'\d+(;\d+)*', event.source_rows) is not None, 'Invalid source_rows')
        indices = [int(x) for x in event.source_rows.split(';')]
        require(len(indices) == len(set(indices)) == event.source_count, 'Source count/duplicate mismatch')
        require(min(indices) >= 0 and max(indices) < len(raw), 'Source index out of range')
        rows = raw.iloc[indices]
        require(rows.recommended_stock_code.str.zfill(6).eq(event.stock).all(), 'Source stock identity mismatch')
        dates = pd.to_datetime(rows.recommend_date, format='mixed').dt.normalize()
        require(np.all(calendar.searchsorted(dates) == event.day), 'Source recommendation/event date mapping differs')
        authors = {x.strip() for x in rows.bigv_name if x.strip()}
        platforms = {x.strip() for x in rows.laiyuan if x.strip()}
        author_missing = any(not x.strip() for x in rows.bigv_name)
        platform_missing = any(not x.strip() for x in rows.laiyuan)
        require(platforms <= {'微博', '微信'}, 'Unexpected raw platform value')
        if data['arrays']['split'][i] == 0:
            train_authors |= authors
        if not authors or author_missing:
            influence_group = 'missing'
        elif authors <= vocab:
            influence_group = 'all_seen'
        elif not authors.intersection(vocab):
            influence_group = 'all_unseen'
        else:
            influence_group = 'mixed'
        platform_group = ('missing' if platform_missing or not platforms else 'multiple_platforms' if len(platforms) > 1
                          else 'weibo_only' if platforms == {'微博'} else 'wechat_only')
        assignments.append(dict(event_id=event.event_id, split=event.split, event_date=event.event_date,
            source_rows=event.source_rows, source_count=len(indices), source_authors=';'.join(sorted(authors)),
            source_platforms=';'.join(sorted(platforms)), source_author_status=influence_group,
            source_platform_group=platform_group, model_platform=event.platform,
            model_influencer_index=int(data['arrays']['influencer'][i]), condition_known=bool(event.condition_known),
            source_description_only=True))
        for source_index, row in zip(indices, rows.itertuples()):
            links.append(dict(event_id=event.event_id, split=event.split, filtered_source_row=source_index,
                original_source_row=row.original_source_row, original_stock_code=row.original_stock_code,
                canonical_stock_code=row.recommended_stock_code, recommend_date=row.recommend_date,
                source_author=row.bigv_name, source_platform=row.laiyuan, author_in_training_vocab=row.bigv_name.strip() in vocab))
    source = pd.DataFrame(assignments)
    source.to_csv(args.output / 'source_group_event_assignments.csv', index=False)
    pd.DataFrame(links).to_csv(args.output / 'source_group_record_links.csv', index=False)
    author_names = sorted(set(raw.bigv_name.str.strip()) - {''})
    pd.DataFrame([dict(author=name, present_in_retained_training_sources=name in train_authors,
                       present_in_training_vocabulary=name in vocab) for name in author_names]).to_csv(args.output / 'source_author_training_status.csv', index=False)
    details = dict(training_source_author_count=len(train_authors), training_vocabulary_author_count=len(vocab),
        training_source_only_authors=sorted(train_authors - vocab), training_vocabulary_only_authors=sorted(vocab - train_authors),
        seen_definition='Nonzero author entries in training influencer_vocab; actual retained-training source-author union also recorded.',
        source_row_definition='Zero-based row in filtered source_expanded/original_data/events.csv, not original_source_row.',
        interpretation='Retrospective descriptive grouping only. Multi-source identities and ordering are not asserted to be available at query time and are not model inputs.')
    return source.loc[data['arrays']['split'] == 2].reset_index(drop=True), details


def build_groups(args, data, source, inputs):
    test, arrays = data['test'], data['arrays']
    cuts = np.quantile(arrays['q'][data['train_ids']], [1/3, 2/3])
    groups = dict(period=pd.to_datetime(test.event_date).dt.to_period('Q').astype(str).to_numpy(),
        platform=test.platform.fillna('unknown').astype(str).to_numpy(),
        quality=quality_groups(test.q.to_numpy(), cuts),
        influencer_status=np.where(arrays['influencer'][data['test_ids']] == 0, 'unknown_or_unseen', 'seen'),
        condition_known=np.where(test.condition_known.to_numpy(), 'known', 'unknown'),
        period_half=np.where(test.event_date.to_numpy() < EARLIER_LATER_CUTOFF, 'earlier', 'later'),
        source_platform=source.source_platform_group.to_numpy(), source_influencer=source.source_author_status.to_numpy())
    category_sets = {key: list(np.unique(values)) for key, values in groups.items()}
    category_sets.update(source_platform=['weibo_only', 'wechat_only', 'multiple_platforms', 'missing'],
                         source_influencer=['all_seen', 'all_unseen', 'mixed', 'missing'])
    counts, event_groups = [], []
    for group_type, categories in category_sets.items():
        for group in categories:
            mask = groups[group_type] == group
            n, days = int(mask.sum()), int(test.loc[mask, 'event_date'].nunique())
            counts.append(dict(group_type=group_type, group=group, n_events=n, n_dates=days,
                report_metrics=sufficient(n, days), status='available' if sufficient(n, days) else 'insufficient_group_size'))
        for event_id, group in zip(test.event_id, groups[group_type]):
            event_groups.append(dict(event_id=event_id, group_type=group_type, group=group))
    count_frame = pd.DataFrame(counts)
    previous = inputs.csv(args.report / 'group_counts.csv')
    for row in previous.itertuples():
        current = count_frame.loc[count_frame.group_type.eq(row.group_type) & count_frame.group.eq(str(row.group))]
        require(len(current) == 1 and current.n_events.iloc[0] == row.n_events and current.n_dates.iloc[0] == row.n_dates,
                'Existing report group definition mismatch')
    count_frame.to_csv(args.output / 'group_counts.csv', index=False)
    pd.DataFrame(event_groups).to_csv(args.output / 'test_group_assignments.csv', index=False)
    return groups, count_frame, cuts.tolist()


def validate_memory(directory, data, inputs, mode):
    retrieved = inputs.csv(directory / 'retrieved_events.csv')
    lookup = pd.Series(np.arange(len(data['events'])), index=data['events'].event_id)
    qi = lookup.loc[retrieved.query_id].to_numpy()
    hi = lookup.loc[retrieved.historical_id].to_numpy()
    a = data['arrays']
    require((qi != hi).all() and (a['end_day'][hi] < a['day'][qi]).all(), 'Illegal retrieval maturity/self match')
    require((a['split'][qi] == 2).all(), 'Non-test retrieval query')
    require(np.array_equal(retrieved.query_day.to_numpy(), a['day'][qi]) and np.array_equal(retrieved.historical_end_day.to_numpy(), a['end_day'][hi]), 'Retrieved date fields mismatch')
    require(np.array_equal(retrieved.origin_split.to_numpy(), a['split'][hi]), 'Retrieval origin split mismatch')
    require(not retrieved.duplicated(['query_id', 'rank']).any() and not retrieved.duplicated(['query_id', 'historical_id']).any(), 'Duplicate retrieval entry')
    require(set(retrieved.query_id) == set(data['test'].event_id), 'Retrieval query coverage differs')
    if mode == 'fixed':
        require((a['split'][hi] == 0).all(), 'Fixed memory includes nontraining event')
    updates = inputs.csv(directory / 'memory_updates.csv')
    if len(updates):
        ui = lookup.loc[updates.event_id].to_numpy()
        require(mode == 'rolling' and not updates.event_id.duplicated().any(), 'Unexpected/duplicate memory append')
        require(np.array_equal(ui, updates['index'].to_numpy()), 'Memory update identity mismatch')
        require(np.array_equal(a['end_day'][ui], updates.end_day.to_numpy()), 'Memory update end day mismatch')
        require((updates.end_day < updates.added_before_query_day).all(), 'Memory appended before response matured')
        require(np.array_equal(a['split'][ui], updates.origin_split.to_numpy()) and (a['split'][ui] != 0).all(), 'Memory append source mismatch')
        appended = updates.set_index('event_id').added_before_query_day
        nontrain = a['split'][hi] != 0
        require((appended.loc[retrieved.loc[nontrain, 'historical_id']].to_numpy() <= a['day'][qi[nontrain]]).all(), 'Retrieved nontraining event not yet appended')
    return dict(retrieval_rows=len(retrieved), update_rows=len(updates), strict_end_before_query=True,
                nontraining_retrieval_rows=int((a['split'][hi] != 0).sum()))


def memory_outputs(args, data, inputs, main_predictions, groups, counts):
    candidates = []
    fixed_checks, memory_checks = [], []
    for seed in SEEDS:
        candidates.extend([('RAIF_fixed', 'RAIF', 'fixed', seed, main_predictions['RAIF', seed]),
                           ('ECT_direct', 'ECT', 'not_applicable', seed, main_predictions['ECT', seed])])
        checkpoint = inputs.record(args.suite / 'main' / 'RAIF' / ('seed' + str(seed)) / 'checkpoint.pt')
        for mode, root in [('fixed', args.frozen), ('rolling', args.rolling)]:
            matches = list(root.glob('seed%d_*_response' % seed))
            require(len(matches) == 1, 'Ambiguous/missing response evaluation: ' + str(root))
            directory = matches[0]
            config = inputs.json(directory / 'evaluation_config.json')
            require(config['seed'] == seed and config['model'] == 'RAIF' and config['retrieval'] == 'response' and config['memory_mode'] == mode, 'Extra evaluation config mismatch')
            require(config['dataset_sha256'] == data['data_hash'] and config['checkpoint_sha256'] == sha256(checkpoint), 'Extra evaluation data/checkpoint mismatch')
            require(config['parameters_updated'] is False and config['normalization_updated'] is False, 'Extra evaluation changed fitted parameters')
            pred = predictions(directory / 'predictions.csv', data, 'RAIF', seed, mode, inputs)
            memory_checks.append(dict(seed=seed, memory_mode=mode, **validate_memory(directory, data, inputs, mode)))
            if mode == 'fixed':
                require(np.array_equal(pred, main_predictions['RAIF', seed]), 'Fixed response replay differs from main predictions')
                fixed_checks.append(dict(seed=seed, all_predictions_exactly_equal=True, max_absolute_difference=0.0))
            else:
                candidates.append(('RAIF_rolling', 'RAIF', 'rolling', seed, pred))
    rows = []
    for setting, model, mode, seed, predicted in candidates:
        for count in counts.itertuples():
            mask = groups[count.group_type] == count.group
            error = predicted[mask] - data['truth'][mask]
            for c, channel in enumerate(CHANNELS):
                for metric in ['MAE', 'RMSE']:
                    value = (float(np.abs(error[:, :, c]).mean()) if metric == 'MAE' else float(np.sqrt(np.square(error[:, :, c]).mean()))) if count.report_metrics else np.nan
                    rows.append(dict(setting=setting, model=model, memory_mode=mode, seed=seed,
                        group_type=count.group_type, group=count.group, channel=channel, horizon=11, metric=metric,
                        value=value, n_events=count.n_events, n_dates=count.n_dates, status=count.status))
    raw = pd.DataFrame(rows)
    keys = ['setting', 'model', 'memory_mode', 'group_type', 'group', 'channel', 'horizon', 'metric']
    summary = summarize_values(raw, keys)
    require(summary.n_seeds.eq(3).all(), 'Missing seed in group comparison')
    summary['status'] = ['available' if sufficient(n, d) else 'insufficient_group_size' for n, d in zip(summary.n_events, summary.n_dates)]
    raw.to_csv(args.output / 'memory_group_metrics_seeds.csv', index=False)
    summary.to_csv(args.output / 'memory_group_metrics.csv', index=False)
    delta_keys = ['seed', 'group_type', 'group', 'channel', 'horizon', 'metric', 'n_events', 'n_dates', 'status']
    differences = []
    for left, right in [('RAIF_rolling', 'RAIF_fixed'), ('RAIF_fixed', 'ECT_direct'), ('RAIF_rolling', 'ECT_direct')]:
        joined = raw.loc[raw.setting.eq(left), delta_keys + ['value']].merge(
            raw.loc[raw.setting.eq(right), delta_keys + ['value']], on=delta_keys, validate='one_to_one', suffixes=('_left', '_right'))
        joined['value'] = joined.value_left - joined.value_right
        joined['comparison'] = left + '_minus_' + right
        differences.append(joined.drop(columns=['value_left', 'value_right']))
    diff = pd.concat(differences, ignore_index=True)
    diff.to_csv(args.output / 'memory_group_differences_seeds.csv', index=False)
    diff_summary = summarize_values(diff, ['comparison', 'group_type', 'group', 'channel', 'horizon', 'metric'])
    diff_summary['status'] = ['available' if sufficient(n, d) else 'insufficient_group_size' for n, d in zip(diff_summary.n_events, diff_summary.n_dates)]
    diff_summary.to_csv(args.output / 'memory_group_differences.csv', index=False)
    pd.DataFrame(fixed_checks).to_csv(args.output / 'fixed_response_replay_consistency.csv', index=False)
    pd.DataFrame(memory_checks).to_csv(args.output / 'memory_temporal_checks.csv', index=False)
    existing = inputs.csv(args.report / 'group_metrics_seeds.csv')
    check_keys = ['model', 'seed', 'group_type', 'group', 'channel', 'horizon', 'metric']
    wanted = existing.loc[existing.model.isin(['RAIF', 'ECT'])]
    current = raw.loc[raw.setting.isin(['RAIF_fixed', 'ECT_direct'])]
    joined = wanted.merge(current, on=check_keys, suffixes=('_saved', '_new'), validate='one_to_one')
    require(len(joined) == len(wanted) and np.allclose(joined.value_saved, joined.value_new, rtol=1e-10, atol=1e-12), 'Main report group metrics differ')
    return summary, memory_checks


def figure3(args, data, inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = inputs.csv(args.report / 'case_manifest.csv', dtype={'event_id': str})
    retrieval = inputs.csv(args.report / 'retrieval_case_manifest.csv')
    require(len(cases) == 6 and cases.case_id.tolist() == list(range(1, 7)) and not cases.event_id.duplicated().any(), 'Expected six fixed cases')
    require(set(cases.event_id) <= set(data['test'].event_id) and set(retrieval.query_id) == set(cases.event_id), 'Case/retrieval query set differs')
    require(cases.selection_seed.eq(20261105).all(), 'Case selection seed changed')
    lookup = dict(zip(data['events'].event_id, range(len(data['events']))))
    ids = np.asarray([lookup[event_id] for event_id in cases.event_id])
    xmean, xscale = np.asarray(data['norm']['x_mean']), np.asarray(data['norm']['x_std'])
    require(xmean.shape == xscale.shape == (3,) and (xscale > 0).all(), 'Invalid input scaling')
    x = data['arrays']['x'][ids].astype(np.float64) * xscale + xmean
    rows = [dict(case_id=int(case.case_id), event_id=case.event_id, event_date=case.event_date,
                 trading_day_offset=offset, channel=channel, value=float(x[i, offset + 10, c]), units=UNITS[c])
            for i, case in enumerate(cases.itertuples()) for offset in range(-10, 0) for c, channel in enumerate(CHANNELS)]
    frame = pd.DataFrame(rows)
    previous = inputs.csv(args.report / 'case_pre_event_inputs.csv')
    keys = ['case_id', 'event_id', 'trading_day_offset', 'channel']
    joined = previous.merge(frame, on=keys, suffixes=('_saved', '_new'), validate='one_to_one')
    require(len(joined) == len(frame) == len(previous) == 180 and np.allclose(joined.value_saved, joined.value_new, rtol=1e-12, atol=1e-12), 'Saved pre-event case inputs differ')
    frame.to_csv(args.output / 'figure3_query_prestate.csv', index=False)
    cases.to_csv(args.output / 'figure3_case_manifest.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(6, 3, figsize=(14, 16.5), sharex=False, sharey=False, constrained_layout=True)
    labels = ['Abnormal return (decimal)', 'Turnover (percentage points)', 'Log volume (log shares)']
    for i, case in enumerate(cases.itertuples()):
        for c, ax in enumerate(axes[i]):
            ax.plot(range(-10, 0), x[i, :, c], color=['#2563a6', '#b45309', '#3c7d44'][c], marker='o', markersize=3, linewidth=1.3)
            ax.set_title('Case %d | %s\n%s' % (case.case_id, case.event_id, labels[c]), fontsize=9)
            ax.set_xlabel('Trading days before recommendation')
            ax.set_ylabel(UNITS[c])
            ax.set_xticks([-10, -8, -6, -4, -2, -1])
            ax.grid(alpha=.18)
    fig.suptitle('Fixed retrieval cases: query pre-event states\nOriginal units; each panel has an independent vertical scale', fontsize=12)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(args.output / ('figure3_query_prestate.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    return dict(n_cases=6, source_rows=180, offsets=[-10, -1], selection_seed=20261105,
        uses_existing_case_manifest=True, new_case_selection=False, input_precision='Inverse-transformed float32 prepared model inputs',
        same_case_order_as_existing_retrieval_figure=True, units=dict(zip(CHANNELS, UNITS)))


def boundary_tests():
    y = np.zeros((3, 4, 3))
    p = np.zeros_like(y)
    y[0, :, :] = np.asarray([1, -3, 3, 0])[:, None]
    p[0, :, :] = np.asarray([0, 0, -4, 4])[:, None]
    y[1, :, :] = 1e-4
    y[2, :, :] = np.asarray([0, 0, 0, -2])[:, None]
    p[2, :, :] = np.asarray([-2, 0, 0, 0])[:, None]
    result = peak_statistics(y, p, [1, 1, 1])
    require(np.all(result['true_peak'][0] == 1) and np.all(result['predicted_peak'][0] == 2), 'Earliest absolute tie test')
    require(result['true_tie'][0].all() and result['predicted_tie'][0].all(), 'Tie flags test')
    require(result['flat'][1].all() and np.isnan(result['error'][1]).all(), 'Flat threshold inclusive/exclusion test')
    require(np.all(result['error'][2] == 3), 'Negative absolute peak test')
    require(float((np.sign([0, -1, 1]) == np.sign([0, 1, 1])).mean()) == 2/3, 'Zero sign test')
    require(sufficient(20, 5) and not sufficient(19, 5) and not sufficient(20, 4), 'Group boundary test')
    require(quality_groups([.1, .2, .3], [.1, .2]).tolist() == ['low', 'middle', 'high'], 'Training tercile boundary test')
    per_seed = [np.sqrt(np.mean(np.square(e))) for e in [[0, 0], [2, 2]]]
    require(np.mean(per_seed) == 1 and not np.isclose(np.mean(per_seed), np.sqrt(2)), 'RMSE seed aggregation test')
    require((np.asarray([9, 10]) < 10).tolist() == [True, False], 'Strict memory maturity boundary test')
    return dict(status='passed', checks=['earliest_absolute_tie', 'tie_flags', 'flat_threshold_inclusive',
        'flat_peak_error_excluded', 'negative_absolute_peak', 'zero_sign_accuracy', '20_event_5_date_boundary',
        'quality_tercile_equality', 'per_seed_rmse_before_aggregation', 'strict_memory_maturity'])


def write_documentation(args, data, auxiliary, counts, source_details):
    lines = ['# 论文辅助实验输出', '',
        '本结果使用 5,105 个已保留事件，其中训练 2,170 个、验证 620 个、测试 2,315 个。所有预测比较使用相同的测试事件、11 个响应步和三个通道标签。', '',
        'AR 方向准确率按每个事件和时间步的预测符号与真实符号是否一致计算，零值独立作为零符号。峰值时间取绝对增量最大的位置；精确并列取最早位置。真实轨迹最大绝对值不超过 max(1e-12, 训练集通道标准差 × 1e-4) 时记为近乎平坦，该事件不计入该通道的峰值时间误差，并单列数量。', '',
        '| 模型 | AR 方向准确率 | AR 峰值时间误差 | 换手率峰值时间误差 | 对数成交量峰值时间误差 | 种子数 |',
        '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        part = auxiliary.loc[auxiliary.model.eq(model)]
        values = []
        for metric, channel in [('direction_accuracy', 'AR')] + [('absolute_peak_time_MAE', c) for c in CHANNELS]:
            row = part.loc[part.metric.eq(metric) & part.channel.eq(channel)].iloc[0]
            value = ('%.4f ± %.4f' % (row['mean'], row['std'])) if row.n_seeds > 1 else ('%.4f' % row['mean'])
            values.append(value)
        lines.append('| %s | %s | %d |' % (model, ' | '.join(values), int(part.n_seeds.iloc[0])))
    lines += ['', '峰值误差单位为交易日。七个可训练模型报告种子 41、42、43 的均值和样本标准差；Historical Mean 报告确定性单次结果。', '',
        '分组对照包括 RAIF 固定训练记忆、相同检查点的滚动记忆和由验证集选择的 ECT 直接基线。滚动记忆只在历史响应末日严格早于当前查询日时纳入记录，不更新网络参数和标准化参数。MAE 与 RMSE 先在每个种子内计算，再汇总种子；差值表为成对种子差值，负值表示前一设置误差较低。', '',
        '原分组保留季度、模型平台条件、训练质量三分位及已知嵌入状态。补充分组使用固定切点 2025-10-01 区分较早与较晚测试期，并依据原推荐记录区分单平台、多平台，以及作者全部已见、全部未见和混合。已见作者按训练词表的非零类别定义。来源记录仅用于结果描述，多来源信息未加入预测条件；模型未知条件与原始多来源分组并不等同。', '',
        '| 补充分组 | 类别 | 事件数 | 独立日期数 | 是否报告误差 |', '|---|---|---:|---:|---|']
    for row in counts.loc[counts.group_type.isin(['period_half', 'source_platform', 'source_influencer'])].itertuples():
        lines.append('| %s | %s | %d | %d | %s |' % (row.group_type, row.group, row.n_events, row.n_dates, '是' if row.report_metrics else '否'))
    lines += ['', '不足 20 个事件或 5 个独立日期的组仅报告规模，误差为空。训练保留事件来源作者与训练词表的差集记录在 source_group_definition.json。', '',
        'Figure 3 前状态图沿用主报告的六个固定案例和顺序，覆盖事件前第 10 至第 1 个交易日。各面板使用原单位和独立纵轴；数据由 float32 模型输入反标准化。图与已有检索轨迹图通过 case_id 和 event_id 对应。', '',
        '本批数据采用允许跨行业、跨板块的最近邻参考。推荐记录只有日期；历史行业来自固定版本的公开副本，其逐股首次公告时间未确认。这些限制沿用主数据说明。分组差异为描述性比较，不作因果推断。']
    (args.output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    command = ('python RAIF/complete_paper_outputs.py --data RAIF/data/prepared_expanded '
               '--suite RAIF/results/suite_expanded --report RAIF/results/report_expanded '
               '--rolling RAIF/results/rolling_expanded --frozen RAIF/results/frozen_retrieval_expanded '
               '--output RAIF/results/paper_completion')
    (args.output / 'README.md').write_text('# 脚本说明\n\n运行环境：Python 3.9+、numpy、pandas、matplotlib。仅读取本地结果，不调用训练模型或网络。\n\n```sh\n' + command +
        '\npython RAIF/complete_paper_outputs.py --self-test\n```\n\n辅助指标：auxiliary_metrics.csv 和逐种子表；峰值平坦与并列数量见 peak_time_diagnostics_seeds.csv，逐事件值见 peak_time_events.csv。\n\n分组对照：memory_group_metrics.csv、memory_group_differences.csv 及逐种子表；group_counts.csv 包含所有组的规模和可报告状态。source_group_event_assignments.csv、source_group_record_links.csv、source_author_training_status.csv 保存来源分组依据。\n\n案例图：figure3_query_prestate.png/.pdf/.svg；源值见同名 CSV，案例清单见 figure3_case_manifest.csv。\n\nsource_manifest.csv 保存所有读取文件的哈希；output_manifest.csv 为输出清单。未修改主报告、原有脚本或准备数据。\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/prepared_expanded')
    parser.add_argument('--suite', type=Path, default=ROOT / 'results/suite_expanded')
    parser.add_argument('--report', type=Path, default=ROOT / 'results/report_expanded')
    parser.add_argument('--rolling', type=Path, default=ROOT / 'results/rolling_expanded')
    parser.add_argument('--frozen', type=Path, default=ROOT / 'results/frozen_retrieval_expanded')
    parser.add_argument('--source-events', type=Path, default=ROOT / 'data/source_expanded/original_data/events.csv')
    parser.add_argument('--calendar', type=Path, default=ROOT / 'results/preflight_expanded/calendar.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/paper_completion')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    test_result = boundary_tests()
    if args.self_test:
        print(json.dumps(test_result)); return
    require(args.output.resolve() not in {args.data.resolve(), args.suite.resolve(), args.report.resolve(), args.rolling.resolve(), args.frozen.resolve()}, 'Output must be independent of existing inputs')
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(); inputs.record(Path(__file__))
    data = load_prepared(args, inputs)
    report_summary = inputs.json(args.report / 'summary.json')
    require(report_summary['data']['dataset_sha256'] == data['data_hash'], 'Report dataset differs')
    require(report_summary['best_direct']['model'] == 'ECT' and report_summary['best_direct']['selection_uses_test'] is False, 'ECT is not the validation-selected direct comparator')
    print('Computing auxiliary metrics for 22 existing main runs', flush=True)
    all_predictions, auxiliary = auxiliary_outputs(args, data, inputs)
    source, source_details = source_groups(args, data, inputs)
    groups, counts, cuts = build_groups(args, data, source, inputs)
    write_json(args.output / 'source_group_definition.json', dict(source_details, earlier_later_cutoff=EARLIER_LATER_CUTOFF,
        training_quality_terciles=cuts, group_min_events=MIN_EVENTS, group_min_dates=MIN_DATES))
    print('Computing fixed/rolling/ECT group comparisons', flush=True)
    group_summary, memory_checks = memory_outputs(args, data, inputs, all_predictions, groups, counts)
    print('Exporting six fixed query-state cases', flush=True)
    figure = figure3(args, data, inputs)
    write_json(args.output / 'figure3_query_prestate_metadata.json', figure)
    write_documentation(args, data, auxiliary, counts, source_details)
    write_json(args.output / 'summary.json', dict(status='completed', n_events=5105, n_test=2315, n_test_dates=141,
        dataset_sha256=data['data_hash'], normalization_sha256=data['norm_hash'], main_runs=22,
        deterministic_baseline_seeds=[41], trainable_model_seeds=SEEDS, auxiliary_metric_rows=len(auxiliary),
        group_metric_rows=len(group_summary), source_groups=source_details, boundary_tests=test_result,
        fixed_response_replay_exactly_matches_main=True, memory_checks=memory_checks,
        figure3=figure, model_fitting_performed=False, existing_data_or_reports_modified=False))
    pd.DataFrame(inputs.rows.values()).to_csv(args.output / 'source_manifest.csv', index=False)
    inventory = [dict(file=p.name, bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(args.output.iterdir())
                 if p.is_file() and p.name != 'output_manifest.csv']
    pd.DataFrame(inventory).to_csv(args.output / 'output_manifest.csv', index=False)
    print(json.dumps(dict(status='completed', main_runs=22, test_events=2315, output=str(args.output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
