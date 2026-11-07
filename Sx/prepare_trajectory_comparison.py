"""Build paired full-trajectory targets without changing the primary cohort."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare import MarketPanel, CHANNELS, EPS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def fit_scaling(values, split):
    values, split = np.asarray(values, np.float64), np.asarray(split)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('Expected finite [N,T,3] trajectories')
    if len(values) != len(split) or not np.any(split == 0):
        raise ValueError('Missing/alignment-invalid training partition')
    training = values[split == 0]
    return training.mean(axis=(0, 1)), np.maximum(training.std(axis=(0, 1)), EPS)


def align_pairs(pairs, events):
    selected = pairs.loc[pairs.event_id.isin(events.event_id)].copy()
    if selected.event_id.duplicated().any() or set(selected.event_id) != set(events.event_id):
        raise ValueError('Each retained event must have exactly one fixed reference')
    selected = selected.set_index('event_id').loc[events.event_id].reset_index()
    if not np.array_equal(selected.weight.to_numpy(), np.ones(len(events))):
        raise ValueError('Reference weights must be exactly one')
    if (selected.control.to_numpy() == events.stock.to_numpy()).any():
        raise ValueError('Self references are forbidden')
    for column in ['included_after_outcome_check', 'retained_for_model']:
        if not selected[column].eq(True).all():
            raise ValueError('Reference is not part of the retained complete cohort')
    if not np.array_equal(pd.to_datetime(selected.selected_at), pd.to_datetime(events.event_date)):
        raise ValueError('Reference selection dates differ from the event dates')
    return selected


def align_labels(labels, events, source_version):
    keys = ['event_id', 'tau', 'channel']
    if labels.duplicated(keys).any() or set(labels.event_id) != set(events.event_id):
        raise ValueError('Unexpected/duplicate raw label events')
    grid = pd.MultiIndex.from_product([events.event_id, range(11), CHANNELS], names=keys)
    labels = labels.set_index(keys).reindex(grid)
    if len(labels) != len(events) * 33 or labels.isna().any().any():
        raise ValueError('Incomplete raw label grid')
    if set(labels.label_version) != {source_version}:
        raise ValueError('Raw labels and primary dataset versions differ')
    expected_available = np.repeat((pd.to_datetime(events.end_date) + pd.Timedelta(days=1)).to_numpy(), 33)
    if not np.array_equal(pd.to_datetime(labels.target_available_at).to_numpy(), expected_available):
        raise ValueError('Raw labels have inconsistent availability times')
    if not np.array_equal(pd.to_datetime(labels.event_date).to_numpy(), np.repeat(pd.to_datetime(events.event_date).to_numpy(), 33)):
        raise ValueError('Raw labels have inconsistent event dates')
    values = labels[['y', 'baseline', 'delta']].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite raw trajectories')
    np.testing.assert_allclose(values[:, 0] - values[:, 1], values[:, 2], rtol=1e-12, atol=1e-12)
    return tuple(values[:, i].reshape(len(events), 11, 3) for i in range(3))


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    primary = root / 'data/prepared_expanded'
    source = root / 'data/source_expanded'
    preflight = root / 'results/preflight_expanded'
    external = root / 'data/external'
    source_paths = [primary / f for f in ['dataset.npz', 'normalization.json', 'events.csv', 'labels.csv', 'reference_pairs.csv']]
    source_paths += [source / 'control_stock_finder/data/A_share_daily.parquet', source / 'original_data/events.csv',
                     preflight / 'calendar.csv', preflight / 'indices.parquet',
                     external / 'historical_market_cap.parquet', external / 'industry_history.parquet',
                     Path(__file__).with_name('prepare.py')]
    manifests = [{'path': str(p.relative_to(root)), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in source_paths]
    original_primary_hashes = {row['path']: row['sha256'] for row in manifests[:5]}
    normal = json.loads((primary / 'normalization.json').read_text())
    if not normal.get('ready_for_formal_training') or normal['q_rule'] != 'exp_only':
        raise ValueError('The current primary dataset must be formally ready and exp_only')
    with np.load(primary / 'dataset.npz', allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    events = pd.read_csv(primary / 'events.csv', dtype={'event_id': str, 'stock': str})
    assert len(events) == 5105 and not events.event_id.duplicated().any()
    assert arrays['x'].shape == (5105, 10, 3) and arrays['y'].shape == (5105, 11, 3)
    assert np.bincount(arrays['split']).tolist() == [2170, 620, 2315]
    np.testing.assert_array_equal(events.day.to_numpy(), arrays['day'])
    np.testing.assert_array_equal(events.end_day.to_numpy(), arrays['end_day'])
    np.testing.assert_array_equal(events.split.to_numpy(), np.array(['train', 'validation', 'test'])[arrays['split']])
    pairs = align_pairs(pd.read_csv(primary / 'reference_pairs.csv', dtype={'event_id': str, 'control': str}), events)
    labels = pd.read_csv(primary / 'labels.csv', dtype={'event_id': str})
    raw_y, raw_b, raw_delta = align_labels(labels, events, normal['label_version'])
    ymean, ystd = np.asarray(normal['y_mean']), np.asarray(normal['y_std'])
    canonical_delta = arrays['y'].astype(np.float64) * ystd + ymean
    tolerance = 8 * np.finfo(np.float32).eps * (np.abs(raw_delta) + ystd + np.abs(ymean))
    assert np.all(np.abs(raw_delta - canonical_delta) <= tolerance)
    query_x = arrays['x'].astype(np.float64) * np.asarray(normal['x_std']) + np.asarray(normal['x_mean'])
    reference_x = np.empty_like(query_x)
    panel = MarketPanel(source, preflight, external, length=10, expanded=True)
    print(json.dumps({'stage': 'panel_loaded', 'stocks': len(panel.codes)}), flush=True)
    rows = []
    max_query_error = max_raw_y_error = max_raw_b_error = 0.0
    for di, day in enumerate(np.unique(arrays['day'])):
        ids = np.flatnonzero(arrays['day'] == day)
        state = panel.state(int(day))
        qp = np.array([panel.positions[code] for code in events.stock.iloc[ids]])
        bp = np.array([panel.positions[code] for code in pairs.control.iloc[ids]])
        assert state is not None and state.valid[qp].all() and state.valid[bp].all()
        assert not any(code in panel.recommended.get(int(day), set()) for code in pairs.control.iloc[ids])
        assert int(day) - 1 < int(day) and int(day) - 10 >= 0
        assert np.array_equal(pd.to_datetime(events.event_date.iloc[ids]).to_numpy(), np.repeat(panel.calendar[int(day)].to_datetime64(), len(ids)))
        reference_x[ids] = state.x[bp]
        query_tolerance = 8 * np.finfo(np.float32).eps * (np.abs(state.x[qp]) + np.asarray(normal['x_std']) + np.abs(normal['x_mean']))
        assert np.all(np.abs(query_x[ids] - state.x[qp]) <= query_tolerance)
        max_query_error = max(max_query_error, float(np.abs(query_x[ids] - state.x[qp]).max()))
        # Future outcomes are accessed only for source-target consistency, after
        # identities and both event-preceding input tensors are already fixed.
        observed_y = panel.outcome(int(day), qp, state)
        observed_b = panel.outcome(int(day), bp, state)
        np.testing.assert_allclose(raw_y[ids], observed_y, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(raw_b[ids], observed_b, rtol=1e-12, atol=1e-12)
        max_raw_y_error = max(max_raw_y_error, float(np.abs(raw_y[ids] - observed_y).max()))
        max_raw_b_error = max(max_raw_b_error, float(np.abs(raw_b[ids] - observed_b).max()))
        for pos in ids:
            rows.append(dict(event_index=int(pos), event_id=events.event_id.iloc[pos], recommended_stock=events.stock.iloc[pos],
                             control=pairs.control.iloc[pos], split=events.split.iloc[pos], event_date=events.event_date.iloc[pos],
                             input_start=str(panel.calendar[int(day)-10].date()), input_end=str(panel.calendar[int(day)-1].date()),
                             market_fit_start=str(panel.calendar[max(0,int(day)-100)].date()),
                             market_fit_end=str(panel.calendar[int(day)-10].date()),
                             reference_market_fit_observations=int(state.estimate_count[bp[np.flatnonzero(ids==pos)[0]]])))
        if (di + 1) % 100 == 0:
            print(json.dumps({'stage': 'paired_features', 'dates_completed': di+1}), flush=True)
    assert np.isfinite(reference_x).all()
    alignment = pd.DataFrame(rows).sort_values('event_index')
    alignment.to_csv(output / 'event_alignment.csv', index=False)
    np.savez_compressed(output / 'paired_arrays.npz', query_x=query_x, reference_x=reference_x,
                        raw_y=raw_y, raw_b=raw_b, canonical_delta=canonical_delta,
                        event_id=events.event_id.to_numpy(str), control=pairs.control.to_numpy(str),
                        day=arrays['day'], end_day=arrays['end_day'], split=arrays['split'])
    recipe = dict(channels=list(CHANNELS), n_events=5105, split_counts={'train':2170,'validation':620,'test':2315},
                  length=10, horizon=11, references_per_event=1, market_model_offsets=[-100,-10], minimum_market_fit_pairs=30,
                  input_offsets=[-10,-1], source_dataset_sha256=original_primary_hashes[str((primary/'dataset.npz').relative_to(root))],
                  source_label_version=normal['label_version'], reference_identities_reselected=False,
                  source_references_sha256=digest(primary/'reference_pairs.csv'), original_query_x_preserved=True,
                  reference_input_has_no_recommendation_condition=True, scaling_fit_partition='train',
                  recommended_target_mean='training raw Y mean', recommended_target_scale='primary training Delta standard deviation',
                  reference_target_scaling='training raw B mean and standard deviation',
                  canonical_delta_source='Primary float32 standardized y inverse-transformed using primary y_mean/y_std',
                  raw_trajectories_source='Primary labels.csv y and baseline; matched to MarketPanel.outcome',
                  query_x_precision='Primary float32 standardized x inverse-transformed using primary x_mean/x_std',
                  reference_x_precision='Float64 MarketPanel.state pre-event inputs',
                  future_reference_values_are_targets_only=True, same_day_recommended_stocks_excluded=True,
                  raw_delta_vs_canonical_max_absolute_difference=float(np.abs(raw_delta-canonical_delta).max()))
    outputs = {}
    for name, target, x in [('recommended_full',raw_y,query_x), ('reference_full',raw_b,reference_x)]:
        directory = output / name
        directory.mkdir()
        version = name + '_' + hashlib.sha256(json.dumps(dict(recipe,role=name),sort_keys=True).encode()).hexdigest()[:12]
        ym, empirical_ys = fit_scaling(target, arrays['split'])
        ys = np.asarray(normal['y_std']) if name == 'recommended_full' else empirical_ys
        if name == 'recommended_full':
            xm, xs, input_array = np.asarray(normal['x_mean']), np.asarray(normal['x_std']), arrays['x'].copy()
        else:
            xm, xs = fit_scaling(x, arrays['split'])
            input_array = ((x-xm)/xs).astype(np.float32)
        built = {key:value.copy() for key,value in arrays.items()}
        built['x'], built['y'] = input_array, ((target-ym)/ys).astype(np.float32)
        if name == 'reference_full':
            built['platform'][:] = 0
            built['influencer'][:] = 0
        np.savez_compressed(directory/'dataset.npz', **built)
        current_events = events.copy()
        current_events['source_label_version'] = current_events.label_version
        current_events['label_version'] = version
        current_events['trajectory_stock'] = events.stock if name == 'recommended_full' else pairs.control
        current_events['trajectory_role'] = name
        if name == 'reference_full':
            current_events['platform'], current_events['influencer'], current_events['condition_known'] = '__UNKNOWN__', '__UNKNOWN__', False
        current_events.to_csv(directory/'events.csv',index=False)
        current_norm = dict(normal, x_mean=xm.tolist(), x_std=xs.tolist(), y_mean=ym.tolist(), y_std=ys.tolist(),
                            raw_target_empirical_std=empirical_ys.tolist(),
                            target_scale_source='primary training Delta standard deviation' if name=='recommended_full' else 'training raw B standard deviation',
                            label_version=version, target_type='raw_recommended_trajectory' if name=='recommended_full' else 'raw_reference_trajectory',
                            target_channels=list(CHANNELS), target_is_incremental=False, trajectory_comparison=recipe,
                            source_primary_normalization_sha256=digest(primary/'normalization.json'),
                            normalization_fit_event_ids=events.loc[arrays['split']==0,'event_id'].tolist(),
                            input_scaling='inherited primary training-only scaling' if name=='recommended_full' else 'reference training-only scaling')
        write_json(directory/'normalization.json',current_norm)
        write_json(directory/'config.json',dict(recipe,trajectory_role=name,label_version=version))
        outputs[name] = dict(dataset_sha256=digest(directory/'dataset.npz'),normalization_sha256=digest(directory/'normalization.json'),
                             events_sha256=digest(directory/'events.csv'),label_version=version)
        for field in ['day','end_day','split','q']:
            np.testing.assert_array_equal(built[field],arrays[field])
        assert set(built) == set(arrays)
        np.testing.assert_allclose(built['y'].astype(np.float64)*ys+ym,target,rtol=2e-6,atol=2e-6)
    for path, old_hash in original_primary_hashes.items():
        assert digest(root/path)==old_hash, 'The primary dataset was modified'
    write_json(output/'protocol.json',dict(recipe,datasets=outputs,script_sha256=digest(__file__)))
    pd.DataFrame(manifests).to_csv(output/'source_manifest.csv',index=False)
    verification = dict(status='passed', n_events=5105, split_counts=recipe['split_counts'],
                        all_reference_pre_states_complete=True, reference_inputs_strictly_before_event=True,
                        no_self_or_same_day_recommended_references=True, train_only_scaling=True,
                        recommended_standardized_x_byte_identical=True, primary_files_unchanged=True,
                        maximum_query_prestate_float32_roundtrip_error=max_query_error,
                        maximum_raw_y_source_difference=max_raw_y_error, maximum_raw_b_source_difference=max_raw_b_error,
                        maximum_raw_delta_canonical_difference=recipe['raw_delta_vs_canonical_max_absolute_difference'],
                        shared_canonical_delta_for_all_methods=True, paired_arrays_sha256=digest(output/'paired_arrays.npz'),
                        event_alignment_sha256=digest(output/'event_alignment.csv'), datasets=outputs)
    write_json(output/'verification.json',verification)
    print(json.dumps(verification),flush=True)
    return verification


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();build(a.root,a.output or a.root/'data/trajectory_comparison')
