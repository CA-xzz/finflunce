"""Stage M=1 datasets with the revised q formula, preserving all fixed labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


FORMULA = "max(exp(-balance/s_b - mean_distance/s_d), 1e-8)"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def migrate_one(source, destination, display_name, expected_count):
    """Copy then change q/provenance only; no market or outcome reconstruction."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    source_hashes = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    normal, config = read(source/'normalization.json'), read(source/'config.json')
    arrays = dict(np.load(source/'dataset.npz', allow_pickle=False))
    events = pd.read_csv(source/'events.csv', dtype={'event_id': str, 'stock': str})
    quality = pd.read_csv(source/'reference_quality.csv', dtype={'event_id': str}).set_index('event_id').loc[events.event_id].reset_index()
    assert len(events) == expected_count and config['m'] == 1
    assert quality.effective_n.eq(1).all() and quality.n_references.eq(1).all()
    assert events.event_id.is_unique and quality.event_id.equals(events.event_id)
    training = arrays['split'] == 0
    sb, sd = normal['quality_balance_scale'], normal['quality_distance_scale']
    np.testing.assert_allclose(sb, max(float(quality.loc[training, 'balance'].median()), 1e-8), rtol=1e-12)
    np.testing.assert_allclose(sd, max(float(quality.loc[training, 'mean_distance'].median()), 1e-8), rtol=1e-12)
    assert normal['normalization_fit_event_ids'] == events.loc[training, 'event_id'].tolist()
    q = np.maximum(np.exp(-quality.balance.to_numpy()/sb - quality.mean_distance.to_numpy()/sd), 1e-8).astype(np.float32)
    assert np.isfinite(q).all() and (q > 0).all() and (q <= 1).all()
    old_q, old_version = arrays['q'].copy(), normal['label_version']
    policy = {'quality_effective_n_factor': False, 'quality_neff_factor_enabled': False,
              'q_rule': 'exp_only', 'quality_formula': FORMULA,
              'quality_scale_fit': 'Unchanged medians of retained training events only',
              'reference_count': 1, 'reference_display_name': display_name}
    provenance = {'source_dataset_sha256': source_hashes['dataset.npz'],
                  'source_normalization_sha256': source_hashes['normalization.json'],
                  'source_config_sha256': source_hashes['config.json'],
                  'source_label_version': old_version,
                  'purpose': 'Remove the effective-N multiplier from q while preserving every event, fixed control, input, target, and training normalization scale',
                  'source_reference_mode': config['reference_mode'],
                  'model_training_required': True}
    config.update(policy, quality_revision=provenance)
    frozen_distance_source = None
    if config['reference_mode'] == 'strict':
        assert config['strict_criteria']['ar_correlation_minimum'] == .7
        assert config['industry_policy'] == 'historical_effective_interval'
        config['protocol_adoption'] = 'User adopted correlation >=0.7 after coverage diagnostics; fixed before this experiment revision; no preregistration claim'
        frozen = config.get('frozen_distance_fit')
        if frozen:
            frozen_distance_source = Path(frozen['configuration'])
            if not frozen_distance_source.exists():
                roots = {'RAIF', 'Sx'}  # Older provenance records retain the original directory name.
                ancestors = [p for p in source.resolve().parents if p.name in roots]
                parts = frozen_distance_source.parts
                marker = next((i for i, part in enumerate(parts) if part in roots), None)
                if not ancestors or marker is None:
                    raise FileNotFoundError(frozen_distance_source)
                frozen_distance_source = ancestors[0].joinpath(*parts[marker+1:])
            assert digest(frozen_distance_source) == frozen['sha256']
            config['frozen_distance_fit'] = dict(frozen,
                configuration='provenance/distance_fit_config.json', path_basis='prepared_dataset_directory')
    config.pop('label_version', None)
    version = config['reference_mode'] + '_' + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    config['label_version'] = version
    normal.update(policy, quality_revision=provenance, label_version=version)
    if config['reference_mode'] == 'strict':
        normal['readiness_meaning'] = 'Source structure, complete numerical labels and temporal partition checks passed under the user-adopted retrospective historical-effective industry policy. Original industry publication timing remains unverified.'
        normal['protocol_adoption'] = config['protocol_adoption']
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('verification.json'))
    if frozen_distance_source is not None:
        (destination/'provenance').mkdir(exist_ok=True)
        shutil.copy2(frozen_distance_source, destination/'provenance/distance_fit_config.json')
    arrays['q'] = q
    np.savez_compressed(destination/'dataset.npz', **arrays)
    q_by_id = dict(zip(events.event_id, q))
    for filename in ['events.csv', 'reference_quality.csv']:
        path = destination/filename
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            row['q'] = str(q_by_id[row['event_id']])
            if 'label_version' in row:
                row['label_version'] = version
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    # String replacement preserves every raw label's numeric representation.
    for filename in ['labels.csv', 'evaluation_manifest.csv']:
        path = destination/filename
        path.write_text(path.read_text().replace(old_version, version))
    write(destination/'config.json', config)
    write(destination/'normalization.json', normal)
    summary = read(destination/'preparation_summary.json')
    summary.update(label_version=version, quality_revision='single_reference_no_effective_n_factor',
                   reference_display_name=display_name)
    write(destination/'preparation_summary.json', summary)
    tuning = read(destination/'reference_tuning_protocol.json')
    tuning.update(objective='User-adopted single-reference construction; fixed reference identities and distance fit retained',
                  outcome_labels_used=False, quality_effective_n_factor=False,
                  quality_formula=FORMULA, q_scales_fitted_on='training events only')
    write(destination/'reference_tuning_protocol.json', tuning)
    write(destination/'quality_revision_source_manifest.json', source_hashes)
    migrated = dict(np.load(destination/'dataset.npz', allow_pickle=False))
    original = dict(np.load(source/'dataset.npz', allow_pickle=False))
    for key in original:
        if key != 'q':
            np.testing.assert_array_equal(original[key], migrated[key])
    np.testing.assert_array_equal(migrated['q'], q)
    for filename in ['reference_pairs.csv', 'coverage.csv', 'exclusions.csv']:
        assert digest(source/filename) == digest(destination/filename)
    assert (source/'labels.csv').read_text().replace(old_version, version) == (destination/'labels.csv').read_text()
    new_events = pd.read_csv(destination/'events.csv', dtype={'event_id':str, 'stock':str})
    pd.testing.assert_frame_equal(events.drop(columns=['q', 'label_version']), new_events.drop(columns=['q', 'label_version']))
    assert source_hashes == {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    verification = {'status':'passed', 'reference_display_name':display_name, 'reference_mode':config['reference_mode'],
                    'events':len(events), 'split_counts':events.split.value_counts().to_dict(),
                    'event_ids_unchanged':True, 'reference_pairs_byte_identical':True,
                    'all_arrays_except_q_identical':True, 'raw_labels_identical_except_version':True,
                    'x_y_normalization_and_quality_scales_unchanged':True,
                    'quality_formula':FORMULA, 'quality_effective_n_factor':False,
                    'quality_neff_factor_enabled':False, 'q_rule':'exp_only',
                    'q_uses_pre_event_matching_quality_and_training_medians_only':True,
                    'q_changed_events':int(np.count_nonzero(old_q != q)),
                    'q_at_floor':int(np.count_nonzero(q == np.float32(1e-8))),
                    'label_version':version, 'dataset_sha256':digest(destination/'dataset.npz'),
                    'normalization_sha256':digest(destination/'normalization.json')}
    if frozen_distance_source is not None:
        verification['frozen_distance_fit_embedded'] = True
    write(destination/'verification.json', verification)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--staging', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    staging = args.staging or root/'data/single_reference_staging'
    report = args.report or root/'results/single_reference_revision'
    if staging.exists():
        raise FileExistsError(staging)
    tasks = [('prepared_expanded','prepared_expanded','Nearest Neighbor',5105),
             ('reference_variants_strict640/strict','reference_variants_expanded/strict','Strict Matching',640),
             ('reference_variants_expanded/nearest','reference_variants_expanded/nearest','Constrained Nearest Neighbor',2269)]
    records = []
    for source, target, display, count in tasks:
        result = migrate_one(root/'data'/source, staging/target, display, count)
        result['staging_relative_path'] = target
        records.append(result)
    event_tables = [pd.read_csv(staging/target/'events.csv') for _,target,_,_ in tasks]
    common = {}
    for split in ['train', 'test']:
        ids = set.intersection(*[set(e.loc[e.split.eq(split), 'event_id']) for e in event_tables])
        common[split] = {'events':len(ids), 'dates':int(event_tables[1].loc[event_tables[1].event_id.isin(ids), 'event_date'].nunique())}
    write(staging/'migration_manifest.json', {'status':'prepared', 'schemes':records, 'comparison_datasets_prepared':False})
    report.mkdir(exist_ok=True, parents=True)
    write(report/'data_migration_verification.json', {'status':'passed', 'reference_count':1,
        'quality_formula':FORMULA, 'quality_effective_n_factor':False,
        'quality_neff_factor_enabled':False, 'q_rule':'exp_only', 'schemes':records,
        'old_sources_modified':False, 'comparison_preparation':'Run after placing these datasets at their final active paths',
        'expected_comparison_intersection':common,
        'script_sha256':digest(Path(__file__))})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
