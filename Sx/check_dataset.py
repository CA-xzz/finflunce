"""Verify published data hashes, tensor shapes, event order and temporal splits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import hashlib


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def check(root):
    manifest = json.loads((root / 'data/manifest.json').read_text())
    for entry in manifest['files']:
        path = root / entry['path']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'File differs: {path}')
    checked = []
    for path in sorted((root / 'data').rglob('dataset.npz')):
        events = pd.read_csv(path.parent / 'events.csv')
        normal = json.loads((path.parent / 'normalization.json').read_text())
        with np.load(path, allow_pickle=False) as a:
            n = len(events)
            assert events.event_id.is_unique and a['x'].shape[0] == a['y'].shape[0] == n
            assert a['x'].shape[-1] == 3 and a['y'].shape[1:] == (11, 3)
            assert all(np.isfinite(a[k]).all() for k in a.files)
            assert np.isin(a['split'], [0, 1, 2]).all()
            assert (a['q'] > 0).all() and (a['q'] <= 1).all()
            np.testing.assert_array_equal(a['end_day'], a['day'] + 10)
            np.testing.assert_array_equal(a['split'], events.split.map({'train': 0, 'validation': 1, 'test': 2}))
            ids = [np.flatnonzero(a['split'] == i) for i in range(3)]
            for i in range(2):
                assert a['end_day'][ids[i]].max() < a['day'][ids[i+1]].min()
            assert np.all(np.asarray(normal['y_std']) > 0)
            checked.append({'dataset': path.parent.relative_to(root).as_posix(), 'events': n,
                            'splits': [len(part) for part in ids]})
    protocol = json.loads((root / 'REFERENCE_PROTOCOL.json').read_text())
    by_name = {item['dataset']: item for item in checked}
    for scheme in protocol['schemes']:
        name = 'data/prepared_expanded' if scheme['scheme'] == 'expanded_nearest' else 'data/reference_variants_expanded/' + scheme['scheme']
        assert by_name[name]['events'] == scheme['events']
        assert by_name[name]['splits'] == [scheme['splits'][k] for k in ('train', 'validation', 'test')]
    print(json.dumps({'status': 'passed', 'hashed_files': len(manifest['files']), 'datasets': checked}, indent=2))


if __name__ == '__main__':
    check(Path(__file__).resolve().parent)
