"""Tests for q migration and the three-scheme reference comparison."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import numpy as np
import pandas as pd

import test_prepare as fixtures
from migrate_single_reference import migrate_one
from prepare import run as prepare
from prepare_windows import run as windows
from reference_experiments import run as compare


class SingleReferenceTests(unittest.TestCase):
    def make_sources(self, root):
        paths = {}
        f = fixtures.PreparationTests()
        for name, train, offset in [('strict',['b','c'],-10.), ('nearest',['a','b','c','d'],50.),
                                    ('expanded_nearest',['a','b','c','d'],100.)]:
            path = f.make_comparison_fixture(root, name, train, offset)
            normal = json.loads((path/'normalization.json').read_text())
            normal.update(quality_effective_n_factor=False, quality_neff_factor_enabled=False,
                          q_rule='exp_only', quality_distance_scale=4.)
            arrays = dict(np.load(path/'dataset.npz'))
            qf = pd.read_csv(path/'reference_quality.csv')
            arrays['q'] = np.maximum(np.exp(-qf.balance.to_numpy()/2.-qf.mean_distance.to_numpy()/4.), 1e-8).astype(np.float32)
            np.savez_compressed(path/'dataset.npz', **arrays)
            qf['q'] = arrays['q']; qf.to_csv(path/'reference_quality.csv', index=False)
            events = pd.read_csv(path/'events.csv'); events['q'] = arrays['q']; events.to_csv(path/'events.csv',index=False)
            (path/'normalization.json').write_text(json.dumps(normal))
            (path/'config.json').write_text(json.dumps({'reference_mode':name, 'm':1,
                'quality_effective_n_factor':False, 'quality_neff_factor_enabled':False, 'q_rule':'exp_only'}))
            paths[name] = path
        return paths

    def test_three_schemes_only_and_intersection_q_omits_factor(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); sources = self.make_sources(root)
            plans = compare(None, sources['strict'], sources['nearest'], root/'comparison',
                            {'expanded_nearest':sources['expanded_nearest']}, single_reference=True)
            self.assertEqual(len(plans), 6)
            self.assertEqual({p['scheme'] for p in plans}, {'strict','nearest','expanded_nearest'})
            targets = None
            for plan in plans:
                path = Path(plan['dataset']); a = dict(np.load(path/'dataset.npz'))
                normal = json.loads((path/'normalization.json').read_text())
                self.assertFalse(normal['quality_effective_n_factor'])
                t = dict(np.load(path/'evaluation_targets.npz'))
                if targets is None: targets = t['truth']
                np.testing.assert_array_equal(t['truth'], targets)
                if plan['group'] == 'intersect_training':
                    self.assertEqual(normal['quality_balance_scale'], 2.)
                    self.assertEqual(normal['quality_distance_scale'], 4.)
                    np.testing.assert_allclose(a['q'][:2], [np.exp(-1), np.exp(-3)], rtol=1e-6)
                    self.assertEqual(normal['normalization_fit_event_ids'], ['b','c'])

    def test_three_scheme_mode_rejects_soft_and_multi_reference(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); sources = self.make_sources(root)
            with self.assertRaisesRegex(ValueError, 'do not accept a soft'):
                compare(sources['nearest'], sources['strict'], sources['nearest'],root/'bad1',
                        {'expanded_nearest':sources['expanded_nearest']},single_reference=True)
            with self.assertRaisesRegex(ValueError, 'exactly expanded_nearest'):
                compare(None,sources['strict'],sources['nearest'],root/'bad2',single_reference=True)
            cfg = json.loads((sources['strict']/'config.json').read_text());cfg['m']=3
            (sources['strict']/'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'require M=1'):
                compare(None,sources['strict'],sources['nearest'],root/'bad3',
                        {'expanded_nearest':sources['expanded_nearest']},single_reference=True)

    def test_migration_preserves_arrays_labels_and_training_scales(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); args = fixtures.PreparationTests().build_fixture(root)
            args.reference_mode = 'nearest'
            prepare(args)
            old = dict(np.load(args.output/'dataset.npz'))
            output = root/'migrated'
            result = migrate_one(args.output, output, 'Constrained Nearest Neighbor', len(old['q']))
            new = dict(np.load(output/'dataset.npz'))
            for key in old:
                if key != 'q': np.testing.assert_array_equal(old[key], new[key])
            self.assertGreater(result['q_changed_events'], 0)
            self.assertFalse(result['quality_effective_n_factor'])
            self.assertEqual(result['q_rule'], 'exp_only')

    def test_preparation_and_windows_keep_exp_only_policy(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); args = fixtures.PreparationTests().build_fixture(root)
            args.reference_mode, args.quality_effective_n_factor = 'expanded_nearest', 'none'
            prepare(args)
            base = args.output
            normal = json.loads((base/'normalization.json').read_text())
            self.assertFalse(normal['quality_neff_factor_enabled'])
            self.assertEqual(normal['q_rule'], 'exp_only')
            windows(SimpleNamespace(data=base, source=args.source, preflight=args.preflight,
                                    external=args.external, output=root/'windows'))
            for length in [5,10,20]:
                path = root/'windows'/f'L{length}'
                a = dict(np.load(path/'dataset.npz')); n = json.loads((path/'normalization.json').read_text())
                q = pd.read_csv(path/'reference_quality.csv')
                expected = np.maximum(np.exp(-q.balance/n['quality_balance_scale']-q.mean_distance/n['quality_distance_scale']),1e-8).astype(np.float32)
                np.testing.assert_allclose(a['q'],expected,rtol=1e-6,atol=0)
                self.assertFalse(n['quality_effective_n_factor'])


if __name__ == '__main__':
    unittest.main()
