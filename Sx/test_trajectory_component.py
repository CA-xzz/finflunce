"""Independent CPU checks for full-trajectory training and Delta evaluation.

Synthetic model checks and temporary outputs never train a production model or
modify the source datasets, checkpoints, training core or model implementation.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import train as core
from models import DirectForecaster, RAFT
import train_trajectory_component as component

ROOT = Path(__file__).resolve().parent


def configuration():
    return dict(length=10, features=3, horizon=11, channels=3, n_platforms=3,
                n_influencers=4, hidden=8, heads=2, layers=1, dropout=0., k=5,
                raft_temperature=.1)


def tiny_data():
    generator = torch.Generator().manual_seed(171)
    n = 9
    return dict(x=torch.randn(n, 10, 3, generator=generator),
                y=torch.randn(n, 11, 3, generator=generator), q=torch.ones(n),
                platform=torch.arange(n) % 3, influencer=torch.arange(n) % 4,
                day=torch.tensor([0, 20, 40, 60, 100, 120, 160, 180, 200]),
                end_day=torch.tensor([10, 30, 50, 70, 110, 130, 170, 190, 210]),
                split=torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2]), index=torch.arange(n))


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.level = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, query, memory=None, cached_keys=None, labels=None):
        n = len(query['x'])
        return dict(prediction=self.level.expand(n, 11, 3),
                    retrieval_loss=self.level.new_zeros(n))


class ComponentModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_structure_and_initialization_equal_original_models(self):
        for name, base_class in [('ECT', DirectForecaster), ('RAFT', RAFT)]:
            with self.subTest(model=name):
                torch.manual_seed(91)
                base = base_class(name, **configuration()) if name == 'ECT' else base_class(**configuration())
                torch.manual_seed(91)
                wrapped = component.build_component(name, configuration())
                self.assertEqual(list(base.state_dict()), list(wrapped.state_dict()))
                self.assertEqual(sum(p.numel() for p in base.parameters()), sum(p.numel() for p in wrapped.parameters()))
                for key in base.state_dict():
                    torch.testing.assert_close(base.state_dict()[key], wrapped.state_dict()[key], rtol=0, atol=0)

    def test_wrappers_remove_future_query_fields_and_labels_before_base_call(self):
        data = tiny_data(); memory = core.subset(data, torch.arange(4))
        for name, base_class in [('ECT', DirectForecaster), ('RAFT', RAFT)]:
            with self.subTest(model=name):
                model = component.build_component(name, configuration()).eval()
                query = core.subset(data, torch.tensor([6, 7]))
                expected = model(query, memory)['prediction'].detach()
                query.update(y=object(), raw_y=object(), baseline=object(), canonical_delta=object(), future_market=object())
                original = base_class.forward
                calls = []
                def checked(instance, inputs, memory=None, cached_keys=None, labels=None):
                    self.assertEqual(set(inputs), set(component.QUERY_FIELDS))
                    self.assertIsNone(labels)
                    if name == 'ECT': self.assertIsNone(memory)
                    calls.append(True)
                    return original(instance, inputs, memory, cached_keys, labels)
                with patch.object(base_class, 'forward', checked):
                    actual = model(query, memory, labels=object())['prediction'].detach()
                self.assertEqual(calls, [True])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_predict_excludes_query_targets_before_any_model_call(self):
        data = tiny_data(); data['y'] = object(); data['future_raw_y'] = object()
        class Spy(ConstantModel):
            def forward(inner, query, memory=None, cached_keys=None, labels=None):
                self.assertEqual(set(query), set(component.QUERY_FIELDS))
                self.assertIsNone(labels)
                return super().forward(query, memory, cached_keys, labels)
        prediction, records = component.predict(Spy(), data, torch.tensor([6, 7, 8]), None, 2, collect=True)
        self.assertEqual(prediction.shape, (3, 11, 3)); self.assertEqual(records, [])
        self.assertTrue(np.isfinite(prediction).all())

    def test_raft_response_values_change_predictions_but_not_histories_or_scores(self):
        data = tiny_data(); memory = core.subset(data, torch.arange(4)); query = core.subset(data, torch.tensor([6, 7]))
        model = component.build_component('RAFT', configuration()).eval()
        changed = dict(memory, y=memory['y'] + 17)
        with torch.no_grad():
            before = model(query, memory); after = model(query, changed)
        for field in ['selected', 'selected_valid', 'selection_group', 'similarity']:
            torch.testing.assert_close(before[field], after[field], rtol=0, atol=0)
        self.assertFalse(torch.allclose(before['prediction'], after['prediction']))
        self.assertEqual(set(before['selection_group'].flatten().tolist()), {1, 2, 4})

    def test_strict_maturity_same_day_self_and_nontraining_exclusion(self):
        data = tiny_data(); memory = core.subset(data, torch.arange(6))
        memory['day'] = torch.tensor([0, 10, 20, 40, 0, 0]); memory['end_day'] = memory['day'] + 10
        memory['split'] = torch.tensor([0, 0, 0, 0, 1, 0])
        query = core.subset(data, torch.tensor([5])); query['day'] = torch.tensor([20])
        model = component.build_component('RAFT', configuration()).eval()
        with torch.no_grad():
            result = model(query, memory)
            self.assertEqual(set(result['selected'][result['selected_valid']].tolist()), {0})
            changed = dict(memory, y=memory['y'].clone()); changed['y'][1:] += 10000
            torch.testing.assert_close(model(query, changed)['prediction'], result['prediction'], rtol=0, atol=0)
            boundary = dict(query, day=torch.tensor([10]))
            self.assertFalse(model(boundary, memory)['selected_valid'].any())
            torch.testing.assert_close(model(boundary, memory)['prediction'], model(boundary, None)['prediction'], rtol=0, atol=0)


class ValidationChecks(unittest.TestCase):
    def test_incremental_metric_uses_predicted_B_and_delta_channel_scales(self):
        yhat = np.array([[[10., 40., 90.]]]); bhat = np.array([[[3., 8., 10.]]])
        delta = np.array([[[5., 20., 20.]]]); scale = np.array([2., 3., 4.])
        self.assertEqual(component.incremental_validation(yhat, bhat, delta, scale), (1 + 4 + 15) / 3)
        self.assertNotEqual(component.incremental_validation(yhat, bhat + 7, delta, scale), (1 + 4 + 15) / 3)

    def test_incremental_metric_rejects_invalid_scales_and_unaligned_inputs(self):
        valid = np.zeros((2, 11, 3))
        for scale in [[1, 2], [1, 0, 3], [1, -2, 3], [1, np.inf, 3], [1, np.nan, 3]]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                component.incremental_validation(valid, valid, valid, scale)
        with self.assertRaises(ValueError): component.incremental_validation(valid, valid[:1], valid, [1, 2, 3])
        with self.assertRaises(ValueError): component.incremental_validation(valid + np.nan, valid, valid, [1, 2, 3])

    def test_direct_evaluation_ignores_component_y_and_restores_global_callback(self):
        data = tiny_data(); val = torch.tensor([4, 5]); normal = dict(y_mean=[1., 2., 3.], y_std=[2., 3., 4.])
        standard = np.ones((2, 11, 3), dtype=np.float32); b = np.ones_like(standard) * 5; canonical = np.ones_like(standard) * 2
        original = core.evaluate
        expected = component.incremental_validation(standard * normal['y_std'] + normal['y_mean'], b, canonical, [2., 4., 8.])
        with patch.object(component, 'predict', return_value=(standard, [])):
            with component.validation_evaluator(data, normal, val, b, canonical, [2., 4., 8.]):
                value, _, _ = core.evaluate(None, data, val, None, 2)
                self.assertEqual(value, expected)
                data['y'][val] = float('nan')
                self.assertEqual(core.evaluate(None, data, val, None, 2)[0], expected)
                for other, indices, collect in [(dict(data), val, False), (data, torch.tensor([6, 7]), False), (data, val, True)]:
                    with self.assertRaises(ValueError): core.evaluate(None, other, indices, None, 2, collect=collect)
        self.assertIs(core.evaluate, original)
        with self.assertRaisesRegex(RuntimeError, 'test interruption'):
            with component.validation_evaluator(data, normal, val, b, canonical, [2., 4., 8.]):
                raise RuntimeError('test interruption')
        self.assertIs(core.evaluate, original)

    def test_reference_evaluation_is_standardized_B_mae(self):
        data = tiny_data(); val = torch.tensor([4, 5]); prediction = np.ones((2, 11, 3), dtype=np.float32)
        expected = float(torch.abs(torch.ones_like(data['y'][val]) - data['y'][val]).mean())
        with patch.object(component, 'predict', return_value=(prediction, [])):
            with component.validation_evaluator(data, {'y_mean':[100]*3, 'y_std':[200]*3}, val, None, None, None):
                self.assertEqual(core.evaluate(None, data, val, None, 2)[0], expected)

    def test_real_fit_optimizes_Y_but_selects_checkpoint_by_final_delta(self):
        data = tiny_data(); train_ids = torch.arange(4); val = torch.tensor([4, 5])
        data['y'][train_ids] = 1.; data['y'][val] = 0.
        normal = {'y_mean':[10., 10., 10.], 'y_std':[1., 2., 3.]}
        b = np.full((2, 11, 3), 10.); canonical = np.ones((2, 11, 3)) * np.array([2., 4., 6.])
        model = ConstantModel()
        with tempfile.TemporaryDirectory(prefix='Sx-trajectory-fit-') as directory:
            args = SimpleNamespace(output=Path(directory), lr=.1, weight_decay=0., epochs=3, batch_size=4,
                                   timeout_seconds=60, lambda_ret=.1, huber_delta=1., patience=3)
            with redirect_stdout(io.StringIO()), component.validation_evaluator(data, normal, val, b, canonical, normal['y_std']):
                best, epoch, _ = core.fit_stage(model, data, train_ids, val, core.subset(data, train_ids), args, {}, time.monotonic())
            history = pd.read_csv(Path(directory)/'training_history.csv')
            # Y validation is zero, so growing positive predictions worsen Y MAE.
            # Final Delta requires standardized prediction 2, so the same growth improves it.
            self.assertEqual(epoch, 3); self.assertGreater(model.level.item(), .2)
            self.assertTrue(np.all(np.diff(history.validation_mae) < 0))
            self.assertTrue(np.all(np.diff(history.forecast_loss) < 0))
            self.assertAlmostEqual(history.forecast_loss.iloc[0], .5, places=6)
            self.assertAlmostEqual(best, 2-model.level.item(), places=6)
            checkpoint = torch.load(Path(directory)/'checkpoint.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 3)


class PreparedDataAndExecutionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_datasets_preserve_ids_splits_and_train_only_target_scales(self):
        source = ROOT/'data/prepared_expanded'; prepared = ROOT/'data/trajectory_comparison'
        with np.load(source/'dataset.npz') as p, np.load(prepared/'paired_arrays.npz') as pairs:
            primary = dict(p); paired = dict(pairs)
        normal = component.read_json(source/'normalization.json'); mask = primary['split'] == 0
        canonical = primary['y'].astype(float) * normal['y_std'] + normal['y_mean']
        np.testing.assert_array_equal(paired['canonical_delta'], canonical)
        self.assertEqual(len(primary['y']), 5105)
        for kind, values in [('recommended_full', paired['raw_y']), ('reference_full', paired['raw_b'])]:
            arrays = dict(np.load(prepared/kind/'dataset.npz')); n = component.read_json(prepared/kind/'normalization.json')
            events = pd.read_csv(prepared/kind/'events.csv')
            np.testing.assert_array_equal(events.event_id, paired['event_id'])
            for key in ['split','day','end_day']: np.testing.assert_array_equal(arrays[key], primary[key])
            np.testing.assert_allclose(n['y_mean'], values[mask].mean((0,1)), rtol=0, atol=1e-12)
            scale = normal['y_std'] if kind == 'recommended_full' else values[mask].std((0,1))
            np.testing.assert_allclose(n['y_std'], scale, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(arrays['y'], ((values-n['y_mean'])/n['y_std']).astype(np.float32))
            if kind == 'recommended_full': np.testing.assert_array_equal(arrays['x'], primary['x'])
            else:
                self.assertFalse(arrays['platform'].any()); self.assertFalse(arrays['influencer'].any())
                np.testing.assert_allclose(n['x_mean'], paired['reference_x'][mask].mean((0,1)), atol=1e-12, rtol=0)
                np.testing.assert_allclose(n['x_std'], paired['reference_x'][mask].std((0,1)), atol=1e-12, rtol=0)

    def execution_fixture(self, directory, role='ect_direct'):
        root = Path(directory)
        (root/'data').symlink_to(ROOT/'data', target_is_directory=True)
        (root/'results').mkdir()
        (root/'results/suite_expanded').symlink_to(ROOT/'published_results/suite_expanded', target_is_directory=True)
        for name in ['models.py','train.py','train_trajectory_component.py']:
            (root/name).symlink_to(ROOT/name)
        reference = root/'results/trajectory_comparison/components/reference/seed41'
        reference.mkdir(parents=True)
        (reference/'checkpoint.pt').write_bytes(b'SYNTHETIC_TEST_ONLY_NOT_A_MODEL')
        arrays = dict(np.load(ROOT/'data/prepared_expanded/dataset.npz'))
        for split, number in [('validation',1), ('test',2)]:
            ids = np.flatnonzero(arrays['split']==number)
            np.savez(reference/(split+'_predictions.npz'), event_indices=ids,
                     prediction=np.zeros((len(ids),11,3)), truth=np.zeros((len(ids),11,3)))
        config = {'role':'reference', 'seed':41,
                  'primary_data_hash':core.sha256(ROOT/'data/prepared_expanded/dataset.npz'),
                  'code_hashes':{name:core.sha256(ROOT/name) for name in ['models.py','train.py','train_trajectory_component.py']},
                  'data_hash':core.sha256(ROOT/'data/trajectory_comparison/reference_full/dataset.npz'),
                  'normalization_hash':core.sha256(ROOT/'data/trajectory_comparison/reference_full/normalization.json')}
        core.write_json(reference/'config.json', config)
        status = {'status':'completed','checkpoint_sha256':core.sha256(reference/'checkpoint.pt'),
                  'validation_predictions_sha256':core.sha256(reference/'validation_predictions.npz'),
                  'test_predictions_sha256':core.sha256(reference/'test_predictions.npz')}
        core.write_json(reference/'status.json', status)
        options = SimpleNamespace(root=root, role=role, seed=41, device='cpu', threads=1, timeout_seconds=60)
        return options, reference

    def test_changed_frozen_B_checkpoint_and_prediction_rejected_before_training(self):
        for filename in ['checkpoint.pt','validation_predictions.npz','test_predictions.npz']:
            with self.subTest(artifact=filename), tempfile.TemporaryDirectory(prefix='Sx-trajectory-hash-') as directory:
                options, reference = self.execution_fixture(directory)
                with (reference/filename).open('ab') as handle: handle.write(b'changed')
                with patch.object(core, 'fit_stage') as fit, self.assertRaises(AssertionError): component.execute(options)
                fit.assert_not_called()

    def test_reference_nonzero_event_category_rejected_before_training(self):
        with tempfile.TemporaryDirectory(prefix='Sx-trajectory-category-') as directory:
            options, reference = self.execution_fixture(directory, 'reference')
            # The temporary fake reference outputs would otherwise activate resume.
            for path in reference.iterdir(): path.unlink()
            original = core.load_data
            def invalid_category(path, device):
                data, events, normal = original(path, device)
                data['platform'][0] = 1
                return data, events, normal
            with patch.object(core, 'load_data', invalid_category), patch.object(core, 'fit_stage') as fit:
                with self.assertRaises(AssertionError): component.execute(options)
                fit.assert_not_called()

    def test_direct_execution_inherits_settings_and_both_roles_bind_same_B(self):
        with tempfile.TemporaryDirectory(prefix='Sx-trajectory-component-') as directory:
            options, reference = self.execution_fixture(directory)
            original_evaluate = core.evaluate
            configs = []
            def fake_fit(model, data, train_ids, val_ids, memory, args, model_config, started):
                # Read-only CPU checks of the integration; no optimizer or production training.
                self.assertIsNot(core.evaluate, original_evaluate)
                self.assertEqual(len(memory['y']), 2170)
                self.assertTrue(bool((memory['split']==0).all()))
                source = component.read_json(ROOT/'published_results/suite_expanded/main'/('RAFT' if isinstance(model, RAFT) else 'ECT')/'seed41/config.json')
                for key in ['epochs','patience','batch_size','lr','weight_decay','huber_delta','lambda_ret']:
                    self.assertEqual(getattr(args,key), source[key])
                torch.save({'model_state':model.state_dict(), 'epoch':0, 'validation_mae':0., 'model_config':model_config}, args.output/'checkpoint.pt')
                return 0., 0, 0.
            def fake_predict(model, data, indices, memory, batch_size, cache=None, collect=False):
                return np.zeros((len(indices),11,3),dtype=np.float32), []
            for role in ['ect_direct','raft_direct']:
                options.role = role
                with redirect_stdout(io.StringIO()), patch.object(core,'fit_stage',fake_fit), patch.object(component,'predict',fake_predict):
                    component.execute(options)
                self.assertIs(core.evaluate, original_evaluate)
                output = options.root/'results/trajectory_comparison/components'/role/'seed41'
                configs.append(component.read_json(output/'config.json'))
                for split in ['validation','test']:
                    archive = np.load(output/(split+'_predictions.npz'))
                    self.assertEqual(archive['prediction'].shape, archive['truth'].shape)
            for field, filename in [('reference_checkpoint_sha256','checkpoint.pt'),
                                    ('reference_validation_predictions_sha256','validation_predictions.npz'),
                                    ('reference_test_predictions_sha256','test_predictions.npz')]:
                self.assertEqual(configs[0][field], configs[1][field])
                self.assertEqual(configs[0][field], core.sha256(reference/filename))
            self.assertEqual({c['training_target_scale'] for c in configs}, {'primary_train_Delta_std'})
            self.assertEqual({c['validation_objective'] for c in configs}, {'standardized_incremental_MAE_with_frozen_predicted_B'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
