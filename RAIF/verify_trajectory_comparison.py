"""Independent acceptance of the five-method trajectory comparison.

No training or source edits. --replay-only --device cuda emits full original-
batch replay evidence to stdout. A later CPU invocation can consume that JSON
with --replay-evidence and writes only the final independent_validation.json.
Run CUDA replay only after the experiment owner releases the device.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch

SEEDS = (41, 42, 43)
METHODS = ('ECT-Direct', 'ECT-Incremental', 'RAFT-Direct', 'RAFT-Incremental', 'RAIF')
ROLES = ('reference', 'ect_direct', 'raft_direct')
CHANNELS = ('AR', 'turnover', 'log_volume')
QUERY_FIELDS = ('x', 'platform', 'influencer', 'day', 'index')
OPTIMIZER_FIELDS = ('epochs', 'patience', 'batch_size', 'lr', 'weight_decay', 'huber_delta', 'lambda_ret')
PRIMARY_HASH = '313470657a33a77eaed458af7854dc9ac87f0a516dc1614f445488f68c16eb0e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(actual, expected, message=''):
    np.testing.assert_array_equal(actual, expected, err_msg=message)


def close(actual, expected, atol=1e-12, rtol=1e-12):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def inverse(values, normal):
    return values.astype(np.float64) * np.asarray(normal['y_std']) + np.asarray(normal['y_mean'])


class Review:
    def __init__(self, root):
        self.root = root.resolve(); self.inputs = {}; self.component_records = []
        self.datasets = {}; self.main = {}; self.components = {}; self.retrieval_rows = 0
        self.retrieval_pairs_equal = 0; self.validation_max_difference = 0.
        sys.path.insert(0, str(self.root))
        from models import build_model, legal_memory
        self.build_model = build_model; self.legal_memory = legal_memory
        self.destination = self.root/'results/trajectory_comparison'

    def bind(self, path, expected=None):
        path = Path(path); key = str(path.relative_to(self.root)); value = sha(path)
        assert expected is None or value == expected, 'Hash differs: ' + key
        assert self.inputs.get(key, value) == value, 'Input changed: ' + key
        self.inputs[key] = value
        return value

    def read(self, path):
        self.bind(path); return json.loads(Path(path).read_text())

    def npz(self, path):
        self.bind(path)
        with np.load(path, allow_pickle=False) as a: return dict(a)

    def csv(self, path):
        self.bind(path); return pd.read_csv(path)

    def unchanged(self):
        for name, value in self.inputs.items(): assert sha(self.root/name) == value, name

    def load_dataset(self, directory):
        data = self.npz(directory/'dataset.npz'); normal = self.read(directory/'normalization.json')
        events = self.csv(directory/'events.csv')
        assert len(events) == 5105 and events.event_id.is_unique
        assert all(np.isfinite(a).all() for a in data.values())
        exact(data['end_day'], data['day']+10)
        indices = [np.flatnonzero(data['split']==i) for i in range(3)]
        assert [len(i) for i in indices] == [2170, 620, 2315]
        for i in range(2): assert data['end_day'][indices[i]].max() < data['day'][indices[i+1]].min()
        assert normal['normalization_fit_event_ids'] == events.iloc[indices[0]].event_id.tolist()
        assert np.asarray(normal['y_std']).shape == (3,) and np.all(np.asarray(normal['y_std'])>0)
        assert np.isfinite(normal['y_std']).all()
        return dict(path=directory, arrays=data, normal=normal, events=events, indices=indices, truth=inverse(data['y'], normal))

    def prepare(self):
        main = self.load_dataset(self.root/'data/prepared_expanded'); self.primary = main
        self.bind(main['path']/'dataset.npz', PRIMARY_HASH)
        assert main['normal']['q_rule'] == 'exp_only' and main['normal']['quality_neff_factor_enabled'] is False
        directory = self.root/'data/trajectory_comparison'
        verified = self.read(directory/'verification.json'); protocol = self.read(directory/'protocol.json')
        assert verified['status'] == 'passed' and protocol['n_events'] == 5105
        self.bind(directory/'paired_arrays.npz', verified['paired_arrays_sha256'])
        self.bind(directory/'event_alignment.csv', verified['event_alignment_sha256'])
        pairs = self.npz(directory/'paired_arrays.npz')
        exact(pairs['event_id'], main['events'].event_id); exact(pairs['canonical_delta'], main['truth'])
        for name in ('day','end_day','split'): exact(pairs[name], main['arrays'][name])
        training = main['indices'][0]
        for kind, raw_name in [('recommended_full','raw_y'),('reference_full','raw_b')]:
            data = self.load_dataset(directory/kind); normal = data['normal']; arrays = data['arrays']
            hashes = verified['datasets'][kind]
            for file, field in [('dataset.npz','dataset_sha256'),('normalization.json','normalization_sha256'),('events.csv','events_sha256')]:
                self.bind(data['path']/file, hashes[field])
            exact(data['events'].event_id, main['events'].event_id)
            for name in ('day','end_day','split','q'): exact(arrays[name], main['arrays'][name])
            close(normal['y_mean'], pairs[raw_name][training].mean((0,1)))
            if kind == 'recommended_full':
                for name in ('x','platform','influencer'): exact(arrays[name], main['arrays'][name])
                exact(normal['y_std'], main['normal']['y_std'])
            else:
                assert not arrays['platform'].any() and not arrays['influencer'].any()
                close(normal['y_std'], pairs[raw_name][training].std((0,1)))
                close(normal['x_mean'], pairs['reference_x'][training].mean((0,1)))
                close(normal['x_std'], pairs['reference_x'][training].std((0,1)))
                exact(arrays['x'], ((pairs['reference_x']-normal['x_mean'])/normal['x_std']).astype(np.float32))
            exact(arrays['y'], ((pairs[raw_name]-normal['y_mean'])/normal['y_std']).astype(np.float32))
            self.datasets[kind] = data
        proof = self.read(self.root/'results/single_reference_revision/independent_predictions_validation.json')
        assert proof['status'] == 'passed' and proof['primary_dataset_sha256'] == PRIMARY_HASH
        records = {r['run']:r for r in proof['original_device_replay']['runs']}
        self.main_evidence_sha256 = self.bind(self.root/'results/single_reference_revision/independent_predictions_validation.json')
        for model in ('ECT','RAFT','RAIF'):
            for seed in SEEDS:
                run = self.root/f'results/suite_expanded/main/{model}/seed{seed}'
                record = records[str(run.relative_to(self.root))]
                assert record['exact_equal'] and record['max_raw_difference'] == 0 and record['n_test'] == 2315
                self.bind(run/'checkpoint.pt', record['checkpoint_sha256'])
                self.bind(run/'predictions.npz', record['prediction_sha256'])
                self.bind(main['path']/'dataset.npz', record['data_sha256'])
                self.bind(main['path']/'normalization.json', record['normalization_sha256'])
                config = self.read(run/'config.json'); status = self.read(run/'status.json')
                assert config['model'] == model and config['seed'] == seed and status['status'] == 'completed'
                self.bind(main['path']/'dataset.npz', config['data_hash'])
                self.bind(main['path']/'normalization.json', config['normalization_hash'])
                for name in ('models.py','train.py'): self.bind(self.root/name, config['code_hashes'][name])
                archive = self.npz(run/'predictions.npz')
                exact(archive['event_indices'], main['indices'][2]); exact(archive['truth'], main['truth'][main['indices'][2]])
                self.main[(model,seed)] = dict(path=run, config=config, archive=archive)
        experiment = self.read(self.destination/'experiment_protocol.json')
        assert experiment['methods'] == list(METHODS) and experiment['seeds'] == list(SEEDS)
        assert len(experiment['frozen_incremental_sources']) == 9
        for source in experiment['frozen_incremental_sources']:
            for path, value in source['files'].items(): self.bind(self.root/path, value)
        return {'status':'data_and_source_checks_passed', 'n_events':5105, 'reused_main_models':9, 'new_component_checks_pending':True}

    def retrieval(self, run, split, indices, seed):
        frame = self.csv(run/f'{split}_retrieved_events.csv'); events = self.primary['events'].set_index('event_id')
        assert len(frame) == len(indices)*15 and set(frame.selection_group) == {1,2,4}
        assert set(frame.query_id) == set(self.primary['events'].iloc[indices].event_id)
        assert not frame.duplicated(['query_id','rank']).any()
        assert not frame.duplicated(['query_id','selection_group','historical_id']).any()
        assert frame.groupby(['query_id','selection_group']).size().eq(5).all()
        historical = events.loc[frame.historical_id]; queries = events.loc[frame.query_id]
        assert historical.split.eq('train').all()
        assert (historical.end_day.to_numpy() < queries.day.to_numpy()).all()
        assert not (frame.query_id == frame.historical_id).any()
        exact(frame.end_day, historical.end_day); exact(frame.query_day, queries.day)
        assert np.isfinite(frame.similarity).all()
        self.retrieval_rows += len(frame)
        if split == 'test':
            original = self.csv(self.main[('RAFT',seed)]['path']/'retrieved_events.csv')
            keys = ['query_id','rank']; values = ['query_id','historical_id','rank','selection_group','end_day']
            a = frame.sort_values(keys).reset_index(drop=True); b = original.sort_values(keys).reset_index(drop=True)
            exact(a[values].to_numpy(), b[values].to_numpy(), 'RAFT Direct/Incremental selected histories')
            close(a.similarity, b.similarity)
            self.retrieval_pairs_equal += len(frame)

    def check_components(self):
        for seed in SEEDS:
            for role in ROLES:
                run = self.destination/f'components/{role}/seed{seed}'; model = 'RAFT' if role == 'raft_direct' else 'ECT'
                config = self.read(run/'config.json'); status = self.read(run/'status.json')
                assert (config['role'],config['model'],config['seed']) == (role,model,seed)
                assert (status['role'],status['model'],status['seed'],status['status']) == (role,model,seed,'completed')
                dataset = self.datasets['reference_full' if role == 'reference' else 'recommended_full']
                self.bind(dataset['path']/'dataset.npz', config['data_hash'])
                self.bind(dataset['path']/'normalization.json', config['normalization_hash'])
                self.bind(self.primary['path']/'dataset.npz', config['primary_data_hash'])
                self.bind(self.primary['path']/'normalization.json', config['primary_normalization_hash'])
                source = self.main[(model,seed)]
                self.bind(source['path']/'config.json', config['source_main_config_sha256'])
                assert config['model_config'] == source['config']['model_config']
                assert config['optimizer_settings'] == {key:source['config'][key] for key in OPTIMIZER_FIELDS}
                assert config['optimizer_settings']['batch_size'] == 64
                assert config['n_train'] == 2170 and config['n_validation'] == 620 and config['n_test'] == 2315
                assert config['n_memory'] == (2170 if model == 'RAFT' else 0)
                assert config['query_future_inputs'] is False and config['true_reference_future_used_for_prediction'] is False
                assert config['query_fields'] == list(QUERY_FIELDS)
                for code, value in config['code_hashes'].items(): self.bind(self.root/code, value)
                checkpoint_hash = self.bind(run/'checkpoint.pt', status['checkpoint_sha256'])
                checkpoint = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=True)
                assert checkpoint['model_config'] == config['model_config']
                assert all(torch.isfinite(v).all() for v in checkpoint['model_state'].values())
                history = self.csv(run/'training_history.csv')
                assert len(history) > 0 and np.isfinite(history[['loss','forecast_loss','retrieval_loss','validation_mae']]).all().all()
                exact(history.epoch, np.arange(1,len(history)+1))
                best_epoch = int(history.loc[history.validation_mae.idxmin(),'epoch'])
                assert best_epoch == status['best_epoch'] == checkpoint['epoch']
                close(checkpoint['validation_mae'], history.validation_mae.min())
                close(status['validation_standardized_mae'], history.validation_mae.min())
                assert len(history) <= config['optimizer_settings']['epochs']
                assert len(history) == config['optimizer_settings']['epochs'] or len(history)-best_epoch >= config['optimizer_settings']['patience']
                assert history.retrieval_loss.eq(0).all()
                expected_objective = 'standardized_B_MAE' if role == 'reference' else 'standardized_incremental_MAE_with_frozen_predicted_B'
                assert config['validation_objective'] == status['validation_objective'] == expected_objective
                assert config['training_target_scale'] == ('train_control_B_std' if role == 'reference' else 'primary_train_Delta_std')
                predictions = {}
                for name, partition in [('validation',1),('test',2)]:
                    file = run/f'{name}_predictions.npz'; self.bind(file, status[f'{name}_predictions_sha256'])
                    archive = self.npz(file); indices = dataset['indices'][partition]
                    exact(archive['event_indices'], indices); exact(archive['truth'], dataset['truth'][indices])
                    assert archive['prediction'].dtype == np.float64 and archive['prediction'].shape == (len(indices),11,3)
                    assert np.isfinite(archive['prediction']).all()
                    predictions[name] = archive
                    if role == 'raft_direct': self.retrieval(run,name,indices,seed)
                indices = dataset['indices'][1]
                if role == 'reference':
                    value = np.abs((predictions['validation']['prediction']-dataset['truth'][indices])/dataset['normal']['y_std']).mean()
                    close(value,status['validation_standardized_mae'],atol=2e-7,rtol=2e-6)
                else:
                    reference = self.components[('reference',seed)]
                    for field, filename in [('checkpoint','checkpoint.pt'),('validation_predictions','validation_predictions.npz'),('test_predictions','test_predictions.npz')]:
                        assert config[f'reference_{field}_sha256'] == reference['status'][f'{field}_sha256']
                        self.bind(reference['path']/filename,config[f'reference_{field}_sha256'])
                    value = np.abs((predictions['validation']['prediction']-reference['predictions']['validation']['prediction']-
                                    self.primary['truth'][indices])/self.primary['normal']['y_std']).mean()
                    close(value,status['validation_standardized_mae'])
                self.validation_max_difference = max(self.validation_max_difference,abs(float(value)-status['validation_standardized_mae']))
                self.components[(role,seed)] = dict(path=run, config=config, status=status, predictions=predictions,
                                                   dataset=dataset, checkpoint=checkpoint)
                self.component_records.append(dict(role=role,seed=seed,best_epoch=best_epoch,epochs_run=len(history),
                    validation_metric_recomputed=float(value),validation_metric_saved=status['validation_standardized_mae'],
                    checkpoint_sha256=checkpoint_hash))
            for field in ['checkpoint','validation_predictions','test_predictions']:
                assert self.components[('ect_direct',seed)]['config'][f'reference_{field}_sha256'] == self.components[('raft_direct',seed)]['config'][f'reference_{field}_sha256']
        # Check every training-query eligibility mask independently of predictions.
        arrays = self.primary['arrays']; ids = self.primary['indices'][0]
        query = {'day':torch.as_tensor(arrays['day'][ids]),'index':torch.as_tensor(ids)}
        memory = {'end_day':torch.as_tensor(arrays['end_day'][ids]),'index':torch.as_tensor(ids),'split':torch.as_tensor(arrays['split'][ids])}
        expected = (arrays['end_day'][ids][None,:] < arrays['day'][ids][:,None]) & (ids[None,:] != ids[:,None])
        exact(self.legal_memory(query,memory).numpy(),expected)

    def replay(self, device_name):
        assert device_name == 'cuda', 'Exact acceptance requires the original CUDA implementation'
        device = torch.device(device_name); assert torch.cuda.is_available()
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.mha.set_fastpath_enabled(True)
        records = []
        for seed in SEEDS:
            for role in ROLES:
                component = self.components[(role,seed)]; cfg = component['config']; dataset = component['dataset']
                assert str(torch.__version__) == cfg['torch_version']
                assert torch.cuda.get_device_name(device) == cfg['gpu']
                arrays = dataset['arrays']
                tensors = {name:torch.as_tensor(value,device=device,dtype=torch.float32 if name in ('x','y','q') else torch.long) for name,value in arrays.items()}
                tensors['index'] = torch.arange(len(arrays['x']),device=device)
                memory = {name:value[dataset['indices'][0]] for name,value in tensors.items()}
                model = self.build_model(cfg['model'],**cfg['model_config']).to(device)
                model.load_state_dict(component['checkpoint']['model_state'],strict=True); model.eval()
                assert sum(p.numel() for p in model.parameters()) == cfg['parameters']
                with torch.no_grad():
                    cache = model.refresh_memory(memory) if cfg['model'] == 'RAFT' else None
                    for split, part in [('validation',1),('test',2)]:
                        ids = torch.as_tensor(dataset['indices'][part],device=device); pieces = []
                        for batch in ids.split(cfg['optimizer_settings']['batch_size']):
                            query = {key:tensors[key][batch] for key in QUERY_FIELDS}
                            pieces.append(model(query,memory,cache,labels=None)['prediction'].cpu().numpy())
                        standardized = np.concatenate(pieces); physical = inverse(standardized,dataset['normal'])
                        actual = component['predictions'][split]['prediction']
                        exact(physical,actual,f'{role}/seed{seed}/{split}: original CUDA replay')
                        record = dict(role=role,seed=seed,split=split,n_events=len(ids),exact_equal=True,maximum_absolute_difference=0.,
                            checkpoint_sha256=cfg.get('checkpoint_sha256',component['status']['checkpoint_sha256']),
                            prediction_sha256=component['status'][f'{split}_predictions_sha256'],data_sha256=cfg['data_hash'],
                            normalization_sha256=cfg['normalization_hash'],config_sha256=self.bind(component['path']/'config.json'),
                            batch_size=cfg['optimizer_settings']['batch_size'])
                        if split == 'validation' and role == 'reference':
                            # Reproduce the original float32 device reduction as well as raw-unit verification.
                            val_mae = (torch.as_tensor(standardized,device=device)-tensors['y'][ids]).abs().mean().item()
                            assert val_mae == component['status']['validation_standardized_mae']
                            record['float32_validation_mae_exact'] = True
                        records.append(record)
                del model, memory, tensors
        self.unchanged()
        return dict(status='passed',device='cuda',torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(device),
                    created_utc=datetime.now(timezone.utc).isoformat(),original_batch_boundaries=True,query_fields=list(QUERY_FIELDS),
                    query_outcomes_omitted=True,checkpoint_weights_unchanged=True,records=records)

    def bind_replay(self, evidence):
        assert evidence['status'] == 'passed' and evidence['device'] == 'cuda'
        assert evidence['original_batch_boundaries'] and evidence['query_outcomes_omitted']
        assert evidence['query_fields'] == list(QUERY_FIELDS) and len(evidence['records']) == 18
        expected = {(role,seed,split) for role in ROLES for seed in SEEDS for split in ('validation','test')}
        assert {(r['role'],r['seed'],r['split']) for r in evidence['records']} == expected
        for record in evidence['records']:
            component = self.components[(record['role'],record['seed'])]; cfg = component['config']
            assert record['exact_equal'] and record['maximum_absolute_difference'] == 0.
            assert record['n_events'] == (620 if record['split']=='validation' else 2315)
            assert record['batch_size'] == 64 and evidence['torch_version'] == cfg['torch_version']
            assert evidence['gpu'] == cfg['gpu']
            for field, value in [('data_sha256',cfg['data_hash']),('normalization_sha256',cfg['normalization_hash']),
                                  ('checkpoint_sha256',component['status']['checkpoint_sha256']),
                                  ('prediction_sha256',component['status'][record['split']+'_predictions_sha256'])]: assert record[field] == value
            self.bind(component['path']/'config.json',record['config_sha256'])
            if record['role']=='reference' and record['split']=='validation': assert record['float32_validation_mae_exact']

    def final_tables(self):
        test = self.primary['indices'][2]; truth = self.primary['truth'][test]
        n_dates = self.primary['events'].iloc[test].event_date.nunique(); records = []
        for method in METHODS:
            for seed in SEEDS:
                archive = self.npz(self.destination/f'predictions/{method}/seed{seed}/predictions.npz')
                exact(archive['event_indices'],test); exact(archive['truth'],truth)
                if method.endswith('-Direct'):
                    role = 'ect_direct' if method.startswith('ECT') else 'raft_direct'
                    y = self.components[(role,seed)]['predictions']['test']['prediction']
                    b = self.components[('reference',seed)]['predictions']['test']['prediction']
                    exact(archive['predicted_y'],y); exact(archive['predicted_b'],b); expected = y-b
                else:
                    model = method.replace('-Incremental',''); expected = self.main[(model,seed)]['archive']['prediction']
                exact(archive['prediction'],expected)
                error = archive['prediction']-truth
                for i,channel in enumerate(CHANNELS):
                    for metric,value in [('MAE',np.mean(np.abs(error[:,:,i]))),('RMSE',np.sqrt(np.mean(error[:,:,i]**2)))]:
                        records.append(dict(method=method,seed=seed,channel=channel,metric=metric,value=float(value)))
        frame = pd.DataFrame(records).set_index(['method','seed','channel','metric']).sort_index()
        saved = self.csv(self.destination/'metrics_seeds.csv').set_index(frame.index.names).sort_index()
        assert len(saved) == 90 and not saved.index.duplicated().any() and frame.index.equals(saved.index)
        assert saved.n_events.eq(2315).all() and saved.n_dates.eq(n_dates).all() and saved.horizon.eq(11).all()
        close(saved.value,frame.value)
        table = self.csv(self.destination/'table2_trajectory_comparison.csv')
        assert table.method.tolist() == list(METHODS) and table.n_seeds.eq(3).all()
        assert table.n_events.eq(2315).all() and table.n_dates.eq(n_dates).all() and table.horizon.eq(11).all()
        comparisons = []
        for row in table.itertuples():
            for channel in CHANNELS:
                for metric in ('MAE','RMSE'):
                    values = [frame.loc[(row.method,seed,channel,metric),'value'] for seed in SEEDS]
                    for statistic,value in [('mean',np.mean(values)),('std',np.std(values,ddof=1))]:
                        actual = getattr(row,f'{channel}_{metric}_{statistic}'); close(actual,value)
                        comparisons.append(abs(actual-value))
        export = self.read(self.destination/'export_verification.json'); assert export['status'] == 'passed'
        for collection in ('input_sha256','output_sha256'):
            for path,value in export[collection].items(): self.bind(self.root/path,value)
        return dict(final_prediction_files=15,test_events=2315,test_dates=int(n_dates),
                    seed_metric_values_recomputed=90,aggregate_mean_sd_values_recomputed=60,
                    maximum_seed_metric_difference=float(np.abs(saved.value-frame.value).max()),
                    maximum_aggregate_difference=float(max(comparisons)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-only',action='store_true')
    parser.add_argument('--replay-only',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--replay-evidence',type=Path)
    args = parser.parse_args(); torch.set_num_threads(2)
    review = Review(args.root); prepared = review.prepare()
    if args.data_only:
        review.unchanged(); print(json.dumps(prepared,indent=2)); return
    review.check_components()
    if args.replay_evidence:
        evidence = json.loads(args.replay_evidence.read_text())
    else:
        evidence = review.replay(args.device)
    review.bind_replay(evidence)
    if args.replay_only:
        print(json.dumps(evidence,indent=2)); return
    table = review.final_tables(); review.unchanged()
    result = dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),methods=list(METHODS),seeds=list(SEEDS),
        new_components=9,reused_main_models=9,main_events=5105,train_events=2170,validation_events=620,
        **table,all_final_truth_exactly_primary=True,direct_predictions_exactly_Yhat_minus_shared_Bhat=True,
        validation_best_epochs_verified=True,training_target_scale_and_source_configuration_verified=True,
        shared_reference_checkpoint_validation_test_hashes_verified=True,reference_category_indices_all_zero=True,
        raft_retrieval_rows_checked=review.retrieval_rows,raft_direct_incremental_test_retrieval_rows_exact=review.retrieval_pairs_equal,
        mature_training_histories_only=True,self_retrieval_excluded=True,training_queries_with_independent_availability_masks=2170,
        component_validation_reduction_max_difference=review.validation_max_difference,component_checks=review.component_records,
        original_device_replay=evidence,original_main_replay_report_sha256=review.main_evidence_sha256,
        source_files_unchanged=True,input_sha256=review.inputs,
        interpretation='Direct methods additionally use control pre-event inputs and a shared B forecaster; total input information and parameter count are not equal across all five methods',
        script_sha256=sha(Path(__file__)))
    output=review.destination/'independent_validation.json'
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','new_components','final_prediction_files','seed_metric_values_recomputed','aggregate_mean_sd_values_recomputed','raft_retrieval_rows_checked','raft_direct_incremental_test_retrieval_rows_exact')},indent=2))


if __name__ == '__main__': main()
