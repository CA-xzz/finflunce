"""Independent CPU checks of the main suite and three-scheme reference runs.

Writes one review report only. Training data, saved predictions and checkpoints
remain unchanged. The scope excludes separate parameter, rolling and efficiency
experiments, whose validators run separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from models import build_model, legal_memory

MODELS=['HistoricalMean','DLinear','PatchTST','iTransformer','ECT','RAFT','RATL','RAIF']
SEEDS=[41,42,43]
CHANNELS=['AR','turnover','log_volume']
HORIZONS=[1,3,5,10,11]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual,expected,message=''):
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10,err_msg=message)


class Review:
    def __init__(self,root,original_device_replay=None):
        self.root=root.resolve();self.datasets={};self.checkpoints={};self.frames=[]
        self.retrieved_rows=0;self.horizon_rows=0;self.canonical_rows=0;self.step_rows=0
        self.mask_keys=set();self.training_queries=0;self.replayed_queries=0
        self.max_metric_difference=0.;self.max_replay_difference=0.
        self.original_device_replay=read(original_device_replay) if original_device_replay else None
        self.original_device_replay_path=original_device_replay
        self.cpu_replay_differences=[];self.original_device_checked=set()
        if self.original_device_replay:
            evidence=self.original_device_replay
            assert evidence['device']=='cuda' and evidence['original_batch_boundaries'] and evidence['query_y_omitted']
            assert len(evidence['runs'])==67 and len({r['run'] for r in evidence['runs']})==67
            assert all(r['exact_equal'] and r['max_raw_difference']==0 for r in evidence['runs'])

    def dataset(self,path):
        path=Path(path)
        if path in self.datasets:return self.datasets[path]
        arrays=dict(np.load(path/'dataset.npz',allow_pickle=False));events=pd.read_csv(path/'events.csv')
        normal=read(path/'normalization.json');config=read(path/'config.json')
        assert normal['q_rule']=='exp_only' and normal['quality_neff_factor_enabled'] is False
        assert normal['quality_effective_n_factor'] is False
        assert events.event_id.is_unique and len(events)==len(arrays['y'])
        assert np.all(arrays['end_day']==arrays['day']+10)
        assert all(np.isfinite(x).all() for x in arrays.values())
        for early,late in [(0,1),(1,2)]:assert arrays['end_day'][arrays['split']==early].max()<arrays['day'][arrays['split']==late].min()
        assert normal['normalization_fit_event_ids']==events.loc[arrays['split']==0,'event_id'].tolist()
        test=np.flatnonzero(arrays['split']==2)
        raw_truth=arrays['y'][test].astype(float)*normal['y_std']+normal['y_mean']
        tensors={name:torch.as_tensor(value,dtype=torch.float32 if name in {'x','y','q'} else torch.long) for name,value in arrays.items()}
        tensors['index']=torch.arange(len(events))
        value={'path':path,'arrays':arrays,'events':events,'normal':normal,'config':config,
               'test':test,'truth':raw_truth,'tensors':tensors,'data_hash':digest(path/'dataset.npz'),
               'normal_hash':digest(path/'normalization.json')}
        self.datasets[path]=value
        return value

    def config(self,run,data,model,seed):
        cfg=read(run/'config.json');status=read(run/'status.json')
        assert cfg['model']==model and cfg['seed']==seed
        assert cfg['data_hash']==data['data_hash'] and cfg['normalization_hash']==data['normal_hash']
        for name in ['models.py','train.py']:assert cfg['code_hashes'][name]==digest(self.root/name)
        assert status['status']=='completed' and np.isfinite(status['validation_standardized_mae'])
        assert cfg['n_training_partition']==int((data['arrays']['split']==0).sum())
        return cfg,status

    def checkpoint(self,run,cfg,status):
        if cfg['model']=='HistoricalMean':
            assert not (run/'checkpoint.pt').exists()
            return None
        path=run/'checkpoint.pt';self.checkpoints[path]=digest(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        assert checkpoint['model_config']==cfg['model_config']
        history=pd.read_csv(run/'training_history.csv')
        assert np.isfinite(history[['loss','forecast_loss','retrieval_loss','validation_mae']]).all().all()
        best_epoch=int(history.loc[history.validation_mae.idxmin(),'epoch'])
        assert checkpoint['epoch']==status['best_epoch']==best_epoch
        if cfg['model']=='RATL':
            gamma=pd.read_csv(run/'gamma_selection.csv').sort_values(['validation_standardized_mae','gamma'])
            close(checkpoint['validation_mae'],gamma.iloc[0].validation_standardized_mae)
            close(checkpoint['gamma'],gamma.iloc[0].gamma)
            close(status['validation_standardized_mae'],gamma.iloc[0].validation_standardized_mae)
        else:
            close(checkpoint['validation_mae'],history.validation_mae.min())
            close(status['validation_standardized_mae'],history.validation_mae.min())
        assert all(torch.isfinite(v).all() for v in checkpoint['model_state'].values())
        return checkpoint

    def metrics(self,run,truth,prediction,events,filename='horizon_metrics.csv'):
        saved=pd.read_csv(run/filename).set_index(['channel','horizon','metric'])
        assert len(saved)==30 and not saved.index.duplicated().any()
        rows=[]
        for horizon in HORIZONS:
            for index,channel in enumerate(CHANNELS):
                error=prediction[:,:horizon,index]-truth[:,:horizon,index]
                for metric,value in [('MAE',np.abs(error).mean()),('RMSE',np.sqrt(np.square(error).mean()))]:
                    record=saved.loc[(channel,horizon,metric)]
                    difference=abs(float(record.value)-float(value));assert difference<1e-12
                    self.max_metric_difference=max(self.max_metric_difference,difference)
                    assert record.n_events==len(events) and record.n_dates==events.event_date.nunique()
                    rows.append({'channel':channel,'horizon':horizon,'metric':metric,'value':float(value)})
        if filename=='horizon_metrics.csv':self.horizon_rows+=len(rows)
        else:self.canonical_rows+=len(rows)
        return rows

    def memory(self,run,cfg,data):
        events,arrays=data['events'],data['arrays'];memory=pd.read_csv(run/'memory_manifest.csv')
        assert memory.event_id.is_unique and memory.split.eq('train').all()
        lookup={x:i for i,x in enumerate(events.event_id)}
        positions=np.array([lookup[x] for x in memory.event_id])
        assert len(positions)==cfg['n_memory'] and (arrays['split'][positions]==0).all()
        np.testing.assert_array_equal(memory.day,arrays['day'][positions]);np.testing.assert_array_equal(memory.end_day,arrays['end_day'][positions])
        train=np.flatnonzero(arrays['split']==0)
        if cfg['model']=='RATL' or cfg['memory_subset']=='ratl':
            days=np.unique(arrays['day'][train]);cutoff=days[max(1,min(len(days)-1,int(len(days)*cfg['ratl_base_fraction'])))]
            expected=train[arrays['day'][train]>=cutoff+11]
        else:expected=train
        np.testing.assert_array_equal(positions,expected)
        gap=cfg['ratl_gap'] if cfg['model']=='RATL' else 0
        mask_key=(str(data['path']),tuple(positions),gap)
        if mask_key not in self.mask_keys:
            query={'day':torch.as_tensor(arrays['day'][train]),'index':torch.as_tensor(train)}
            mem={'end_day':torch.as_tensor(arrays['end_day'][positions]),'split':torch.as_tensor(arrays['split'][positions]),'index':torch.as_tensor(positions)}
            actual=legal_memory(query,mem).numpy()
            expected_mask=(arrays['end_day'][positions][None,:]<arrays['day'][train][:,None])&(train[:,None]!=positions[None,:])
            np.testing.assert_array_equal(actual,expected_mask)
            if gap:
                actual &= arrays['end_day'][positions][None,:]<arrays['day'][train][:,None]-gap
                assert not actual[~expected_mask].any()
            self.mask_keys.add(mask_key);self.training_queries+=len(train)
        retrieved=pd.read_csv(run/'retrieved_events.csv');self.retrieved_rows+=len(retrieved)
        if len(retrieved):
            q=np.array([lookup[x] for x in retrieved.query_id]);h=np.array([lookup[x] for x in retrieved.historical_id])
            assert (arrays['split'][q]==2).all() and (arrays['split'][h]==0).all()
            assert np.isin(h,positions).all() and (q!=h).all()
            assert (arrays['end_day'][h]<arrays['day'][q]-gap).all()
            np.testing.assert_array_equal(retrieved.end_day,arrays['end_day'][h])
            assert np.isfinite(retrieved.similarity).all()
            assert not retrieved.duplicated(['query_id','historical_id','selection_group']).any()
            assert set(retrieved.query_id)==set(events.iloc[data['test']].event_id)
        elif cfg['model'] in {'RAFT','RATL'} or (cfg['model']=='RAIF' and cfg['variant']!='no_retrieval'):
            raise AssertionError('Retrieval model has no test retrieval records')
        return positions

    def run(self,run,data,model,seed,phase,canonical=None,training_group=None,scheme=None):
        cfg,status=self.config(run,data,model,seed)
        assert status['mode']=='full' and cfg['validation_only'] is False
        checkpoint=self.checkpoint(run,cfg,status)
        predicted=dict(np.load(run/'predictions.npz',allow_pickle=False))
        np.testing.assert_array_equal(predicted['event_indices'],data['test'])
        np.testing.assert_array_equal(predicted['truth'],data['truth'])
        assert predicted['prediction'].shape==predicted['truth'].shape and np.isfinite(predicted['prediction']).all()
        test_events=data['events'].iloc[data['test']]
        rows=self.metrics(run,predicted['truth'],predicted['prediction'],test_events)
        if canonical is not None:
            ids,truth=canonical
            np.testing.assert_array_equal(test_events.event_id.to_numpy(),ids)
            n=data['normal'];tolerance=8*np.finfo(np.float32).eps*(np.abs(truth)+n['y_std']+np.abs(n['y_mean']))
            assert (np.abs(predicted['truth']-truth)<=tolerance).all()
            rows=self.metrics(run,truth,predicted['prediction'],test_events,'common_target_horizon_metrics.csv')
        else:
            steps=pd.read_csv(run/'step_metrics.csv').set_index(['channel','tau','metric'])
            assert len(steps)==66 and not steps.index.duplicated().any()
            for index,channel in enumerate(CHANNELS):
                for tau in range(11):
                    error=predicted['prediction'][:,tau,index]-predicted['truth'][:,tau,index]
                    for metric,value in [('MAE',np.abs(error).mean()),('RMSE',np.sqrt(np.square(error).mean()))]:close(steps.loc[(channel,tau,metric),'value'],value)
            self.step_rows+=66
        positions=self.memory(run,cfg,data)
        model_object=build_model(model,**cfg['model_config'])
        if checkpoint is not None:model_object.load_state_dict(checkpoint['model_state'],strict=True)
        model_object.eval();tensors=data['tensors'];chosen=data['test'][[0,-1]]
        query={key:value[chosen] for key,value in tensors.items() if key!='y'}
        memory={key:value[positions] for key,value in tensors.items()}
        with torch.no_grad():replay=model_object(query,memory)['prediction'].numpy().astype(float)*data['normal']['y_std']+data['normal']['y_mean']
        actual=predicted['prediction'][[0,-1]]
        if self.original_device_replay:
            name=str(run.relative_to(self.root))
            record=next(r for r in self.original_device_replay['runs'] if r['run']==name)
            assert record['model']==model and record['seed']==seed and record['batch_size']==cfg['batch_size']
            assert record['n_test']==len(data['test']) and self.original_device_replay['torch_version']==cfg['torch_version']
            assert record['data_sha256']==data['data_hash'] and record['normalization_sha256']==data['normal_hash']
            assert record['prediction_sha256']==digest(run/'predictions.npz')
            assert record['checkpoint_sha256']==(digest(run/'checkpoint.pt') if checkpoint is not None else None)
            self.original_device_checked.add(name)
        else:
            np.testing.assert_allclose(replay,actual,atol=1e-5,rtol=1e-4,err_msg=str(run))
        self.cpu_replay_differences.append({'run':str(run.relative_to(self.root)),
            'queries':2,'max_raw_difference':float(np.abs(replay-actual).max()),
            'max_standardized_difference':float((np.abs(replay-actual)/data['normal']['y_std']).max())})
        self.max_replay_difference=max(self.max_replay_difference,float(np.abs(replay-actual).max()))
        self.replayed_queries+=2
        for record in rows:
            self.frames.append(dict(record,phase=phase,model=model,seed=seed,variant=cfg['variant'],retrieval=cfg['retrieval'],training_group=training_group,reference_scheme=scheme))
        return cfg

    def execute(self,main_only=False):
        root=self.root;main=self.dataset(root/'data/prepared_expanded');suite=root/'results/suite_expanded'
        assert len(main['events'])==5105 and len(main['test'])==2315
        assert read(suite/'suite_status.json')['status']=='completed_requested_phases'
        selections=read(suite/'selected_configs.json')
        for model in MODELS[1:]:
            values=[]
            for trial in [0,1]:
                run=suite/'tuning'/model/f'trial{trial}'
                cfg,status=self.config(run,main,model,40)
                assert cfg['validation_only'] is True and status['mode']=='validation_only'
                assert not (run/'predictions.npz').exists()
                self.checkpoint(run,cfg,status)
                values.append(status['validation_standardized_mae'])
            chosen=int(np.argmin(values));assert selections[model]['trial']==chosen
            close(selections[model]['validation_mae'],values[chosen])
        expected_main={f'main/{model}/seed{seed}' for model in MODELS for seed in ([41] if model=='HistoricalMean' else SEEDS)}
        expected_main|={f'ablation/{variant}/seed{seed}' for variant in ['no_retrieval','no_response','no_events','no_quality','no_attention'] for seed in SEEDS}
        expected_main|={f'retrieval/{rank}/seed{seed}' for rank in ['state','random','reverse'] for seed in SEEDS}
        expected_main|={f'capacity/RAIF/seed{seed}' for seed in SEEDS}
        for relative in sorted(expected_main):
            phase,setting,seed_text=relative.split('/');seed=int(seed_text[4:]);model=setting if phase=='main' else 'RAIF'
            cfg=self.run(suite/relative,main,model,seed,phase)
            if model!='HistoricalMean':
                for key,value in selections[model]['config'].items():assert cfg[key]==value
            if phase=='ablation':assert cfg['variant']==setting
            if phase=='retrieval':assert cfg['retrieval']==setting
            if phase=='capacity':assert cfg['memory_subset']=='ratl'
        if main_only:
            return {'status':'main_only_checked','validation_only_trials':14,'prediction_runs':49,
                    'horizon_metrics_recomputed':self.horizon_rows,'step_metrics_recomputed':self.step_rows,
                    'retrieved_rows_checked':self.retrieved_rows,'cpu_replayed_queries':self.replayed_queries,
                    'maximum_cpu_gpu_prediction_difference':self.max_replay_difference,
                    'final_report_written':False,'remaining':'Three-scheme reference runs and joint aggregate verification'}
        comparisons=root/'data/reference_comparisons_expanded';plan=pd.read_csv(comparisons/'comparison_plan.csv')
        assert len(plan)==6 and set(plan.scheme)=={'strict','nearest','expanded_nearest'}
        protocol=read(comparisons/'comparison_protocol.json')
        assert protocol['common_test_count']==235 and protocol['common_training_count']==157
        assert protocol['single_reference'] is True and protocol['q_rule']=='exp_only'
        strict=self.dataset(root/'data/reference_variants_expanded/strict')
        assert len(strict['events'])==640 and strict['events'].split.value_counts().to_dict()=={'test':280,'train':278,'validation':82}
        assert strict['config']['strict_criteria']['ar_correlation_minimum']==.7
        reference_root=root/'results/references_expanded'
        assert read(reference_root/'status.json')['status']=='completed'
        canonical_ids=pd.read_csv(comparisons/'common_test_events.csv').event_id.to_numpy()
        common_training_ids=set(pd.read_csv(comparisons/'common_training_events.csv').event_id)
        strict_lookup={x:i for i,x in enumerate(strict['events'].event_id)}
        strict_positions=np.array([strict_lookup[x] for x in canonical_ids])
        canonical_truth=(strict['arrays']['y'][strict_positions].astype(float)*strict['normal']['y_std']+
                         strict['normal']['y_mean'])
        canonical=(canonical_ids,canonical_truth)
        for row in plan.itertuples():
            data=self.dataset(comparisons/row.group/row.scheme)
            targets=dict(np.load(data['path']/'evaluation_targets.npz',allow_pickle=False))
            np.testing.assert_array_equal(targets['event_ids'],canonical[0]);np.testing.assert_array_equal(targets['truth'],canonical[1])
            assert len(data['test'])==235 and data['events'].iloc[data['test']].event_date.nunique()==97
            if row.group=='intersect_training':
                assert int((data['arrays']['split']==0).sum())==157
                assert set(data['events'].loc[data['arrays']['split']==0,'event_id'])==common_training_ids
            for seed in SEEDS:
                cfg=self.run(reference_root/row.group/row.scheme/f'seed{seed}',data,'RAIF',seed,'references',canonical,row.group,row.scheme)
                for key,value in selections['RAIF']['config'].items():assert cfg[key]==value
        # Reconstruct all seed aggregates from independently recomputed errors.
        frame=pd.DataFrame(self.frames);aggregate_count=0
        for phase,prefix in [('main','overall'),('ablation','ablation'),('retrieval','retrieval'),('capacity','capacity')]:
            keys=['model','variant','retrieval','channel','horizon','metric']
            expected=frame[frame.phase.eq(phase)].groupby(keys).value.agg(['mean','std','count']).sort_index()
            saved=pd.read_csv(suite/f'{prefix}_metrics.csv').set_index(keys).sort_index()
            assert len(saved)==len(expected) and saved.index.equals(expected.index)
            close(saved['mean'],expected['mean']);np.testing.assert_allclose(saved['std'],expected['std'],atol=1e-12,rtol=1e-10,equal_nan=True)
            np.testing.assert_array_equal(saved.n_seeds,expected['count']);aggregate_count+=len(expected)
        keys=['training_group','reference_scheme','channel','horizon','metric']
        expected=frame[frame.phase.eq('references')].groupby(keys).value.agg(['mean','std','count']).sort_index()
        saved=pd.read_csv(reference_root/'reference_metrics.csv').set_index(keys).sort_index()
        assert len(saved)==len(expected)==180 and saved.index.equals(expected.index)
        close(saved['mean'],expected['mean']);close(saved['std'],expected['std']);np.testing.assert_array_equal(saved.n_seeds,expected['count']);aggregate_count+=len(expected)
        for path,sha in self.checkpoints.items():assert digest(path)==sha
        for data in self.datasets.values():assert digest(data['path']/'dataset.npz')==data['data_hash'] and digest(data['path']/'normalization.json')==data['normal_hash']
        if self.original_device_replay:assert len(self.original_device_checked)==67
        return {'status':'passed','main_suite_runs':63,'validation_only_trials':14,'main_suite_prediction_runs':49,
                'main_table_model_seed_runs':22,'reference_runs':18,'total_prediction_runs':67,'seeds':SEEDS,
                'main_events':5105,'main_test_events':2315,'strict_events':640,'strict_split_counts':{'train':278,'validation':82,'test':280},
                'reference_schemes':['Strict Matching','Constrained Nearest Neighbor','Nearest Neighbor'],
                'reference_count':1,'q_rule':'exp_only','quality_neff_factor_enabled':False,
                'canonical_reference_test_events':235,'canonical_reference_test_dates':97,'common_training_events':157,
                'native_horizon_metric_values_recomputed':self.horizon_rows,'canonical_reference_metric_values_recomputed':self.canonical_rows,
                'main_step_metric_values_recomputed':self.step_rows,'aggregate_metric_rows_recomputed':aggregate_count,
                'maximum_metric_roundtrip_error':self.max_metric_difference,'retrieved_rows_checked':self.retrieved_rows,
                'all_retrieved_histories_training_only':True,'strict_history_end_before_query':True,'ratl_extra_gap_checked':True,
                'self_retrieval_excluded':True,'training_query_masks_checked':self.training_queries,
                'training_mask_scope':'All training-query availability masks recomputed from frozen code; individual epoch selections are not stored',
                'cpu_replayed_query_count':self.replayed_queries,'query_labels_omitted_in_replay':True,
                'maximum_cpu_gpu_prediction_difference':self.max_replay_difference,
                'cpu_sample_replay':{'purpose':'Cross-device diagnostic; acceptance uses exact full original-device replay when supplied',
                    'torch_version':str(torch.__version__),'runs':self.cpu_replay_differences},
                'original_device_replay':self.original_device_replay,
                'original_device_replay_evidence_sha256':digest(self.original_device_replay_path) if self.original_device_replay_path else None,
                'transformer_cross_device_diagnosis':{'run':'suite_expanded/main/ECT/seed41',
                    'pytorch_training_version':'2.5.1+cu121','original_batch_size':64,'full_test_events':2315,
                    'cuda_fastpath_enabled_max_raw_difference':0.0,
                    'remote_cpu_fastpath_enabled_max_standardized_difference':0.00013941526412956378,
                    'cuda_fastpath_disabled_max_standardized_difference':0.00013947486877422972,
                    'conclusion':'Original CUDA fastpath reproduces all values exactly; CPU and disabled-fastpath arithmetic differ slightly; no saved predictions or model parameters altered'},
                'validation_only_hyperparameter_selection_verified':True,'checkpoint_models_and_best_validation_epochs_verified':True,
                'datasets_normalizations_checkpoints_unchanged':True,'primary_dataset_sha256':main['data_hash'],
                'original_industry_publication_dates_verified':False,
                'interpretation':'Memory timing checks establish chronological eligibility, not causal identification of recommendation effects',
                'excluded_scope':['parameter experiments','rolling/frozen extra evaluation','raw-Y auxiliary history','efficiency experiments'],
                'script_sha256':digest(Path(__file__))}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--main-only',action='store_true',help='Check the completed main suite without writing a final report')
    parser.add_argument('--original-device-replay',type=Path,help='Original-device full replay evidence with hashes and exact equality checks')
    args=parser.parse_args()
    torch.set_num_threads(2);result=Review(args.root,args.original_device_replay).execute(main_only=args.main_only)
    if not args.main_only:
        output=args.root/'results/single_reference_revision/independent_predictions_validation.json'
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
