"""CPU-only audit of serial training and aligned batch-one efficiency outputs.

This reads measured samples and saved predictions; it does not train, invoke
CUDA, change checkpoints, or infer whole-host exclusivity from GPU checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ['HistoricalMean','DLinear','PatchTST','iTransformer','ECT','RAFT','RATL','RAIF']
EXPECTED = {(m,s) for m in MODELS for s in ([41] if m=='HistoricalMean' else [41,42,43])}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, message='numeric mismatch'):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, err_msg=message)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_samples(path, summary, queries, options):
    samples = pd.read_csv(path)
    n, repeats = len(queries), int(options['repeats'])
    assert len(samples)==n*repeats and samples.batch_size.eq(1).all()
    assert not samples.duplicated(['repeat','dataset_index']).any()
    assert set(samples['repeat'])==set(range(repeats))
    assert set(samples.dataset_index)==set(queries)
    assert samples.groupby('dataset_index').size().eq(repeats).all()
    assert np.isfinite(samples.inference_ms).all() and samples.inference_ms.gt(0).all()
    assert summary['inference_batch_size']==1
    assert summary['inference_samples']==len(samples)
    assert summary['inference_queries']==n and summary['inference_repeats']==repeats
    assert summary['inference_warmup_calls']==options['warmup']
    for field,value in [('inference_event_ms',samples.inference_ms.mean()),
                        ('inference_sample_std_ms',samples.inference_ms.std(ddof=1)),
                        ('inference_median_ms',samples.inference_ms.median()),
                        ('inference_p95_ms',samples.inference_ms.quantile(.95))]:close(summary[field],value,field)
    retrieval=samples.retrieval_ms.dropna()
    if len(retrieval):
        assert len(retrieval)==len(samples) and retrieval.gt(0).all()
        close(summary['retrieval_event_ms'],retrieval.mean())
        close(summary['retrieval_sample_std_ms'],retrieval.std(ddof=1))
    else:
        assert pd.isna(summary.get('retrieval_event_ms'))
    values_mb=912*11*3*4/1e6
    model=summary['model']
    if model=='RATL':close(summary['canonical_memory_mb'],summary['memory_cache_tensor_mb'])
    elif model in {'HistoricalMean','RAFT','RAIF'}:
        close(summary['canonical_memory_mb'],summary['memory_cache_tensor_mb']+values_mb)
    else:assert summary['canonical_memory_mb']==0
    return len(samples)


def gpu_checks(directory, expected_idle):
    gpu=read(directory/'gpu_checks.json'); companions=read(directory/'companion_gpu_checks.json')
    assert gpu and companions
    initial=gpu[0]
    assert initial['stage']=='before_cuda_allocation' and not initial['pids']
    assert initial['memory_used_mib']<=64 and initial['utilization']==0
    for record in gpu:
        assert record['index']==initial['index']
        assert record['additional_reported_pids']==[]
        maximum=record.get('maximum_memory_mib')
        if maximum is not None: assert record['actual_memory_mib']<=maximum
        owned=record.get('owned_reported_pid')
        if owned is not None: assert set(record['pids']).issubset({owned})
    observed={int(x['index']) for x in companions}
    assert observed==set(expected_idle)
    for record in companions:
        assert not record['pids'] and record['memory_used_mib']<=64 and record['utilization']==0
    return {'gpu_checks':len(gpu),'companion_gpu_checks':len(companions),
            'observed_gpu_index':int(initial['index']),'companion_gpu_indices':sorted(observed),
            'whole_host_exclusivity_claimed':False}


def manifests(directory, data, events, query_count):
    common=pd.read_csv(directory/'common_memory_manifest.csv')
    query=pd.read_csv(directory/'batch1_query_manifest.csv')
    for frame,split in [(common,0),(query,2)]:
        assert frame.dataset_index.is_unique and frame.event_id.is_unique
        ids=frame.dataset_index.to_numpy()
        np.testing.assert_array_equal(frame.event_id,events.iloc[ids].event_id)
        np.testing.assert_array_equal(frame.day,data['day'][ids])
        np.testing.assert_array_equal(frame.end_day,data['end_day'][ids])
        assert (data['split'][ids]==split).all()
    test=np.flatnonzero(data['split']==2)
    expected=test[np.linspace(0,len(test)-1,min(query_count,len(test)),dtype=int)]
    np.testing.assert_array_equal(query.dataset_index,expected)
    assert len(common)==912
    return common.dataset_index.to_numpy(),query.dataset_index.to_numpy()


def summary_check(path, source, keys):
    summary=pd.read_csv(path)
    for row in summary.itertuples():
        frame=source[row.measurement] if 'measurement' in summary else source
        values=pd.to_numeric(frame.loc[frame.model.eq(row.model),row.metric],errors='coerce').dropna()
        assert row.n_seeds==len(values)
        close(row.mean,values.mean())
        if len(values)==1:assert pd.isna(row.std)
        else:close(row.std,values.std(ddof=1))
    assert not summary.duplicated(keys).any()


def check_code(root, protocol):
    for name,sha in protocol['code_hashes'].items(): assert digest(root/name)==sha,name
    assert digest(root/'benchmark_serial_training.py')==protocol['benchmark_script_sha256']


def serial(root, directory=None, verify_code=True):
    directory=directory or root/'results/serial_training_expanded'
    protocol,status=read(directory/'protocol.json'),read(directory/'status.json')
    assert status['status']=='completed' and status['training_runs']==status['inference_runs']==22
    assert status['no_observed_contention'] is True
    data_path=root/'data/prepared_expanded';data=dict(np.load(data_path/'dataset.npz'))
    events=pd.read_csv(data_path/'events.csv'); normal=read(data_path/'normalization.json')
    for key,file in [('dataset_sha256','dataset.npz'),('normalization_sha256','normalization.json')]:assert protocol[key]==digest(data_path/file)
    if verify_code:check_code(root,protocol)
    assert protocol['common_memory_sha256']==digest(directory/'common_memory_manifest.csv')
    options=protocol['options'];common,queries=manifests(directory,data,events,options['query_count'])
    assert protocol['n_memory']==912 and protocol['n_test']==2315 and protocol['n_train']==2170
    costs=pd.read_csv(directory/'training_timing.csv');inference=pd.read_csv(directory/'batch1_timing.csv')
    assert set(zip(costs.model,costs.seed))==set(zip(inference.model,inference.seed))==EXPECTED
    assert len(costs)==len(inference)==22
    stage_count=batch_count=visits=sample_count=0;minimum_bookkeeping=np.inf
    checkpoints=[]
    for row in costs.itertuples():
        run=directory/'training'/row.model/f'seed{row.seed}'
        config=read(run/'config.json');state=read(run/'status.json');stages=read(run/'stage_timing.json')
        assert state['status']=='completed' and state['mode']=='efficiency_only'
        assert config['model']==row.model and config['seed']==row.seed and config['n_memory']==912
        assert config['efficiency_only'] and not config['test_metrics_generated']
        assert config['data_hash']==protocol['dataset_sha256'] and config['normalization_hash']==protocol['normalization_sha256']
        assert config['benchmark_script_sha256']==protocol['benchmark_script_sha256']
        population=pd.read_csv(run/'training_population.csv')
        pids=population.dataset_index.to_numpy()
        np.testing.assert_array_equal(population.event_id,events.iloc[pids].event_id)
        assert (data['split'][pids]==0).all()
        ends=np.sort(data['end_day'][common])
        for stage,pop in population.groupby('stage',sort=False):
            ids=pop.dataset_index.to_numpy();gap=config['ratl_gap'] if stage=='residual' else 0
            expected_mature=np.searchsorted(ends,data['day'][ids]-gap,side='left')
            np.testing.assert_array_equal(pop.mature_common_candidates,expected_mature)
            if stage=='residual':np.testing.assert_array_equal(ids,common)
            elif stage=='main':np.testing.assert_array_equal(ids,np.flatnonzero(data['split']==0))
        if row.model=='HistoricalMean':
            assert not stages and row.training_seconds==0 and row.optimizer_steps==0
        elif row.model=='RATL':assert {s['stage'] for s in stages}=={'base','residual'}
        else:assert len(stages)==1 and stages[0]['stage']=='main'
        for stage in stages:
            name=stage['stage'];batch=jsonl(run/f'{name}_batches_samples.jsonl');val=jsonl(run/f'{name}_validation_samples.jsonl');cache=jsonl(run/f'{name}_cache_samples.jsonl')
            stage_count+=1;batch_count+=len(batch);visits+=sum(x['n_events'] for x in batch)
            assert all(np.isfinite(x['seconds']) and x['seconds']>0 for x in batch+val+cache)
            assert len(batch)==stage['optimizer_steps'] and len(val)==stage['epochs']
            assert {x['epoch'] for x in val}==set(range(1,stage['epochs']+1))
            population_ids=population.loc[population.stage.eq(name),'dataset_index'].to_numpy()
            assert len(population_ids)==stage['n_train']
            for epoch in range(1,stage['epochs']+1):
                batches=[b for b in batch if b['epoch']==epoch]
                ids=[i for b in batches for i in b['dataset_indices']]
                assert len(ids)==len(set(ids))==len(population_ids)
                assert set(ids)==set(population_ids)
                assert all(len(b['dataset_indices'])==b['n_events'] and 0<b['n_events']<=options['batch_size'] for b in batches)
            close(stage['optimization_seconds'],sum(x['seconds'] for x in batch))
            close(stage['validation_seconds'],sum(x['seconds'] for x in val))
            close(stage['training_cache_seconds'],sum(x['seconds'] for x in cache if x['phase']=='training'))
            close(stage['stage_wall_seconds']-stage['monitor_seconds'],stage['stage_seconds'])
            close(stage['optimization_seconds']+stage['validation_seconds']+stage['training_cache_seconds']+stage['bookkeeping_seconds'],stage['stage_seconds'])
            assert stage['bookkeeping_seconds']>=-1e-8
            minimum_bookkeeping=min(minimum_bookkeeping,stage['bookkeeping_seconds'])
            assert sum(x['n_events'] for x in batch)==stage['total_training_event_visits']==stage['epochs']*stage['n_train']
            history=pd.read_csv(run/('base_training_history.csv' if name=='base' else 'training_history.csv'))
            assert len(history)==stage['epochs']
            assert int(history.loc[history.validation_mae.idxmin(),'epoch'])==stage['best_epoch']
            close(history.validation_mae.min(),stage['validation_standardized_mae'])
        close(row.training_seconds,sum(s['stage_seconds'] for s in stages)+row.ratl_gamma_seconds)
        close(row.training_stage_wall_seconds,sum(s['stage_wall_seconds'] for s in stages)+row.ratl_gamma_seconds)
        close(row.training_monitor_seconds,sum(s['monitor_seconds'] for s in stages))
        for key in ['optimization_seconds','validation_seconds','training_cache_seconds']:
            close(getattr(row,key),sum(s[key] for s in stages))
        assert row.optimizer_steps==sum(s['optimizer_steps'] for s in stages)
        assert row.total_epochs==sum(s['epochs'] for s in stages)
        close(state['training_seconds'],row.training_seconds)
        if row.model=='RATL':
            gamma=pd.read_csv(run/'gamma_selection.csv')
            assert len(gamma)==11 and gamma.seconds.gt(0).all()
            assert row.ratl_gamma_seconds>=gamma.seconds.sum()+row.ratl_gamma_cache_seconds
            close(row.ratl_base_seconds,stages[0]['stage_seconds']);close(row.ratl_residual_seconds,stages[1]['stage_seconds'])
        if (run/'checkpoint.pt').exists():checkpoints.append({'path':str(run/'checkpoint.pt'),'sha256':digest(run/'checkpoint.pt')})
    for row in inference.itertuples():
        path=directory/'batch1'/row.model/f'seed{row.seed}'
        summary=read(path/'summary.json')
        sample_count+=validate_samples(path/'samples.csv',summary,queries,options)
        assert summary['checkpoint_role']=='original_main_checkpoint' and summary['common_memory_events']==912
        source=root/'results/suite_expanded/main'/row.model/f'seed{row.seed}'
        if row.model!='HistoricalMean':assert summary['checkpoint_sha256']==digest(source/'checkpoint.pt')
        for key in ['inference_event_ms','canonical_memory_mb','parameters']:
            close(summary[key],getattr(row,key))
    summary_check(directory/'summary.csv',{'same_capacity_retraining':costs,'main_checkpoint_batch1':inference},['measurement','model','metric'])
    monitoring=gpu_checks(directory,options['idle_gpus'])
    assert stage_count==24
    assert sample_count==22*len(queries)*options['repeats']
    report={'status':'passed','model_seed_runs':22,'training_stages':stage_count,
            'optimizer_batch_samples':batch_count,'training_event_visits':visits,
            'batch1_samples':sample_count,'batch1_queries':len(queries),'repeats':options['repeats'],
            'common_memory_events':len(common),'minimum_bookkeeping_seconds':float(minimum_bookkeeping),
            'accuracy_recomputed':False,'dataset_sha256':protocol['dataset_sha256'],
            'normalization_sha256':protocol['normalization_sha256'],**monitoring}
    return report,checkpoints


def aligned(root, directory=None, serial_directory=None, verify_code=True):
    directory=directory or root/'results/efficiency_aligned_expanded'
    source=serial_directory or root/'results/serial_training_expanded'
    protocol,status=read(directory/'protocol.json'),read(directory/'status.json')
    assert status['status']=='completed' and status['completed_runs']==22 and status['no_observed_contention'] is True
    assert status['same_checkpoint_accuracy_training_and_inference'] is True
    assert protocol['parameters_updated'] is protocol['normalization_updated'] is protocol['hyperparameters_selected'] is False
    source_protocol=read(source/'protocol.json')
    assert protocol['source_protocol_sha256']==digest(source/'protocol.json')
    assert protocol['source_training_timing_sha256']==digest(source/'training_timing.csv')
    if verify_code:
        check_code(root,source_protocol)
        assert protocol['evaluator_sha256']==digest(root/'evaluate_efficiency_outputs.py')
    data_path=root/'data/prepared_expanded';a=dict(np.load(data_path/'dataset.npz'));normal=read(data_path/'normalization.json');events=pd.read_csv(data_path/'events.csv')
    assert protocol['dataset_sha256']==digest(data_path/'dataset.npz') and protocol['normalization_sha256']==digest(data_path/'normalization.json')
    options=source_protocol['options'];common,queries=manifests(directory,a,events,options['query_count'])
    for key,file in [('common_memory_sha256','common_memory_manifest.csv'),('query_manifest_sha256','batch1_query_manifest.csv')]:assert protocol[key]==digest(directory/file)
    assert digest(directory/'common_memory_manifest.csv')==digest(source/'common_memory_manifest.csv')
    assert digest(directory/'batch1_query_manifest.csv')==digest(source/'batch1_query_manifest.csv')
    test=np.flatnonzero(a['split']==2);truth=a['y'][test].astype(float)*normal['y_std']+normal['y_mean']
    costs=pd.read_csv(source/'training_timing.csv').set_index(['model','seed']);timing=pd.read_csv(directory/'aligned_timing.csv')
    assert len(timing)==22 and set(zip(timing.model,timing.seed))==EXPECTED
    metric_count=sample_count=0;max_metric_error=max_cost_error=0.;checkpoints=[];metric_records=[]
    for row in timing.itertuples():
        run=directory/row.model/f'seed{row.seed}';training=source/'training'/row.model/f'seed{row.seed}'
        cfg=read(run/'evaluation_config.json');train_cfg=read(training/'config.json')
        assert cfg['same_checkpoint_accuracy_training_and_inference'] is True and cfg['accuracy_recomputed'] is True
        assert cfg['parameters_updated'] is False and cfg['replaces_main_results'] is False
        assert cfg['checkpoint_role']=='same_capacity_efficiency_checkpoint'
        assert cfg['inference_common_memory_events']==cfg['training_common_memory_events']==912
        assert train_cfg['data_hash']==protocol['dataset_sha256'] and train_cfg['normalization_hash']==protocol['normalization_sha256']
        assert train_cfg['model']==row.model and train_cfg['seed']==row.seed
        if row.model!='HistoricalMean':
            checkpoint=training/'checkpoint.pt';sha=digest(checkpoint)
            assert sha==cfg['checkpoint_sha256']==row.checkpoint_sha256
            assert checkpoint.stat().st_mtime<=protocol['started_unix']+1
            checkpoints.append({'path':str(checkpoint),'sha256':sha})
        pred=dict(np.load(run/'predictions.npz'));np.testing.assert_array_equal(pred['event_indices'],test)
        np.testing.assert_array_equal(pred['truth'],truth)
        assert pred['prediction'].shape==truth.shape==(2315,11,3) and np.isfinite(pred['prediction']).all()
        metrics=pd.read_csv(run/'horizon_metrics.csv').set_index(['channel','horizon','metric'])
        assert len(metrics)==30 and not metrics.index.duplicated().any()
        for index,channel in enumerate(['AR','turnover','log_volume']):
            for horizon in [1,3,5,10,11]:
                error=pred['prediction'][:,:horizon,index]-truth[:,:horizon,index]
                for metric,value in [('MAE',np.abs(error).mean()),('RMSE',np.sqrt(np.square(error).mean()))]:
                    difference=abs(metrics.loc[(channel,horizon,metric),'value']-value)
                    assert difference<1e-12;max_metric_error=max(max_metric_error,float(difference));metric_count+=1
                    metric_records.append({'model':row.model,'seed':row.seed,'channel':channel,'horizon':horizon,'metric':metric,'value':value})
        # evaluate_efficiency_outputs computes this in float32 normalized units.
        standardized=((pred['prediction']-normal['y_mean'])/normal['y_std']).astype(np.float32)
        expected_mae=float(np.abs(standardized-a['y'][test]).mean())
        np.testing.assert_allclose(cfg['test_standardized_mae'],expected_mae,atol=2e-7,rtol=1e-6)
        for key,value in costs.loc[(row.model,row.seed)].items():
            if isinstance(value,(float,int,np.number)) and not isinstance(value,(bool,np.bool_)):
                difference=abs(float(cfg[key])-float(value));assert difference<1e-10
                max_cost_error=max(max_cost_error,difference)
        sample_count+=validate_samples(run/'batch1_samples.csv',cfg,queries,options)
        for key in ['inference_event_ms','retrieval_event_ms','canonical_memory_mb','training_seconds','test_standardized_mae']:
            if not pd.isna(cfg[key]):close(cfg[key],getattr(row,key))
    computed=pd.DataFrame(metric_records)
    keys=['model','channel','horizon','metric']
    expected=computed.groupby(keys).value.agg(['count','mean','std']).sort_index()
    saved=pd.read_csv(directory/'accuracy_summary.csv').set_index(keys).sort_index()
    np.testing.assert_array_equal(saved.n_seeds,expected['count']);close(saved['mean'],expected['mean'])
    np.testing.assert_allclose(saved['std'],expected['std'],atol=1e-12,rtol=1e-12,equal_nan=True)
    summary_check(directory/'summary.csv',timing,['model','metric'])
    monitoring=gpu_checks(directory,options['idle_gpus'])
    report={'status':'passed','aligned_runs':22,'test_events_per_run':len(test),'memory_events':912,
            'channel_horizon_metric_values_recomputed':metric_count,'maximum_metric_csv_roundtrip_error':max_metric_error,
            'maximum_cost_csv_roundtrip_error':max_cost_error,'batch1_samples':sample_count,
            'checkpoint_hashes_unchanged':True,'checkpoint_mtimes_precede_evaluation':True,
            'dataset_sha256':protocol['dataset_sha256'],'normalization_sha256':protocol['normalization_sha256'],**monitoring}
    return report,checkpoints


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['serial','aligned','all'],default='all')
    args=p.parse_args();root=args.root.resolve()
    for name,checker in [('serial_training_expanded',serial),('efficiency_aligned_expanded',aligned)]:
        if args.mode!='all' and not name.startswith('serial' if args.mode=='serial' else 'efficiency'):continue
        report,checkpoints=checker(root)
        for record in checkpoints:assert digest(record['path'])==record['sha256']
        assert digest(root/'data/prepared_expanded/dataset.npz')==report['dataset_sha256']
        assert digest(root/'data/prepared_expanded/normalization.json')==report['normalization_sha256']
        report['validator_sha256']=digest(Path(__file__))
        report['frozen_checkpoint_bytes_unchanged_during_validation']=True
        report['source_dataset_and_normalization_unchanged_during_validation']=True
        output=root/'results'/name/'independent_validation.json'
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'directory':name,**report}),flush=True)


if __name__=='__main__':main()
