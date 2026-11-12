"""Replay the nine incremental models used by the trajectory comparison."""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import train
import evaluate_extras as extra


def replay(root, device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_path = root / 'data/prepared_expanded'
    data, events, normal = train.load_data(data_path, device)
    ids = (data['split'] == 2).nonzero().flatten()
    records = []
    for name in ('ECT', 'RAFT', 'RAIF'):
        for seed in (41, 42, 43):
            directory = root / f'results/suite_expanded/main/{name}/seed{seed}'
            config = extra.read_run(directory, data_path)
            model = extra.load_checkpoint(directory, config, device)
            memory_ids = extra.memory_indices(directory, config, data, events)
            replay_ids, values, _, _, _ = extra.predict_fixed(model, data, events, memory_ids, config['batch_size'])
            torch.testing.assert_close(replay_ids, ids, rtol=0, atol=0)
            prediction = values.astype(np.float64) * np.asarray(normal['y_std']) + np.asarray(normal['y_mean'])
            with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['event_indices'], ids.cpu().numpy())
                np.testing.assert_array_equal(prediction, saved['prediction'])
            record = dict(run=directory.relative_to(root).as_posix(), exact_equal=True,
                          n_test=len(ids), max_raw_difference=0.,
                          data_sha256=train.sha256(data_path / 'dataset.npz'),
                          normalization_sha256=train.sha256(data_path / 'normalization.json'),
                          checkpoint_sha256=train.sha256(directory / 'checkpoint.pt'),
                          prediction_sha256=train.sha256(directory / 'predictions.npz'))
            records.append(record)
            print(record['run'], 'passed', flush=True)
    output = root / 'results/single_reference_revision/independent_predictions_validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    train.write_json(output, dict(status='passed', scope='Nine incremental models for trajectory comparison',
        original_device_replay=dict(device=device.type, original_batch_boundaries=True,
                                    query_y_omitted=True, runs=records)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    replay(args.root.resolve(), torch.device(args.device))
