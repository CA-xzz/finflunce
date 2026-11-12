"""Supplementary-evaluation checks on temporary development data only."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from models import build_model
from train import load_data, partition_ratl, sha256, subset
from evaluate_extras import (ExclusiveGPU, common_memory, execute, load_checkpoint,
                             memory_indices, predict_fixed, predict_rolling, process_identity, read_run)


def fixture(directory, names=("RAIF",)):
    directory = Path(directory)
    data_path = directory / "data"
    data_path.mkdir()
    rng = np.random.default_rng(9)
    days = np.array([0, 15, 30, 45, 60, 75, 90, 105,
                     150, 150, 160, 161, 170, 180,
                     220, 220, 230, 231, 240, 250])
    split = np.array([0] * 8 + [1] * 6 + [2] * 6)
    n = len(days)
    q = rng.uniform(.2, .9, n).astype(np.float32)
    np.savez(data_path / "dataset.npz", x=rng.normal(size=(n, 10, 3)).astype(np.float32),
             y=rng.normal(size=(n, 11, 3)).astype(np.float32), q=q,
             platform=np.arange(n) % 2, influencer=np.arange(n) % 3,
             day=days, end_day=days + 10, split=split)
    pd.DataFrame(dict(event_id=["DEVELOPMENT_%02d" % i for i in range(n)],
                      event_date=(pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")).astype(str),
                      platform=np.arange(n) % 2, influencer=np.arange(n) % 3,
                      q=q, label_version="DEVELOPMENT_SYNTHETIC_NOT_PAPER")).to_csv(data_path / "events.csv", index=False)
    (data_path / "normalization.json").write_text(json.dumps(dict(ready_for_formal_training=True,
        y_mean=[.1, .2, .3], y_std=[1., 2., 3.], n_platforms=2, n_influencers=3)))
    runs = []
    for name in names:
        directory_run = directory / name
        directory_run.mkdir()
        cfg = dict(length=10, features=3, horizon=11, channels=3, n_platforms=2, n_influencers=3,
                   hidden=16, heads=4, key_dim=8, layers=1, k=5, dropout=0., variant="full", retrieval="response")
        torch.manual_seed(12)
        model = build_model(name, **cfg)
        config = dict(model=name, model_config=cfg, variant="full", retrieval="response", seed=12,
                      data_hash=sha256(data_path / "dataset.npz"),
                      normalization_hash=sha256(data_path / "normalization.json"),
                      code_hashes={"models.py": sha256(Path(__file__).with_name("models.py"))})
        (directory_run / "config.json").write_text(json.dumps(config))
        (directory_run / "status.json").write_text(json.dumps({"status": "completed", "mode": "full"}))
        if name != "HistoricalMean":
            torch.save(dict(model_state=model.state_dict(), model_config=cfg, epoch=0), directory_run / "checkpoint.pt")
            pd.DataFrame(dict(epoch=[1, 2], seconds=[.1, .2])).to_csv(directory_run / "training_history.csv", index=False)
        if name == "RATL":
            pd.DataFrame(dict(epoch=[1, 2], seconds=[.3, .4])).to_csv(directory_run / "base_training_history.csv", index=False)
        runs.append(directory_run)
    return data_path, runs


def arguments(data, runs, output, mode):
    return SimpleNamespace(data=data, runs=runs, output=Path(output), mode=mode,
                           device="cpu", batch_size=64, warmup=1, repeats=2, memory_size=0, threads=1)


class SupplementaryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rolling_maturity_same_day_and_source_immutable(self):
        with tempfile.TemporaryDirectory(prefix="Sx-extras-development-") as temporary:
            data_path, runs = fixture(temporary)
            data, events, _ = load_data(data_path, torch.device("cpu"))
            original = {key: value.clone() for key, value in data.items()}
            config = read_run(runs[0], data_path)
            model = load_checkpoint(runs[0], config, torch.device("cpu"))
            initial = memory_indices(runs[0], config, data, events)
            ids, predicted, records, updates, days = predict_rolling(model, data, events, initial, 1)
            self.assertEqual(len(ids), 6)
            self.assertEqual(predicted.shape, (6, 11, 3))
            for update in updates:
                self.assertLess(update["end_day"], update["added_before_query_day"])
                self.assertIn(update["origin_split"], [1, 2])
            for record in records:
                self.assertLess(record["historical_end_day"], record["query_day"])
                self.assertNotEqual(record["query_id"], record["historical_id"])
            at_220 = next(day for day in days if day["query_day"] == 220)
            at_230 = next(day for day in days if day["query_day"] == 230)
            at_231 = next(day for day in days if day["query_day"] == 231)
            self.assertEqual(at_220["mature_validation"], 6)
            self.assertEqual(at_230["mature_test"], 0)
            self.assertEqual(at_231["mature_test"], 2)
            for key in data:
                torch.testing.assert_close(data[key], original[key], rtol=0, atol=0)
            other = predict_rolling(model, data, events, initial, 64)
            np.testing.assert_allclose(predicted, other[1], rtol=1e-5, atol=1e-6)

    def test_future_or_query_label_cannot_change_earlier_prediction(self):
        with tempfile.TemporaryDirectory(prefix="Sx-extras-development-") as temporary:
            data_path, runs = fixture(temporary)
            data, events, _ = load_data(data_path, torch.device("cpu"))
            config = read_run(runs[0], data_path)
            model = load_checkpoint(runs[0], config, torch.device("cpu"))
            initial = memory_indices(runs[0], config, data, events)
            original = predict_rolling(model, data, events, initial, 64)
            changed = {key: value.clone() for key, value in data.items()}
            changed["y"][-1] += 1000  # This event never matures before another query.
            second = predict_rolling(model, changed, events, initial, 64)
            np.testing.assert_allclose(original[1], second[1], rtol=0, atol=0)
            changed["y"][14:16] += 1000
            third = predict_rolling(model, changed, events, initial, 64)
            np.testing.assert_allclose(original[1][:3], third[1][:3], rtol=0, atol=0)

    def test_frozen_strategy_exports_are_distinct_and_source_preserved(self):
        with tempfile.TemporaryDirectory(prefix="Sx-extras-development-") as temporary:
            data_path, runs = fixture(temporary)
            before = sha256(runs[0] / "checkpoint.pt")
            args = arguments(data_path, runs, Path(temporary) / "frozen", "frozen")
            execute(args)
            table = pd.read_csv(args.output / "metrics.csv")
            self.assertEqual(set(table.retrieval), {"response", "state", "random", "reverse"})
            self.assertEqual(set(table.evaluation_kind), {"frozen_retrieval_replacement"})
            self.assertEqual(set(table.memory_mode), {"fixed"})
            self.assertEqual(before, sha256(runs[0] / "checkpoint.pt"))
            response_path = next(args.output.glob("*_response/predictions.npz"))
            data, events, normal = load_data(data_path, torch.device("cpu"))
            config = read_run(runs[0], data_path)
            model = load_checkpoint(runs[0], config, torch.device("cpu"))
            initial = memory_indices(runs[0], config, data, events)
            expected = predict_fixed(model, data, events, initial, 64)[1]
            expected = expected * np.array(normal["y_std"]) + np.array(normal["y_mean"])
            np.testing.assert_allclose(np.load(response_path)["prediction"], expected, rtol=1e-6)

    def test_rolling_output_metadata_and_rows(self):
        with tempfile.TemporaryDirectory(prefix="Sx-extras-development-") as temporary:
            data_path, runs = fixture(temporary)
            args = arguments(data_path, runs, Path(temporary) / "rolling", "rolling")
            execute(args)
            table = pd.read_csv(args.output / "metrics.csv")
            self.assertEqual(set(table.memory_mode), {"rolling"})
            self.assertEqual(set(table.evaluation_kind), {"frozen_encoder_rolling_memory"})
            predictions = pd.read_csv(next(args.output.glob("*/predictions.csv")))
            self.assertEqual(len(predictions), 6 * 11 * 3)
            config = json.loads(next(args.output.glob("*/evaluation_config.json")).read_text())
            self.assertFalse(config["parameters_updated"])

    def test_timing_eight_models_common_memory_no_accuracy(self):
        with tempfile.TemporaryDirectory(prefix="Sx-extras-development-") as temporary:
            names = ["RAIF", "RAFT", "RATL", "ECT", "DLinear", "PatchTST", "iTransformer", "HistoricalMean"]
            data_path, runs = fixture(temporary, names)
            args = arguments(data_path, runs, Path(temporary) / "timing", "timing")
            execute(args)
            table = pd.read_csv(args.output / "timing.csv")
            self.assertEqual(len(table), 8)
            self.assertEqual(table.common_memory_events.nunique(), 1)
            self.assertEqual(set(table.batch_size), {6})
            self.assertTrue((table.inference_batch_ms > 0).all())
            self.assertFalse(table.training_timing_exclusive.any())
            self.assertFalse(table.accuracy_recomputed.any())
            self.assertFalse(table.inference_exclusive_requested.any())
            ratl = table[table.model == "RATL"].iloc[0]
            self.assertAlmostEqual(ratl.logged_training_epoch_seconds, 1.0)
            self.assertGreater(ratl.retrieval_batch_ms, 0)
            self.assertFalse((args.output / "metrics.csv").exists())

    def test_ratl_cache_restores_saved_base_and_out_of_sample_memory(self):
        with tempfile.TemporaryDirectory(prefix="Sx-extras-development-") as temporary:
            data_path, runs = fixture(temporary, ["RATL"])
            data, events, _ = load_data(data_path, torch.device("cpu"))
            config = read_run(runs[0], data_path)
            model = load_checkpoint(runs[0], config, torch.device("cpu"))
            indices = memory_indices(runs[0], config, data, events)
            train_ids = (data["split"] == 0).nonzero().flatten()
            early, late, _ = partition_ratl(data, train_ids)
            self.assertTrue(torch.equal(indices, late))
            memory = subset(data, indices)
            cache = model.refresh_memory(memory)
            torch.testing.assert_close(cache["residuals"], memory["y"] - model.base(memory)["prediction"])
            self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
            self.assertFalse(set(early.tolist()) & set(late.tolist()))

    def test_common_memory_intersection(self):
        actual = common_memory([torch.tensor([1, 2, 4, 7]), torch.tensor([0, 2, 4, 5])])
        self.assertEqual(actual.tolist(), [2, 4])
        with self.assertRaises(ValueError):
            common_memory([torch.tensor([1]), torch.tensor([2])])

    def test_gpu_guard_counts_own_context_and_rejects_external_memory(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            guard = ExclusiveGPU(torch.device("cuda"))
        idle = dict(index=0, uuid="GPU-test", name="test", memory_used_mib=2., utilization=0., pids=[], observed_unix=0.)
        with patch("evaluate_extras.gpu_snapshot", return_value=idle):
            guard.initial()
        guard.context_mib = 300.
        own = dict(idle, memory_used_mib=450.)
        with patch("evaluate_extras.gpu_snapshot", return_value=own), patch("torch.cuda.memory_reserved", return_value=128 * 2**20):
            guard.check("own_context_and_reservation")
        observed = guard.records[-1]
        self.assertEqual(observed["actual_memory_mib"], 450.)
        self.assertEqual(observed["torch_reserved_mib"], 128.)
        self.assertEqual(observed["expected_memory_mib"], 428.)
        external = dict(idle, memory_used_mib=900.)
        with patch("evaluate_extras.gpu_snapshot", return_value=external), patch("torch.cuda.memory_reserved", return_value=128 * 2**20):
            with self.assertRaises(RuntimeError):
                guard.check("external_allocation")
        with patch("evaluate_extras.gpu_snapshot", return_value=dict(idle, utilization=90.)):
            with self.assertRaises(RuntimeError):
                guard.initial()

    def test_process_identity_reads_host_and_container_nspid(self):
        with patch("os.getpid", return_value=50123), patch("pathlib.Path.read_text", return_value="Name:\tpython\nNSpid:\t4032108\t50123\n"):
            identity = process_identity()
        self.assertEqual(identity["local_pid"], 50123)
        self.assertEqual(identity["namespace_pids"], [4032108, 50123])
        self.assertEqual(set(identity["pid_candidates"]), {4032108, 50123})
        with patch("os.getpid", return_value=50123), patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
            self.assertEqual(process_identity()["pid_candidates"], [50123])

    def test_gpu_guard_infers_host_pid_after_real_library_warmup(self):
        identity = dict(local_pid=50123, namespace_pids=[50123], pid_candidates=[50123])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), patch("evaluate_extras.process_identity", return_value=identity):
            guard = ExclusiveGPU(torch.device("cuda"))
        idle = dict(index=0, uuid="GPU-test", name="test", memory_used_mib=2., utilization=0., pids=[], observed_unix=0.)
        warmed = dict(idle, memory_used_mib=460., pids=[4032108])
        with patch("evaluate_extras.gpu_snapshot", side_effect=[idle, warmed]), patch("evaluate_extras.warm_cuda_libraries") as warm, patch("torch.cuda.memory_reserved", return_value=0):
            guard.initial()
            guard.establish()
        warm.assert_called_once_with(torch.device("cuda"))
        self.assertEqual(guard.context_mib, 460.)
        self.assertEqual(guard.owned_reported_pid, 4032108)
        self.assertTrue(guard.ownership_method.startswith("inferred_unique_new_pid"))
        self.assertIn("does not prove", guard.records[-1]["exclusivity_limitation"])
        active = dict(warmed, memory_used_mib=588.)
        with patch("evaluate_extras.gpu_snapshot", return_value=active), patch("torch.cuda.memory_reserved", return_value=128 * 2**20):
            guard.check("own_host_pid_and_lazy_allocations")
        self.assertEqual(guard.records[-1]["expected_memory_mib"], 588.)
        self.assertEqual(guard.records[-1]["additional_reported_pids"], [])
        with patch("evaluate_extras.gpu_snapshot", return_value=dict(active, pids=[4032108, 4032110])), patch("torch.cuda.memory_reserved", return_value=128 * 2**20):
            with self.assertRaisesRegex(RuntimeError, "contention"):
                guard.check("additional_host_pid")
        self.assertEqual(guard.records[-1]["additional_reported_pids"], [4032110])

    def test_gpu_guard_prefers_nspid_and_rejects_ambiguous_processes(self):
        identity = dict(local_pid=50123, namespace_pids=[4032108, 50123], pid_candidates=[4032108, 50123])
        idle = dict(index=0, uuid="GPU-test", name="test", memory_used_mib=2., utilization=0., pids=[], observed_unix=0.)
        for pids, succeeds in [([4032108], True), ([4032108, 4032110], False), ([4032110, 4032111], False)]:
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), patch("evaluate_extras.process_identity", return_value=identity):
                guard = ExclusiveGPU(torch.device("cuda"))
            with patch("evaluate_extras.gpu_snapshot", side_effect=[idle, dict(idle, memory_used_mib=460., pids=pids)]), patch("evaluate_extras.warm_cuda_libraries"), patch("torch.cuda.memory_reserved", return_value=0):
                guard.initial()
                if succeeds:
                    guard.establish()
                    self.assertEqual(guard.ownership_method, "proc_self_status_NSpid")
                    self.assertEqual(guard.owned_reported_pid, 4032108)
                else:
                    with self.assertRaisesRegex(RuntimeError, "Another GPU process"):
                        guard.establish()

    def test_gpu_guard_keeps_initial_idle_and_empty_table_checks(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            guard = ExclusiveGPU(torch.device("cuda"))
        idle = dict(index=0, uuid="GPU-test", name="test", memory_used_mib=2., utilization=0., pids=[], observed_unix=0.)
        for busy in [dict(idle, pids=[4032108]), dict(idle, utilization=1.), dict(idle, memory_used_mib=65.)]:
            with patch("evaluate_extras.gpu_snapshot", return_value=busy):
                with self.assertRaisesRegex(RuntimeError, "not idle"):
                    guard.initial()
        with patch("evaluate_extras.gpu_snapshot", side_effect=[idle, dict(idle, memory_used_mib=460.)]), patch("evaluate_extras.warm_cuda_libraries"), patch("torch.cuda.memory_reserved", return_value=0):
            guard.initial()
            guard.establish()
        self.assertIsNone(guard.owned_reported_pid)
        self.assertIn("process_table_empty", guard.ownership_method)
        with patch("evaluate_extras.gpu_snapshot", return_value=dict(idle, memory_used_mib=460., pids=[4032110])), patch("torch.cuda.memory_reserved", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "contention"):
                guard.check("unattributed_process_after_calibration")

    def test_mismatched_data_hash_rejected(self):
        with tempfile.TemporaryDirectory(prefix="Sx-extras-development-") as temporary:
            data_path, runs = fixture(temporary)
            config_path = runs[0] / "config.json"
            config = json.loads(config_path.read_text())
            config["data_hash"] = "different"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "different dataset"):
                read_run(runs[0], data_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
