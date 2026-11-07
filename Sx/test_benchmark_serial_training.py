"""CPU-only development checks; temporary synthetic data are not paper results."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import benchmark_serial_training as bench
import evaluate_extras as extras
import train as core
from models import build_model, legal_memory


def fixture(root):
    root = Path(root)
    prepared = root / "data"
    prepared.mkdir()
    rng = np.random.default_rng(51)
    days = np.array([0, 20, 40, 60, 80, 100, 150, 170, 190, 240, 260, 280])
    n = len(days)
    split = np.array([0] * 6 + [1] * 3 + [2] * 3)
    q = np.linspace(.3, .8, n).astype(np.float32)
    np.savez(prepared / "dataset.npz", x=rng.normal(size=(n, 10, 3)).astype(np.float32),
        y=rng.normal(size=(n, 11, 3)).astype(np.float32), q=q, platform=np.arange(n) % 2,
        influencer=np.arange(n) % 3, day=days, end_day=days + 10, split=split)
    events = pd.DataFrame(dict(event_id=["DEVELOPMENT_%02d" % i for i in range(n)],
        event_date=(pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")).astype(str),
        platform=np.arange(n) % 2, influencer=np.arange(n) % 3, q=q, label_version="DEVELOPMENT_SYNTHETIC_NOT_PAPER"))
    events.to_csv(prepared / "events.csv", index=False)
    core.write_json(prepared / "normalization.json", dict(y_mean=[0.] * 3, y_std=[1.] * 3,
        n_platforms=2, n_influencers=3, ready_for_formal_training=True))
    suite = root / "suite"
    selection = {}
    model_config = dict(length=10, features=3, horizon=11, channels=3, n_platforms=2, n_influencers=3,
        hidden=8, heads=2, layers=1, key_dim=4, k=2, kernel=3, dropout=.1, variant="full",
        retrieval="response", t_delta=1., t_z=.1, t_r=.2, lambda_q=1., raft_temperature=.1,
        ratl_block=8, ratl_temperature=1., ratl_teacher_temperature=1., ratl_gap=11)
    data, _, _ = core.load_data(prepared, torch.device("cpu"))
    all_train = (data["split"] == 0).nonzero().flatten()
    early, late, _ = core.partition_ratl(data, all_train)
    for name in bench.MODELS:
        if name != "HistoricalMean":
            selection[name] = dict(trial=0, config=dict(hidden=8, lr=.001), validation_mae=.7)
        for seed in ((41,) if name == "HistoricalMean" else bench.SEEDS):
            path = suite / "main" / name / ("seed%d" % seed)
            path.mkdir(parents=True)
            config = dict(model=name, seed=seed, epochs=2, patience=2, batch_size=4, hidden=8,
                lr=.001, weight_decay=.0001, lambda_ret=.1, huber_delta=1., variant="full", retrieval="response",
                ratl_base_fraction=.6, ratl_gap=11, ratl_teacher_weight=.4, memory_subset="all",
                n_memory=len(late) if name == "RATL" else len(all_train),
                n_train=len(late) if name == "RATL" else len(all_train),
                model_config=model_config, data_hash=core.sha256(prepared / "dataset.npz"),
                normalization_hash=core.sha256(prepared / "normalization.json"),
                code_hashes={file: core.sha256(Path(__file__).with_name(file)) for file in ["models.py", "train.py"]})
            core.write_json(path / "config.json", config)
            core.write_json(path / "status.json", dict(status="completed", mode="full", development_fixture=True))
            if name != "HistoricalMean":
                bench.seed_all(seed)
                model = build_model(name, **model_config)
                torch.save(dict(model_state=model.state_dict(), model_config=model_config), path / "checkpoint.pt")
    core.write_json(suite / "selected_configs.json", selection)
    options = bench.parse_args(["--data", str(prepared), "--suite", str(suite), "--output", str(root / "benchmark"),
        "--device", "cpu", "--expected-memory", "2", "--batch-size", "4", "--threads", "1",
        "--warmup", "1", "--repeats", "2", "--query-count", "3"])
    return options, data, events, model_config


class TimingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_instrumented_optimizer_loop_matches_frozen_core(self):
        with tempfile.TemporaryDirectory(prefix="Sx-timing-development-") as root:
            options, data, events, config = fixture(root)
            ids = (data["split"] == 0).nonzero().flatten()
            val = (data["split"] == 1).nonzero().flatten()
            memory = core.subset(data, ids[-2:])
            args = SimpleNamespace(epochs=2, patience=2, batch_size=4, lr=.001, weight_decay=.0001,
                timeout_seconds=60., lambda_ret=.1, huber_delta=1., ratl_teacher_weight=.4)
            for model_name in ("RAIF", "DLinear"):
                states = []
                for instrumented in (False, True):
                    args.output = Path(root) / (model_name + str(instrumented))
                    args.output.mkdir()
                    bench.seed_all(42)
                    model = build_model(model_name, **config)
                    with redirect_stdout(io.StringIO()):
                        if instrumented:
                            record = bench.timed_stage(model, data, ids, val, memory, args, config,
                                __import__("time").monotonic(), bench.QuietGPU(torch.device("cpu")), "main")
                            self.assertEqual(record["optimizer_steps"], 4)
                            self.assertEqual(record["total_training_event_visits"], 12)
                            raw = [json.loads(line) for line in (args.output / "main_batches_samples.jsonl").read_text().splitlines()]
                            for epoch in (1, 2):
                                seen = [i for row in raw if row["epoch"] == epoch for i in row["dataset_indices"]]
                                self.assertEqual(sorted(seen), ids.tolist())
                        else:
                            core.fit_stage(model, data, ids, val, memory, args, config, __import__("time").monotonic())
                    states.append({key: value.clone() for key, value in model.state_dict().items()})
                self.assertEqual(set(states[0]), set(states[1]))
                for key in states[0]:
                    torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_batch_one_has_no_query_outcomes_and_no_amortization(self):
        with tempfile.TemporaryDirectory(prefix="Sx-timing-development-") as root:
            _, data, _, config = fixture(root)
            model = build_model("RAFT", **config)
            memory = core.subset(data, torch.tensor([4, 5]))
            queries = torch.tensor([9, 10, 11])
            seen = []

            def check(module, args):
                seen.append(args[0]["index"].item())
                self.assertEqual(len(args[0]["x"]), 1)
                self.assertNotIn("y", args[0])
                self.assertTrue(bool(legal_memory(args[0], args[1]).all()))

            handle = model.register_forward_pre_hook(check)
            summary, samples = bench.batch_one_benchmark(model, data, memory, queries, 1, 2,
                bench.QuietGPU(torch.device("cpu")))
            handle.remove()
            self.assertEqual(len(seen), 7)
            self.assertEqual(summary["inference_batch_size"], 1)
            self.assertEqual(summary["inference_samples"], 6)
            self.assertAlmostEqual(summary["inference_event_ms"], np.mean([row["inference_ms"] for row in samples]))
            self.assertEqual([row["dataset_index"] for row in samples], [9, 10, 11] * 2)

    def test_complete_cpu_protocol_separates_provenance_and_never_scores_test(self):
        with tempfile.TemporaryDirectory(prefix="Sx-timing-development-") as root:
            options, _, _, _ = fixture(root)
            original = core.evaluate
            calls = []

            def only_validation(model, data, ids, *args, **kwargs):
                self.assertTrue(bool((data["split"][ids] == 1).all()))
                calls.append(model.__class__.__name__)
                return original(model, data, ids, *args, **kwargs)

            with patch.object(core, "evaluate", only_validation), redirect_stdout(io.StringIO()):
                bench.execute(options)
            status = json.loads((options.output / "status.json").read_text())
            self.assertEqual(status["training_runs"], 22)
            self.assertEqual(status["inference_runs"], 22)
            training = pd.read_csv(options.output / "training_timing.csv")
            inference = pd.read_csv(options.output / "batch1_timing.csv")
            self.assertTrue((training.training_common_memory_events == 2).all())
            self.assertTrue((inference.inference_batch_size == 1).all())
            self.assertEqual(training.loc[training.model == "HistoricalMean", "training_seconds"].item(), 0.)
            self.assertEqual(training.loc[training.model == "HistoricalMean", "optimization_seconds"].item(), 0.)
            aggregate = pd.read_csv(options.output / "summary.csv")
            self.assertTrue(aggregate.loc[aggregate.model == "HistoricalMean", "std"].isna().all())
            stages = json.loads((options.output / "training/RATL/seed41/stage_timing.json").read_text())
            self.assertEqual([row["stage"] for row in stages], ["base", "residual"])
            self.assertEqual([row["n_train"] for row in stages], [3, 2])
            self.assertTrue((training.loc[training.model == "RATL", "ratl_gamma_seconds"] > 0).all())
            self.assertFalse(list(options.output.rglob("predictions*")))
            self.assertFalse(list(options.output.rglob("*metrics.csv")))
            self.assertNotIn("test_standardized_mae", (options.output / "status.json").read_text())
            self.assertTrue(calls)
            with self.assertRaises(FileExistsError):
                bench.execute(options)

    def test_memory_and_configuration_mismatch_fail(self):
        with tempfile.TemporaryDirectory(prefix="Sx-timing-development-") as root:
            options, data, events, _ = fixture(root)
            runs = bench.discover_runs(options.suite, options.data)
            with self.assertRaisesRegex(ValueError, "expected 912"):
                bench.common_ids(runs, data, events)
            common = bench.common_ids(runs, data, events, 2)
            self.assertEqual(common.tolist(), [4, 5])
            ratl = next(config for _, config in runs if config["model"] == "RATL")
            with self.assertRaisesRegex(ValueError, "residual partition"):
                bench.training_partitions(ratl, data, common[:1])
            file = options.suite / "selected_configs.json"
            selected = json.loads(file.read_text())
            selected["RAFT"]["config"]["hidden"] = 16
            core.write_json(file, selected)
            with self.assertRaisesRegex(ValueError, "validation selection"):
                bench.discover_runs(options.suite, options.data)

    def test_exception_restores_original_training_functions(self):
        original_evaluate, original_adam = core.evaluate, torch.optim.AdamW
        model = build_model("RAFT", length=10, features=3, horizon=11, channels=3, n_platforms=2, n_influencers=3)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with bench.instrument_stage(model, torch.device("cpu"), bench.QuietGPU(torch.device("cpu")), "main"):
                raise RuntimeError("injected")
        self.assertIs(core.evaluate, original_evaluate)
        self.assertIs(torch.optim.AdamW, original_adam)
        self.assertNotIn("refresh_memory", vars(model))
        self.assertFalse(model._forward_pre_hooks)

    def test_companion_gpu_busy_rejects_even_without_visible_process_ids(self):
        guard = bench.QuietGPU(torch.device("cpu"))
        guard.index = "0"
        guard.idle_gpus = ("1",)
        record = dict(index=1, uuid="GPU-test", name="test", memory_used_mib=1000., utilization=90., pids=[])
        with patch.object(extras, "gpu_snapshot", return_value=record), self.assertRaisesRegex(RuntimeError, "GPU 1"):
            guard.check_companions("development")
        self.assertEqual(len(guard.idle_records), 1)


if __name__ == "__main__":
    unittest.main()
