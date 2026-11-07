"""CPU-only auxiliary-RAFT checks using temporary synthetic development data."""
from contextlib import redirect_stderr, redirect_stdout
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

import train as core
from models import RAFT, legal_memory
from train_raft_raw_history import (EXPERIMENT, QUERY_FIELDS, RawHistoryRAFT, argv_context,
                                   execute, frozen_arguments, load_training_raw_history,
                                   parse_args, restore_checkpoint)


def fixture(root):
    root = Path(root)
    data_path = root / "data"
    data_path.mkdir()
    rng = np.random.default_rng(16)
    n = 12
    split = np.array([0] * 6 + [1] * 3 + [2] * 3)
    days = np.array([0, 20, 40, 60, 80, 100, 150, 170, 190, 240, 260, 280])
    delta = rng.normal(size=(n, 11, 3)) * np.array([.03, 2., .7])
    baseline = rng.normal(size=(n, 11, 3)) * np.array([.01, .5, .1]) + np.array([0., 7., 15.])
    raw_y = baseline + delta
    mean, std = delta[:6].mean((0, 1)), delta[:6].std((0, 1))
    q = np.linspace(.3, .8, n).astype(np.float32)
    np.savez(data_path / "dataset.npz", x=rng.normal(size=(n, 10, 3)).astype(np.float32),
             y=((delta - mean) / std).astype(np.float32), q=q, platform=np.arange(n) % 2,
             influencer=np.arange(n) % 3, day=days, end_day=days + 10, split=split)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")
    end_dates = dates + pd.Timedelta(days=10)
    events = pd.DataFrame(dict(event_id=["DEVELOPMENT_%02d" % i for i in range(n)], event_date=dates.astype(str),
                              end_date=end_dates.astype(str), platform=np.arange(n) % 2, influencer=np.arange(n) % 3,
                              q=q, label_version="DEVELOPMENT_SYNTHETIC_NOT_PAPER"))
    events.to_csv(data_path / "events.csv", index=False)
    normal = dict(y_mean=mean.tolist(), y_std=std.tolist(), n_platforms=2, n_influencers=3,
                  ready_for_formal_training=True)
    core.write_json(data_path / "normalization.json", normal)
    labels = pd.DataFrame(dict(event_id=np.repeat(events.event_id, 33), tau=np.tile(np.repeat(np.arange(11), 3), n),
        channel=np.tile(core.CHANNELS, n * 11), y=raw_y.reshape(-1), baseline=baseline.reshape(-1), delta=delta.reshape(-1),
        label_version="DEVELOPMENT_SYNTHETIC_NOT_PAPER", target_available_at=np.repeat((end_dates + pd.Timedelta(days=1)).astype(str), 33)))
    labels.to_csv(data_path / "labels.csv", index=False)
    suite = root / "suite"
    main = suite / "main/RAFT/seed41"
    tuning = suite / "tuning/RAFT/trial0"
    main.mkdir(parents=True)
    tuning.mkdir(parents=True)
    with argv_context(["train.py", "--data", str(data_path), "--output", str(main), "--model", "RAFT",
                       "--device", "cpu", "--epochs", "2", "--patience", "2", "--hidden", "8", "--timing-repeats", "1", "--threads", "1"]):
        args = core.parse_args()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(data_hash=core.sha256(data_path / "dataset.npz"), normalization_hash=core.sha256(data_path / "normalization.json"),
                  code_hashes={name: core.sha256(Path(__file__).with_name(name)) for name in ["models.py", "train.py"]})
    core.write_json(main / "config.json", config)
    core.write_json(main / "status.json", dict(status="completed", mode="full", test_standardized_mae=123456.))
    core.write_json(tuning / "config.json", dict(config, output=str(tuning), seed=40, validation_only=True))
    core.write_json(tuning / "status.json", dict(status="completed", mode="validation_only"))
    core.write_json(suite / "selected_configs.json", {"RAFT": {"trial": 0, "config": {"hidden": 8, "lr": .001}, "validation_mae": .9}})
    options = SimpleNamespace(data=data_path, output=root / "auxiliary", main_run=main, selection=None,
                              labels=None, seed=41, device="cpu", threads=1)
    return options, raw_y


def model_config():
    return dict(length=10, features=3, horizon=11, channels=3, n_platforms=2, n_influencers=3,
                hidden=8, k=3, raft_temperature=.1)


class RawHistoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def load(self, options):
        data, events, normal = core.load_data(options.data, torch.device("cpu"))
        ids, values, metadata = load_training_raw_history(options.data / "labels.csv", data, events, normal)
        return data, events, normal, ids, values, metadata

    def test_training_only_scaling_and_no_raw_outcomes_in_query_data(self):
        with tempfile.TemporaryDirectory(prefix="Sx-raft-raw-development-") as directory:
            options, raw = fixture(directory)
            data, events, normal, ids, values, metadata = self.load(options)
            np.testing.assert_allclose(metadata["raw_y_mean"], raw[:6].mean((0, 1)), rtol=0, atol=1e-14)
            np.testing.assert_allclose(metadata["raw_y_std"], raw[:6].std((0, 1)), rtol=0, atol=1e-14)
            self.assertEqual(metadata["fit_dataset_indices"], list(range(6)))
            self.assertEqual(set(data), {"x", "y", "q", "platform", "influencer", "day", "end_day", "split", "index"})
            labels = pd.read_csv(options.data / "labels.csv")
            external = ~labels.event_id.isin(events.event_id.iloc[:6])
            labels["y"] = labels.y.astype(object)
            labels.loc[external, "y"] = "UNREADABLE_VALIDATION_OR_TEST_RAW_Y"
            labels.to_csv(options.data / "labels.csv", index=False)
            ids2, values2, metadata2 = load_training_raw_history(options.data / "labels.csv", data, events, normal)
            torch.testing.assert_close(values, values2, rtol=0, atol=0)
            self.assertEqual(metadata["raw_y_mean"], metadata2["raw_y_mean"])
            self.assertTrue(torch.equal(ids, ids2))

    def test_same_parameter_structure_and_only_memory_values_replaced(self):
        with tempfile.TemporaryDirectory(prefix="Sx-raft-raw-development-") as directory:
            options, _ = fixture(directory)
            data, _, _, ids, values, _ = self.load(options)
            torch.manual_seed(88)
            baseline = RAFT(**model_config()).eval()
            torch.manual_seed(88)
            wrapper = RawHistoryRAFT(ids, values, **model_config()).eval()
            self.assertEqual(list(baseline.state_dict()), list(wrapper.state_dict()))
            self.assertEqual(sum(p.numel() for p in baseline.parameters()), sum(p.numel() for p in wrapper.parameters()))
            for key in baseline.state_dict():
                torch.testing.assert_close(baseline.state_dict()[key], wrapper.state_dict()[key], rtol=0, atol=0)
            query = core.subset(data, torch.tensor([9, 10]))
            memory = core.subset(data, ids)
            expected = baseline({k: query[k] for k in QUERY_FIELDS}, dict(memory, y=values))["prediction"]
            actual = wrapper(query, memory)["prediction"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            different_delta = dict(memory, y=memory["y"] + 10000)
            torch.testing.assert_close(wrapper(query, different_delta)["prediction"], actual, rtol=0, atol=0)

    def test_query_labels_removed_before_base_forward(self):
        with tempfile.TemporaryDirectory(prefix="Sx-raft-raw-development-") as directory:
            options, _ = fixture(directory)
            data, _, _, ids, values, _ = self.load(options)
            model = RawHistoryRAFT(ids, values, **model_config()).eval()
            query = core.subset(data, torch.tensor([9]))
            query["raw_y"] = torch.full((1, 11, 3), float("nan"))
            query["future_y"] = object()
            query["baseline"] = object()
            memory = core.subset(data, ids)
            original = RAFT.forward
            seen = []
            def checked(instance, inputs, history, cached_keys=None, labels=None):
                self.assertEqual(set(inputs), set(QUERY_FIELDS))
                self.assertIsNone(labels)
                seen.append(True)
                return original(instance, inputs, history, cached_keys, labels)
            with patch.object(RAFT, "forward", checked):
                actual = model(query, memory, labels=torch.full((1, 11, 3), float("nan")))["prediction"]
            self.assertEqual(seen, [True])
            self.assertTrue(torch.isfinite(actual).all())

    def test_strict_maturity_self_and_nontraining_memory(self):
        with tempfile.TemporaryDirectory(prefix="Sx-raft-raw-development-") as directory:
            options, _ = fixture(directory)
            data, _, _, ids, values, _ = self.load(options)
            model = RawHistoryRAFT(ids, values, **model_config()).eval()
            memory = core.subset(data, ids)
            query = core.subset(data, torch.tensor([1]))  # day20: only end10 is mature.
            result = model(query, memory)
            self.assertEqual(set(result["selected"][result["selected_valid"]].tolist()), {0})
            with torch.no_grad():
                model._history_values[1:] += 100000  # Self and future responses.
            torch.testing.assert_close(model(query, memory)["prediction"], result["prediction"], rtol=0, atol=0)
            at_boundary = dict(query, day=torch.tensor([10]), index=torch.tensor([20]))
            out = model(at_boundary, memory)
            self.assertFalse(out["selected_valid"].any())
            torch.testing.assert_close(out["prediction"], model(at_boundary, None)["prediction"], rtol=0, atol=0)
            same_identity = dict(query, index=torch.tensor([0]))
            self.assertFalse(model(same_identity, memory)["selected_valid"].any())
            external = dict(memory, split=memory["split"].clone())
            external["split"][0] = 1
            with self.assertRaisesRegex(ValueError, "only training"):
                model(query, external)

    def test_label_alignment_target_and_availability_validation(self):
        for defect in ["duplicate", "missing", "delta", "availability", "version"]:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory(prefix="Sx-raft-raw-development-") as directory:
                options, _ = fixture(directory)
                labels = pd.read_csv(options.data / "labels.csv")
                if defect == "duplicate": labels = pd.concat([labels, labels.iloc[:1]])
                elif defect == "missing": labels = labels.iloc[1:]
                elif defect == "delta": labels.loc[0, "delta"] += 5
                elif defect == "availability": labels.loc[0, "target_available_at"] = "2020-01-01"
                else: labels.loc[0, "label_version"] = "WRONG_VERSION"
                labels.to_csv(options.data / "labels.csv", index=False)
                with self.assertRaises(ValueError): self.load(options)

    def test_frozen_main_validation_selection_no_new_hyperparameter_flags(self):
        with tempfile.TemporaryDirectory(prefix="Sx-raft-raw-development-") as directory:
            options, _ = fixture(directory)
            args, metadata = frozen_arguments(options)
            self.assertEqual((args.hidden, args.lr, args.epochs, args.patience), (8, .001, 2, 2))
            self.assertFalse(metadata["hyperparameters_retuned"])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--data", "a", "--output", "b", "--main-run", "c", "--seed", "41", "--lr", ".5"])
            path = options.main_run / "config.json"
            config = json.loads(path.read_text()); config["hidden"] = 64; core.write_json(path, config)
            with self.assertRaisesRegex(ValueError, "frozen validation selection"):
                frozen_arguments(options)

    def test_cpu_training_exports_delta_and_checkpoint_restores(self):
        with tempfile.TemporaryDirectory(prefix="Sx-raft-raw-development-") as directory:
            options, raw_y = fixture(directory)
            original_load, original_build, original_write = core.load_data, core.build_model, core.write_json
            protected = [options.data / "dataset.npz", options.data / "labels.csv", Path(__file__).with_name("models.py"), Path(__file__).with_name("train.py")]
            hashes = {str(path): core.sha256(path) for path in protected}
            with redirect_stdout(io.StringIO()): execute(options)
            self.assertIs(core.load_data, original_load); self.assertIs(core.build_model, original_build); self.assertIs(core.write_json, original_write)
            self.assertEqual(hashes, {str(path): core.sha256(path) for path in protected})
            config = json.loads((options.output / "config.json").read_text())
            self.assertEqual(config["auxiliary_experiment"], EXPERIMENT)
            self.assertEqual(config["model"], EXPERIMENT)
            self.assertEqual(config["base_model"], "RAFT")
            self.assertTrue(config["checkpoint_requires_wrapper"])
            self.assertEqual(config["raw_history"]["labels_sha256"], core.sha256(options.data / "labels.csv"))
            metrics = pd.read_csv(options.output / "horizon_metrics.csv")
            self.assertEqual(set(metrics.model), {EXPERIMENT})
            saved = np.load(options.output / "predictions.npz")
            model, data, events, normal = restore_checkpoint(options.output, options.data)
            test_ids = (data["split"] == 2).nonzero().flatten(); train_ids = (data["split"] == 0).nonzero().flatten()
            truth = data["y"][test_ids].numpy().astype(np.float64) * np.array(normal["y_std"]) + np.array(normal["y_mean"])
            np.testing.assert_allclose(saved["truth"], truth, rtol=0, atol=0)
            self.assertFalse(np.allclose(saved["truth"], raw_y[9:]))
            _, prediction, _ = core.evaluate(model, data, test_ids, core.subset(data, train_ids), 64)
            prediction = prediction.astype(np.float64) * np.array(normal["y_std"]) + np.array(normal["y_mean"])
            np.testing.assert_allclose(saved["prediction"], prediction, rtol=0, atol=0)
            status_hash = core.sha256(options.output / "status.json")
            with self.assertRaises(FileExistsError): execute(options)
            self.assertEqual(status_hash, core.sha256(options.output / "status.json"))

    def test_failed_label_load_restores_scoped_training_functions(self):
        with tempfile.TemporaryDirectory(prefix="Sx-raft-raw-development-") as directory:
            options, _ = fixture(directory)
            (options.data / "labels.csv").unlink()
            previous = (core.load_data, core.build_model, core.write_json)
            with self.assertRaises(FileNotFoundError): execute(options)
            self.assertEqual(previous, (core.load_data, core.build_model, core.write_json))
            self.assertEqual(json.loads((options.output / "status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
