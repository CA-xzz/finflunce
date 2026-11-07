"""Local CPU checks for aligned efficiency evaluation, with synthetic fixtures."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

import benchmark_serial_training as bench
import evaluate_efficiency_outputs as aligned
import evaluate_extras as extras
import train as core
from models import build_model
from test_benchmark_serial_training import fixture


class AlignmentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def options(self, original):
        return aligned.parse_args(["--data", str(original.data), "--source", str(original.output),
            "--output", str(original.output.parent / "aligned"), "--device", "cpu", "--threads", "1",
            "--expected-memory", "2", "--batch-size", "4"])

    def test_full_alignment_same_checkpoint_cost_latency_and_original_units(self):
        with tempfile.TemporaryDirectory(prefix="Sx-aligned-development-") as root:
            original, data, events, _ = fixture(root)
            original.mode = "train"
            with redirect_stdout(io.StringIO()):
                bench.execute(original)
            options = self.options(original)
            checkpoint_hashes = {str(path): core.sha256(path) for path in original.output.rglob("checkpoint.pt")}
            with redirect_stdout(io.StringIO()):
                aligned.execute(options)
            status = json.loads((options.output / "status.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["completed_runs"], 22)
            times = pd.read_csv(options.output / "aligned_timing.csv")
            self.assertTrue(times.same_checkpoint_accuracy_training_and_inference.all())
            self.assertTrue((times.inference_batch_size == 1).all())
            self.assertTrue((times.inference_common_memory_events == 2).all())
            source = pd.read_csv(original.output / "training_timing.csv")
            np.testing.assert_allclose(times.training_seconds, source.training_seconds, rtol=0, atol=0)
            metrics = pd.read_csv(options.output / "horizon_metrics.csv")
            self.assertEqual(len(metrics), 22 * 5 * 3 * 2)
            self.assertEqual(set(metrics.horizon), {1, 3, 5, 10, 11})
            summary = pd.read_csv(options.output / "summary.csv")
            self.assertTrue(summary.loc[summary.model == "HistoricalMean", "std"].isna().all())
            for row in times.itertuples():
                path = options.output / row.model / ("seed%d" % row.seed)
                predicted = dict(np.load(path / "predictions.npz"))
                np.testing.assert_array_equal(predicted["event_indices"], [9, 10, 11])
                np.testing.assert_array_equal(predicted["truth"], data["y"][[9, 10, 11]].numpy())
                raw = pd.read_csv(path / "batch1_samples.csv")
                self.assertEqual(len(raw), 6)
                self.assertAlmostEqual(raw.inference_ms.mean(), row.inference_event_ms)
                checkpoint = original.output / "training" / row.model / ("seed%d" % row.seed) / "checkpoint.pt"
                if checkpoint.exists():
                    self.assertEqual(row.checkpoint_sha256, core.sha256(checkpoint))
                values = metrics[(metrics.model == row.model) & (metrics.seed == row.seed) &
                                 (metrics.horizon == 11) & (metrics.channel == "AR") & (metrics.metric == "MAE")]
                self.assertAlmostEqual(values.value.item(), np.abs(predicted["prediction"][:, :, 0] - predicted["truth"][:, :, 0]).mean())
            self.assertEqual(checkpoint_hashes, {str(path): core.sha256(path) for path in original.output.rglob("checkpoint.pt")})
            with self.assertRaises(FileExistsError):
                aligned.execute(options)

    def test_prediction_ignores_query_labels_and_never_passes_them_to_forward(self):
        with tempfile.TemporaryDirectory(prefix="Sx-aligned-development-") as root:
            _, data, _, config = fixture(root)
            model = build_model("RAIF", **config)
            memory = core.subset(data, torch.tensor([4, 5]))

            def check(module, args):
                self.assertNotIn("y", args[0])
                self.assertEqual(set(args[1]["split"].tolist()), {0})

            handle = model.register_forward_pre_hook(check)
            guard = bench.QuietGPU(torch.device("cpu"))
            ids, first = aligned.predict_test(model, data, memory, 2, guard, 60)
            changed = dict(data, y=data["y"].clone())
            changed["y"][ids] = 1e6
            _, second = aligned.predict_test(model, changed, memory, 2, guard, 60)
            handle.remove()
            np.testing.assert_array_equal(first, second)

    def test_manifest_identity_and_split_binding(self):
        with tempfile.TemporaryDirectory(prefix="Sx-aligned-development-") as root:
            original, data, events, _ = fixture(root)
            path = Path(root) / "manifest.csv"
            events.iloc[[4, 5]].assign(dataset_index=[4, 5]).to_csv(path, index=False)
            self.assertEqual(aligned.bound_manifest(path, data, events, 0).tolist(), [4, 5])
            changed = pd.read_csv(path)
            changed.loc[0, "event_id"] = events.event_id.iloc[0]
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "event IDs"):
                aligned.bound_manifest(path, data, events, 0)
            events.iloc[[9]].assign(dataset_index=[9]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "required split"):
                aligned.bound_manifest(path, data, events, 0)

    def test_original_unit_conversion(self):
        with tempfile.TemporaryDirectory(prefix="Sx-aligned-development-") as root:
            _, data, events, config = fixture(root)
            normal = dict(y_mean=[1., 2., 3.], y_std=[2., 3., 4.])
            ids = torch.tensor([9, 10, 11])
            output = Path(root) / "export"
            output.mkdir()
            run_config = dict(model="RAIF", seed=41, n_memory=2)
            predictions = np.zeros((3, 11, 3), dtype=np.float32)
            aligned.export_accuracy(output, ids, predictions, data, events, normal, run_config, "development")
            values = dict(np.load(output / "predictions.npz"))
            np.testing.assert_array_equal(values["truth"], data["y"][ids].numpy().astype(float) * normal["y_std"] + normal["y_mean"])
            np.testing.assert_array_equal(values["prediction"], np.broadcast_to(normal["y_mean"], (3, 11, 3)))


if __name__ == "__main__":
    unittest.main()
