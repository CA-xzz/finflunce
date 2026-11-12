"""Independent checks of common-target metrics, without training models."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from run_references import common_target_metrics


class CommonTargetMetricsTests(unittest.TestCase):
    def fixture(self, root, name="run", truth_offset=0.):
        dataset, run = root / "dataset", root / name
        dataset.mkdir(exist_ok=True)
        run.mkdir()
        pd.DataFrame({"event_id": ["train0", "test0", "test1"],
                      "event_date": ["2020-01-01", "2021-01-01", "2021-01-02"]}).to_csv(
            dataset / "events.csv", index=False)
        normal = {"y_mean": [10., -20., 1.], "y_std": [2., 5., .5]}
        (dataset / "normalization.json").write_text(json.dumps(normal))
        truth = np.arange(66, dtype=np.float64).reshape(2, 11, 3) / 13.
        error = (np.arange(1, 3)[:, None, None] * np.arange(1, 12)[None, :, None]
                 * np.arange(1, 4)[None, None, :] / 10.)
        error[:, 1::2] *= -1
        arrays = dict(event_indices=np.array([1, 2]), truth=truth + truth_offset,
                      prediction=truth + error)
        np.savez(run / "predictions.npz", **arrays)
        rows = [{"channel": channel, "horizon": horizon, "metric": metric,
                 "value": 987., "seed": 41, "n_events": 2, "n_dates": 2}
                for horizon in [1, 3, 5, 10, 11]
                for channel in ["AR", "turnover", "log_volume"] for metric in ["MAE", "RMSE"]]
        pd.DataFrame(rows).to_csv(run / "horizon_metrics.csv", index=False)
        targets = {"event_ids": np.array(["test0", "test1"]), "truth": truth}
        return dataset, run, targets, arrays

    def test_exact_common_truth_metrics_ignore_scheme_roundtrip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, first, targets, _ = self.fixture(root, "first", truth_offset=1e-6)
            _, second, _, _ = self.fixture(root, "second", truth_offset=-1e-6)
            originals = {path: (path / "horizon_metrics.csv").read_bytes() for path in [first, second]}
            first_metrics = common_target_metrics(dataset, first, targets)
            second_metrics = common_target_metrics(dataset, second, targets)
            pd.testing.assert_frame_equal(first_metrics, second_metrics)
            for row in first_metrics.itertuples():
                scale = {"AR": 1, "turnover": 2, "log_volume": 3}[row.channel] / 10.
                h = row.horizon
                expected = (1.5 * (h + 1) / 2. * scale if row.metric == "MAE" else
                            np.sqrt(2.5 * (h + 1) * (2 * h + 1) / 6.) * scale)
                self.assertAlmostEqual(row.value, expected, places=12)
                self.assertTrue(row.common_original_unit_target)
            for path in [first, second]:
                self.assertEqual((path / "horizon_metrics.csv").read_bytes(), originals[path])
                self.assertTrue((path / "common_target_horizon_metrics.csv").exists())

    def test_event_order_mismatch_fails(self):
        with TemporaryDirectory() as directory:
            dataset, run, targets, arrays = self.fixture(Path(directory))
            arrays["event_indices"] = np.array([2, 1])
            np.savez(run / "predictions.npz", **arrays)
            with self.assertRaises(AssertionError):
                common_target_metrics(dataset, run, targets)
            self.assertFalse((run / "common_target_horizon_metrics.csv").exists())

    def test_roundtrip_difference_outside_tolerance_fails(self):
        with TemporaryDirectory() as directory:
            dataset, run, targets, _ = self.fixture(Path(directory), truth_offset=.01)
            with self.assertRaisesRegex(ValueError, "beyond float32 rounding"):
                common_target_metrics(dataset, run, targets)
            self.assertFalse((run / "common_target_horizon_metrics.csv").exists())

    def test_nonfinite_values_fail(self):
        for location in ["canonical_truth", "run_truth", "prediction"]:
            with self.subTest(location=location), TemporaryDirectory() as directory:
                dataset, run, targets, arrays = self.fixture(Path(directory))
                if location == "canonical_truth":
                    targets["truth"][0, 0, 0] = np.nan
                else:
                    arrays["truth" if location == "run_truth" else "prediction"][0, 0, 0] = np.nan
                    np.savez(run / "predictions.npz", **arrays)
                with self.assertRaises(ValueError):
                    common_target_metrics(dataset, run, targets)
                self.assertFalse((run / "common_target_horizon_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
