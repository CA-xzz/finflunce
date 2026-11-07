"""Checks for incomplete results, fixed targets, and seed-level aggregation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from summarize_supplementary import (Report, aggregate_metrics, metric_rows, run,
                                    verify_frozen_response, digest, validate_reference_plan)


def fixture(root):
    path = root / "data/prepared_expanded"
    path.mkdir(parents=True)
    arrays = dict(x=np.ones((6, 10, 3), dtype=np.float32),
                  y=np.arange(198, dtype=np.float32).reshape(6, 11, 3) / 200,
                  split=np.array([0, 0, 1, 2, 2, 2]), day=np.arange(6) * 15,
                  end_day=np.arange(6) * 15 + 10, q=np.ones(6),
                  platform=np.ones(6, dtype=np.int64), influencer=np.ones(6, dtype=np.int64))
    np.savez(path / "dataset.npz", **arrays)
    pd.DataFrame(dict(event_id=[f"event{i}" for i in range(6)],
        event_date=pd.date_range("2020-01-01", periods=6).astype(str),
        label_version="DEVELOPMENT_TEST_ONLY")).to_csv(path / "events.csv", index=False)
    (path / "normalization.json").write_text(json.dumps(dict(y_mean=[.1, 1., 2.], y_std=[.2, 3., 5.],
        label_version="DEVELOPMENT_TEST_ONLY", ready_for_formal_training=True,
        q_rule="exp_only", quality_neff_factor_enabled=False)))
    return path


class SupplementarySummaryTests(unittest.TestCase):
    def test_previous_quality_rule_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = fixture(root)
            original = json.loads((path / "normalization.json").read_text())
            for field, old_value in [("q_rule", "exp_times_effective_n"), ("quality_neff_factor_enabled", True)]:
                changed = dict(original, **{field: old_value})
                (path / "normalization.json").write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "exponential-only quality rule"):
                    Report(root, root / "report").dataset(path)

    def test_reference_plan_requires_only_three_schemes(self):
        plan = pd.DataFrame([dict(group=group, scheme=scheme, variant="full")
                             for group in ["full_training", "intersect_training"]
                             for scheme in ["strict", "nearest", "expanded_nearest"]])
        self.assertIs(validate_reference_plan(plan), plan)
        extra = pd.concat([plan, pd.DataFrame([dict(group="full_training", scheme="unrequested", variant="full")])], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unexpected schemes"):
            validate_reference_plan(extra)
        changed = plan.copy()
        changed.loc[0, "variant"] = "no_quality"
        with self.assertRaisesRegex(ValueError, "six single-reference"):
            validate_reference_plan(changed)

    def test_missing_experiments_are_incomplete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = run(root, root / "report")
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(set(result["remaining_experiments"]), {"suite", "parameters", "references", "rolling", "frozen_retrieval", "serial_timing", "raft_raw_history"})
            self.assertEqual(result["actual_training_runs"], 0)
            self.assertFalse((root / "results").exists())
            self.assertTrue((root / "report/SUPPLEMENTARY.md").exists())

    def test_rmse_computed_per_seed_before_mean(self):
        frames = []
        truth = np.zeros((2, 11, 3))
        for seed, magnitude in zip([41, 42, 43], [1., 2., 3.]):
            frame = metric_rows(truth, truth + magnitude, ["d0", "d1"])
            for key, value in dict(experiment="parameters", comparison="K=5", model="RAIF", variant="full", retrieval="response", target_set="main_test", seed=seed).items():
                frame[key] = value
            frames.append(frame)
        result = aggregate_metrics(pd.concat(frames, ignore_index=True))
        np.testing.assert_allclose(result["mean"], 2.)
        np.testing.assert_allclose(result["std"], 1.)
        self.assertTrue(result.status.eq("completed").all())
        self.assertNotAlmostEqual(result.loc[result.metric == "RMSE", "mean"].iloc[0], np.sqrt(14 / 3))
        incomplete = aggregate_metrics(pd.concat(frames[:2], ignore_index=True))
        self.assertTrue(incomplete.status.eq("incomplete").all())

    def test_reference_canonical_targets_and_saved_metrics_are_checked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = fixture(root)
            report = Report(root, root / "report")
            data = report.dataset(dataset_path)
            canonical = data["truth"] + 1e-7
            output = root / "run"
            output.mkdir()
            prediction = data["truth"] + .1
            np.savez(output / "predictions.npz", event_indices=data["test_indices"], truth=data["truth"], prediction=prediction)
            config = dict(data=str(dataset_path), model="RAIF", seed=41, variant="full", retrieval="response",
                          data_hash=digest(dataset_path / "dataset.npz"), normalization_hash=digest(dataset_path / "normalization.json"))
            (output / "config.json").write_text(json.dumps(config))
            (output / "status.json").write_text(json.dumps(dict(status="completed", mode="full", validation_standardized_mae=.1)))
            dates = data["events"].event_date.to_numpy()[data["test_indices"]]
            for filename, truth in [("horizon_metrics.csv", data["truth"]), ("common_target_horizon_metrics.csv", canonical)]:
                metrics = metric_rows(truth, prediction, dates).assign(model="RAIF", seed=41, common_original_unit_target=True)
                metrics.to_csv(output / filename, index=False)
            result = report.training_run("references", "test", output, "RAIF", 41, target_name="strict_common_test", canonical=(data["test_ids"], canonical))
            self.assertIsNotNone(result)
            self.assertEqual(report.inventory[0]["status"], "completed")
            np.testing.assert_allclose(report.frames[0].value, metric_rows(canonical, prediction, dates).value)
            wrong = pd.read_csv(output / "common_target_horizon_metrics.csv")
            wrong.loc[0, "value"] += .01
            wrong.to_csv(output / "common_target_horizon_metrics.csv", index=False)
            failing = Report(root, root / "bad_report")
            result = failing.training_run("references", "test", output, "RAIF", 41, target_name="strict_common_test", canonical=(data["test_ids"], canonical))
            self.assertIsNone(result)
            self.assertEqual(failing.inventory[0]["status"], "incomplete")
            self.assertFalse(failing.frames)

    def test_prediction_identity_and_finiteness(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = Report(root, root / "report")
            data = report.dataset(fixture(root))
            output = root / "run"
            output.mkdir()
            for change in ["indices", "truth", "prediction"]:
                arrays = dict(event_indices=data["test_indices"].copy(), truth=data["truth"].copy(), prediction=data["truth"].copy())
                if change == "indices":
                    arrays["event_indices"] = arrays["event_indices"][::-1]
                else:
                    arrays[change][0, 0, 0] = np.nan
                np.savez(output / "predictions.npz", **arrays)
                with self.subTest(change=change), self.assertRaises((ValueError, AssertionError)):
                    report.prediction(output, data)

    def test_frozen_response_requires_same_predictions_at_channel_scale(self):
        baseline = np.ones((2, 11, 3))
        normal = dict(y_mean=[.1, 1., 2.], y_std=[.2, 3., 5.])
        verify_frozen_response(baseline + 1e-7, baseline, normal)
        changed = baseline.copy()
        changed[1, 5, 0] += .001
        with self.assertRaisesRegex(ValueError, "Frozen response predictions differ"):
            verify_frozen_response(changed, baseline, normal)
        changed[1, 5, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid frozen response"):
            verify_frozen_response(changed, baseline, normal)


if __name__ == "__main__":
    unittest.main()
