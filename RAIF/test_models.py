"""Small deterministic development checks; these are not paper experiments."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from models import RAIF, RAFT, RATL, build_model, legal_memory
from train import objective, load_data, run, partition_ratl


def config():
    return dict(length=10, features=4, horizon=11, channels=3,
                n_platforms=3, n_influencers=7, hidden=16, heads=4,
                key_dim=8, k=5, dropout=0.0)


def fixtures():
    torch.manual_seed(17)
    memory = dict(x=torch.randn(7, 10, 4), y=torch.randn(7, 11, 3),
                  q=torch.tensor([0.1, 0.2, 0.7, 0.4, 0.9, 0.3, 0.6]),
                  platform=torch.tensor([0, 1, 2, 0, 1, 2, 0]),
                  influencer=torch.arange(7), day=torch.tensor([0, 3, 10, 15, 20, 30, 40]),
                  end_day=torch.tensor([10, 13, 20, 25, 30, 40, 50]),
                  index=torch.arange(7), split=torch.tensor([0, 0, 0, 0, 0, 1, 2]))
    query = dict(x=torch.randn(3, 10, 4), y=torch.randn(3, 11, 3), q=torch.tensor([0.2, 0.6, 0.9]),
                 platform=torch.tensor([0, 1, 2]), influencer=torch.tensor([2, 3, 4]),
                 day=torch.tensor([10, 20, 60]), end_day=torch.tensor([20, 30, 70]),
                 index=torch.tensor([7, 8, 9]), split=torch.tensor([0, 1, 2]))
    return query, memory


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_shapes_and_finite_backward(self):
        query, memory = fixtures()
        for name in ["RAIF", "ECT", "DLinear", "PatchTST", "iTransformer", "HistoricalMean", "RAFT", "RATL"]:
            with self.subTest(model=name):
                model = build_model(name, **config())
                result = model(query, memory, labels=query["y"])
                self.assertEqual(tuple(result["prediction"].shape), (3, 11, 3))
                self.assertTrue(torch.isfinite(result["prediction"]).all())
                if name != "HistoricalMean":
                    loss, _, _ = objective(result, query["y"], query["q"], model)
                    loss.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_maturity_no_self_and_train_partition(self):
        query, memory = fixtures()
        expected = torch.tensor([[False] * 7,
                                 [True, True, False, False, False, False, False],
                                 [True, True, True, True, True, False, False]])
        self.assertTrue(torch.equal(legal_memory(query, memory), expected))
        query["index"][2] = 0
        self.assertFalse(legal_memory(query, memory)[2, 0])
        model = RAIF(**config()).eval()
        output = model(query, memory)
        for row in range(3):
            actual = output["selected"][row][output["selected_valid"][row]]
            self.assertTrue(legal_memory(query, memory)[row, actual].all())

    def test_future_targets_and_states_cannot_affect_prediction(self):
        query, memory = fixtures()
        query = {key: value[1:2] for key, value in query.items()}
        model = RAIF(**config()).eval()
        first = model(query, memory)["prediction"]
        altered = {key: value.clone() for key, value in memory.items()}
        forbidden = ~legal_memory(query, memory)[0]
        altered["x"][forbidden] *= 1000
        altered["y"][forbidden] += 1000
        altered["q"][forbidden] = 1e-8
        query["y"] = query["y"] + 10000
        second = model(query, altered)["prediction"]
        torch.testing.assert_close(first, second, rtol=0, atol=1e-6)

    def test_zero_candidate_exact_direct_fallback(self):
        query, memory = fixtures()
        query = {key: value[:1] for key, value in query.items()}
        model = RAIF(**config()).eval()
        with_memory = model(query, memory, labels=query["y"])
        direct = model(query, None)["prediction"]
        torch.testing.assert_close(with_memory["prediction"], direct, rtol=0, atol=0)
        self.assertEqual(with_memory["retrieval_loss"].item(), 0)

    def test_query_target_changes_supervision_not_prediction(self):
        query, memory = fixtures()
        model = RAIF(**config()).eval()
        first = model(query, memory, labels=query["y"])
        second = model(query, memory, labels=query["y"] * 10)
        torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(first["retrieval_loss"], second["retrieval_loss"]))

    def test_quality_ablation_removes_all_three_uses(self):
        query, memory = fixtures()
        model = RAIF(**config(), variant="no_quality").eval()
        changed_memory = {key: value.clone() for key, value in memory.items()}
        changed_memory["q"] = torch.flip(memory["q"], dims=[0])
        first = model(query, memory, labels=query["y"])
        second = model(query, changed_memory, labels=query["y"])
        torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)
        torch.testing.assert_close(first["retrieval_loss"], second["retrieval_loss"], rtol=0, atol=0)
        a = objective(first, query["y"], query["q"], model)[0]
        b = objective(second, query["y"], torch.flip(query["q"], dims=[0]), model)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(model.lambda_q, 0)

    def test_zero_keys_and_each_ablation(self):
        query, memory = fixtures()
        for variant in RAIF.VARIANTS:
            with self.subTest(variant=variant):
                model = RAIF(**config(), variant=variant)
                with torch.no_grad():
                    model.key.weight.zero_()
                output = model(query, memory, labels=query["y"])
                loss, _, _ = objective(output, query["y"], query["q"], model)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                if variant in {"no_response", "no_retrieval"}:
                    self.assertEqual(output["retrieval_loss"].sum().item(), 0)

    def test_target_distribution_matches_formula_on_full_legal_set(self):
        query, memory = fixtures()
        model = RAIF(**config()).eval()
        _, z = model.encode(query)
        _, mz = model.encode(memory)
        mask = legal_memory(query, memory)
        actual = model.retrieval_loss(z, mz, mask, query["y"], memory)
        for row in [1, 2]:
            indices = mask[row].nonzero().flatten()
            distances = ((query["y"][row] - memory["y"][indices]) ** 2).mean((1, 2))
            target = (-distances / model.t_delta + memory["q"][indices].log()).softmax(0)
            predicted_log = (z[row] @ mz[indices].T / model.t_z).log_softmax(0)
            torch.testing.assert_close(actual[row], -(target * predicted_log).sum())

    def test_no_event_ablation_ignores_ids(self):
        query, memory = fixtures()
        model = RAIF(**config(), variant="no_events").eval()
        before = model(query, memory)["prediction"]
        query["platform"] = (query["platform"] + 1) % 3
        query["influencer"] = (query["influencer"] + 2) % 7
        memory["platform"] = (memory["platform"] + 1) % 3
        memory["influencer"] = (memory["influencer"] + 2) % 7
        torch.testing.assert_close(before, model(query, memory)["prediction"], rtol=0, atol=0)

    def test_random_retrieval_stable_and_legal(self):
        query, memory = fixtures()
        model = RAIF(**config(), retrieval="random").eval()
        first, second = model(query, memory), model(query, memory)
        self.assertTrue(torch.equal(first["selected"], second["selected"]))
        for row in range(3):
            selected = first["selected"][row][first["selected_valid"][row]]
            self.assertTrue(legal_memory(query, memory)[row, selected].all())

    def test_formal_readiness_gate(self):
        with tempfile.TemporaryDirectory(prefix="Sx-test-") as temporary:
            directory = Path(temporary)
            (directory / "normalization.json").write_text(json.dumps({"ready_for_formal_training": False}))
            with self.assertRaisesRegex(ValueError, "Formal training blocked"):
                load_data(directory, torch.device("cpu"))

    def test_ratl_base_is_frozen_and_additional_gap(self):
        query, memory = fixtures()
        model = RATL(**config()).eval()
        cache = model.refresh_memory(memory)
        first = model(query, memory, cache, labels=query["y"])
        second = model(query, memory, cache, labels=query["y"] + 10)
        torch.testing.assert_close(first["prediction"], second["prediction"], rtol=0, atol=0)
        for row in range(3):
            historical = first["selected"][row][first["selected_valid"][row]]
            self.assertTrue((memory["end_day"][historical] < query["day"][row] - 11).all())
        model.train()
        self.assertFalse(model.base.training)
        result = model(query, memory, cache, labels=query["y"])
        objective(result, query["y"], query["q"], model, 0.4)[0].backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base.parameters()))
        model.eval()
        model.gamma.fill_(0)
        output = model(query, memory, cache)
        torch.testing.assert_close(output["prediction"], output["base_prediction"], rtol=0, atol=0)

    def test_raft_pearson_keys_match_manual_calculation(self):
        query, memory = fixtures()
        model = RAFT(**config())
        keys = model.multiscale(query["x"])
        first = query["x"] - query["x"][:, -1:, :]
        flat = first.flatten(1)
        flat -= flat.mean(1, keepdim=True)
        expected = torch.nn.functional.normalize(flat, dim=1)
        torch.testing.assert_close(keys[0], expected)
        for row in range(3):
            out = model(query, memory)
            indices = out["selected"][row][out["selected_valid"][row]]
            self.assertTrue(legal_memory(query, memory)[row, indices].all())

    def test_development_training_and_export(self):
        """Temporary synthetic files are deleted and never become paper results."""
        with tempfile.TemporaryDirectory(prefix="Sx-development-") as temporary:
            directory = Path(temporary)
            rng = np.random.default_rng(5)
            n = 24
            split = np.array([0] * 16 + [1] * 4 + [2] * 4)
            days = np.r_[np.arange(16) * 12, np.arange(4) * 12 + 230, np.arange(4) * 12 + 310]
            quality = rng.uniform(0.1, 0.9, n).astype(np.float32)
            np.savez(directory / "dataset.npz", x=rng.normal(size=(n, 10, 4)).astype(np.float32),
                     y=rng.normal(size=(n, 11, 3)).astype(np.float32), q=quality,
                     platform=np.arange(n) % 3, influencer=np.arange(n) % 7,
                     day=days, end_day=days + 10, split=split)
            dates = (pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")).astype(str)
            pd.DataFrame(dict(event_id=["DEVELOPMENT_%d" % i for i in range(n)], event_date=dates,
                              platform=np.arange(n) % 3, influencer=np.arange(n) % 7,
                              q=quality, label_version="DEVELOPMENT_SYNTHETIC_NOT_PAPER")).to_csv(directory / "events.csv", index=False)
            (directory / "normalization.json").write_text(json.dumps(dict(
                ready_for_formal_training=True, y_mean=[0, 0, 0], y_std=[1, 2, 3], n_platforms=3, n_influencers=7)))
            for model_name in ["RAIF", "HistoricalMean", "ECT", "DLinear", "PatchTST", "iTransformer", "RAFT", "RATL"]:
                with self.subTest(model=model_name):
                    output = directory / model_name
                    args = SimpleNamespace(data=directory, output=output, model=model_name, seed=3,
                                           epochs=2, patience=2, batch_size=8, hidden=16, heads=4, layers=1,
                                           key_dim=8, kernel=3, dropout=0.0, lr=1e-3, weight_decay=1e-4,
                                           k=3, variant="full", retrieval="response", t_delta=1.0,
                                           t_z=0.1, t_r=0.2, lambda_q=1.0, lambda_ret=0.1,
                                           huber_delta=1.0, device="cpu", threads=1,
                                           timeout_seconds=60, timing_repeats=2)
                    run(args)
                    status = json.loads((output / "status.json").read_text())
                    self.assertEqual(status["status"], "completed")
                    self.assertEqual(status["n_test"], 4)
                    self.assertEqual(len(pd.read_csv(output / "predictions.csv")), 132)
                    horizon = pd.read_csv(output / "horizon_metrics.csv")
                    overall = pd.read_csv(output / "overall_metrics.csv")
                    self.assertEqual(len(horizon), 30)
                    np.testing.assert_allclose(horizon[horizon.horizon == 11].value, overall.value)
                    if model_name == "RATL":
                        stages = pd.read_csv(output / "training_stages.csv")
                        self.assertTrue({"base_fit", "residual_fit"}.issubset(set(stages.phase)))
                        self.assertEqual(len(pd.read_csv(output / "gamma_selection.csv")), 11)
                    args.output = directory / (model_name + "_validation")
                    args.validation_only = True
                    run(args)
                    self.assertFalse((args.output / "predictions.csv").exists())
                    self.assertFalse((args.output / "overall_metrics.csv").exists())
                    status = json.loads((args.output / "status.json").read_text())
                    self.assertEqual(status["mode"], "validation_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
