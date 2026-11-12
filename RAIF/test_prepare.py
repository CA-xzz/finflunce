"""Development tests for information timing and label construction."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from prepare import (HistoricalAttributes, market_fit, fit_distance, select_reference,
                     fixed_label, encode_categories, MarketPanel, run, attributes_ready, strict_candidates,
                     ranked_candidates, event_timing_checks, board, index_code)


class PreparationTests(unittest.TestCase):
    def test_chinext_renumbered_stock_uses_chinext_index(self):
        for code in ["300114", "301001", "302132"]:
            self.assertEqual(board(code), "CHINEXT")
            self.assertEqual(index_code(code), "sz.399006")
        self.assertEqual(board("000001"), "SZSE")
        self.assertEqual(board("688001"), "STAR")
        self.assertEqual(board("920001"), "BSE")

    def test_readiness_does_not_claim_vintage_archives(self):
        evidence = {"historical_market_cap_records_verified": True,
                    "historical_industry_records_verified": True,
                    "conservative_availability_rule_verified": True,
                    "historical_vintage_archive_verified": False,
                    "provenance": "unit-test fixture", "availability_policy": "prior close and source update time"}
        self.assertTrue(attributes_ready(evidence))
        evidence["conservative_availability_rule_verified"] = False
        self.assertFalse(attributes_ready(evidence))
        self.assertFalse(attributes_ready({"historical_market_cap_verified": True,
                                           "historical_industry_verified": True, "provenance": "legacy booleans"}))

    def test_market_fit_uses_valid_pairs_and_minimum(self):
        market = np.linspace(-0.03, 0.04, 40)[:, None]
        stock = 0.002 + 1.3 * market
        alpha, beta, n = market_fit(stock, market)
        np.testing.assert_allclose(alpha, [0.002], atol=1e-12)
        np.testing.assert_allclose(beta, [1.3], atol=1e-12)
        stock[:11] = np.nan
        alpha, beta, n = market_fit(stock, market)
        self.assertEqual(n[0], 29)
        self.assertTrue(np.isnan(alpha[0]))

    def test_historical_attributes_disallow_future_and_boundaries(self):
        cap = pd.DataFrame({"code": ["000001", "000002"], "date": ["2024-01-01"] * 2,
                            "total_market_cap": [100., 200.], "available_at": ["2024-01-02", "2024-01-03"]})
        sector = pd.DataFrame({"code": ["000001", "000001", "000002"],
            "industry": ["old", "new", "late"], "effective_from": ["2020-01-01", "2024-01-02", "2020-01-01"],
            "effective_to": ["2024-01-02", None, None], "available_at": ["2020-01-01", "2024-01-02", "2024-01-03"]})
        attrs = HistoricalAttributes(cap, sector, ["000001", "000002"])
        values, names = attrs.at(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
        np.testing.assert_allclose(values[0], 100)
        self.assertTrue(np.isnan(values[1]))
        self.assertEqual(names.tolist(), ["new", ""])
        values, _ = attrs.at(pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-02"))
        self.assertTrue(np.isnan(values).all())

    def test_overlapping_industry_intervals_fail(self):
        cap = pd.DataFrame({"code": ["000001"], "date": ["2024-01-01"],
                            "total_market_cap": [100.], "available_at": ["2024-01-02"]})
        sector = pd.DataFrame({"code": ["000001", "000001"], "industry": ["A", "B"],
            "effective_from": ["2020-01-01", "2023-01-01"], "effective_to": [None, None],
            "available_at": ["2020-01-01", "2023-01-01"]})
        attrs = HistoricalAttributes(cap, sector, ["000001"])
        with self.assertRaises(ValueError):
            attrs.at(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))

    def test_fixed_control_missing_future_rejects_entire_label(self):
        query = np.ones((11, 3))
        controls = np.stack([np.ones((11, 3)) * 2, np.ones((11, 3)) * 4])
        weights = np.array([0.75, 0.25])
        result = fixed_label(query, controls, weights)
        np.testing.assert_allclose(result[0], -1.5)
        controls[1, 3, 0] = np.nan
        self.assertIsNone(fixed_label(query, controls, weights))

    def test_reference_weights_and_quality_statistics(self):
        distances = np.array([1., 2., 10.])
        query = np.array([0., 0.])
        candidates = np.array([[1., 0.], [0., 1.], [4., 4.]])
        chosen, weights, balance, mean_distance, neff = select_reference(
            distances, query, candidates, m=3, temperature=2., maximum=2.)
        self.assertEqual(chosen.tolist(), [0, 1])
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(balance, np.mean((weights @ candidates[:2]) ** 2))
        self.assertAlmostEqual(neff, 1 / np.sum(weights ** 2))
        self.assertIsNone(select_reference(distances, query, candidates, 3, 2., 0.5))

    def test_training_vocab_and_scaling(self):
        values = np.array(["A", "__UNKNOWN__", "B", "C"])
        encoded, vocab = encode_categories(values, np.array([True, True, False, False]))
        self.assertEqual(encoded.tolist(), [1, 0, 0, 0])
        self.assertEqual(vocab, {"__UNKNOWN__": 0, "A": 1})
        fit = fit_distance(np.array([[1., 5.], [2., 5.], [3., 5.]]), 0.1)
        np.testing.assert_allclose(fit.mean, [2., 5.])
        self.assertTrue(np.isfinite(fit.inverse).all())

    def build_fixture(self, root):
        source, preflight, external = [root / name for name in ["source", "preflight", "external"]]
        for directory in [source / "control_stock_finder/data", source / "original_data", preflight, external]:
            directory.mkdir(parents=True)
        dates = pd.bdate_range("2022-01-03", periods=420)
        rng = np.random.default_rng(34)
        market = rng.normal(0, 0.01, len(dates))
        indices = pd.DataFrame({"date": dates, "index_code": "sz.399001", "index_ret": market, "index_close": 1000.})
        indices.to_parquet(preflight / "indices.parquet", index=False)
        pd.DataFrame({"day": np.arange(len(dates)), "date": dates}).to_csv(preflight / "calendar.csv", index=False)
        frames = []
        for i, code in enumerate(["000001", "000002", "000003", "000004"]):
            returns = 0.0002 + market * (0.9 + i * 0.05) + rng.normal(0, 0.003, len(dates))
            frames.append(pd.DataFrame({"date": dates, "code": code, "close": 10 * np.cumprod(1 + returns),
                                       "volume": 1e6 + rng.uniform(0, 1e4, len(dates)),
                                       "turnover": 1 + rng.uniform(0, .2, len(dates))}))
        pd.concat(frames).to_parquet(source / "control_stock_finder/data/A_share_daily.parquet", index=False)
        event_days = list(range(110, 250)) + list(range(285, 305)) + list(range(345, 365))
        events = pd.DataFrame({"day": event_days, "stock": "000001"})
        events["event_date"] = dates[events.day]
        events["end_day"] = events.day + 10
        events["end_date"] = dates[events.end_day]
        events["event_id"] = events.event_date.dt.strftime("000001_%Y-%m-%d")
        events["split"] = np.where(events.day < 250, "train", np.where(events.day < 305, "validation", "test"))
        events["platform"], events["influencer"] = "P", "A"
        events.loc[events.split == "test", "influencer"] = "UNSEEN"
        events["exclusion_reason"] = ""
        events.to_csv(preflight / "events_manifest.csv", index=False)
        raw = pd.DataFrame({"recommend_date": events.event_date.dt.strftime("%Y-%m-%d"), "recommended_stock_code": "000001"})
        raw.to_csv(source / "original_data/events.csv", index=False)
        caps = []
        for code in ["000001", "000002", "000003", "000004"]:
            caps.append(pd.DataFrame({"code": code, "date": dates, "available_at": dates + pd.Timedelta(days=1),
                                      "total_market_cap": 1e9}))
        pd.concat(caps).to_parquet(external / "historical_market_cap.parquet", index=False)
        pd.DataFrame({"code": ["000001", "000002", "000003", "000004"], "industry": "A",
                      "effective_from": "2020-01-01", "effective_to": None,
                      "available_at": "2020-01-01"}).to_parquet(external / "industry_history.parquet", index=False)
        return SimpleNamespace(source=source, preflight=preflight, external=external, output=root / "output",
                               length=10, ridge=.1, m_grid=[1, 3], quantile_grid=[.8, 1.], temperature_grid=[1.])

    def test_calendar_suspension_and_future_independence(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            panel = MarketPanel(args.source, args.preflight, args.external, 10)
            before = panel.state(120)
            expected = before.v.copy()
            panel.returns[120:] = 1e6
            panel.turnover[120:] = 1e6
            panel.log_volume[120:] = 1e6
            panel._last_state = None
            after = panel.state(120)
            np.testing.assert_allclose(expected, after.v, equal_nan=True)
            panel.returns[115, 0] = np.nan
            panel._last_state = None
            self.assertFalse(panel.state(120).valid[0])

    def test_suspension_retained_close_does_not_bridge_returns(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            path = args.source / "control_stock_finder/data/A_share_daily.parquet"
            daily = pd.read_parquet(path)
            stock = daily.code == "000001"
            suspended_date = daily.loc[stock, "date"].iloc[115]
            daily.loc[stock & (daily.date == suspended_date), "volume"] = 0
            daily.to_parquet(path, index=False)
            panel = MarketPanel(args.source, args.preflight, args.external, 10)
            self.assertTrue(np.isnan(panel.returns[115, 0]))
            self.assertTrue(np.isnan(panel.returns[116, 0]))
            self.assertTrue(np.isfinite(panel.returns[117, 0]))

    def test_end_to_end_development_data_and_formal_gate(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            run(args)
            arrays = np.load(args.output / "dataset.npz")
            normal = json.loads((args.output / "normalization.json").read_text())
            self.assertFalse(normal["ready_for_formal_training"])
            self.assertTrue(normal["reference_hyperparameters_tuned"])
            self.assertEqual(arrays["y"].shape[1:], (11, 3))
            self.assertTrue(np.all(arrays["influencer"][arrays["split"] == 2] == 0))
            training = arrays["split"] == 0
            np.testing.assert_allclose(arrays["y"][training].mean(axis=(0, 1)), np.zeros(3), atol=1e-6)
            self.assertTrue(np.all(arrays["q"] > 0))
            pairs = pd.read_csv(args.output / "reference_pairs.csv", dtype={"control": str})
            self.assertFalse((pairs.control == "000001").any())
            protocol = json.loads((args.output / "reference_tuning_protocol.json").read_text())
            self.assertFalse(protocol["outcome_labels_used"])
            self.assertTrue(protocol["common_selection_events_include_unmatched"])

    def test_strict_criteria_and_single_nearest_modes(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            panel = MarketPanel(args.source, args.preflight, args.external, 10)
            state = panel.state(120)
            path = np.linspace(-.01, .01, 10)
            panel.returns[110:120] = path[:, None]
            panel.market[110:120] = 0
            panel.turnover[110:120] = 1
            state.alpha[:] = 0
            state.beta[:] = 0
            self.assertEqual(strict_candidates(panel, state, 0, np.array([1, 2, 3])).tolist(), [1, 2, 3])
            panel.returns[110:120, 1] *= -1
            panel.turnover[110:120, 2] = 2
            panel.returns[110:120, 3] *= 3
            self.assertEqual(len(strict_candidates(panel, state, 0, np.array([1, 2, 3]))), 0)
            # End-to-end strict fixture uses identical return paths, not relaxed
            # thresholds, to make a nonempty known-positive strict sample.
            source_path = args.source / "control_stock_finder/data/A_share_daily.parquet"
            frame = pd.read_parquet(source_path)
            close = frame.loc[frame.code == "000001", "close"].to_numpy()
            for code in ["000002", "000003", "000004"]:
                frame.loc[frame.code == code, "close"] = close
            frame.to_parquet(source_path, index=False)
            for mode in ["strict", "nearest"]:
                args.reference_mode, args.output = mode, Path(directory) / mode
                run(args)
                pairs = pd.read_csv(args.output / "reference_pairs.csv")
                self.assertTrue((pairs.groupby("event_id").size() == 1).all())
                np.testing.assert_allclose(pairs.weight, 1)
                config = json.loads((args.output / "config.json").read_text())
                self.assertEqual(config["reference_mode"], mode)
                self.assertFalse(config["reference_hyperparameters_tuned"])

    def test_effective_industry_keeps_interval_bounds_and_cap_timing(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            path = args.external / "industry_history.parquet"
            industry = pd.read_parquet(path)
            industry["available_at"] = "2030-01-01"
            industry.loc[industry.code == "000003", "effective_from"] = "2030-01-01"
            industry.to_parquet(path, index=False)
            cap_path = args.external / "historical_market_cap.parquet"
            cap = pd.read_parquet(cap_path)
            cap.loc[cap.code == "000002", "available_at"] = "2030-01-01"
            cap.to_parquet(cap_path, index=False)
            original = MarketPanel(args.source, args.preflight, args.external, 10).state(120)
            effective = MarketPanel(args.source, args.preflight, args.external, 10,
                                    industry_policy="effective").state(120)
            self.assertFalse(original.valid.any())
            self.assertEqual(effective.valid.tolist(), [True, False, False, True])
            self.assertTrue(np.isnan(effective.cap[1]))
            self.assertEqual(effective.industry[2], "")

    def test_correlation_revision_changes_only_requested_cutoff(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            panel = MarketPanel(args.source, args.preflight, args.external, 10)
            state = panel.state(120)
            u = np.linspace(-1., 1., 10)
            u /= np.linalg.norm(u)
            v = np.arange(10.) ** 2
            v -= v.mean()
            v -= (v @ u) * u
            v /= np.linalg.norm(v)
            panel.returns[110:120, 0] = .01 * u
            panel.returns[110:120, 1] = .01 * (.75 * u + np.sqrt(1 - .75 ** 2) * v)
            panel.market[110:120] = 0
            panel.turnover[110:120] = 1
            state.alpha[:] = state.beta[:] = 0
            self.assertEqual(strict_candidates(panel, state, 0, np.array([1])).tolist(), [])
            self.assertEqual(strict_candidates(panel, state, 0, np.array([1]), .7).tolist(), [1])
            panel.turnover[110:120, 1] = 2
            self.assertEqual(strict_candidates(panel, state, 0, np.array([1]), .7).tolist(), [])

    def make_comparison_fixture(self, root, name, training_ids, offset):
        path = root / name
        path.mkdir()
        ids = training_ids + ["v0", "v1", "t0", "t1"]
        days = {"a": 10, "b": 25, "c": 40, "d": 55, "v0": 100, "v1": 120, "t0": 200, "t1": 220}
        n = len(ids)
        raw_x = np.stack([np.full((10, 3), days[event_id] / 10.) for event_id in ids])
        raw_y = np.stack([np.full((11, 3), days[event_id] / 5. + offset) for event_id in ids])
        split = np.array([0] * len(training_ids) + [1, 1, 2, 2], dtype=np.int8)
        train = split == 0
        normal = {"x_mean": raw_x[train].mean((0, 1)).tolist(), "x_std": raw_x[train].std((0, 1)).tolist(),
            "y_mean": raw_y[train].mean((0, 1)).tolist(), "y_std": raw_y[train].std((0, 1)).tolist(),
            "label_version": name + "_fixture", "ready_for_formal_training": True,
            "n_platforms": 3, "n_influencers": 5, "quality_balance_scale": 2., "quality_distance_scale": 2.}
        normal["x_std"] = np.maximum(normal["x_std"], 1e-8).tolist()
        normal["y_std"] = np.maximum(normal["y_std"], 1e-8).tolist()
        q = np.full(n, .25, dtype=np.float32)
        dates = [str((pd.Timestamp("2020-01-01") + pd.Timedelta(days=days[event_id])).date()) for event_id in ids]
        events = pd.DataFrame({"event_id": ids, "event_date": dates, "split": np.array(["train", "validation", "test"])[split],
            "platform": ["P" if event_id in {"b", "c"} else "OUTSIDE" for event_id in ids],
            "influencer": ["I" if event_id in {"b", "c"} else "UNSEEN" for event_id in ids],
            "q": q, "label_version": normal["label_version"]})
        np.savez_compressed(path / "dataset.npz", x=((raw_x - normal["x_mean"]) / normal["x_std"]).astype(np.float32),
            y=((raw_y - normal["y_mean"]) / normal["y_std"]).astype(np.float32), q=q, platform=np.ones(n, dtype=np.int64),
            influencer=np.ones(n, dtype=np.int64), day=np.array([days[event_id] for event_id in ids]),
            end_day=np.array([days[event_id] + 10 for event_id in ids]), split=split)
        events.to_csv(path / "events.csv", index=False)
        (path / "normalization.json").write_text(json.dumps(normal))
        (path / "config.json").write_text(json.dumps({"reference_mode": name}))
        pd.DataFrame({"event_id": ids, "balance": [1. if event_id == "b" else 3. if event_id == "c" else 1000. for event_id in ids],
            "mean_distance": [2. if event_id == "b" else 6. if event_id == "c" else 2000. for event_id in ids],
            "effective_n": 1., "q": q}).to_csv(path / "reference_quality.csv", index=False)
        events.groupby("split").size().rename("n_events").to_csv(path / "coverage.csv")
        return path

    def test_reference_comparison_common_targets_and_intersection_scaling(self):
        from reference_experiments import run as compare, Prepared
        with TemporaryDirectory() as directory:
            root = Path(directory)
            soft = self.make_comparison_fixture(root, "soft", ["a", "b", "c"], 100.)
            strict = self.make_comparison_fixture(root, "strict", ["b", "c"], -10.)
            nearest = self.make_comparison_fixture(root, "nearest", ["b", "c", "d"], 50.)
            source = Prepared.load(soft, "soft")
            plans = compare(soft, strict, nearest, root / "comparison")
            self.assertEqual(len(plans), 8)
            all_targets = []
            for plan in plans:
                path = Path(plan["dataset"])
                with np.load(path / "evaluation_targets.npz") as archive:
                    all_targets.append(archive["truth"])
                with np.load(path / "dataset.npz") as archive:
                    arrays = dict(archive)
                normal = json.loads((path / "normalization.json").read_text())
                self.assertTrue(normal["ready_for_formal_training"])
                if plan["group"] == "full_training" and plan["scheme"] == "soft_quality":
                    np.testing.assert_array_equal(arrays["y"][arrays["split"] == 0], source.arrays["y"][source.arrays["split"] == 0])
                    np.testing.assert_array_equal(arrays["y"][arrays["split"] == 1], source.arrays["y"][source.arrays["split"] == 1])
                if plan["group"] == "intersect_training" and plan["scheme"] == "soft_quality":
                    self.assertEqual(normal["normalization_fit_event_ids"], ["b", "c"])
                    np.testing.assert_allclose(normal["x_mean"], [3.25] * 3, atol=1e-6)
                    np.testing.assert_allclose(normal["y_mean"], [106.5] * 3, atol=1e-6)
                    self.assertEqual(normal["quality_balance_scale"], 2.)
                    self.assertEqual(normal["quality_distance_scale"], 4.)
                    self.assertEqual(normal["platform_vocab"], {"__UNKNOWN__": 0, "P": 1})
                    self.assertTrue((arrays["platform"][arrays["split"] != 0] == 0).all())
            for targets in all_targets[1:]:
                np.testing.assert_array_equal(targets, all_targets[0])
            np.testing.assert_array_equal(np.load(soft / "dataset.npz")["y"], source.arrays["y"])

    def test_empty_common_strict_test_fails(self):
        from reference_experiments import run as compare
        with TemporaryDirectory() as directory:
            root = Path(directory)
            soft = self.make_comparison_fixture(root, "soft", ["a", "b", "c"], 100.)
            strict = self.make_comparison_fixture(root, "strict", ["b", "c"], -10.)
            nearest = self.make_comparison_fixture(root, "nearest", ["b", "c", "d"], 50.)
            events = pd.read_csv(strict / "events.csv")
            events.loc[events.split == "test", "event_id"] = ["outside0", "outside1"]
            events.to_csv(strict / "events.csv", index=False)
            quality = pd.read_csv(strict / "reference_quality.csv")
            quality.loc[quality.event_id.isin(["t0", "t1"]), "event_id"] = ["outside0", "outside1"]
            quality.to_csv(strict / "reference_quality.csv", index=False)
            with self.assertRaisesRegex(ValueError, "No common complete test"):
                compare(soft, strict, nearest, root / "comparison")

    def test_frozen_reference_sweep_preserves_state_fit(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            run(args)
            base = args.output
            arrays_base = dict(np.load(base / "dataset.npz"))
            config_base = json.loads((base / "config.json").read_text())
            args.fixed_reference_config = base / "config.json"
            args.output = Path(directory) / "frozen_same"
            run(args)
            arrays_same = dict(np.load(args.output / "dataset.npz"))
            for name in arrays_base:
                np.testing.assert_array_equal(arrays_base[name], arrays_same[name])
            args.m_override, args.distance_multiplier = 1, 1.
            args.output = Path(directory) / "m1"
            run(args)
            config = json.loads((args.output / "config.json").read_text())
            self.assertEqual(config["m"], 1)
            self.assertEqual(config["distance_maximum"], config_base["distance_maximum"])
            self.assertEqual(config["temperature"], config_base["temperature"])
            self.assertEqual(config["state_mean"], config_base["state_mean"])
            self.assertEqual(config["state_std"], config_base["state_std"])
            self.assertEqual(config["inverse_covariance"], config_base["inverse_covariance"])
            self.assertFalse(config["reference_hyperparameters_tuned"])
            args.m_override, args.distance_multiplier = None, .5
            args.output = Path(directory) / "dhalf"
            run(args)
            config = json.loads((args.output / "config.json").read_text())
            self.assertEqual(config["m"], config_base["m"])
            self.assertEqual(config["distance_maximum"], config_base["distance_maximum"] * .5)
            self.assertEqual(config["temperature"], config_base["temperature"])
            bad = dict(config_base)
            bad["source_hashes"] = {}
            (Path(directory) / "bad.json").write_text(json.dumps(bad))
            args.fixed_reference_config, args.output = Path(directory) / "bad.json", Path(directory) / "bad_output"
            with self.assertRaisesRegex(ValueError, "different source files"):
                run(args)

    def test_extra_reference_variant_and_readiness_inheritance(self):
        from reference_experiments import run as compare
        with TemporaryDirectory() as directory:
            root = Path(directory)
            soft = self.make_comparison_fixture(root, "soft", ["a", "b", "c"], 100.)
            strict = self.make_comparison_fixture(root, "strict", ["b", "c"], -10.)
            nearest = self.make_comparison_fixture(root, "nearest", ["b", "c", "d"], 50.)
            normal = json.loads((nearest / "normalization.json").read_text())
            normal["ready_for_formal_training"] = False
            (nearest / "normalization.json").write_text(json.dumps(normal))
            plans = compare(soft, strict, nearest, root / "comparison", {"M1": soft})
            self.assertEqual(len(plans), 10)
            self.assertEqual(sum(plan["scheme"] == "M1" for plan in plans), 2)
            for plan in plans:
                normal = json.loads((Path(plan["dataset"]) / "normalization.json").read_text())
                self.assertFalse(normal["ready_for_formal_training"])

    def test_expanded_reference_variant_keeps_own_training_and_common_strict_test(self):
        from reference_experiments import run as compare, Prepared
        with TemporaryDirectory() as directory:
            root = Path(directory)
            soft = self.make_comparison_fixture(root, "soft", ["a", "b", "c"], 100.)
            strict = self.make_comparison_fixture(root, "strict", ["b", "c"], -10.)
            nearest = self.make_comparison_fixture(root, "nearest", ["b", "c", "d"], 50.)
            expanded = self.make_comparison_fixture(root, "expanded_nearest", ["a", "b", "c", "d"], -100.)
            # One expanded test event is outside the other schemes' test sets.
            for filename in ["events.csv", "reference_quality.csv"]:
                frame = pd.read_csv(expanded / filename)
                frame.loc[frame.event_id == "t1", "event_id"] = "expanded_only"
                frame.to_csv(expanded / filename, index=False)
            source = Prepared.load(expanded, None)
            strict_source = Prepared.load(strict, "strict")
            expected_truth = strict_source.raw("y")[strict_source.events.event_id == "t0"]
            plans = compare(soft, strict, nearest, root / "comparison", {"expanded_nearest": expanded})
            self.assertEqual(len(plans), 10)
            for plan in plans:
                path = Path(plan["dataset"])
                with np.load(path / "evaluation_targets.npz") as targets:
                    np.testing.assert_array_equal(targets["event_ids"], ["t0"])
                    np.testing.assert_array_equal(targets["truth"], expected_truth)
                if plan["scheme"] != "expanded_nearest":
                    continue
                generated = Prepared.load(path, "expanded_nearest")
                self.assertEqual(generated.normal["training_label_version"], "expanded_nearest_fixture")
                self.assertEqual(generated.normal["evaluation_label_version"], "strict_fixture")
                train = generated.arrays["split"] == 0
                if plan["group"] == "full_training":
                    np.testing.assert_array_equal(generated.arrays["y"][train], source.arrays["y"][source.arrays["split"] == 0])
                    np.testing.assert_array_equal(generated.arrays["y"][generated.arrays["split"] == 1], source.arrays["y"][source.arrays["split"] == 1])
                else:
                    self.assertEqual(generated.normal["normalization_fit_event_ids"], ["b", "c"])
                    np.testing.assert_allclose(generated.normal["y_mean"], [-93.5] * 3, atol=1e-6)
                    np.testing.assert_allclose(generated.normal["x_mean"], [3.25] * 3, atol=1e-6)
                    self.assertEqual(generated.normal["quality_balance_scale"], 2.)
                    self.assertEqual(generated.normal["quality_distance_scale"], 4.)
                    self.assertEqual(generated.normal["platform_vocab"], {"__UNKNOWN__": 0, "P": 1})
                positions = [source.events.index[source.events.event_id == event_id][0]
                             for event_id in generated.events.loc[train, "event_id"]]
                np.testing.assert_allclose(generated.raw("y")[train], source.raw("y")[positions], atol=1e-6)
            with np.load(expanded / "dataset.npz") as unchanged:
                for key in source.arrays:
                    np.testing.assert_array_equal(unchanged[key], source.arrays[key])

    def test_expanded_ignores_attribute_and_board_gates_and_ties_by_code(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            panel = MarketPanel(args.source, args.preflight, args.external, 10, expanded=True)
            state = panel.state(120)
            self.assertEqual(state.v.shape[1], 14)
            state.cap[:] = [1., 1e10, np.nan, 1e-10]
            state.industry[:] = ["Q", "X", "", "Y"]
            panel.boards[:] = ["A", "B", "C", "D"]
            candidates = panel.candidates(state, 0)
            self.assertEqual(candidates.tolist(), [1, 2, 3])
            state.v[:] = state.v[0]
            fit = fit_distance(np.array([np.zeros(14), np.ones(14), np.ones(14) * 2]), .1)
            ids, distances, _, _ = ranked_candidates(state, 0, candidates, fit, 1)
            self.assertEqual(panel.codes[ids[0]], "000002")
            self.assertEqual(distances[0], 0)
            panel.recommended[120].add("000002")
            self.assertEqual(panel.candidates(state, 0).tolist(), [2, 3])

    def test_effective_industry_diagnostic_does_not_fabricate_availability(self):
        cap = pd.DataFrame(columns=["code", "date", "total_market_cap", "available_at"])
        industry = pd.DataFrame({"code": ["000001"], "industry": ["historical"],
            "effective_from": ["2020-01-01"], "effective_to": [None], "available_at": ["2026-06-30"]})
        attributes = HistoricalAttributes(cap, industry, ["000001", "000002"])
        _, observable = attributes.at(pd.Timestamp("2023-12-29"), pd.Timestamp("2024-01-01"))
        effective, flags = attributes.at_effective(pd.Timestamp("2024-01-01"))
        self.assertEqual(observable.tolist(), ["", ""])
        self.assertEqual(effective.tolist(), ["historical", ""])
        self.assertFalse(flags.any())
        self.assertEqual(attributes.industry.available_at.iloc[0], pd.Timestamp("2026-06-30"))

    def test_expanded_without_attributes_and_no_future_replacement(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            (args.external / "historical_market_cap.parquet").unlink()
            (args.external / "industry_history.parquet").unlink()
            source_path = args.source / "control_stock_finder/data/A_share_daily.parquet"
            frame = pd.read_parquet(source_path)
            first = frame[frame.code == "000001"].reset_index(drop=True)
            for column in ["close", "volume", "turnover"]:
                positions = frame.index[frame.code == "000002"][:345]
                frame.loc[positions, column] = first[column].iloc[:345].to_numpy()
            failed_date = first.date.iloc[345]
            frame.loc[(frame.code == "000002") & (frame.date == failed_date), "volume"] = 0
            frame.to_parquet(source_path, index=False)
            args.reference_mode = "expanded_nearest"
            run(args)
            normal = json.loads((args.output / "normalization.json").read_text())
            config = json.loads((args.output / "config.json").read_text())
            self.assertTrue(normal["ready_for_formal_training"])
            self.assertEqual(normal["attribute_verification"], {})
            self.assertTrue(normal["attributes_diagnostic_only"])
            self.assertFalse(normal["market_data_timing_checks"]["attributes_required_for_matching"])
            self.assertEqual(len(config["state_mean"]), args.length + 4)
            self.assertEqual(config["m"], 1)
            self.assertIsNone(config["distance_maximum"])
            diagnostic = pd.read_csv(args.output / "matching_diagnostics.csv", dtype={"stock": str, "control": str})
            target = diagnostic[pd.to_datetime(diagnostic.event_date) == failed_date].iloc[0]
            self.assertEqual(target.control, "000002")
            self.assertTrue(target.prestate_available)
            self.assertTrue(target.reference_available)
            self.assertFalse(target.complete_label)
            self.assertFalse(target.included)
            pairs = pd.read_csv(args.output / "reference_pairs.csv", dtype={"control": str})
            match = pairs[pairs.event_id == target.event_id]
            self.assertEqual(len(match), 1)
            self.assertEqual(match.control.iloc[0], "000002")
            self.assertFalse(match.included_after_outcome_check.iloc[0])
            self.assertTrue(diagnostic.query_industry_effective.isna().all())
            self.assertTrue(diagnostic.market_cap_ratio.isna().all())
            quality = pd.read_csv(args.output / "reference_quality.csv")
            np.testing.assert_allclose(quality.effective_n, 1)
            self.assertTrue((quality.q <= .5).all())
            flow = pd.read_csv(args.output / "sample_flow.csv")
            self.assertEqual(flow[(flow.group == "all") & (flow.stage == "input_events")].n_events.iloc[0], 180)

    def test_expanded_timing_check_rejects_future_window_mismatch(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            events = pd.read_csv(args.preflight / "events_manifest.csv", parse_dates=["event_date", "end_date"])
            calendar = pd.DatetimeIndex(pd.read_csv(args.preflight / "calendar.csv", parse_dates=["date"]).date)
            self.assertTrue(event_timing_checks(events, calendar)["event_calendar_alignment"])
            events.loc[0, "end_day"] += 1
            with self.assertRaisesRegex(ValueError, "Invalid event response windows"):
                event_timing_checks(events, calendar)

    def test_industry_recovery_diagnostics_preserve_expanded_matching(self):
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            industry_path = args.external / "industry_history.parquet"
            industry = pd.read_parquet(industry_path)
            industry["original_industry_code"] = "123456"
            industry["update_time"] = "2026-06-30"
            industry["available_at"] = "2026-07-01"
            industry.to_parquet(industry_path, index=False)
            events = pd.read_csv(args.preflight / "events_manifest.csv", dtype={"stock": str})
            recovered = events.iloc[[0, 1, 160]][["event_id", "stock", "event_date"]].copy()
            recovered["industry"], recovered["industry_name_zh"] = "A", "Industry A"
            recovered["original_industry_code"], recovered["effective_from"] = "123456", "2020-01-01"
            recovered["effective_to"], recovered["publication_time"] = "", ""
            recovered["update_time"], recovered["publication_time_verified"] = "2026-06-30", False
            recovered["reason"] = "later_source_update"
            recovery = Path(directory) / "recovery"
            recovery.mkdir()
            recovered.to_parquet(recovery / "event_recovery.parquet", index=False)
            industry.rename(columns={"code": "stock"}).to_parquet(recovery / "stock_intervals.parquet", index=False)
            (recovery / "industry_level2_names.json").write_text(json.dumps({"A": "Industry A"}))
            for name in ["source_evidence.json", "source_manifest.json"]:
                (recovery / name).write_text(json.dumps({"source": "synthetic unit-test fixture"}))
            source_path = args.source / "control_stock_finder/data/A_share_daily.parquet"
            daily = pd.read_parquet(source_path)
            failed_date = pd.Timestamp(recovered.event_date.iloc[-1])
            daily.loc[(daily.code == "000001") & (daily.date == failed_date), "volume"] = 0
            daily.to_parquet(source_path, index=False)
            args.reference_mode = "expanded_nearest"
            run(args)
            baseline = dict(np.load(args.output / "dataset.npz"))
            args.industry_recovery, args.output = recovery, Path(directory) / "with_recovery"
            run(args)
            result = dict(np.load(args.output / "dataset.npz"))
            for name in baseline:
                np.testing.assert_array_equal(result[name], baseline[name])
            verification = pd.read_csv(args.output / "industry_recovery_verification.csv")
            self.assertTrue(verification.all_match.all())
            self.assertEqual(len(verification), 3)
            summary = json.loads((args.output / "industry_recovery_summary.json").read_text())
            self.assertEqual(summary["n_included"], 2)
            self.assertEqual(summary["n_excluded"], 1)
            assignments = pd.read_csv(args.output / "industry_assignments.csv")
            self.assertFalse(assignments.publication_time_verified.any())
            self.assertTrue(assignments.publication_time.isna().all())
            self.assertTrue((assignments.original_update_time == "2026-06-30").all())
            self.assertTrue((assignments.update_proxy_available_at == "2026-07-01").all())
            self.assertTrue((assignments.industry_name_zh == "Industry A").all())
            self.assertFalse(assignments.used_for_model_matching.any())

    def test_window_comparison_preserves_expanded_mode_without_attributes(self):
        from prepare_windows import run as prepare_windows
        with TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            (args.external / "historical_market_cap.parquet").unlink()
            (args.external / "industry_history.parquet").unlink()
            args.reference_mode = "expanded_nearest"
            run(args)
            windows = SimpleNamespace(data=args.output, source=args.source, preflight=args.preflight,
                                      external=args.external, output=Path(directory) / "windows")
            prepare_windows(windows)
            targets, event_ids = [], []
            for length in [5, 10, 20]:
                path = windows.output / f"L{length}"
                normal = json.loads((path / "normalization.json").read_text())
                self.assertEqual(normal["source_reference_mode"], "expanded_nearest")
                self.assertEqual(normal["reference_mode"], "expanded_nearest")
                self.assertEqual(normal["reference_length"], 10)
                self.assertTrue(normal["attributes_diagnostic_only"])
                arrays = dict(np.load(path / "dataset.npz"))
                self.assertEqual(arrays["x"].shape[1], length)
                with np.load(path / "evaluation_targets.npz") as target:
                    targets.append(target["truth"])
                    event_ids.append(target["event_ids"])
            for i in [1, 2]:
                np.testing.assert_array_equal(targets[0], targets[i])
                np.testing.assert_array_equal(event_ids[0], event_ids[i])


if __name__ == "__main__":
    unittest.main()
