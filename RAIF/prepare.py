"""Construct calendar-aligned RAIF soft-reference labels from dated attributes.

No snapshot fallback exists. Reference hyperparameters are chosen using only
pre-event feature reconstruction on an internal chronological training split.
Future outcomes are accessed only after control identities/weights are fixed.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

HORIZON = 11
CHANNELS = ("AR", "turnover", "log_volume")
EPS = 1e-8


def code6(value):
    text = str(value).split(".")[0] if str(value).endswith(".0") else str(value)
    match = re.search(r"\d{6}", text)
    return match.group(0) if match else text.zfill(6)


def board(code):
    if code.startswith("688"):
        return "STAR"
    if code.startswith("30"):
        return "CHINEXT"
    if code.startswith("6"):
        return "SSE"
    if code.startswith(("0", "2")):
        return "SZSE"
    return "BSE"


def index_code(code):
    return {"STAR": "sh.000688", "CHINEXT": "sz.399006", "SSE": "sh.000001",
            "SZSE": "sz.399001", "BSE": "sh.000905"}[board(code)]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def attributes_ready(evidence):
    """Readiness confirms declared checks, not unavailable historical vintages."""
    checks = ("historical_market_cap_records_verified", "historical_industry_records_verified",
              "conservative_availability_rule_verified")
    return (all(evidence.get(name) is True for name in checks)
            and bool(evidence.get("provenance")) and bool(evidence.get("availability_policy")))


class HistoricalAttributes:
    def __init__(self, cap, industry, codes):
        cap = cap.copy()
        industry = industry.copy()
        for data, required in [(cap, {"code", "date", "total_market_cap", "available_at"}),
                               (industry, {"code", "industry", "effective_from", "effective_to", "available_at"})]:
            if not required.issubset(data):
                raise ValueError(f"Missing historical attribute columns: {required - set(data)}")
            data["code"] = data.code.map(code6)
        for col in ("date", "available_at"):
            cap[col] = pd.to_datetime(cap[col], errors="raise")
        for col in ("effective_from", "effective_to", "available_at"):
            industry[col] = pd.to_datetime(industry[col], errors="raise")
        if cap[["code", "date"]].duplicated().any():
            raise ValueError("Duplicate stock/date market capitalization")
        if cap[["date", "available_at"]].isna().any().any():
            raise ValueError("Market-cap availability date is missing")
        if industry[["effective_from", "available_at", "industry"]].isna().any().any():
            raise ValueError("Industry effective/availability dates or classification are missing")
        if ((industry.effective_to.notna()) & (industry.effective_to <= industry.effective_from)).any():
            raise ValueError("Invalid industry interval: use [effective_from,effective_to)")
        self.codes = pd.Index(codes)
        self.cap_by_date = {date: frame.set_index("code") for date, frame in cap.groupby("date")}
        self.industry = industry.sort_values(["code", "effective_from", "available_at"])
        self._industry_by_code = None

    def at(self, previous_date, query_date):
        caps = np.full(len(self.codes), np.nan)
        frame = self.cap_by_date.get(pd.Timestamp(previous_date))
        if frame is not None:
            valid = frame[frame.available_at <= query_date]
            caps = pd.to_numeric(valid.total_market_cap, errors="coerce").reindex(self.codes).to_numpy(dtype=float, copy=True)
            caps[caps <= 0] = np.nan
        hist = self.industry
        valid = hist[(hist.effective_from <= query_date) & (hist.available_at <= query_date) &
                     (hist.effective_to.isna() | (hist.effective_to > query_date))]
        # Overlapping observable intervals cannot be resolved by choosing a later
        # revision silently; a malformed source must be corrected explicitly.
        if valid.code.duplicated().any():
            raise ValueError(f"Overlapping industry intervals at {query_date}")
        sectors = valid.set_index("code").industry.reindex(self.codes).fillna("").astype(str).to_numpy()
        return caps, sectors

    def at_effective(self, query_date, codes=None):
        """Retrospective effective classification for diagnostics only.

        A record's effective date is not promoted to a verified publication date.
        The returned flag separately preserves the source-update-time proxy.
        """
        hist = self.industry
        valid = hist[(hist.effective_from <= query_date) &
                     (hist.effective_to.isna() | (hist.effective_to > query_date))]
        if valid.code.duplicated().any():
            raise ValueError(f"Overlapping effective industry intervals at {query_date}")
        indexed = valid.set_index("code")
        target_codes = self.codes if codes is None else pd.Index(codes)
        effective = indexed.industry.reindex(target_codes).fillna("").astype(str).to_numpy()
        observable = (indexed.available_at <= query_date).reindex(target_codes, fill_value=False).to_numpy(bool)
        return effective, observable

    def effective_record(self, code, query_date):
        if self._industry_by_code is None:
            self._industry_by_code = {key: frame for key, frame in self.industry.groupby("code")}
        hist = self._industry_by_code.get(code)
        if hist is None:
            return None
        rows = hist[(hist.effective_from <= query_date) &
                    (hist.effective_to.isna() | (hist.effective_to > query_date))]
        if len(rows) > 1:
            raise ValueError(f"Overlapping effective industry intervals for {code} at {query_date}")
        return None if rows.empty else rows.iloc[0]


@dataclass
class DayState:
    day: int
    x: np.ndarray
    v: np.ndarray
    valid: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    cap: np.ndarray
    industry: np.ndarray
    estimate_count: np.ndarray


def market_fit(stock_returns, market_returns, min_obs=30):
    valid = np.isfinite(stock_returns) & np.isfinite(market_returns)
    count = valid.sum(axis=0)
    denom = np.maximum(count, 1)
    sx = np.where(valid, market_returns, 0).sum(axis=0)
    sy = np.where(valid, stock_returns, 0).sum(axis=0)
    sxx = np.where(valid, market_returns * market_returns, 0).sum(axis=0)
    sxy = np.where(valid, stock_returns * market_returns, 0).sum(axis=0)
    variance = sxx - sx * sx / denom
    good = (count >= min_obs) & (variance > 1e-12)
    beta = np.full(stock_returns.shape[1], np.nan)
    beta[good] = (sxy[good] - sx[good] * sy[good] / denom[good]) / variance[good]
    alpha = sy / denom - beta * sx / denom
    return alpha, beta, count


class MarketPanel:
    def __init__(self, source, preflight, external, length, expanded=False, industry_policy="update_proxy"):
        self.calendar = pd.DatetimeIndex(pd.read_csv(preflight / "calendar.csv", parse_dates=["date"]).date)
        if not self.calendar.is_monotonic_increasing or self.calendar.has_duplicates:
            raise ValueError("Invalid common trading calendar")
        self.length = length
        self.expanded = expanded
        if industry_policy not in {"update_proxy", "effective"}:
            raise ValueError("Unknown industry policy")
        self.industry_policy = industry_policy
        daily = pd.read_parquet(source / "control_stock_finder/data/A_share_daily.parquet",
                                columns=["date", "code", "close", "volume", "turnover"])
        daily["date"] = pd.to_datetime(daily.date).dt.normalize()
        daily["code"] = daily.code.map(code6)
        daily = daily[daily.date.isin(self.calendar)]
        if daily[["date", "code"]].duplicated().any():
            raise ValueError("Duplicate stock-date daily observations")
        self.codes = np.array(sorted(daily.code.unique()))
        self.positions = {code: i for i, code in enumerate(self.codes)}
        self.boards = np.array([board(code) for code in self.codes])
        arrays = {}
        for name in ["close", "volume", "turnover"]:
            daily[name] = pd.to_numeric(daily[name], errors="coerce")
            arrays[name] = daily.pivot(index="date", columns="code", values=name).reindex(
                index=self.calendar, columns=self.codes).to_numpy(float)
        close, volume = arrays["close"], arrays["volume"]
        present = np.isfinite(volume) & (volume > 0) & np.isfinite(close) & (close > 0)
        # Returns use consecutive common-calendar closes; no forward filling or
        # row compression across a suspension is permitted.
        self.returns = np.full(close.shape, np.nan)
        np.divide(close[1:], close[:-1], out=self.returns[1:],
                  where=present[1:] & present[:-1])
        self.returns[1:] -= 1
        self.returns[~present] = np.nan
        self.turnover = np.where(present, arrays["turnover"], np.nan)
        self.log_volume = np.full(volume.shape, np.nan)
        np.log(volume, out=self.log_volume, where=present)
        indices = pd.read_parquet(preflight / "indices.parquet")
        indices["date"] = pd.to_datetime(indices.date)
        index_matrix = indices.pivot(index="date", columns="index_code", values="index_ret").reindex(self.calendar)
        self.market = np.stack([index_matrix[index_code(code)].to_numpy(float) for code in self.codes], axis=1)
        cap_path, industry_path = external / "historical_market_cap.parquet", external / "industry_history.parquet"
        cap = (pd.read_parquet(cap_path) if not expanded or cap_path.exists() else
               pd.DataFrame(columns=["code", "date", "total_market_cap", "available_at"]))
        industry = (pd.read_parquet(industry_path) if not expanded or industry_path.exists() else
                    pd.DataFrame(columns=["code", "industry", "effective_from", "effective_to", "available_at"]))
        self.attributes = HistoricalAttributes(cap, industry, self.codes)
        raw = pd.read_csv(source / "original_data/events.csv", dtype={"recommended_stock_code": str})
        raw_dates = pd.to_datetime(raw.recommend_date).dt.normalize()
        raw_pos = self.calendar.searchsorted(raw_dates)
        self.recommended = {}
        # Even recommendations omitted by deduplication or a partition embargo
        # exclude a stock from that day's controls.
        for pos, code in zip(raw_pos, raw.recommended_stock_code.map(code6)):
            self.recommended.setdefault(int(pos), set()).add(code)
        self._last_state = None

    def state(self, day):
        if self._last_state is not None and self._last_state.day == day:
            return self._last_state
        n, length = len(self.codes), self.length
        if day < length or day < 10 or day >= len(self.calendar):
            return None
        lo, hi = max(0, day - 100), day - 10 + 1
        alpha, beta, count = market_fit(self.returns[lo:hi], self.market[lo:hi])
        history = slice(day - length, day)
        ar = self.returns[history] - alpha[None, :] - beta[None, :] * self.market[history]
        x = np.stack([ar, self.turnover[history], self.log_volume[history]], axis=-1).transpose(1, 0, 2)
        cap, sectors = self.attributes.at(self.calendar[day - 1], self.calendar[day])
        if self.industry_policy == "effective":
            sectors, _ = self.attributes.at_effective(self.calendar[day])
        raw = self.returns[history]
        with np.errstate(invalid="ignore", divide="ignore"):
            columns = [ar.T, (np.prod(1 + raw, axis=0) - 1)[:, None],
                       self.turnover[history].mean(axis=0)[:, None],
                       self.log_volume[history].mean(axis=0)[:, None], raw.std(axis=0, ddof=1)[:, None]]
            if not self.expanded:
                columns.append(np.log(cap)[:, None])
            v = np.concatenate(columns, axis=1)
        valid = np.isfinite(v).all(axis=1) & np.isfinite(x).all(axis=(1, 2)) & (count >= 30)
        if not self.expanded:
            valid &= sectors != ""
        state = DayState(day, x, v, valid, alpha, beta, cap, sectors, count)
        self._last_state = state
        return state

    def outcome(self, day, positions, state):
        if day + HORIZON > len(self.calendar):
            return np.full((len(positions), HORIZON, 3), np.nan)
        future = slice(day, day + HORIZON)
        ar = self.returns[future, positions] - state.alpha[positions][None, :] - (
            state.beta[positions][None, :] * self.market[future, positions])
        return np.stack([ar, self.turnover[future, positions], self.log_volume[future, positions]], axis=-1).transpose(1, 0, 2)

    def candidates(self, state, query):
        allowed = state.valid.copy()
        if not self.expanded:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = state.cap / state.cap[query]
            allowed &= (self.boards == self.boards[query]) & (state.industry == state.industry[query])
            allowed &= (ratio >= 0.5) & (ratio <= 2)
        for code in self.recommended.get(state.day, set()):
            if code in self.positions:
                allowed[self.positions[code]] = False
        allowed[query] = False
        return np.flatnonzero(allowed)


@dataclass
class DistanceFit:
    mean: np.ndarray
    std: np.ndarray
    inverse: np.ndarray

    def transform(self, values):
        return (values - self.mean) / self.std


def fit_distance(values, ridge):
    if len(values) < 3:
        raise ValueError("At least three training states are required to fit the distance")
    mean, std = values.mean(axis=0), np.maximum(values.std(axis=0), EPS)
    transformed = (values - mean) / std
    covariance = np.cov(transformed, rowvar=False)
    inverse = np.linalg.inv(covariance + ridge * np.eye(values.shape[1]))
    return DistanceFit(mean, std, inverse)


def ranked_candidates(state, query, candidates, fit, max_m=5):
    v_query = fit.transform(state.v[query])
    v_controls = fit.transform(state.v[candidates])
    diff = v_controls - v_query
    distances = np.maximum(np.einsum("ij,jk,ik->i", diff, fit.inverse, diff), 0)
    order = np.argsort(distances, kind="stable")[:max_m]
    return candidates[order], distances[order], v_query, v_controls[order]


def select_reference(distances, v_query, v_controls, m, temperature, maximum):
    selected = np.flatnonzero(distances <= maximum)[:m]
    if not len(selected):
        return None
    chosen = distances[selected]
    logits = -(chosen - chosen.min()) / max(temperature, EPS)
    weights = np.exp(logits)
    weights /= weights.sum()
    balance = float(np.mean((v_query - weights @ v_controls[selected]) ** 2))
    return selected, weights, balance, float(weights @ chosen), float(1 / (weights @ weights))


def fixed_label(query_outcome, selected_outcomes, weights):
    """Reject an incomplete fixed reference; never reweight the survivors."""
    if not np.isfinite(query_outcome).all() or not np.isfinite(selected_outcomes).all():
        return None
    baseline = np.einsum("j,jhc->hc", weights, selected_outcomes)
    return query_outcome - baseline, baseline


def strict_candidates(panel, state, query, candidates, correlation_minimum=.8):
    """Apply the paper's fixed strict criteria on ten pre-event sessions."""
    if state.day < 10 or not len(candidates):
        return candidates[:0]
    positions = np.r_[query, candidates]
    history = slice(state.day - 10, state.day)
    raw = panel.returns[history, positions]
    market = panel.market[history, positions]
    ar = raw - state.alpha[positions][None, :] - state.beta[positions][None, :] * market
    turnover = panel.turnover[history, positions]
    finite = np.isfinite(ar).all(axis=0) & np.isfinite(raw).all(axis=0) & np.isfinite(turnover).all(axis=0)
    if not finite[0]:
        return candidates[:0]
    centered = ar - ar.mean(axis=0)
    norm = np.sqrt(np.sum(centered * centered, axis=0))
    corr = np.full(len(candidates), np.nan)
    np.divide(np.sum(centered[:, :1] * centered[:, 1:], axis=0), norm[0] * norm[1:],
              out=corr, where=(norm[0] * norm[1:]) > 0)
    volatility = raw.std(axis=0, ddof=1)
    mean_turnover = turnover.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        volatility_ratio = volatility[1:] / volatility[0]
        turnover_ratio = mean_turnover[1:] / mean_turnover[0]
    keep = finite[1:] & (corr >= correlation_minimum) & (np.abs(ar[:, 1:].sum(axis=0) - ar[:, 0].sum()) <= .2)
    keep &= (volatility_ratio >= .5) & (volatility_ratio <= 2.)
    keep &= (turnover_ratio >= .7) & (turnover_ratio <= 1.5)
    return candidates[keep]


def scan_training_states(panel, events):
    rows, states = [], []
    for row in events[events.split == "train"].itertuples():
        state = panel.state(int(row.day))
        query = panel.positions.get(str(row.stock))
        if state is not None and query is not None and state.valid[query]:
            rows.append(row.Index)
            states.append(state.v[query])
    if not rows:
        raise ValueError("No training states with complete history and historical attributes")
    return np.array(rows), np.stack(states)


def collect_rankings(panel, events, indices, fit, max_m):
    result = []
    for index in indices:
        row = events.loc[index]
        state = panel.state(int(row.day))
        query = panel.positions[str(row.stock)]
        candidates = panel.candidates(state, query)
        ids, distances, vq, vc = ranked_candidates(state, query, candidates, fit, max_m)
        result.append((int(index), ids, distances, vq, vc))
    return result


def tune_reference(panel, events, training_indices, training_states, ridge, m_values, quantiles, temperatures):
    train_dates = events.loc[training_indices, "day"].to_numpy()
    days = np.unique(train_dates)
    if len(days) < 10:
        raise ValueError("Insufficient training dates for reference parameter selection")
    cut = int(days[max(1, int(len(days) * 0.8))])
    # No response is used by this procedure. A temporal gap is nonetheless kept
    # so its fit/selection periods match the experiment's time-isolation rule.
    fit_mask = train_dates + HORIZON - 1 < cut
    validation_mask = train_dates >= cut + HORIZON
    if fit_mask.sum() < 10 or validation_mask.sum() < 5:
        raise ValueError("Insufficient observations in internal reference fitting/selection periods")
    fit = fit_distance(training_states[fit_mask], ridge)
    rankings_fit = collect_rankings(panel, events, training_indices[fit_mask], fit, max(m_values))
    rankings_val = collect_rankings(panel, events, training_indices[validation_mask], fit, max(m_values))
    nonempty = [row[2] for row in rankings_fit if len(row[2])]
    if not nonempty:
        raise ValueError("No controls in reference-fitting interval")
    distribution = np.concatenate(nonempty)
    temperature_base = max(float(np.median(distribution)), EPS)
    nearest_balances = [float(np.mean((row[3] - row[4][0]) ** 2)) for row in rankings_fit if len(row[2])]
    failure_penalty = max(2 * float(np.quantile(nearest_balances, 0.95)), EPS)
    trials = []
    for m in m_values:
        for quantile in quantiles:
            maximum = float(np.quantile(distribution, quantile))
            for factor in temperatures:
                scores, coverage = [], 0
                for _, ids, distances, vq, vc in rankings_val:
                    selected = select_reference(distances, vq, vc, m, temperature_base * factor, maximum)
                    scores.append(failure_penalty if selected is None else selected[2])
                    coverage += selected is not None
                trials.append({"m": int(m), "distance_quantile": float(quantile), "temperature_factor": float(factor),
                               "objective": float(np.mean(scores)), "n_selection_events": len(scores),
                               "covered": int(coverage), "failure_penalty": failure_penalty})
    best = min(trials, key=lambda row: (row["objective"], -row["covered"], row["m"], row["temperature_factor"]))
    return best, trials, {"selection_start_day": cut + HORIZON, "fit_last_day_exclusive": cut - HORIZON + 1,
        "fit_event_ids": events.loc[training_indices[fit_mask], "event_id"].tolist(),
        "selection_event_ids": events.loc[training_indices[validation_mask], "event_id"].tolist(),
        "objective": "Mean standardized pre-event state reconstruction MSE; unmatched events receive a fixed fit-only 2*p95 nearest-control MSE penalty.",
        "outcome_labels_used": False, "common_selection_events_include_unmatched": True}


def encode_categories(values, training):
    names = sorted(set(values[training]) - {"__UNKNOWN__"})
    vocabulary = {"__UNKNOWN__": 0, **{name: i + 1 for i, name in enumerate(names)}}
    return np.array([vocabulary.get(value, 0) for value in values], dtype=np.int64), vocabulary


def event_timing_checks(events, calendar):
    if events.event_id.duplicated().any() or not events.split.isin(["train", "validation", "test", "excluded"]).all():
        raise ValueError("Duplicate event IDs or invalid temporal split")
    eligible = events[events.split != "excluded"]
    day = eligible.day.to_numpy(dtype=float)
    if not np.isfinite(day).all() or not np.array_equal(day, np.floor(day)):
        raise ValueError("Noninteger trading-day positions")
    day = day.astype(int)
    end = eligible.end_day.to_numpy(dtype=int)
    if (day < 0).any() or (end >= len(calendar)).any() or not np.array_equal(end, day + 10):
        raise ValueError("Invalid event response windows")
    if not np.array_equal(calendar[day].to_numpy(), pd.to_datetime(eligible.event_date).to_numpy()):
        raise ValueError("Event dates do not match common-calendar positions")
    if not np.array_equal(calendar[end].to_numpy(), pd.to_datetime(eligible.end_date).to_numpy()):
        raise ValueError("Response dates do not match common-calendar positions")
    for earlier, later in [("train", "validation"), ("validation", "test")]:
        a, b = eligible[eligible.split == earlier], eligible[eligible.split == later]
        if a.empty or b.empty or a.end_day.max() >= b.day.min():
            raise ValueError("Empty or overlapping temporal partitions")
    return {"calendar_unique_sorted": True, "daily_stock_date_keys_unique": True,
            "explicit_market_index_returns_loaded": True, "event_calendar_alignment": True,
            "response_length": HORIZON, "partition_response_windows_disjoint": True,
            "market_model_estimation_offsets": [-100, -10], "minimum_estimation_pairs": 30,
            "suspension_calendar_gaps_not_compressed": True,
            "attributes_required_for_matching": False,
            "intraday_publication_timestamps_verified": False}


def save_matching_flow(output, diagnostics):
    frame = pd.DataFrame(diagnostics)
    frame.to_csv(output / "matching_diagnostics.csv", index=False)
    flow = []
    for group in ["all", "eligible", "train", "validation", "test", "excluded"]:
        rows = frame if group == "all" else frame[frame.temporally_eligible] if group == "eligible" else frame[frame.split == group]
        for stage, column in [("input_events", None), ("complete_prestate", "prestate_available"),
                              ("fixed_nearest_reference", "reference_available"),
                              ("complete_fixed_reference_label", "complete_label"), ("retained_for_model", "included")]:
            flow.append({"group": group, "stage": stage, "n_events": len(rows) if column is None else int(rows[column].sum())})
    pd.DataFrame(flow).to_csv(output / "sample_flow.csv", index=False)


def date_text(value):
    if pd.isna(value) or str(value).strip() == "":
        return ""
    return str(pd.Timestamp(value).date())


def industry_assignment(attributes, code, query_date, names):
    row = attributes.effective_record(code, query_date)
    result = {"stock": code, "query_date": date_text(query_date), "assignment_found": row is not None,
        "industry": "", "industry_name_zh": "", "original_industry_code": "", "effective_from": "",
        "effective_to": "", "original_update_time": "", "update_proxy_available_at": "",
        "update_proxy_observable": False, "publication_time": "", "publication_time_verified": False,
        "source_update_is_first_publication": False,
        "industry_name_scope": "SW2021 display dictionary; historical names are not reconstructed",
        "evidence_scope": "Retrospective historical effective assignment; individual publication time is unknown",
        "used_for_model_matching": False}
    if row is not None:
        result.update(industry=str(row.industry), industry_name_zh=names.get(str(row.industry), ""),
            original_industry_code=str(row.get("original_industry_code", "")),
            effective_from=date_text(row.effective_from), effective_to=date_text(row.effective_to),
            original_update_time=date_text(row.get("update_time", "")),
            update_proxy_available_at=date_text(row.available_at), update_proxy_observable=bool(row.available_at <= query_date))
    return result


def load_industry_recovery(directory, attributes, output):
    recovery = pd.read_parquet(directory / "event_recovery.parquet")
    if recovery.event_id.duplicated().any():
        raise ValueError("Duplicate recovered event IDs")
    names = json.loads((directory / "industry_level2_names.json").read_text())
    recovery["stock"] = recovery.stock.map(code6)
    recovery["event_date"] = pd.to_datetime(recovery.event_date)
    rows = []
    for date, group in recovery.groupby("event_date"):
        effective, proxy = attributes.at_effective(date, group.stock.to_numpy())
        for position, (_, original) in enumerate(group.iterrows()):
            detail = industry_assignment(attributes, original.stock, date, names)
            checks = {
                "industry_matches": str(original.industry) == effective[position] == detail["industry"],
                "original_code_matches": str(original.original_industry_code) == detail["original_industry_code"],
                "display_name_matches": str(original.industry_name_zh) == detail["industry_name_zh"],
                "effective_from_matches": date_text(original.effective_from) == detail["effective_from"],
                "effective_to_matches": date_text(original.effective_to) == detail["effective_to"],
                "original_update_matches": date_text(original.update_time) == detail["original_update_time"],
                "publication_unknown_preserved": not bool(original.publication_time_verified) and date_text(original.publication_time) == "",
            }
            rows.append({"event_id": original.event_id, "stock": original.stock, "event_date": date_text(date),
                         "recovered_industry": str(original.industry), "at_effective_industry": effective[position],
                         "update_proxy_observable": bool(proxy[position]), **checks, "all_match": all(checks.values())})
    verification = pd.DataFrame(rows)
    verification.to_csv(output / "industry_recovery_verification.csv", index=False)
    if verification.empty or not verification.all_match.all():
        raise ValueError("Industry recovery differs from the original effective-history records; see verification CSV")
    metadata = {"source_directory": str(directory), "events_verified": len(recovery),
                "stocks_verified": int(recovery.stock.nunique()), "effective_record_comparison_passed": True,
                "publication_times_verified": 0, "matching_features_changed": False,
                "verification_scope": "Effective classification, interval bounds, original code and source update agree; original publication times remain unknown"}
    return recovery, names, metadata


def save_recovery_outcomes(output, recovery, diagnostics, metadata):
    frame = pd.DataFrame(diagnostics)
    fields = ["event_id", "split", "temporally_eligible", "prestate_available", "reference_available",
              "complete_label", "included", "control", "exclusion_reason", "data_exclusion_reason"]
    outcome = recovery.rename(columns={"reason": "previous_exclusion_reason"}).merge(
        frame[fields], on="event_id", how="left", validate="one_to_one", indicator=True)
    absent = outcome["_merge"] == "left_only"
    outcome.loc[absent, "exclusion_reason"] = "not_in_current_event_manifest"
    outcome["included"] = outcome.included.fillna(False).astype(bool)
    outcome.to_csv(output / "industry_recovery_outcomes.csv", index=False)
    summary = dict(metadata, n_included=int(outcome.included.sum()), n_excluded=int((~outcome.included).sum()),
        n_absent_from_current_manifest=int(absent.sum()),
        excluded_reasons=outcome.loc[~outcome.included, "exclusion_reason"].value_counts(dropna=False).to_dict(),
        included_by_split=outcome.loc[outcome.included, "split"].value_counts().to_dict())
    write_json(output / "industry_recovery_summary.json", summary)


def run(args):
    start = time.monotonic()
    reference_mode = getattr(args, "reference_mode", "soft")
    fixed_path = getattr(args, "fixed_reference_config", None)
    m_override = getattr(args, "m_override", None)
    distance_multiplier = getattr(args, "distance_multiplier", 1.0)
    recovery_directory = getattr(args, "industry_recovery", None)
    industry_policy = getattr(args, "industry_policy", "update_proxy")
    strict_correlation = getattr(args, "strict_correlation", .8)
    distance_fit_path = getattr(args, "distance_fit_config", None)
    quality_effective_n_factor = getattr(args, "quality_effective_n_factor", "legacy")
    if quality_effective_n_factor not in {"legacy", "none"}:
        raise ValueError("Invalid effective-N quality factor policy")
    if not 0 <= strict_correlation <= 1:
        raise ValueError("Strict correlation must be between zero and one")
    if (industry_policy != "update_proxy" or strict_correlation != .8 or distance_fit_path is not None) and reference_mode != "strict":
        raise ValueError("Revised industry/correlation/distance settings apply only to the strict comparator")
    expanded = reference_mode == "expanded_nearest"
    if reference_mode not in {"soft", "strict", "nearest", "expanded_nearest"}:
        raise ValueError("Invalid reference mode")
    if recovery_directory is not None and not expanded:
        raise ValueError("Industry recovery diagnostics apply only to expanded_nearest")
    if fixed_path is None and (m_override is not None or distance_multiplier != 1.0):
        raise ValueError("M/distance overrides require a frozen training-selected reference configuration")
    if fixed_path is not None and reference_mode != "soft":
        raise ValueError("Frozen reference sweeps apply only to soft references")
    if m_override is not None and distance_multiplier != 1.0:
        raise ValueError("Change only M or the distance threshold in one reference sweep")
    if distance_multiplier <= 0 or (m_override is not None and m_override < 1):
        raise ValueError("Reference sweep parameters must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "dataset.npz").exists():
        raise FileExistsError("Prepared dataset already exists; choose a fresh output directory")
    required = [args.external / "historical_market_cap.parquet", args.external / "industry_history.parquet"]
    if not expanded and not all(path.exists() for path in required):
        raise FileNotFoundError("Historical data required; no snapshot fallback: " + str([str(p) for p in required if not p.exists()]))
    evidence_path = args.external / "attribute_verification.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    verified = attributes_ready(evidence)
    inputs = [path for path in required if path.exists()] + [args.source / "control_stock_finder/data/A_share_daily.parquet",
                         args.preflight / "events_manifest.csv", args.preflight / "calendar.csv", args.preflight / "indices.parquet"]
    if expanded:
        inputs.append(args.source / "original_data/events.csv")
    if recovery_directory is not None:
        recovery_directory = Path(recovery_directory)
        for name in ["event_recovery.parquet", "stock_intervals.parquet", "industry_level2_names.json",
                     "source_evidence.json", "source_manifest.json"]:
            path = recovery_directory / name
            if not path.exists():
                raise FileNotFoundError("Missing industry recovery evidence: " + str(path))
            inputs.append(path)
    if evidence_path.exists():
        inputs.append(evidence_path)
    source_manifest = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)} for path in inputs]
    pd.DataFrame(source_manifest).to_csv(args.output / "source_manifest.csv", index=False)
    events = pd.read_csv(args.preflight / "events_manifest.csv", dtype={"stock": str, "event_id": str},
                         parse_dates=["event_date", "end_date"])
    events["stock"] = events.stock.map(code6)
    events = events.sort_values(["day", "stock"], kind="stable").reset_index(drop=True)
    panel = MarketPanel(args.source, args.preflight, args.external, args.length, expanded=expanded,
                        industry_policy=industry_policy)
    market_checks = event_timing_checks(events, panel.calendar) if expanded else None
    if expanded:
        # This readiness concerns observable market features and partition timing;
        # it makes no claim that missing industry history has been recovered.
        verified = True
        write_json(args.output / "market_data_timing_checks.json", market_checks)
    recovery, industry_names, recovery_metadata = None, {}, None
    if recovery_directory is not None:
        recovery, industry_names, recovery_metadata = load_industry_recovery(recovery_directory, panel.attributes, args.output)
    print(json.dumps({"stage": "panel_loaded", "stocks": len(panel.codes), "dates": len(panel.calendar)}), flush=True)
    train_indices, train_states = scan_training_states(panel, events)
    base_config = None
    if fixed_path is not None:
        fixed_path = Path(fixed_path)
        base_config = json.loads(fixed_path.read_text())
        if base_config.get("reference_mode") != "soft" or base_config.get("length") != args.length:
            raise ValueError("Frozen configuration must be a soft reference with the same history window")
        if base_config.get("horizon") != HORIZON or base_config.get("channels") != list(CHANNELS):
            raise ValueError("Frozen configuration has incompatible targets")
        current_hashes = {Path(row["path"]).name: row["sha256"] for row in source_manifest}
        if base_config.get("source_hashes") != current_hashes:
            raise ValueError("Frozen configuration was fitted on different source files")
        if not set(base_config["state_fit_event_ids"]).issubset(set(events.loc[events.split == "train", "event_id"])):
            raise ValueError("Frozen distance uses events outside the current training partition")
        fit = DistanceFit(np.asarray(base_config["state_mean"], dtype=float),
                          np.asarray(base_config["state_std"], dtype=float),
                          np.asarray(base_config["inverse_covariance"], dtype=float))
        p = train_states.shape[1]
        if fit.mean.shape != (p,) or fit.std.shape != (p,) or fit.inverse.shape != (p, p):
            raise ValueError("Frozen distance dimensions do not match state features")
        if not all(np.isfinite(x).all() for x in [fit.mean, fit.std, fit.inverse]) or (fit.std <= 0).any():
            raise ValueError("Invalid frozen distance parameters")
        best = {"m": int(m_override if m_override is not None else base_config["m"])}
        maximum = float(base_config["distance_maximum"]) * distance_multiplier
        temperature = float(base_config["temperature"])
        if not np.isfinite(maximum) or maximum < 0 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid frozen distance threshold or temperature")
        trials = []
        tuning = {"objective": "Frozen training-selected base configuration; only explicit sweep parameter changed",
                  "outcome_labels_used": False, "base_configuration": str(fixed_path),
                  "base_sha256": digest(fixed_path), "m_override": m_override,
                  "distance_multiplier": float(distance_multiplier)}
    elif distance_fit_path is not None:
        base_config = json.loads(Path(distance_fit_path).read_text())
        current_hashes = {Path(row["path"]).name: row["sha256"] for row in source_manifest}
        if base_config.get("source_hashes") != current_hashes:
            raise ValueError("Frozen distance uses different source files")
        if base_config.get("length") != args.length or base_config.get("reference_mode") != "strict":
            raise ValueError("Frozen distance must come from the original strict comparator with the same window")
        if not set(base_config["state_fit_event_ids"]).issubset(set(events.loc[events.split == "train", "event_id"])):
            raise ValueError("Frozen distance contains nontraining events")
        fit = DistanceFit(np.asarray(base_config["state_mean"], dtype=float),
                          np.asarray(base_config["state_std"], dtype=float),
                          np.asarray(base_config["inverse_covariance"], dtype=float))
        dimension = train_states.shape[1]
        if fit.mean.shape != (dimension,) or fit.std.shape != (dimension,) or fit.inverse.shape != (dimension, dimension):
            raise ValueError("Frozen distance dimensions do not match")
        if not all(np.isfinite(x).all() for x in [fit.mean, fit.std, fit.inverse]) or (fit.std <= 0).any():
            raise ValueError("Invalid frozen distance")
    else:
        fit = fit_distance(train_states, args.ridge)
    if fixed_path is None and reference_mode == "soft":
        best, trials, tuning = tune_reference(panel, events, train_indices, train_states, args.ridge,
                                              args.m_grid, args.quantile_grid, args.temperature_grid)
        train_rankings = collect_rankings(panel, events, train_indices, fit, max(args.m_grid))
        distances = np.concatenate([row[2] for row in train_rankings if len(row[2])])
        maximum = float(np.quantile(distances, best["distance_quantile"]))
        temperature = max(float(np.median(distances)), EPS) * best["temperature_factor"]
    elif fixed_path is None:
        best, trials = {"m": 1}, []
        tuning = {"objective": "Fixed prespecified comparator; no reference hyperparameter selection", "outcome_labels_used": False}
        maximum, temperature = np.inf, 1.0
    pd.DataFrame(trials).to_csv(args.output / "reference_tuning.csv", index=False)
    write_json(args.output / "reference_tuning_protocol.json", tuning)
    config = {"length": args.length, "horizon": HORIZON, "channels": list(CHANNELS), "ridge": args.ridge,
              "reference_mode": reference_mode, "m": best["m"], "distance_maximum": maximum if np.isfinite(maximum) else None, "temperature": temperature,
              "selected_relative_parameters": best, "reference_hyperparameters_tuned": reference_mode == "soft" and fixed_path is None,
              "state_fit_event_ids": events.loc[train_indices, "event_id"].tolist(),
              "state_mean": fit.mean.tolist(), "state_std": fit.std.tolist(), "inverse_covariance": fit.inverse.tolist(),
              "state_definition": "L ARs, compounded raw return, mean turnover, mean log(volume), sample SD raw returns, log(prior-close total cap)",
              "market_model_window": [-100, -10], "minimum_market_model_observations": 30,
              "industry_interval_convention": "[effective_from,effective_to)",
              "source_hashes": {Path(row["path"]).name: row["sha256"] for row in source_manifest},
              "scope": "soft labels only; strict/nearest-reference common-target comparisons are not generated"}
    if reference_mode != "soft":
        config["scope"] = "Reference-specific training labels; common strict test targets are prepared separately by reference_experiments.py"
    if reference_mode == "strict":
        config["strict_criteria"] = {"lookback": 10, "ar_correlation_minimum": strict_correlation,
            "absolute_sum_ar_difference_maximum": .2, "raw_return_std_ratio": [.5, 2.],
            "mean_turnover_ratio": [.7, 1.5], "ranking": "training-fitted Mahalanobis", "n_references": 1}
    if industry_policy == "effective":
        config["industry_policy"] = "historical_effective_interval"
        config["industry_publication_times_verified"] = False
        config["scope"] = "User-adopted strict comparator with historical-effective industry intervals; original publication timestamps remain unverified"
    if distance_fit_path is not None:
        config["state_fit_event_ids"] = base_config["state_fit_event_ids"]
        config["ridge"] = base_config["ridge"]
        config["frozen_distance_fit"] = {"configuration": str(distance_fit_path),
            "sha256": digest(Path(distance_fit_path)), "purpose": "Hold training-fit distance fixed while revising industry eligibility and correlation threshold"}
    if quality_effective_n_factor == "none":
        config["quality_effective_n_factor"] = False
        config["quality_neff_factor_enabled"] = False
        config["q_rule"] = "exp_only"
        config["quality_formula"] = "max(exp(-balance/s_b - mean_distance/s_d), 1e-8)"
    if expanded:
        config["scope"] = "One nearest nonrecommended stock from all cached market stocks with complete observable pre-event core features"
        config["state_definition"] = "L ARs, compounded raw return, mean turnover, mean log(volume), sample SD raw returns; no market cap or industry feature"
        config["candidate_constraints"] = {"same_day_not_recommended": True, "exclude_query_stock": True,
            "complete_core_prestate": True, "same_industry": False, "same_board": False,
            "market_cap_ratio": None, "distance_threshold": None}
        config["distance_tie_break"] = "Ascending stock code; stable order on equal Mahalanobis distance"
        config["attributes_diagnostic_only"] = True
        config["diagnostic_industry_basis"] = "Historical effective interval, with source-update-time proxy observability recorded separately; not verified original publication time"
        if recovery_metadata is not None:
            config["industry_recovery"] = recovery_metadata
    if fixed_path is not None:
        config["frozen_reference_config"] = tuning
        config["state_fit_event_ids"] = base_config["state_fit_event_ids"]
        config["ridge"] = base_config["ridge"]
        config["scope"] = "Frozen-reference parameter sweep; evaluate changed labels on a common strict target via reference_experiments.py"
    version = reference_mode + "_" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    config["label_version"] = version
    write_json(args.output / "config.json", config)
    x_rows, y_rows, outcome_rows, baseline_rows = [], [], [], []
    kept, references, quality, exclusions, diagnostics, industry_records = [], [], [], [], [], []
    for row in events.itertuples():
        reason = str(row.exclusion_reason) if row.split == "excluded" else ""
        eligible = row.split != "excluded"
        diagnostic = {"event_id": row.event_id, "event_date": str(row.event_date), "stock": row.stock,
            "split": row.split, "temporally_eligible": eligible, "prestate_available": False,
            "reference_available": False, "complete_label": False, "included": False,
            "control": "", "candidate_count": 0, "industry_same": None, "board_same": None,
            "market_cap_ratio": np.nan, "distance": np.nan,
            "query_industry": "", "control_industry": "", "query_total_market_cap": np.nan,
            "control_total_market_cap": np.nan, "query_industry_effective": "", "control_industry_effective": "",
            "query_industry_update_proxy_observable": None, "control_industry_update_proxy_observable": None,
            "exclusion_reason": reason, "data_exclusion_reason": ""}
        if expanded:
            diagnostics.append(diagnostic)
            industry_records.append(dict(industry_assignment(panel.attributes, row.stock, row.event_date, industry_names),
                                         event_id=row.event_id, role="query"))

        def exclude(data_reason):
            diagnostic["data_exclusion_reason"] = data_reason
            if eligible:
                diagnostic["exclusion_reason"] = data_reason
            exclusions.append({"event_id": row.event_id, "reason": diagnostic["exclusion_reason"]})

        if not eligible and not expanded:
            exclusions.append({"event_id": row.event_id, "reason": reason})
            continue
        day, query = int(row.day), panel.positions.get(row.stock)
        state = panel.state(day)
        if query is None:
            exclude("stock_absent_from_cached_universe")
            continue
        if state is None:
            exclude("insufficient_common_calendar_history")
            continue
        effective_industry = observable_industry = None
        if expanded:
            effective_industry, observable_industry = panel.attributes.at_effective(panel.calendar[day])
            diagnostic["query_industry_effective"] = str(effective_industry[query])
            if effective_industry[query]:
                diagnostic["query_industry_update_proxy_observable"] = bool(observable_industry[query])
        if not state.valid[query]:
            if not expanded and not np.isfinite(state.cap[query]):
                reason = "missing_prior_close_historical_market_cap"
            elif not expanded and state.industry[query] == "":
                reason = "missing_observable_effective_industry"
            elif state.estimate_count[query] < 30:
                reason = "fewer_than_30_market_model_observations"
            elif not np.isfinite(state.x[query]).all():
                reason = "incomplete_adjacent_session_pre_event_channels"
            else:
                reason = "invalid_market_model_or_matching_state"
            exclude(reason)
            continue
        diagnostic["prestate_available"] = True
        diagnostic["query_industry"] = str(state.industry[query])
        diagnostic["query_total_market_cap"] = float(state.cap[query])
        candidates = panel.candidates(state, query)
        diagnostic["candidate_count"] = len(candidates)
        if reference_mode == "strict":
            candidates = strict_candidates(panel, state, query, candidates, strict_correlation)
        ids, distance, vq, vc = ranked_candidates(state, query, candidates, fit, best["m"])
        selection = select_reference(distance, vq, vc, best["m"], temperature, maximum)
        if selection is None:
            exclude("no_reference_within_frozen_threshold" if reference_mode == "soft" else "no_" + reference_mode + "_reference")
            continue
        select, weights, balance, mean_distance, effective_n = selection
        chosen = ids[select]
        diagnostic["reference_available"] = True
        if expanded:
            control = chosen[0]
            industry_records.append(dict(industry_assignment(panel.attributes, panel.codes[control], row.event_date, industry_names),
                                         event_id=row.event_id, role="control"))
            diagnostic.update(control=panel.codes[control], distance=float(distance[select[0]]),
                board_same=bool(panel.boards[query] == panel.boards[control]),
                control_industry=str(state.industry[control]), control_total_market_cap=float(state.cap[control]),
                control_industry_effective=str(effective_industry[control]))
            if effective_industry[control]:
                diagnostic["control_industry_update_proxy_observable"] = bool(observable_industry[control])
            if effective_industry[query] and effective_industry[control]:
                diagnostic["industry_same"] = bool(effective_industry[query] == effective_industry[control])
            if np.isfinite(state.cap[[query, control]]).all():
                diagnostic["market_cap_ratio"] = float(state.cap[control] / state.cap[query])
        # Identity and weight records are fixed before accessing either outcome.
        pair_rows = [{"event_id": row.event_id, "control": panel.codes[c], "distance": float(distance[s]),
                      "weight": float(w), "selected_at": str(panel.calendar[day].date()),
                      "market_cap_date": str(panel.calendar[day - 1].date())}
                     for c, s, w in zip(chosen, select, weights)]
        outcomes = panel.outcome(day, np.r_[query, chosen], state)
        label = fixed_label(outcomes[0], outcomes[1:], weights)
        diagnostic["complete_label"] = label is not None
        for pair in pair_rows:
            pair["included_after_outcome_check"] = label is not None
            if expanded:
                pair["retained_for_model"] = label is not None and eligible
        references.extend(pair_rows)
        if label is None:
            exclude("incomplete_fixed_reference_or_query_response")
            continue
        if not eligible:
            exclusions.append({"event_id": row.event_id, "reason": diagnostic["exclusion_reason"]})
            continue
        diagnostic["included"] = True
        delta, baseline = label
        kept.append(row.Index)
        x_rows.append(state.x[query])
        y_rows.append(delta)
        outcome_rows.append(outcomes[0])
        baseline_rows.append(baseline)
        quality.append({"event_id": row.event_id, "balance": balance, "mean_distance": mean_distance,
                        "effective_n": effective_n, "n_references": len(chosen), "candidate_count": len(candidates)})
        if len(kept) % 200 == 0:
            print(json.dumps({"stage": "labels", "retained": len(kept), "last_date": str(row.event_date.date())}), flush=True)
    pd.DataFrame(references).to_csv(args.output / "reference_pairs.csv", index=False)
    pd.DataFrame(exclusions).to_csv(args.output / "exclusions.csv", index=False)
    if expanded:
        save_matching_flow(args.output, diagnostics)
        pd.DataFrame(industry_records).to_csv(args.output / "industry_assignments.csv", index=False)
        if recovery is not None:
            save_recovery_outcomes(args.output, recovery, diagnostics, recovery_metadata)
    if not kept:
        raise ValueError("No complete fixed-reference labels")
    retained = events.loc[kept].copy().reset_index(drop=True)
    split = retained.split.map({"train": 0, "validation": 1, "test": 2}).to_numpy(np.int8)
    if any(not np.any(split == s) for s in [0, 1, 2]):
        raise ValueError("At least one temporal split has no complete labels")
    training = split == 0
    x, y = np.stack(x_rows), np.stack(y_rows)
    qframe = pd.DataFrame(quality)
    s_b = max(float(np.median(qframe.loc[training, "balance"])), EPS)
    s_d = max(float(np.median(qframe.loc[training, "mean_distance"])), EPS)
    q = np.exp(-qframe.balance.to_numpy() / s_b - qframe.mean_distance.to_numpy() / s_d)
    if quality_effective_n_factor == "legacy":
        q *= qframe.effective_n.to_numpy() / (qframe.effective_n.to_numpy() + 1)
    q = np.maximum(q, EPS).astype(np.float32)
    qframe["q"] = q
    qframe.to_csv(args.output / "reference_quality.csv", index=False)
    x_mean, x_std = x[training].mean(axis=(0, 1)), np.maximum(x[training].std(axis=(0, 1)), EPS)
    y_mean, y_std = y[training].mean(axis=(0, 1)), np.maximum(y[training].std(axis=(0, 1)), EPS)
    platform, platform_vocab = encode_categories(retained.platform.to_numpy(), training)
    influencer, influencer_vocab = encode_categories(retained.influencer.to_numpy(), training)
    np.savez_compressed(args.output / "dataset.npz", x=((x - x_mean) / x_std).astype(np.float32),
        y=((y - y_mean) / y_std).astype(np.float32), q=q, platform=platform, influencer=influencer,
        day=retained.day.to_numpy(np.int64), end_day=retained.end_day.to_numpy(np.int64), split=split)
    retained["q"], retained["label_version"] = q, version
    retained.to_csv(args.output / "events.csv", index=False)
    normal = {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        "n_platforms": len(platform_vocab), "n_influencers": len(influencer_vocab), "platform_vocab": platform_vocab,
        "influencer_vocab": influencer_vocab, "quality_balance_scale": s_b, "quality_distance_scale": s_d,
        "quality_floor": EPS, "normalization_fit_event_ids": retained.loc[training, "event_id"].tolist(),
        "ready_for_formal_training": bool(verified), "attribute_verification": evidence,
        "readiness_meaning": "Source/record checks and the explicitly documented conservative availability rule passed; this flag does not certify original historical data vintages.",
        "reference_hyperparameters_tuned": reference_mode == "soft" and fixed_path is None,
        "reference_mode": reference_mode, "label_version": version,
        "limitations": ["Date-only recommendations cannot verify intraday 15:00 mapping.",
                        "Historical attributes are retrospective downloads; source revision availability is documented separately. Original historical-vintage verification is reported in attribute_verification.",
                        "Stock universe is restricted to the existing cached price data."]}
    if expanded:
        normal["market_data_timing_checks"] = market_checks
        normal["readiness_meaning"] = "Observable market-feature and temporal-partition checks passed. Industry and market cap are diagnostic only; missing historical attributes are not claimed recovered."
        normal["attributes_diagnostic_only"] = True
        normal["diagnostic_industry_basis"] = config["diagnostic_industry_basis"]
        if recovery_metadata is not None:
            normal["industry_recovery"] = recovery_metadata
        normal["limitations"].append("Cross-industry and cross-board controls are permitted; historical-attribute completeness is reported separately in matching_diagnostics.csv.")
    if industry_policy == "effective":
        normal["industry_policy"] = config["industry_policy"]
        normal["industry_publication_times_verified"] = False
        normal["limitations"].append("Matching uses retrospective historical-effective industry intervals; first-publication timestamps are unverified.")
    if quality_effective_n_factor == "none":
        normal["quality_effective_n_factor"] = False
        normal["quality_neff_factor_enabled"] = False
        normal["q_rule"] = "exp_only"
        normal["quality_formula"] = config["quality_formula"]
    write_json(args.output / "normalization.json", normal)
    repeated = retained.loc[retained.index.repeat(HORIZON * 3), ["event_id", "event_date", "end_date"]].reset_index(drop=True)
    repeated["tau"] = np.tile(np.repeat(np.arange(HORIZON), 3), len(retained))
    repeated["channel"] = np.tile(CHANNELS, len(retained) * HORIZON)
    repeated["y"] = np.stack(outcome_rows).reshape(-1)
    repeated["baseline"] = np.stack(baseline_rows).reshape(-1)
    repeated["delta"] = y.reshape(-1)
    repeated["label_version"] = version
    repeated["target_available_at"] = pd.to_datetime(repeated.end_date) + pd.Timedelta(days=1)
    repeated.to_csv(args.output / "labels.csv", index=False)
    retained.groupby("split").agg(n_events=("event_id", "size"), n_dates=("event_date", "nunique"),
        first_date=("event_date", "min"), last_date=("event_date", "max")).to_csv(args.output / "coverage.csv")
    retained.loc[retained.split == "test", ["event_id", "event_date", "label_version"]].to_csv(args.output / "evaluation_manifest.csv", index=False)
    summary = {"status": "prepared" if verified else "prepared_but_formal_training_blocked", "n_events": len(retained),
               "split_counts": retained.split.value_counts().to_dict(), "seconds": time.monotonic() - start,
               "label_version": version, "ready_for_formal_training": bool(verified)}
    write_json(args.output / "preparation_summary.json", summary)
    print(json.dumps(summary), flush=True)


def parser():
    p = argparse.ArgumentParser()
    for name in ["source", "preflight", "external", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--length", type=int, default=10)
    p.add_argument("--reference-mode", choices=["soft", "strict", "nearest", "expanded_nearest"], default="soft")
    p.add_argument("--industry-policy", choices=["update_proxy", "effective"], default="update_proxy")
    p.add_argument("--strict-correlation", type=float, default=.8)
    p.add_argument("--distance-fit-config", type=Path, help="Keep the original strict training-only distance fit for a comparator revision")
    p.add_argument("--quality-effective-n-factor", choices=["legacy", "none"], default="legacy",
                   help="Use none for the single-reference quality formula without an effective-N multiplier")
    p.add_argument("--industry-recovery", type=Path, help="Optional event/interval recovery evidence, diagnostic only for expanded_nearest")
    p.add_argument("--fixed-reference-config", type=Path)
    p.add_argument("--m-override", type=int, choices=[1, 3, 5])
    p.add_argument("--distance-multiplier", type=float, choices=[.5, 1., 2.], default=1.0)
    p.add_argument("--ridge", type=float, default=0.1)
    p.add_argument("--m-grid", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--quantile-grid", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    p.add_argument("--temperature-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.length < 2 or args.ridge <= 0 or min(args.m_grid) < 1 or min(args.temperature_grid) <= 0:
        raise SystemExit("Invalid reference configuration")
    if any(q <= 0 or q > 1 for q in args.quantile_grid):
        raise SystemExit("Distance quantiles must be in (0,1]")
    run(args)
