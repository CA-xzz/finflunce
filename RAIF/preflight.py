"""Inventory real inputs and build calendar-aligned event manifests.

This command does not fit a model or replace missing historical attributes.
Run: python preflight.py --source data/source --output results/preflight
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_code(values):
    return values.astype(str).str.extract(r"(\d{6})", expand=False)


def make_events(raw, calendar):
    raw = raw.copy()
    raw["source_row"] = np.arange(len(raw))
    raw["stock"] = raw.recommended_stock_code.str.zfill(6)
    raw["nominal_date"] = pd.to_datetime(raw.recommend_date, errors="raise").dt.normalize()
    raw = raw.sort_values(["nominal_date", "source_row"], kind="stable")
    last_retained = {}
    keep = []
    for row in raw.itertuples():
        key = (row.bigv_name, row.stock)
        last = last_retained.get(key)
        retained = last is None or (row.nominal_date - last).days >= 10
        keep.append(retained)
        if retained:
            last_retained[key] = row.nominal_date
    raw["retained_after_10_day_rule"] = keep
    records = []
    selected = raw[raw.retained_after_10_day_rule]
    for (stock, date), group in selected.groupby(["stock", "nominal_date"], sort=True):
        pos = int(calendar.searchsorted(date))
        event_date = calendar[pos] if pos < len(calendar) else pd.NaT
        # Dates have no time of day. Multi-source ordering is not reconstructable.
        sources = group[["bigv_name", "laiyuan"]].drop_duplicates()
        known = len(sources) == 1
        records.append({
            "stock": stock, "nominal_date": date, "event_date": event_date,
            "day": pos, "platform": sources.laiyuan.iloc[0] if known else "__UNKNOWN__",
            "influencer": sources.bigv_name.iloc[0] if known else "__UNKNOWN__",
            "condition_known": known, "source_count": len(group),
            "query_timestamp": "", "timestamp_precision": "date_only",
            "source_rows": ";".join(map(str, group.source_row)),
        })
    events = pd.DataFrame(records)
    # Weekend/holiday recommendations of one stock may map to the same date.
    merged = []
    for (stock, date), group in events.groupby(["stock", "event_date"], dropna=False):
        row = group.iloc[0].to_dict()
        if len(group) > 1:
            row["platform"] = row["influencer"] = "__UNKNOWN__"
            row["condition_known"] = False
            row["source_count"] = int(group.source_count.sum())
            row["source_rows"] = ";".join(group.source_rows)
        row["event_id"] = f"{stock}_{date:%Y-%m-%d}" if pd.notna(date) else f"{stock}_unmapped"
        merged.append(row)
    events = pd.DataFrame(merged).sort_values(["event_date", "stock"]).reset_index(drop=True)
    events["end_day"] = events.day + 10
    events["end_date"] = [calendar[p] if p < len(calendar) else pd.NaT for p in events.end_day]
    events["split"] = "excluded"
    events["exclusion_reason"] = ""
    b1, b2 = pd.Timestamp("2024-12-31"), pd.Timestamp("2025-06-30")
    val_start = int(calendar.searchsorted(b1, side="right")) + 11
    test_start = int(calendar.searchsorted(b2, side="right")) + 11
    for row in events.itertuples():
        if pd.isna(row.event_date) or pd.isna(row.end_date):
            reason = "response_beyond_calendar"
        elif row.event_date <= b1:
            reason = "response_crosses_training_boundary" if row.end_date > b1 else ""
            if not reason:
                events.at[row.Index, "split"] = "train"
        elif row.event_date <= b2:
            reason = "validation_embargo" if row.day < val_start else (
                "response_crosses_validation_boundary" if row.end_date > b2 else "")
            if not reason:
                events.at[row.Index, "split"] = "validation"
        else:
            reason = "test_embargo" if row.day < test_start else ""
            if not reason:
                events.at[row.Index, "split"] = "test"
        events.at[row.Index, "exclusion_reason"] = reason
    return raw, events


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = [source / "original_data/events.csv", source / "control_stock_finder/data/A_share_daily.parquet",
             source / "control_stock_finder/data/all_stocks_info.csv"]
    panels = [source / "data" / name for name in ["event_panel.csv", "recommended_event_panel.csv", "control_event_panel.csv"]]
    paths += panels
    sources = [{"file": str(p.relative_to(source)), "bytes": p.stat().st_size,
                "sha256": sha256(p)} for p in paths]
    pd.DataFrame(sources).to_csv(output / "source_manifest.csv", index=False)
    rows = []
    for path in panels:
        frame = pd.read_csv(path, usecols=["date", "index_code", "index_close", "index_ret"])
        frame["date"] = pd.to_datetime(frame.date)
        rows.append(frame)
    index_rows = pd.concat(rows, ignore_index=True)
    grouped = index_rows.groupby(["index_code", "date"])
    spread = grouped[["index_close", "index_ret"]].max() - grouped[["index_close", "index_ret"]].min()
    if (spread.index_close > 1e-6).any() or (spread.index_ret > 1e-10).any():
        raise ValueError("Conflicting archived index observations")
    indices = grouped[["index_close", "index_ret"]].median().reset_index()
    indices.to_parquet(output / "indices.parquet", index=False)
    calendar = pd.DatetimeIndex(indices.loc[indices.index_code == "sh.000001", "date"].unique()).sort_values()
    pd.DataFrame({"day": np.arange(len(calendar)), "date": calendar}).to_csv(output / "calendar.csv", index=False)
    raw = pd.read_csv(paths[0], dtype={"recommended_stock_code": str})
    records, events = make_events(raw, calendar)
    records.to_csv(output / "source_event_decisions.csv", index=False)
    events.to_csv(output / "events_manifest.csv", index=False)
    split = events.groupby("split").agg(n_events=("event_id", "size"), n_dates=("event_date", "nunique"),
                                          first_date=("event_date", "min"), last_date=("event_date", "max"))
    split.to_csv(output / "split_manifest.csv")
    daily = pd.read_parquet(paths[1], columns=["date", "code"])
    info = pd.read_csv(paths[2])
    fields = [
        {"field": "stock_daily", "source": str(paths[1].relative_to(source)), "available": True,
         "point_in_time_verified": True, "available_at": "daily close; use prior sessions for inputs"},
        {"field": "market_index", "source": "archived panel explicit index columns", "available": True,
         "point_in_time_verified": True, "available_at": "daily close"},
        {"field": "recommendation_timestamp", "source": "original_data/events.csv", "available": False,
         "point_in_time_verified": False, "available_at": "date only; no 15:00 mapping verification"},
        {"field": "historical_total_market_cap", "source": "all_stocks_info.csv contains only a static snapshot", "available": False,
         "point_in_time_verified": False, "available_at": "unknown"},
        {"field": "historical_industry", "source": "all_stocks_info.csv contains no effective dates", "available": False,
         "point_in_time_verified": False, "available_at": "unknown"},
    ]
    pd.DataFrame(fields).to_csv(output / "field_availability.csv", index=False)
    flow = [{"stage": "raw_records", "n": len(raw)},
            {"stage": "raw_stock_nominal_date_unique", "n": len(raw.drop_duplicates(["recommended_stock_code", "recommend_date"]))},
            {"stage": "retained_records_after_10_calendar_day_rule", "n": int(records.retained_after_10_day_rule.sum())},
            {"stage": "stock_trading_date_events", "n": len(events)},
            {"stage": "temporally_eligible_before_feature_and_label_checks", "n": int((events.split != "excluded").sum())}]
    pd.DataFrame(flow).to_csv(output / "sample_flow.csv", index=False)
    report = {"status": "BLOCKED_MISSING_HISTORICAL_ATTRIBUTES", "training_started": False,
              "daily_rows": pq.ParquetFile(paths[1]).metadata.num_rows,
              "daily_stocks": int(daily.code.nunique()),
              "daily_first_date": str(daily.date.min().date()), "daily_last_date": str(daily.date.max().date()),
              "calendar_first_date": str(calendar.min().date()), "calendar_last_date": str(calendar.max().date()),
              "raw_influencers": int(raw.bigv_name.nunique()), "static_metadata_stocks": len(info),
              "deduplication": "Keep first; subsequent same-influencer same-stock record retained at >=10 calendar days from last retained record.",
              "temporal_counts_are_not_training_counts": True,
              "blockers": ["Historical total market capitalization before each query date is absent.",
                           "Industry classification with effective dates before each query is absent."],
              "limitations": ["Recommendation time of day is absent; dates cannot verify the 15:00 cutoff.",
                              "27 influencer identifiers in raw records; manuscript says 26.",
                              "7040 counts stock-date merging before applying the stated 10-day repeated-recommendation rule.",
                              "Local stock universe may omit historical delisted securities; candidate-universe coverage must be assessed."],
              "splits": json.loads(split.to_json(orient="index", date_format="iso"))}
    (output / "readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
