"""Merge checked supplemental observations into a separate, reproducible input set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from preflight import make_events, sha256


def code6(values):
    codes = values.astype(str).str.extract(r"(\d{6})", expand=False)
    if codes.isna().any():
        raise ValueError("Invalid security code")
    return codes


def merge_market(original, supplemental):
    original, supplemental = original.copy(), supplemental.copy()
    for frame in (original, supplemental):
        frame["code"] = code6(frame.code)
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        if frame.duplicated(["code", "date"]).any():
            raise ValueError("Duplicate stock-date observations")
    overlap = set(original.code) & set(supplemental.code)
    if overlap:
        raise ValueError("A supplemental stock already has original observations: " + str(sorted(overlap)))
    missing = set(original.columns) - set(supplemental.columns)
    if missing:
        raise ValueError("Supplemental market fields missing: " + str(sorted(missing)))
    return pd.concat([original, supplemental[original.columns]], ignore_index=True).sort_values(["code", "date"])


def merge_indices(original, supplemental):
    original, supplemental = original.copy(), supplemental.copy()
    for frame in (original, supplemental):
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        if frame.duplicated(["index_code", "date"]).any():
            raise ValueError("Duplicate index-date observations")
    for code, frame in supplemental.groupby("index_code"):
        old = original[original.index_code == code]
        if old.empty or set(frame.date) & set(old.date):
            raise ValueError("Index supplements cannot replace original observations")
        if not np.isfinite(frame[["index_close", "index_ret"]].to_numpy()).all():
            raise ValueError("Nonfinite supplemental index values")
    return pd.concat([original, supplemental], ignore_index=True).sort_values(["index_code", "date"])


def select_events(raw, decisions):
    raw = raw.copy()
    raw["original_source_row"] = np.arange(len(raw))
    raw["recommended_stock_code"] = raw.recommended_stock_code.str.zfill(6)
    bindings = {"recommended_stock_code": "original_code", "recommended_stock_name": "original_name",
                "recommend_date": "original_date", "bigv_name": "original_influencer", "laiyuan": "original_platform"}
    required = {"source_row", "decision", "reason", "canonical_code"} | set(bindings.values())
    if not required <= set(decisions):
        raise ValueError("Event classification columns missing")
    if decisions.source_row.duplicated().any() or set(decisions.source_row) != set(range(len(raw))):
        raise ValueError("Event classification must cover each original row exactly once")
    decisions = decisions.set_index("source_row").reindex(range(len(raw)))
    if not decisions.decision.isin(["keep", "exclude"]).all():
        raise ValueError("Unresolved event classification")
    for original_column, decision_column in bindings.items():
        if not np.array_equal(raw[original_column].fillna("").astype(str).to_numpy(),
                              decisions[decision_column].fillna("").astype(str).to_numpy()):
            raise ValueError("Event classification does not match original " + original_column)
    keep = decisions.decision.eq("keep").to_numpy()
    selected = raw.loc[keep].copy()
    selected["original_stock_code"] = selected.recommended_stock_code
    selected["recommended_stock_code"] = decisions.loc[keep, "canonical_code"].to_numpy()
    if selected.recommended_stock_code.str.fullmatch(r"\d{6}").ne(True).any():
        raise ValueError("Kept event lacks a valid canonical code")
    # Original row identifiers remain available even though the filtered file
    # receives its own contiguous row numbers in the event-building function.
    return selected, raw.loc[~keep].assign(exclusion_reason=decisions.loc[~keep, "reason"].to_numpy())


def run(args):
    if args.output_source.exists() or args.output_preflight.exists():
        raise FileExistsError("Choose fresh source and preflight output directories")
    manifest = json.loads((args.supplement / "manifest.json").read_text())
    daily_path = args.source / "control_stock_finder/data/A_share_daily.parquet"
    raw_path = args.source / "original_data/events.csv"
    index_path = args.preflight / "indices.parquet"
    if sha256(daily_path) != manifest["original_market_sha256"]:
        raise ValueError("Original market hash mismatch")
    if sha256(raw_path) != manifest["original_events_sha256"]:
        raise ValueError("Original events hash mismatch")
    if not manifest.get("original_market_unchanged"):
        raise ValueError("Supplement collection did not verify original market preservation")
    if not manifest["combined_indices"]["all_five_indices_passed"]:
        raise ValueError("All market-index overlap checks must pass")
    if sha256(index_path) != manifest["combined_indices"]["original_indices_sha256"]:
        raise ValueError("Original index hash mismatch")
    supplement_stock = args.supplement / "stocks/supplemental_stock_daily.parquet"
    if sha256(supplement_stock) != manifest["combined_stock"]["sha256"]:
        raise ValueError("Supplemental market hash mismatch")
    supplemental_indices = args.supplement / "indices/supplemental_indices.parquet"
    if sha256(supplemental_indices) != manifest["supplemental_indices_sha256"]:
        raise ValueError("Supplemental index hash mismatch")
    # Verify the hashed merged artifact against the actual originals and additions.
    checked_indices = args.supplement / "indices/indices_with_checked_history.parquet"
    if sha256(checked_indices) != manifest["combined_indices"]["sha256"]:
        raise ValueError("Checked index artifact hash mismatch")
    daily = merge_market(pd.read_parquet(daily_path), pd.read_parquet(supplement_stock))
    indices = merge_indices(pd.read_parquet(index_path), pd.read_parquet(supplemental_indices))
    pd.testing.assert_frame_equal(indices.reset_index(drop=True),
        pd.read_parquet(checked_indices).sort_values(["index_code", "date"]).reset_index(drop=True),
        check_dtype=False, check_exact=True)
    decisions_path = args.supplement / "all_event_decisions.csv"
    if sha256(decisions_path) != manifest["event_decisions"]["sha256"]:
        raise ValueError("Event classification hash mismatch")
    decisions = pd.read_csv(decisions_path, dtype={"original_code": str, "canonical_code": str})
    raw = pd.read_csv(raw_path, dtype={"recommended_stock_code": str})
    selected, excluded = select_events(raw, decisions)
    calendar = pd.DatetimeIndex(indices.loc[indices.index_code == "sh.000001", "date"].unique()).sort_values()
    for code, rows in indices.groupby("index_code"):
        if set(rows.date) != set(calendar):
            raise ValueError("Market indices do not share the same full calendar: " + code)
    records, events = make_events(selected.reset_index(drop=True), calendar)
    for path in [args.output_source / "original_data", args.output_source / "control_stock_finder/data", args.output_preflight]:
        path.mkdir(parents=True, exist_ok=True)
    out_daily = args.output_source / "control_stock_finder/data/A_share_daily.parquet"
    daily.to_parquet(out_daily, index=False)
    selected.to_csv(args.output_source / "original_data/events.csv", index=False)
    # Static names are carried only for inventory; they are never historical attributes.
    shutil.copy2(args.source / "control_stock_finder/data/all_stocks_info.csv",
                 args.output_source / "control_stock_finder/data/all_stocks_info.csv")
    indices.to_parquet(args.output_preflight / "indices.parquet", index=False)
    pd.DataFrame({"day": np.arange(len(calendar)), "date": calendar}).to_csv(args.output_preflight / "calendar.csv", index=False)
    records.to_csv(args.output_preflight / "source_event_decisions.csv", index=False)
    events.to_csv(args.output_preflight / "events_manifest.csv", index=False)
    excluded.to_csv(args.output_preflight / "excluded_original_recommendations.csv", index=False)
    decisions.to_csv(args.output_preflight / "asset_classification_decisions.csv", index=False)
    paths = [daily_path, raw_path, index_path, supplement_stock, supplemental_indices, checked_indices,
             decisions_path, args.supplement / "manifest.json", Path(__file__), Path(__file__).with_name("preflight.py")]
    pd.DataFrame([{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths]).to_csv(
        args.output_preflight / "source_manifest.csv", index=False)
    report = dict(original_recommendations=len(raw), kept_recommendations=len(selected),
        excluded_recommendations=len(excluded), exclusion_counts=excluded.exclusion_reason.value_counts().to_dict(),
        canonical_code_changes=int((selected.original_stock_code != selected.recommended_stock_code).sum()),
        stocks=int(daily.code.nunique()), stock_rows=len(daily), stock_daily_sha256=sha256(out_daily),
        calendar_first=str(calendar.min().date()), calendar_last=str(calendar.max().date()),
        events=len(events), split_counts=events.split.value_counts().to_dict(),
        stock_policy="Original series preserved; only entirely absent stocks added from separately documented source.",
        index_policy="Original observations preserved; only missing dates added after overlapping observations agree within source rounding bounds.",
        price_adjustment_limitation="New stocks use Eastmoney qfq; existing stocks use Baostock forward adjustment. No stock mixes providers; both are retrospective downloads.",
        recommendation_time_precision="date only")
    (args.output_preflight / "expanded_input_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["source", "preflight", "supplement", "output-source", "output-preflight"]:
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
