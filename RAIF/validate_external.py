"""Check source records and the declared conservative availability rules."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    cap_path, industry_path = [args.external / name for name in ["historical_market_cap.parquet", "industry_history.parquet"]]
    cap, industry = pd.read_parquet(cap_path), pd.read_parquet(industry_path)
    cap_meta = json.loads((args.external / "market_cap_metadata.json").read_text())
    industry_meta = json.loads((args.external / "industry_metadata.json").read_text())
    events = pd.read_csv(args.preflight / "events_manifest.csv")
    calendar = pd.read_csv(args.preflight / "calendar.csv", parse_dates=["date"])
    expected_days = sorted(set(events.loc[(events.split != "excluded") & (events.day > 0), "day"].astype(int) - 1))
    expected_dates = set(calendar.iloc[expected_days].date)
    if set(cap.date.unique()) != set(np.array(sorted(expected_dates), dtype="datetime64[ns]")):
        # pandas/Timestamp and numpy datetime scalars hash differently; compare strings.
        if set(cap.date.dt.strftime("%Y-%m-%d")) != {str(d.date()) for d in expected_dates}:
            raise ValueError("Historical capitalization does not cover every required prior trading date")
    if len(cap) != cap_meta["row_count"] or cap.date.nunique() != cap_meta["date_count"]:
        raise ValueError("Market cap receipt counts do not match output")
    if cap.duplicated(["code", "date"]).any() or cap[["code", "date", "available_at"]].isna().any().any():
        raise ValueError("Invalid cap record keys or availability")
    if not cap.available_at.eq(cap.date + pd.Timedelta(days=1)).all():
        raise ValueError("Market-cap availability rule mismatch")
    if digest(industry_path) != industry_meta["output_sha256"]:
        raise ValueError("Industry normalization receipt mismatch")
    if industry.duplicated(["code", "effective_from"]).any():
        raise ValueError("Duplicate industry intervals")
    expected_available = industry[["effective_from", "update_time"]].max(axis=1) + pd.Timedelta(days=1)
    if not industry.available_at.eq(expected_available).all():
        raise ValueError("Industry availability rule mismatch")
    expected_end = industry.sort_values(["code", "effective_from"]).groupby("code").effective_from.shift(-1)
    if not industry.effective_to.fillna(pd.Timestamp.max).eq(expected_end.fillna(pd.Timestamp.max)).all():
        raise ValueError("Industry interval closure mismatch")
    product = cap.unadjusted_close * cap.total_shares
    finite = np.isfinite(product) & np.isfinite(cap.total_market_cap) & (cap.total_market_cap > 0)
    relative_difference = ((product[finite] - cap.total_market_cap[finite]).abs() / cap.total_market_cap[finite])
    report = {
        "historical_market_cap_records_verified": True,
        "historical_industry_records_verified": True,
        "conservative_availability_rule_verified": True,
        "historical_vintage_archive_verified": False,
        "official_industry_original_comparison_verified": False,
        "verification_scope": "Record hashes, schema, coverage, unique keys, interval construction and declared availability filtering. Not authentication of original announcement dates or archived vintage equivalence.",
        "provenance": {"market_cap": cap_meta, "industry": industry_meta},
        "availability_policy": "Prior-trading-day total cap becomes available at next calendar midnight. Industry applies only within its effective interval and after both its effective date and source update date, plus one day. Unavailable records are excluded; no snapshot fallback.",
        "record_counts": {"market_cap": len(cap), "market_cap_dates": int(cap.date.nunique()), "industry": len(industry)},
        "units": {"total_market_cap": "CNY", "total_shares": "shares", "unadjusted_close": "CNY/share"},
        "diagnostics": {"nonpositive_or_missing_cap_records": int((~np.isfinite(cap.total_market_cap) | (cap.total_market_cap <= 0)).sum()),
            "cap_equal_close_times_shares_fraction_at_1e_6": float((relative_difference <= 1e-6).mean()),
            "cap_product_relative_difference_max": float(relative_difference.max())},
        "sha256": {"historical_market_cap.parquet": digest(cap_path), "industry_history.parquet": digest(industry_path)},
        "limitations": ["Industry history is a pinned public conversion of the claimed SWS official file; the official original was unavailable for comparison.",
            "The industry's last-update field is not a verified first-publication timestamp; late updated intervals are conservatively excluded.",
            "Both sources are retrospective data versions; past revisions cannot be reconstructed.",
            "Back-mapped securities codes, missing historical listings and date-only recommendations can reduce the eligible sample."]}
    (args.external / "attribute_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "record_checks_passed", "counts": report["record_counts"], "diagnostics": report["diagnostics"],
                      "historical_vintage_archive_verified": False}), flush=True)


if __name__ == "__main__":
    main()
