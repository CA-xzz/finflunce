"""Normalize the pinned public SWS history copy without backfilling late records.

This script checks the downloaded JSON against its recorded SHA256. It cannot
verify that the third-party export is bytewise equivalent to the unavailable
official XLS, nor reconstruct past revisions or publication timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PINNED_COMMIT = "d1c77e0ca5023136399725be6517a5a70394b489"
PINNED_SHA256 = "05bbf0e0b2e3e7ef3a139ab088bbef538331340795eda1c3c0e1628f98e746c9"
PINNED_URL = (
    "https://raw.githubusercontent.com/MORECORIANDERS/sw-map/"
    + PINNED_COMMIT
    + "/sw_stock_industry.json"
)
REQUIRED_FIELDS = {"symbol", "start_date", "industry_code", "update_time"}
CODE_PATTERN = re.compile(r"[0-9]{6}\Z")
DAY_MILLISECONDS = 86_400_000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently accepting the last one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


def load_checked_source(raw_path: Path, metadata_path: Path) -> tuple[list, dict]:
    raw_bytes = raw_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(metadata, dict):
        raise ValueError("Source metadata must be a JSON object")
    if metadata.get("repository_commit") != PINNED_COMMIT:
        raise ValueError("Source metadata does not name the fixed repository commit")
    if metadata.get("source_url") != PINNED_URL:
        raise ValueError("Source URL must point to the pinned commit, not a moving branch")
    actual_hash = sha256_bytes(raw_bytes)
    if metadata.get("sha256") != PINNED_SHA256 or actual_hash != PINNED_SHA256:
        raise ValueError("Raw source SHA256 differs from the pinned export")
    if type(metadata.get("bytes")) is not int or metadata["bytes"] != len(raw_bytes):
        raise ValueError("Raw source byte count differs from its metadata")
    records = json.loads(raw_bytes, object_pairs_hook=unique_object)
    return records, metadata


def normalize_records(records: Any) -> pd.DataFrame:
    if not isinstance(records, list) or not records:
        raise ValueError("History must be a nonempty JSON array")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != REQUIRED_FIELDS:
            raise ValueError("Unexpected history schema at row %d" % index)
        for field in ("symbol", "industry_code"):
            value = record[field]
            if not isinstance(value, str) or CODE_PATTERN.fullmatch(value) is None:
                raise ValueError("%s must be a six-digit string at row %d" % (field, index))
        for field in ("start_date", "update_time"):
            value = record[field]
            # bool is an int subclass, so the exact type check is intentional.
            if type(value) is not int or value % DAY_MILLISECONDS != 0:
                raise ValueError("%s must encode a complete date in integer milliseconds at row %d" % (field, index))
    frame = pd.DataFrame(records).rename(
        columns={"symbol": "code", "industry_code": "original_industry_code", "start_date": "effective_from"}
    )
    for field in ("effective_from", "update_time"):
        frame[field] = pd.to_datetime(frame[field], unit="ms", utc=True, errors="raise").dt.tz_localize(None)
        if frame[field].isna().any() or not frame[field].eq(frame[field].dt.normalize()).all():
            raise ValueError("Invalid date-only values in " + field)
    if frame.duplicated(["code", "effective_from"]).any():
        duplicate = frame.loc[frame.duplicated(["code", "effective_from"], keep=False), ["code", "effective_from"]]
        raise ValueError("Ambiguous duplicate code/effective_from records: " + duplicate.head(5).to_json(orient="records"))
    frame = frame.sort_values(["code", "effective_from"], kind="stable").reset_index(drop=True)
    # Keep every source row, including post-study rows, when closing intervals.
    frame["effective_to"] = frame.groupby("code", sort=False)["effective_from"].shift(-1)
    if (frame.effective_to.notna() & frame.effective_to.le(frame.effective_from)).any():
        raise ValueError("Non-positive effective interval")
    frame["available_at"] = frame[["effective_from", "update_time"]].max(axis=1) + pd.Timedelta(days=1)
    frame["industry"] = frame.original_industry_code.str[:4] + "00"
    return frame[[
        "code", "industry", "effective_from", "effective_to", "available_at",
        "original_industry_code", "update_time",
    ]]


def date_text(value: Any) -> Any:
    return None if pd.isna(value) else pd.Timestamp(value).isoformat()


def build_metadata(frame: pd.DataFrame, source_metadata: dict, raw_path: Path,
                   metadata_path: Path, parquet_path: Path) -> dict:
    lag = (frame.update_time - frame.effective_from).dt.days
    exhausted = frame.effective_to.notna() & frame.available_at.ge(frame.effective_to)
    coverage = source_metadata.get("coverage", {})
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_status": "normalized_public_historical_copy; final experiment readiness is assessed separately",
        "source_kind": "third_party_export_of_claimed_official_SWS_XLS",
        "source_url": PINNED_URL,
        "repository_commit": PINNED_COMMIT,
        "source_note_url": source_metadata.get("source_note_url"),
        "original_source_claimed": source_metadata.get("original_source_claimed"),
        "official_original_byte_comparison": "not_performed: official XLS unavailable",
        "raw_file": str(raw_path.resolve()),
        "raw_sha256": PINNED_SHA256,
        "raw_bytes": raw_path.stat().st_size,
        "source_metadata_file": str(metadata_path.resolve()),
        "source_metadata_sha256": sha256_bytes(metadata_path.read_bytes()),
        "output_file": str(parquet_path.resolve()),
        "output_sha256": sha256_bytes(parquet_path.read_bytes()),
        "normalizer_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "industry_level": "SW level 2: first four original code digits followed by 00",
        "effective_interval": "effective_from <= query_date < effective_to; missing effective_to means open-ended",
        "availability_rule": "available_at = max(effective_from, update_time) + one calendar day; require available_at <= query_date",
        "query_date_semantics": "Date-only event query, evaluated at the beginning of the local trading date. Integer milliseconds in the export encode dates, not verified publication instants.",
        "unknown_interval_rule": "Exclude when the applicable interval is unavailable; never carry a superseded industry through a later effective interval.",
        "rows_after_study_end": "Preserved before deriving effective_to; the consumer must apply effective and availability filters.",
        "limitations": [
            "This is a non-official publication copy from a pinned public repository. Its author describes conversion from the SWS XLS; bytewise equivalence to that XLS has not been verified.",
            "update_time is a source update field, not a verified original announcement time.",
            "Historical revisions cannot be reconstructed from the single exported version.",
            "Requiring both source dates to precede the query removes late-updated records and reduces available sample and candidate coverage.",
            "Historical classification versions are retained. The original six-digit code remains available for version checks; no missing industry is inferred from today's classification.",
        ],
        "counts": {
            "records": len(frame),
            "stocks": int(frame.code.nunique()),
            "level2_codes": int(frame.industry.nunique()),
            "original_codes": int(frame.original_industry_code.nunique()),
            "closed_intervals": int(frame.effective_to.notna().sum()),
            "open_intervals": int(frame.effective_to.isna().sum()),
            "intervals_with_no_available_date": int(exhausted.sum()),
            "update_precedes_start": int((lag < 0).sum()),
            "update_equals_start": int((lag == 0).sum()),
            "update_follows_start": int((lag > 0).sum()),
            "update_minus_start_days_median": float(lag.median()),
            "update_minus_start_days_p95": float(lag.quantile(0.95)),
            "first_effective_from": date_text(frame.effective_from.min()),
            "last_effective_from": date_text(frame.effective_from.max()),
            "first_update_time": date_text(frame.update_time.min()),
            "last_update_time": date_text(frame.update_time.max()),
        },
        "prior_coverage_check": {
            "description": "Source metadata's diagnostic on the original 7040-row event mapping; not the final experiment sample after deduplication, matching or label filtering.",
            "local_stocks": coverage.get("local_stocks"),
            "local_stocks_in_history": coverage.get("local_stocks_in_history"),
            "raw_event_mapping_rows": coverage.get("raw_event_mapping_rows"),
            "event_status_counts": coverage.get("event_status_counts"),
            "event_year_status": coverage.get("event_year_status"),
        },
    }


def main() -> None:
    base = Path(__file__).resolve().parent / "data" / "external"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=base / "raw" / "sw_stock_industry_github.json")
    parser.add_argument("--source-metadata", type=Path, default=base / "raw" / "source_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=base)
    args = parser.parse_args()
    records, source_metadata = load_checked_source(args.raw, args.source_metadata)
    frame = normalize_records(records)
    expected = source_metadata.get("coverage", {})
    if expected.get("rows") != len(frame) or expected.get("stocks") != int(frame.code.nunique()):
        raise ValueError("Record/stock counts disagree with the pinned source metadata")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "industry_history.parquet"
    temporary = args.output_dir / "industry_history.parquet.tmp"
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    restored = pd.read_parquet(temporary, engine="pyarrow")
    pd.testing.assert_frame_equal(frame, restored, check_dtype=True)
    temporary.replace(target)
    metadata = build_metadata(frame, source_metadata, args.raw, args.source_metadata, target)
    metadata_target = args.output_dir / "industry_metadata.json"
    metadata_temporary = args.output_dir / "industry_metadata.json.tmp"
    metadata_temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_temporary.replace(metadata_target)
    print(json.dumps({"output": str(target), "metadata": str(metadata_target), "counts": metadata["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
