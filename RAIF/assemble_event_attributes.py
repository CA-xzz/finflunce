"""Assemble complete descriptive historical attributes without changing model data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    root = Path(root)
    prepared = root / "data/prepared_expanded"
    output = root / "results/industry_recovery"
    cap_dir = root / "data/external/cap_annotation_recovery"
    fixed = [prepared / "dataset.npz", prepared / "normalization.json"]
    frozen = {str(path): digest(path) for path in fixed}
    events = pd.read_csv(prepared / "events.csv", dtype={"event_id": str, "stock": str})
    industry_path = prepared / "industry_assignments.csv"
    industry = pd.read_csv(industry_path, dtype={"stock": str, "industry": str, "original_industry_code": str})
    industry = industry[industry.event_id.isin(events.event_id)].copy().set_index(["event_id", "role"])
    expected = pd.MultiIndex.from_product([events.event_id, ["query", "control"]], names=["event_id", "role"])
    assert industry.index.is_unique and set(industry.index) == set(expected)
    industry = industry.reindex(expected)
    industry["industry_lookup_stock"] = industry.stock
    industry["industry_code_alias_applied"] = False
    alias_path = output / "industry_alias_recovery.csv"
    aliases = pd.read_csv(alias_path, dtype={"canonical_code": str, "industry_lookup_original_code": str,
        "industry": str, "original_industry_code": str})
    for row in aliases.itertuples():
        key = (row.event_id, "query")
        assert not bool(industry.at[key, "assignment_found"])
        assert industry.at[key, "stock"] == row.canonical_code
        assert pd.Timestamp(row.event_date) < pd.Timestamp(row.code_change_effective_date)
        for field in ["industry", "industry_name_zh", "original_industry_code", "effective_from", "effective_to"]:
            industry.at[key, field] = getattr(row, field)
        industry.at[key, "original_update_time"] = row.source_update_time
        industry.at[key, "industry_lookup_stock"] = row.industry_lookup_original_code
        industry.at[key, "industry_code_alias_applied"] = True
        industry.at[key, "assignment_found"] = True
        proxy = max(pd.Timestamp(row.effective_from), pd.Timestamp(row.source_update_time)) + pd.Timedelta(days=1)
        industry.at[key, "update_proxy_available_at"] = proxy.strftime("%Y-%m-%d")
        industry.at[key, "update_proxy_observable"] = proxy <= pd.Timestamp(row.event_date)
    diagnostics_path = prepared / "matching_diagnostics.csv"
    diagnostic = pd.read_csv(diagnostics_path, dtype={"stock": str, "control": str}).set_index("event_id")
    pairs_path = prepared / "reference_pairs.csv"
    pairs = pd.read_csv(pairs_path).set_index("event_id")
    industry["cap_date"] = [pairs.at[event, "market_cap_date"] for event, role in expected]
    industry["total_market_cap_cny"] = [diagnostic.at[event, role + "_total_market_cap"] for event, role in expected]
    industry["market_cap_source"] = "Eastmoney RPT_VALUEANALYSIS_DET dated total-market-cap observation"
    industry["market_cap_publication_time_verified"] = False
    cap_path = cap_dir / "cap_annotation_recovery.csv"
    caps = pd.read_csv(cap_path, dtype={"stock": str})
    for row in caps.itertuples():
        key = (row.event_id, row.role)
        assert pd.isna(industry.at[key, "total_market_cap_cny"])
        assert industry.at[key, "stock"] == row.stock
        assert pd.Timestamp(industry.at[key, "cap_date"]) == pd.Timestamp(row.cap_date)
        assert np.isclose(row.total_shares * row.unadjusted_close_cny, row.total_market_cap_cny, rtol=1e-12)
        assert bool(row.recovered) and row.total_market_cap_cny > 0
        industry.at[key, "total_market_cap_cny"] = row.total_market_cap_cny
        industry.at[key, "market_cap_source"] = "Sina dated total-share history multiplied by Eastmoney same-day unadjusted close"
    assert len(industry) == 2 * len(events) and industry.assignment_found.all()
    assert industry[["industry", "industry_name_zh", "original_industry_code"]].notna().all().all()
    assert np.isfinite(industry.total_market_cap_cny).all() and (industry.total_market_cap_cny > 0).all()
    query_dates = pd.to_datetime(industry.query_date)
    begin, end = pd.to_datetime(industry.effective_from), pd.to_datetime(industry.effective_to)
    assert (begin <= query_dates).all() and ((end > query_dates) | end.isna()).all()
    assert (pd.to_datetime(industry.cap_date) < query_dates).all()
    assert not industry.publication_time_verified.any() and not industry.used_for_model_matching.any()
    table = industry.reset_index()
    table.to_csv(output / "historical_event_attributes.csv", index=False)
    table.to_parquet(output / "historical_event_attributes.parquet", index=False)
    sources = [industry_path, diagnostics_path, pairs_path, alias_path, cap_path,
               cap_dir / "source_metadata.json", Path(__file__)]
    summary = dict(events=len(events), rows=len(table), roles=["query", "control"],
        industry_missing=0, market_cap_missing=0, industry_alias_annotations=len(aliases),
        derived_market_cap_annotations=len(caps), individual_industry_publication_times_verified=False,
        attributes_used_for_primary_matching_or_prediction=False,
        frozen_training_files_unchanged=all(digest(Path(path)) == value for path, value in frozen.items()),
        sources=[dict(path=str(path), sha256=digest(path)) for path in sources], frozen_training_hashes=frozen,
        identity_scope="Only retained equity events; records with unresolved security identities are excluded.")
    assert summary["frozen_training_files_unchanged"]
    (output / "attribute_annotation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"sources", "frozen_training_hashes"}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
