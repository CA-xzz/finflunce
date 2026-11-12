"""Create per-original-row equity decisions without modifying the source CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

from fetch_supplemental_market import (BSE_CANDIDATES, CONFLICTS, INDEX_EVENTS,
                                       STOCK_CODES, now, sha256, write_json)

BSE_SOURCE = "https://dataclouds.cninfo.com.cn/sjother/regulatory/2024/20241213/765a685e5d5545248d554c24fefc6684.pdf"
CODE_ALIASES = {
    "430017": ("920017", "星昊医药", "2023-05-31", "", BSE_SOURCE),
    "830799": ("920799", "艾融软件", "2021-11-15", "2025-05-06", BSE_SOURCE),
    "832982": ("920982", "锦波生物", "2023-07-20", "", BSE_SOURCE),
    "833394": ("920394", "民士达", "2023-04-25", "", BSE_SOURCE),
    "835185": ("920185", "贝特瑞", "2021-11-15", "", BSE_SOURCE),
    "300114": ("302132", "中航成飞", "2010-08-27", "2025-02-17", "https://disc.static.szse.cn/disc/disk03/finalpage/2025-02-15/cedb693a-f5ee-4463-9682-ea33d406b569.PDF"),
}
# Only same-code aliases supported by original issuer announcements are allowed.
NAME_ALIASES = {
    ("002150", "通润装备"): "https://disc.static.szse.cn/disc/disk03/finalpage/2026-01-15/a9c65cb5-2d13-4053-b965-d65c3c36d5b0.PDF",
    ("000697", "炼石航空"): "https://static.cninfo.com.cn/finalpage/2026-04-30/1225262622.PDF",
    ("601211", "国泰君安"): "https://www.sse.com.cn/disclosure/listedinfo/announcement/c/new/2025-04-08/601211_20250408_4UTC.pdf",
    ("600292", "远达环保"): "https://static.cninfo.com.cn/finalpage/2026-01-08/1224922708.PDF",
    ("600807", "济南高新"): "https://paper.cnstock.com/html/2024-12/07/content_2003578.htm",
}
INDEX_COLLISIONS = {
    ("000001", "上证指数"), ("000510", "中证A500"), ("000688", "科创50"),
    ("000807", "食品饮料"), ("000819", "有色金属"), ("000905", "中证500"),
}


def normalize_name(value):
    value = unicodedata.normalize("NFKC", str(value)).upper()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"^(?:S\*ST|SST|\*ST|ST|XD|XR|DR)", "", value)
    value = re.sub(r"-(?:UWD|UW|WD|U|W|D)$", "", value)
    value = re.sub(r"^退市", "", value)
    value = re.sub(r"退$", "", value)
    return value


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    raw_path = source / "original_data/events.csv"
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    master_path = source / "control_stock_finder/data/all_stocks_info.csv"
    master = pd.read_csv(master_path, dtype={"code": str}).set_index("code")["name"].to_dict()
    missing = pd.read_csv(source / "RAIF/results/industry_recovery/missing_price_names.csv", dtype={"stock": str})
    clear_missing_pairs = set(zip(missing[missing.stock.isin(STOCK_CODES)].stock,
                                 missing[missing.stock.isin(STOCK_CODES)].recommended_stock_name))
    rows = []
    for source_row, record in raw.iterrows():
        code = str(record.recommended_stock_code).strip().zfill(6)
        name = str(record.recommended_stock_name).strip()
        canonical, decision, reason, evidence = code, "exclude", "identity_not_verified", ""
        known_name = master.get(code, "")
        if (code, name) in INDEX_COLLISIONS or code in INDEX_EVENTS:
            reason = "non_equity_index_including_same_numeric_code_collision"
        elif re.search(r"ETF|指数|基金|混合|联接|债券", name, flags=re.I) or code.startswith(("50", "51", "52", "53", "56", "58")):
            reason = "non_equity_fund_or_index"
        elif (code, name) in CONFLICTS:
            reason = "name_code_conflict_no_original_recommendation_to_resolve"
            evidence = CONFLICTS[(code, name)][2]
        elif code in {"000602", "600065", "300023"}:
            reason = "original_code_delisted_before_study_period"
        elif code == "688765":
            evidence = "https://www.sse.com.cn/assortment/stock/list/info/company/index.shtml?COMPANY_CODE=688765&tabActive=1"
            reason = "recommendation_precedes_exchange_A_share_listing_2025_10_28"
        elif (code, name) == ("601989", "中国重工"):
            evidence = "https://dataclouds.cninfo.com.cn/sjother/regulatory/2025/20250829/076aae87d5334d9d974d515b3ab04039.pdf"
            if pd.to_datetime(record.recommend_date) < pd.Timestamp("2025-09-05"):
                decision, reason = "keep", "official_equity_identity_recommendation_before_delisting"
            else:
                reason = "recommendation_on_or_after_delisting"
        elif code in CODE_ALIASES:
            canonical, issuer, listed, changed, evidence = CODE_ALIASES[code]
            if normalize_name(name) not in {normalize_name(issuer), "中航电测"}:
                reason = "name_does_not_match_verified_code_alias"
            elif pd.to_datetime(record.recommend_date) < pd.Timestamp(listed):
                reason = "recommendation_precedes_exchange_A_share_listing"
            else:
                decision, reason = "keep", "official_same_issuer_code_alias"
        elif (code, name) in NAME_ALIASES:
            decision, reason, evidence = "keep", "official_same_code_historical_name", NAME_ALIASES[(code, name)]
        elif known_name and normalize_name(name) == normalize_name(known_name):
            decision, reason = "keep", "original_name_matches_equity_master_after_status_suffix_normalization"
        elif (code, name) in clear_missing_pairs:
            decision, reason = "keep", "clear_original_equity_code_and_name_in_missing_price_allowlist"
        elif known_name:
            reason = "name_code_conflict_with_equity_master; unresolved_original_intent"
        elif code.startswith(("40", "43", "83", "87", "92")):
            reason = "NEEQ_or_BSE_listing_identity_not_verified"
        else:
            reason = "fund_or_security_identity_not_verified"
        rows.append(dict(source_row=source_row, decision=decision, reason=reason, canonical_code=canonical,
                         original_code=record.recommended_stock_code, original_name=record.recommended_stock_name, stock_master_name=known_name,
                         original_date=record.recommend_date, original_influencer=record.bigv_name,
                         original_platform=record.laiyuan, recommend_date=record.recommend_date, evidence_url=evidence))
    result = pd.DataFrame(rows)
    assert len(result) == len(raw) and result.source_row.tolist() == list(range(len(raw)))
    assert set(result.decision) <= {"keep", "exclude"}
    result.to_csv(output / "all_event_decisions.csv", index=False)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["event_decisions"] = dict(sha256=sha256(output / "all_event_decisions.csv"), rows=len(result))
    write_json(manifest_path, manifest)
    result[result.decision == "exclude"].groupby(["original_code", "original_name", "stock_master_name", "reason"], dropna=False).agg(n_raw_records=("source_row", "size"), source_rows=("source_row", lambda x: ";".join(map(str, x)))).reset_index().to_csv(output / "all_event_exclusion_pairs.csv", index=False)
    crosswalk = pd.DataFrame([dict(original_code=old, canonical_code=values[0], issuer=values[1],
                                  exchange_A_share_history_start=values[2], code_change_effective_date=values[3],
                                  primary_source_url=values[4], mapping_kind="same_issuer_official_code_change",
                                  application="Preserve original event code; use canonical identity to match one price series. Exclude pre-listing NEEQ observations; never duplicate old/new issuer as separate controls.")
                              for old, values in CODE_ALIASES.items()])
    crosswalk.to_csv(output / "canonical_code_crosswalk.csv", index=False)
    counts = result.groupby(["decision", "reason"]).size().reset_index(name="raw_records")
    counts.to_csv(output / "event_decision_counts.csv", index=False)
    write_json(output / "event_screening_metadata.json", dict(created_utc=now(), raw_source=str(raw_path),
        raw_sha256=sha256(raw_path), stock_master_sha256=sha256(master_path), screening_code_sha256=sha256(__file__),
        n_raw_records=len(raw), decision_counts=result.decision.value_counts().to_dict(),
        source_row_definition="Zero-based physical data-row index in original_data/events.csv, excluding CSV header",
        application="Filter raw records to decision=keep before event deduplication/10-day spacing; retain source_row and original_code. Apply only explicit same-issuer canonical aliases; recompute events and split manifests from expanded calendar.",
        current_name_usage="Identity check only; current stock master is not used for historical industry, cap, or features.",
        unresolved_pairs="Excluded without assigning another company's stock; original recommendation content was not available."))
    print(counts.to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data/supplemental_market")
    args = parser.parse_args()
    build(args.source, args.output)
