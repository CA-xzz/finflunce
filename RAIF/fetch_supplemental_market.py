"""Fetch separately stored missing equities and check index history overlap.

Never edits the original events or market parquet. Only the explicit equity
allow-list is fetched. A code/name conflict does not authorize changing an event.
HTTP authorization/rate-limit failures stop the provider; there are no retries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd
import requests

STOCK_CODES = """000002 000006 000008 000016 000032 000048 000157 000627
000666 000961 002280 002308 002435 002750 003036 300114 300255 300280
300766 301217 600089 600200 600705 600778 600811 600837 600898 603103
603232 603511 603516 603517 603529 603533 603556 603686 603963 605089
605090 605099 688086 688270""".split()
INDEX_CODES = ["sh.000001", "sz.399001", "sz.399006", "sh.000688", "sh.000905"]
INDEX_EVENTS = set("000015 000300 000808 000847 000941 399006 399812 399997 399989 399432 399441 399699".split())
BSE_CANDIDATES = {"430017": "星昊医药", "830799": "艾融软件", "832982": "锦波生物", "833394": "民士达"}
URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
ETF_SOURCE = "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20260109_10815555.shtml"
CONFLICTS = {
    ("300023", "安恒信息"): ("宝德退", "688023", "https://disc.static.szse.cn/disc/disk03/finalpage/2022-06-23/36382004-ef22-490c-bca3-62cbca66b8cb.PDF", "https://star.sse.com.cn/disclosure/listedinfo/announcement/c/new/2025-07-04/688023_20250704_76IB.pdf", "Original code delisted before study; named issuer is a different listed company. Do not recode without original recommendation evidence."),
    ("605090", "中际联合"): ("九丰能源", "605305", "https://www.sse.com.cn/assortment/stock/list/info/turnover/index.shtml?COMPANY_CODE=605090", "https://www.sse.com.cn/assortment/stock/list/info/announcement/index.shtml?productId=605305", "The same original code also has a separate consistent 九丰能源 row. Quarantine this name/code pair only."),
    ("832168", "中科曙光"): ("中科招商", "603019", "https://disc.static.szse.cn/disc/disk01/finalpage/2017-10-24/d9a12d15-d2f8-4275-9835-abe5d22cbebf.PDF", "", "Original code identifies a different NEEQ issuer. No automatic mapping by company name."),
    ("512890", "华泰股份"): ("华泰柏瑞中证红利低波动ETF", "", "https://www.sse.com.cn/disclosure/announcement/general/jjzssgg/c/c_20260408_10814527.shtml", "", "Listed fund code; do not convert a fund-manager/name fragment to an equity."),
    ("560280", "广发证券"): ("工程机械ETF", "", ETF_SOURCE, "", "SSE lists this as a fund. Original name is inconsistent with the fund identity."),
    ("512040", "华宝股份"): ("基金代码，具体名称待核对", "", "", "", "Suspected fund/company-name conflict; unverified and excluded."),
    ("513910", "华夏银行"): ("基金代码，具体名称待核对", "", "", "", "Suspected fund/company-name conflict; unverified and excluded."),
    ("520830", "华泰股份"): ("基金代码，具体名称待核对", "", "", "", "Suspected fund/company-name conflict; unverified and excluded."),
    ("019455", "华泰股份"): ("基金代码，具体名称待核对", "", "", "", "Suspected fund/company-name conflict; unverified and excluded."),
    ("006248", "创业环保"): ("基金代码，具体名称待核对", "", "", "", "Suspected fund/company-name conflict; unverified and excluded."),
    ("006249", "创业环保"): ("基金代码，具体名称待核对", "", "", "", "Suspected fund/company-name conflict; unverified and excluded."),
    ("008809", "安信种苗"): ("基金代码，具体名称待核对", "", "", "", "Suspected fund/company-name conflict; unverified and excluded."),
    ("000154", "沪深300指数"): ("未核实代码", "", "", "", "Name points to an index but supplied code differs from 000300. Do not infer a correction."),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def classifications(source, output):
    events = pd.read_csv(source / "RAIF/results/industry_recovery/missing_price_events.csv", dtype={"stock": str})
    names = pd.read_csv(source / "RAIF/results/industry_recovery/missing_price_names.csv", dtype={"stock": str})
    counts = events.groupby("stock").size()
    crosswalk_path = output / "canonical_code_crosswalk.csv"
    canonical = pd.read_csv(crosswalk_path, dtype=str).set_index("original_code").canonical_code.to_dict() if crosswalk_path.exists() else {}
    rows, conflicts = [], []
    for row in names.itertuples(index=False):
        code, name = row.stock, row.recommended_stock_name
        primary = ""
        if code in STOCK_CODES:
            kind, decision = "SH_SZ_A_share", "fetch_original_code_only"
        elif code in INDEX_EVENTS:
            kind, decision = "index", "exclude_non_equity"
            if code == "000941":
                primary = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000941factsheet.pdf"
            elif code == "000847":
                primary = "https://emt.18.cn/api/quant-help/data/indices.html"
        elif code in {"000602", "600065", "300023"}:
            kind, decision = "A_share_delisted_before_study", "exclude_no_in_period_listing"
            if code == "000602":
                primary = "https://static.cninfo.com.cn/finalpage/2013-08-14/62936762.PDF"
            elif code == "600065":
                primary = "https://www.sse.com.cn/assortment/stock/list/info/turnover/index.shtml?COMPANY_CODE=600065"
        elif code in BSE_CANDIDATES:
            kind, decision = ("BSE_equity_verified_old_code", "use_verified_canonical_code") if code in canonical else ("BSE_equity_old_code_requires_verified_alias", "hold_for_official_code_alias")
            primary = "https://dataclouds.cninfo.com.cn/sjother/regulatory/2024/20241213/765a685e5d5545248d554c24fefc6684.pdf" if code in canonical else "https://www.bse.cn/service/code_mapping.html"
        elif code.startswith(("43", "83", "87")):
            kind, decision = "NEEQ_or_BSE_unverified", "exclude_until_exchange_listing_verified"
        elif code.startswith("40"):
            kind, decision = "delisted_share_transfer", "exclude_non_exchange_A_share"
        elif code.startswith(("50", "51", "52", "53", "56", "58")):
            kind, decision = "exchange_traded_fund", "exclude_non_equity"
        else:
            kind, decision = "fund_or_unresolved_non_equity", "exclude_until_identity_verified"
        conflict = CONFLICTS.get((code, name))
        if conflict:
            identified, other_code, code_url, name_url, note = conflict
            conflicts.append(dict(stock=code, original_name=name, code_identifies=identified,
                                  named_issuer_code=other_code, original_code_primary_source=code_url,
                                  named_issuer_primary_source=name_url, evidence_status="primary_identity_evidence" if code_url else "unverified_suspected_conflict",
                                  action="quarantine_pair_no_automatic_recode", note=note))
            decision = "quarantine_name_code_pair"
            primary = code_url
        rows.append(dict(stock=code, original_name=name, asset_type=kind, missing_event_count=int(counts[code]),
                         canonical_code=canonical.get(code, code),
                         decision=decision, primary_source=primary,
                         evidence_status="primary_source_identified" if primary else "screened_from_original_name_and_code; provider_identity_checked_on_fetch",
                         original_code_changed=False))
    result = pd.DataFrame(rows)
    result.to_csv(output / "asset_classification.csv", index=False)
    pd.DataFrame(conflicts).to_csv(output / "name_code_conflicts.csv", index=False)
    return events, result


class ProviderStopped(RuntimeError):
    pass


class Eastmoney:
    def __init__(self, output, pause):
        self.output, self.pause = output, pause
        self.session = requests.Session()

    def fetch(self, code, start, end, index=False):
        security = code.split(".")[-1]
        market = "1" if code.startswith("sh.") or (not index and security.startswith(("60", "68"))) else "0"
        params = dict(secid=market + "." + security, klt="101", fqt="0" if index else "1",
                      beg=start.replace("-", ""), end=end.replace("-", ""),
                      fields1="f1,f2,f3,f4,f5,f6", fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61")
        tag = "eastmoney_" + code.replace(".", "_") + "_" + params["beg"] + "_" + params["end"] + "_fqt" + params["fqt"]
        raw, meta = self.output / "raw" / (tag + ".json"), self.output / "raw" / (tag + ".metadata.json")
        if raw.exists() and meta.exists():
            metadata = json.loads(meta.read_text())
            if sha256(raw) != metadata["raw_sha256"]:
                raise ValueError("Raw response hash mismatch: " + str(raw))
            payload = json.loads(raw.read_text())
        else:
            metadata = dict(provider="Eastmoney public historical quote endpoint", url=URL,
                            request_parameters=params, requested_start=start, requested_end=end, fetched_utc=now(),
                            adjustment="none (index)" if index else "Eastmoney forward-adjusted (fqt=1), snapshot at fetch time; differs from Baostock algorithm",
                            units=dict(price="index points" if index else "CNY per share", volume_raw="100-share lots", volume_normalized="shares", amount="CNY", turnover="percentage points", pct_chg="percentage points"))
            try:
                response = self.session.get(URL, params=params, timeout=25)
                metadata.update(http_status=response.status_code, resolved_url=response.url)
                if response.status_code in {401, 403, 429}:
                    write_json(meta, dict(metadata, stopped=True, reason="authorization_or_rate_limit; no retries or bypass"))
                    raise ProviderStopped("Provider returned HTTP " + str(response.status_code))
                response.raise_for_status()
                payload = response.json()
                raw.write_bytes(response.content)
                metadata["raw_sha256"] = sha256(raw)
                metadata["raw_file"] = str(raw)
                write_json(meta, metadata)
            except requests.RequestException as error:
                write_json(meta, dict(metadata, error=str(error)))
                raise
            finally:
                time.sleep(self.pause)
        if payload.get("rc") != 0:
            raise ValueError("Provider rc=" + str(payload.get("rc")))
        data = payload.get("data")
        if not data or not data.get("klines"):
            return pd.DataFrame(), dict(metadata, outcome="empty_response")
        if data["code"] != security or int(data["market"]) != int(market):
            raise ValueError("Response identity differs from requested code/market")
        columns = ["date", "open", "close", "high", "low", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]
        frame = pd.DataFrame([line.split(",") for line in data["klines"]], columns=columns)
        frame["date"] = pd.to_datetime(frame.date)
        for column in columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["volume"] *= 100.
        frame["code"] = security
        frame = frame.sort_values("date").reset_index(drop=True)
        validate(frame, start, end)
        return frame, dict(metadata, outcome="data", response_name=data["name"], response_decimal=int(data["decimal"]),
                           n_rows=len(frame), first_date=str(frame.date.min().date()), last_date=str(frame.date.max().date()))


def validate(frame, start, end):
    if frame.date.duplicated().any() or not frame.date.between(pd.Timestamp(start), pd.Timestamp(end)).all():
        raise ValueError("Duplicate or out-of-range dates")
    required = ["close", "volume", "amount", "pct_chg"]
    if not np.isfinite(frame[required].to_numpy()).all() or (frame.close <= 0).any():
        raise ValueError("Nonfinite or nonpositive price/required observation")
    if (frame[["volume", "amount"]] < 0).any().any():
        raise ValueError("Negative traded volume/amount")


def compare_index(frame, code, original, output, metadata):
    current = frame[["date", "close"]].rename(columns={"close": "new_close"}).copy()
    current["new_ret"] = current.new_close.pct_change()
    current["new_prev_close"] = current.new_close.shift()
    old = original[original.index_code == code].copy()
    both = current.merge(old, on="date", how="inner")
    both["close_difference"] = both.new_close - both.index_close
    both["return_difference"] = both.new_ret - both.index_ret
    # Eastmoney index prices have two decimals; old Baostock has four. Bound
    # the return error from independent rounding of current/previous closes.
    half_tick = .5 * 10. ** (-metadata["response_decimal"]) + 0.000051
    both["close_tolerance"] = half_tick
    both["return_tolerance"] = (half_tick / (both.new_prev_close - half_tick)
        + both.new_close.abs() * half_tick / ((both.new_prev_close - half_tick) * both.new_prev_close)) + 1e-10
    both["close_agrees"] = both.close_difference.abs() <= both.close_tolerance
    both["return_agrees"] = both.return_difference.abs() <= both.return_tolerance
    both.to_csv(output / "indices" / (code.replace(".", "_") + "_overlap.csv"), index=False)
    passed = len(both) >= 10 and bool(both.close_agrees.all()) and bool(both.return_agrees.all())
    report = dict(index_code=code, overlap_days=len(both), overlap_pass=passed,
                  max_abs_close_difference=float(both.close_difference.abs().max()) if len(both) else None,
                  max_abs_return_difference=float(both.return_difference.abs().max()) if len(both) else None,
                  close_precision_decimals=metadata["response_decimal"],
                  comparison_rule="at least 10 overlapping sessions; close and close.pct_change must agree within documented rounding bounds",
                  old_overlap_values_preserved=True)
    supplement = current[(current.date >= "2022-09-01") & ~current.date.isin(old.date)].copy()
    supplement = supplement.rename(columns={"new_close": "index_close", "new_ret": "index_ret"})
    supplement["index_code"] = code
    supplement = supplement[["index_code", "date", "index_close", "index_ret"]]
    if passed:
        supplement.to_parquet(output / "indices" / (code.replace(".", "_") + "_supplement.parquet"), index=False)
    report["supplement_rows"] = len(supplement) if passed else 0
    return supplement if passed else None, report


def fetch_canonical_equities(source, output, start, end, pause):
    from screen_event_assets import CODE_ALIASES, normalize_name
    existing = set(pd.read_parquet(source / "control_stock_finder/data/A_share_daily.parquet", columns=["code"]).code)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provider = Eastmoney(output, pause)
    outcomes = []
    targets = dict(CODE_ALIASES)
    targets["601989"] = ("601989", "中国重工", "2009-12-16", "", "https://dataclouds.cninfo.com.cn/sjother/regulatory/2025/20250829/076aae87d5334d9d974d515b3ab04039.pdf")
    for original, (canonical, issuer, listed, changed, evidence) in targets.items():
        record = dict(original_code=original, canonical_code=canonical, issuer=issuer, primary_identity_source=evidence,
                      policy="One price series per issuer, keyed by canonical code; no duplicate old-code candidate")
        if canonical in existing:
            record["outcome"] = "use_existing_original_market_canonical_series; no new provider mixed"
            outcomes.append(record)
            continue
        frame, metadata = provider.fetch(canonical, start, end)
        record.update(metadata)
        if not len(frame):
            outcomes.append(record)
            continue
        if normalize_name(metadata["response_name"]) != normalize_name(issuer):
            raise ValueError("Provider name does not match officially mapped issuer " + canonical)
        record["excluded_pre_exchange_listing_rows"] = int((frame.date < pd.Timestamp(listed)).sum())
        frame = frame[frame.date >= pd.Timestamp(listed)].copy()
        normalized = frame[["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]]
        path = output / "stocks" / (canonical + ".parquet")
        normalized.to_parquet(path, index=False)
        record.update(normalized_file=str(path), normalized_sha256=sha256(path), normalized_rows=len(normalized),
                      normalized_first_date=str(normalized.date.min().date()), normalized_last_date=str(normalized.date.max().date()))
        write_json(output / "stocks" / (canonical + ".metadata.json"), record)
        outcomes.append(record)
        print("canonical stock", original, canonical, issuer, len(frame), flush=True)
    manifest["canonical_stock_outcomes"] = outcomes
    parts = [pd.read_parquet(path) for path in sorted((output / "stocks").glob("[0-9][0-9][0-9][0-9][0-9][0-9].parquet"))]
    combined = pd.concat(parts, ignore_index=True).sort_values(["code", "date"])
    if combined.duplicated(["code", "date"]).any() or set(combined.code) & existing:
        raise ValueError("Canonical supplement duplicates dates or original securities")
    path = output / "stocks" / "supplemental_stock_daily.parquet"
    combined.to_parquet(path, index=False)
    manifest["combined_stock"] = dict(file=str(path), sha256=sha256(path), rows=len(combined), codes=combined.code.nunique())
    manifest["source_code_sha256"] = sha256(__file__)
    mapping_pdf = output / "raw" / "bse_official_code_mapping_20241213.pdf"
    if mapping_pdf.exists():
        manifest["canonical_identity_evidence"] = dict(file=str(mapping_pdf), sha256=sha256(mapping_pdf),
            url="https://dataclouds.cninfo.com.cn/sjother/regulatory/2024/20241213/765a685e5d5545248d554c24fefc6684.pdf",
            evidence="Original BSE regulatory attachment from CNINFO; separately obtained with HTTP200 after no further requests to the HTTP403 BSE webpage")
    write_json(manifest_path, manifest)


def execute(args):
    source, output = args.source.resolve(), args.output.resolve()
    for directory in [output, output / "raw", output / "stocks", output / "indices"]:
        directory.mkdir(parents=True, exist_ok=True)
    if args.canonical_only:
        fetch_canonical_equities(source, output, args.start, args.end, args.pause)
        return
    events, classification = classifications(source, output)
    original_path = source / "control_stock_finder/data/A_share_daily.parquet"
    original = pd.read_parquet(original_path, columns=["code"])
    existing = set(original.code.astype(str))
    manifest_path = output / "manifest.json"
    previous_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = dict(created_utc=now(), original_market_path=str(original_path), original_market_sha256=sha256(original_path),
                    original_events_sha256=sha256(source / "original_data/events.csv"),
                    source_code_sha256=sha256(__file__), stock_request_start=args.start, stock_request_end=args.end,
                    original_files_modified=False, event_codes_automatically_changed=False,
                    stock_policy="Only allow-listed original SH/SZ equity codes wholly absent from original market; no splicing different providers into an existing stock",
                    index_policy="Keep all original overlapping values; add checked missing sessions before, within and after original coverage; no calibration of index levels",
                    known_attempts=[dict(provider="Baostock 00.9.40", attempt="ordinary anonymous bs.login(), socket timeout12", outcome="10002007 network receive error; no history obtained", observed_date="2026-11-05"),
                                    dict(provider="BSE official code mapping page", url="https://www.bse.cn/service/code_mapping.html", outcome="HTTP403; no retries or bypass; old-code candidates kept pending")],
                    securities=[], indices=[])
    if args.indices_only:
        manifest["securities"] = previous_manifest.get("securities", [])
        if "combined_stock" in previous_manifest:
            manifest["combined_stock"] = previous_manifest["combined_stock"]
    if "event_decisions" in previous_manifest:
        manifest["event_decisions"] = previous_manifest["event_decisions"]
    write_json(output / "manifest.json", manifest)
    if args.classify_only:
        return
    provider = Eastmoney(output, args.pause)
    all_stock = []
    failures = 0
    for code in ([] if args.indices_only else STOCK_CODES):
        if code in existing:
            raise ValueError("Refusing to splice a new adjustment source into existing stock " + code)
        try:
            frame, metadata = provider.fetch(code, args.start, args.end)
            metadata.update(original_code=code, original_names=classification[classification.stock == code].original_name.tolist(),
                            eligibility="Only original code/name pairs without a listed conflict; no event relabeling")
            if len(frame):
                path = output / "stocks" / (code + ".parquet")
                normalized = frame[["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]]
                normalized.to_parquet(path, index=False)
                all_stock.append(normalized)
                metadata.update(normalized_file=str(path), normalized_sha256=sha256(path),
                                full_requested_end_reached=frame.date.max() == pd.Timestamp(args.end),
                                missing_values_by_field={c: int(normalized[c].isna().sum()) for c in normalized})
            write_json(output / "stocks" / (code + ".metadata.json"), metadata)
            manifest["securities"].append(metadata)
            failures = 0
            print("stock", code, metadata.get("response_name", ""), len(frame), flush=True)
        except ProviderStopped as error:
            manifest["stopped"] = str(error)
            write_json(output / "manifest.json", manifest)
            raise
        except (requests.RequestException, ValueError) as error:
            failures += 1
            manifest["securities"].append(dict(original_code=code, outcome="error", error=str(error)))
            print("stock failed", code, str(error), flush=True)
            if failures >= 3:
                manifest["stopped"] = "Three consecutive provider failures; no further requests"
                write_json(output / "manifest.json", manifest)
                raise ProviderStopped(manifest["stopped"])
        write_json(output / "manifest.json", manifest)
    if all_stock:
        stock = pd.concat(all_stock, ignore_index=True).sort_values(["code", "date"])
        combined_path = output / "stocks" / "supplemental_stock_daily.parquet"
        stock.to_parquet(combined_path, index=False)
        manifest["combined_stock"] = dict(file=str(combined_path), sha256=sha256(combined_path), rows=len(stock), codes=stock.code.nunique())
    indices_path = source / "RAIF/results/preflight/indices.parquet"
    original_indices = pd.read_parquet(indices_path)
    original_indices["date"] = pd.to_datetime(original_indices.date)
    supplements = []
    for code in INDEX_CODES:
        frame, metadata = provider.fetch(code, "2022-08-31", args.end, index=True)
        if not len(frame):
            manifest["indices"].append(dict(index_code=code, overlap_pass=False, outcome="empty_response"))
            continue
        supplement, report = compare_index(frame, code, original_indices, output, metadata)
        metadata.update(report)
        manifest["indices"].append(metadata)
        write_json(output / "indices" / (code.replace(".", "_") + ".metadata.json"), metadata)
        if supplement is not None:
            supplements.append(supplement)
        print("index", code, report, flush=True)
        write_json(output / "manifest.json", manifest)
    pd.DataFrame([{k: v for k, v in r.items() if k in {"index_code", "overlap_days", "overlap_pass", "max_abs_close_difference", "max_abs_return_difference", "supplement_rows"}} for r in manifest["indices"]]).to_csv(output / "indices" / "overlap_summary.csv", index=False)
    if supplements:
        added = pd.concat(supplements, ignore_index=True)
        added_path = output / "indices" / "supplemental_indices.parquet"
        added.to_parquet(added_path, index=False)
        manifest["supplemental_indices_sha256"] = sha256(added_path)
        combined = pd.concat([original_indices, added], ignore_index=True).sort_values(["index_code", "date"])
        if combined.duplicated(["index_code", "date"]).any():
            raise ValueError("Index merge would duplicate a date")
        combined_path = output / "indices" / "indices_with_checked_history.parquet"
        combined.to_parquet(combined_path, index=False)
        calendar = pd.DatetimeIndex(combined.loc[combined.index_code == "sh.000001", "date"])
        coverage = []
        for code in INDEX_CODES:
            dates = pd.DatetimeIndex(combined.loc[combined.index_code == code, "date"])
            coverage.append(dict(index_code=code, rows=len(dates), missing_master_calendar_dates=len(calendar.difference(dates)),
                                 extra_dates=len(dates.difference(calendar))))
        pd.DataFrame(coverage).to_csv(output / "indices" / "calendar_coverage.csv", index=False)
        manifest["combined_indices"] = dict(file=str(combined_path), sha256=sha256(combined_path), new_rows=len(added),
                                            original_indices_sha256=sha256(indices_path), all_five_indices_passed=len(supplements) == 5,
                                            complete_master_calendar=all(item["missing_master_calendar_dates"] == 0 and item["extra_dates"] == 0 for item in coverage))
    manifest["completed_utc"] = now()
    manifest["original_market_unchanged"] = manifest["original_market_sha256"] == sha256(original_path)
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data/supplemental_market")
    parser.add_argument("--start", default="2022-09-01")
    parser.add_argument("--end", default="2026-03-20")
    parser.add_argument("--pause", type=float, default=.5)
    parser.add_argument("--classify-only", action="store_true")
    parser.add_argument("--indices-only", action="store_true", help="Preserve completed stock downloads and refresh full-span index checks")
    parser.add_argument("--canonical-only", action="store_true", help="Use verified same-issuer code mappings and keep one canonical series")
    execute(parser.parse_args())
