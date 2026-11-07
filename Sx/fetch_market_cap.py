"""Download dated Eastmoney total-market-cap observations with resumable receipts."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
COLUMNS = "SECURITY_CODE,TRADE_DATE,CLOSE_PRICE,TOTAL_MARKET_CAP,TOTAL_SHARES"


def fetch_day(date, directory, pause):
    path = directory / (date + ".json")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("complete") and saved.get("query_date") == date:
            return {"date": date, "rows": len(saved["rows"]), "status": "cached"}
        raise ValueError(f"Incomplete existing response: {path}")
    page, pages, rows, receipts = 1, 1, [], []
    while page <= pages:
        params = dict(reportName="RPT_VALUEANALYSIS_DET", columns=COLUMNS,
                      sortColumns="SECURITY_CODE", sortTypes="1", pageSize="5000",
                      pageNumber=str(page), source="WEB", client="WEB",
                      filter=f"(TRADE_DATE='{date}')")
        request = Request(URL + "?" + urlencode(params), headers={"User-Agent": "Sx-research/1.0"})
        with urlopen(request, timeout=45) as stream:
            body = stream.read()
        result = json.loads(body)
        if not result.get("success") or result.get("result") is None:
            raise RuntimeError(f"{date}: {result.get('message', 'no result')}")
        data = result["result"]
        pages = int(data["pages"])
        if pages > 5:
            raise ValueError(f"Unexpected pagination for one market day: {pages}")
        receipts.append({"page": page, "sha256": hashlib.sha256(body).hexdigest(),
                         "reported_count": data["count"], "retrieved_at": datetime.now(timezone.utc).isoformat()})
        rows.extend(data["data"])
        time.sleep(pause)
        page += 1
    if len(rows) != int(data["count"]):
        raise ValueError(f"{date}: incomplete pages {len(rows)} != {data['count']}")
    if len({row["SECURITY_CODE"] for row in rows}) != len(rows):
        raise ValueError(f"{date}: duplicate securities")
    if any(row["TRADE_DATE"][:10] != date for row in rows):
        raise ValueError(f"{date}: response contains another trading date")
    payload = {"query_date": date, "source_url": URL, "parameters": params,
               "pages": receipts, "complete": True, "rows": rows}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    temporary.replace(path)
    return {"date": date, "rows": len(rows), "status": "downloaded"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    directory = args.output / "raw/market_cap"
    directory.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_csv(args.preflight / "calendar.csv", parse_dates=["date"])
    events = pd.read_csv(args.preflight / "events_manifest.csv")
    positions = sorted(set(events.loc[(events.split != "excluded") & (events.day > 0), "day"].astype(int) - 1))
    dates = calendar.iloc[positions].date.dt.strftime("%Y-%m-%d").tolist()
    if args.limit:
        dates = dates[:args.limit]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_day, date, directory, args.pause): date for date in dates}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                receipt = future.result()
            except Exception as exc:
                receipt = {"date": futures[future], "status": "failed", "error": str(exc)}
                failures.append(receipt)
            with (args.output / "market_cap_receipts.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt) + "\n")
            print(json.dumps({"completed": i, "total": len(dates), **receipt}), flush=True)
    if failures:
        (args.output / "market_cap_failures.json").write_text(json.dumps(failures, indent=2))
        raise SystemExit(f"{len(failures)} dates failed; incomplete data were not promoted")
    frames = [pd.DataFrame(json.loads((directory / (date + ".json")).read_text())["rows"]) for date in dates]
    frame = pd.concat(frames, ignore_index=True).rename(columns={"SECURITY_CODE": "code", "TRADE_DATE": "date",
        "CLOSE_PRICE": "unadjusted_close", "TOTAL_MARKET_CAP": "total_market_cap", "TOTAL_SHARES": "total_shares"})
    frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    frame["available_at"] = frame.date + pd.Timedelta(days=1)
    for col in ["total_market_cap", "total_shares", "unadjusted_close"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.to_parquet(args.output / "historical_market_cap.parquet", index=False)
    metadata = {"source_url": URL, "report": "RPT_VALUEANALYSIS_DET", "units": "CNY",
        "date_count": len(dates), "row_count": len(frame), "requested_dates": dates,
        "availability_rule": "Use prior trading day's closing value; available_at conservatively next calendar midnight.",
        "publication_timestamp_verified": False,
        "first_date": str(frame.date.min().date()), "last_date": str(frame.date.max().date())}
    (args.output / "market_cap_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "complete", "rows": len(frame), "dates": len(dates)}), flush=True)


if __name__ == "__main__":
    main()
