"""Export the pre-event inputs for the already fixed illustrative cases."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def run(data, report):
    cases = pd.read_csv(report / "case_manifest.csv", dtype={"event_id": str})
    events = pd.read_csv(data / "events.csv", dtype={"event_id": str})
    normal = json.loads((data / "normalization.json").read_text())
    with np.load(data / "dataset.npz", allow_pickle=False) as archive:
        x = archive["x"].astype(float) * np.asarray(normal["x_std"]) + np.asarray(normal["x_mean"])
    lookup = pd.Series(np.arange(len(events)), index=events.event_id)
    positions = lookup.reindex(cases.event_id)
    assert positions.notna().all() and not cases.event_id.duplicated().any()
    rows = []
    for case, position in zip(cases.itertuples(), positions.to_numpy(int)):
        for offset, values in zip(range(-x.shape[1], 0), x[position]):
            for channel, value in zip(["AR", "turnover", "log_volume"], values):
                rows.append(dict(case_id=case.case_id, event_id=case.event_id,
                    trading_day_offset=offset, channel=channel, value=float(value)))
    frame = pd.DataFrame(rows)
    assert len(frame) == len(cases) * x.shape[1] * 3 and np.isfinite(frame.value).all()
    frame.to_csv(report / "case_pre_event_inputs.csv", index=False)
    sources = [data / "dataset.npz", data / "normalization.json", report / "case_manifest.csv"]
    metadata = dict(cases=len(cases), rows=len(frame), input_offsets=[-x.shape[1], -1],
        selection="Uses the existing fixed platform/quality-stratified case manifest; no new outcome-based selection",
        units={"AR": "decimal return", "turnover": "percentage points", "log_volume": "natural log of traded shares"},
        precision="Inverse-transformed float32 model inputs",
        sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
    (report / "case_pre_event_inputs_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "rows": len(frame)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.report)
