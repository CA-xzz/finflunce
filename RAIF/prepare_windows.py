"""Vary input history on common events with the original fixed reference labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare import MarketPanel, encode_categories


def run(args):
    normal = json.loads((args.data / "normalization.json").read_text())
    if normal.get("ready_for_formal_training") is not True:
        raise ValueError("Input dataset has not passed its source checks")
    events = pd.read_csv(args.data / "events.csv", dtype={"stock": str, "event_id": str})
    base = dict(np.load(args.data / "dataset.npz"))
    labels = pd.read_csv(args.data / "labels.csv")
    expected_ids = np.repeat(events.event_id.to_numpy(), 33)
    if not np.array_equal(labels.event_id.to_numpy(), expected_ids):
        raise ValueError("Raw label rows are not aligned with event IDs")
    if not np.array_equal(labels.tau.to_numpy(), np.tile(np.repeat(np.arange(11), 3), len(events))):
        raise ValueError("Raw label time-step ordering is inconsistent")
    if not np.array_equal(labels.channel.to_numpy(), np.tile(["AR", "turnover", "log_volume"], len(events) * 11)):
        raise ValueError("Raw label channel ordering is inconsistent")
    raw_y = labels.delta.to_numpy().reshape(len(events), 11, 3)
    config_path = args.data / "config.json"
    source_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    reference_mode = normal.get("reference_mode", source_config.get("reference_mode", "soft"))
    reference_length = int(source_config.get("length", normal.get("reference_length", base["x"].shape[1])))
    panel = MarketPanel(args.source, args.preflight, args.external, length=20,
                        expanded=reference_mode == "expanded_nearest")
    chosen, inputs = [], []
    for row in events.itertuples():
        state = panel.state(int(row.day))
        position = panel.positions.get(row.stock)
        if state is not None and position is not None and np.isfinite(state.x[position]).all():
            chosen.append(row.Index)
            inputs.append(state.x[position])
    chosen = np.asarray(chosen)
    if not len(chosen):
        raise ValueError("No complete common L20 input events")
    x20 = np.stack(inputs)
    subset = events.iloc[chosen].copy().reset_index(drop=True)
    split = base["split"][chosen]
    if any(np.sum(split == i) == 0 for i in [0, 1, 2]):
        raise ValueError("A common-window split is empty")
    training = split == 0
    y = raw_y[chosen]
    y_mean, y_std = y[training].mean((0, 1)), np.maximum(y[training].std((0, 1)), 1e-8)
    platform, pv = encode_categories(subset.platform.to_numpy(), training)
    influencer, iv = encode_categories(subset.influencer.to_numpy(), training)
    quality = pd.read_csv(args.data / "reference_quality.csv").set_index("event_id").loc[subset.event_id].reset_index()
    sb = max(float(np.median(quality.loc[training, "balance"])), 1e-8)
    sd = max(float(np.median(quality.loc[training, "mean_distance"])), 1e-8)
    q = np.exp(-quality.balance.to_numpy() / sb - quality.mean_distance.to_numpy() / sd)
    if normal.get("quality_effective_n_factor", True):
        q = q * quality.effective_n.to_numpy() / (quality.effective_n.to_numpy() + 1)
    q = np.maximum(q, 1e-8).astype(np.float32)
    subset["q"] = quality["q"] = q
    label_version = str(subset.label_version.iloc[0]) + "_commonL20"
    subset["label_version"] = label_version
    for length in [5, 10, 20]:
        out = args.output / f"L{length}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "dataset.npz").exists():
            raise FileExistsError(str(out))
        x = x20[:, -length:, :]
        xm, xs = x[training].mean((0, 1)), np.maximum(x[training].std((0, 1)), 1e-8)
        np.savez_compressed(out / "dataset.npz", x=((x - xm) / xs).astype(np.float32),
            y=((y - y_mean) / y_std).astype(np.float32), q=q, platform=platform, influencer=influencer,
            day=base["day"][chosen], end_day=base["end_day"][chosen], split=split)
        subset.to_csv(out / "events.csv", index=False)
        quality.to_csv(out / "reference_quality.csv", index=False)
        current = dict(normal, x_mean=xm.tolist(), x_std=xs.tolist(), y_mean=y_mean.tolist(), y_std=y_std.tolist(),
            n_platforms=len(pv), n_influencers=len(iv), platform_vocab=pv, influencer_vocab=iv,
            quality_balance_scale=sb, quality_distance_scale=sd, label_version=label_version,
            normalization_fit_event_ids=subset.loc[training, "event_id"].tolist(),
            input_length=length, reference_length=reference_length, source_reference_mode=reference_mode,
            reference_label_source=str(args.data),
            window_comparison=f"Input-window ablation on fixed original L{reference_length} {reference_mode} labels and common L20-complete events; matching rules and labels are not changed with input length.")
        (out / "normalization.json").write_text(json.dumps(current, indent=2, ensure_ascii=False))
        subset.loc[split == 2, ["event_id", "event_date", "label_version"]].to_csv(out / "evaluation_manifest.csv", index=False)
        np.savez_compressed(out / "evaluation_targets.npz", event_ids=subset.loc[split == 2, "event_id"].to_numpy(str),
                            truth=y[split == 2])
        assignments = args.data / "industry_assignments.csv"
        if assignments.exists():
            detail = pd.read_csv(assignments, dtype={"event_id": str, "stock": str})
            detail[detail.event_id.isin(subset.event_id)].to_csv(out / "industry_assignments.csv", index=False)
    args.output.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output / "common_events.csv", index=False)
    (args.output / "protocol.json").write_text(json.dumps({"reference_label_source": str(args.data),
        "common_events": len(chosen), "split_counts": subset.split.value_counts().to_dict(),
        "lengths": [5, 10, 20], "reference_length_fixed": reference_length, "source_reference_mode": reference_mode,
        "reference_weights_and_raw_targets_unchanged": True, "scales_fit_on_common_training_events": True}, indent=2))
    print(json.dumps({"n_common": len(chosen), "split_counts": subset.split.value_counts().to_dict()}))


def main():
    parser = argparse.ArgumentParser()
    for name in ["data", "source", "preflight", "external", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
