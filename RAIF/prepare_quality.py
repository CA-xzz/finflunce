"""Training quality-retention subsets; fixed validation and test events/targets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare import encode_categories


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    d = dict(np.load(args.data / "dataset.npz"))
    n = json.loads((args.data / "normalization.json").read_text())
    e = pd.read_csv(args.data / "events.csv", dtype={"stock": str})
    labels = pd.read_csv(args.data / "labels.csv")
    xraw = d["x"].astype(float) * n["x_std"] + n["x_mean"]
    yraw = labels.delta.to_numpy().reshape(len(e), 11, 3)
    rows = []
    for quantile in [0.25, 0.5, 0.75]:
        cutoff = float(np.quantile(d["q"][d["split"] == 0], quantile))
        keep = (d["split"] != 0) | (d["q"] >= cutoff)
        subset = e[keep].reset_index(drop=True)
        split = d["split"][keep]
        train = split == 0
        x, y = xraw[keep], yraw[keep]
        xm, xs = x[train].mean((0, 1)), np.maximum(x[train].std((0, 1)), 1e-8)
        ym, ys = y[train].mean((0, 1)), np.maximum(y[train].std((0, 1)), 1e-8)
        platform, pv = encode_categories(subset.platform.to_numpy(), train)
        influencer, iv = encode_categories(subset.influencer.to_numpy(), train)
        output = args.output / f"q{int(quantile * 100)}"
        output.mkdir(parents=True, exist_ok=True)
        if (output / "dataset.npz").exists():
            raise FileExistsError(str(output))
        np.savez_compressed(output / "dataset.npz", x=((x - xm) / xs).astype(np.float32),
            y=((y - ym) / ys).astype(np.float32), q=d["q"][keep], split=split,
            day=d["day"][keep], end_day=d["end_day"][keep], platform=platform, influencer=influencer)
        subset.to_csv(output / "events.csv", index=False)
        normal = dict(n, x_mean=xm.tolist(), x_std=xs.tolist(), y_mean=ym.tolist(), y_std=ys.tolist(),
            n_platforms=len(pv), n_influencers=len(iv), platform_vocab=pv, influencer_vocab=iv,
            normalization_fit_event_ids=subset.loc[train, "event_id"].tolist(),
            quality_retention_quantile=quantile, quality_cutoff=cutoff,
            quality_score_scales_frozen_from_original_training=True,
            quality_retention_protocol="Remove lower-quality training examples; refit normalization/vocab on retained train. Preserve original reference rules, q scores, all validation/test events and raw labels.")
        (output / "normalization.json").write_text(json.dumps(normal, indent=2, ensure_ascii=False))
        np.savez_compressed(output / "evaluation_targets.npz", event_ids=subset.loc[split == 2, "event_id"].to_numpy(str), y=y[split == 2])
        assert np.array_equal(y[split == 2], yraw[d["split"] == 2])
        rows.append({"quantile": quantile, "cutoff": cutoff, "n_train": int(train.sum()),
                     "retained_fraction": float(train.sum() / (d["split"] == 0).sum()), "n_test": int((split == 2).sum())})
    pd.DataFrame(rows).to_csv(args.output / "retention.csv", index=False)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
