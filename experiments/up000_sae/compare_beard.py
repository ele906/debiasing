"""Compare parser responses for the two screened beard candidates on common seeds."""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RUN = Path(__file__).resolve().parent
ARCHIVE = RUN.parent / "mid0_sae_selectivity/beard_interventions_20261117_020843_219166"
OUT = RUN / "beard"


def main():
    up = pd.read_csv(OUT / "paired.csv").assign(layer="up.0.0")
    mid = pd.read_csv(ARCHIVE / "results/per_image_scores.csv", low_memory=False)
    mid = mid[(mid.feature_id == 4228) & (mid.method == "multiplicative") & mid.seed.between(30, 34)]
    mid = mid.rename(columns={"prompt_group": "group", "multiplier": "alpha",
                              "delta_fixed_baseline_face_pp": "delta_fixed",
                              "baseline_bearded_proxy": "baseline_bearded"}).assign(layer="mid.0")
    fields = ["layer", "group", "seed", "alpha", "applied_l2", "delta_fixed", "baseline_bearded"]
    both = pd.concat([up[fields], mid[fields]], ignore_index=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    rows = []
    for ax, (direction, sign) in zip(axes, (("decrease", -1), ("increase", 1))):
        active = both[((both.alpha - 1) * sign > 0) & (both.applied_l2 > 0)].copy()
        bounds = active.groupby("layer").applied_l2.agg(["min", "max"])
        common = np.geomspace(bounds["min"].max(), bounds["max"].min(), 9)
        active["dose"] = pd.cut(active.applied_l2, common, labels=False, include_lowest=True)
        for subset in ("all", "baseline_bearded"):
            frame = active if subset == "all" else active[active.baseline_bearded]
            for dose, part in frame.groupby("dose"):
                pairs = part.groupby(["group", "seed", "layer"]).delta_fixed.mean().unstack("layer").dropna()
                if pairs.empty:
                    continue
                by_seed = pairs.groupby("seed").mean()
                for layer in by_seed:
                    rows.append(dict(direction=direction, subset=subset, layer=layer,
                                     center=part.applied_l2.median(), mean=by_seed[layer].mean(),
                                     sem=by_seed[layer].sem(), pairs=len(pairs)))
        plot = pd.DataFrame(rows)
        for layer, part in plot[(plot.direction == direction) & (plot.subset == "all")].groupby("layer"):
            ax.errorbar(part.center, part["mean"], yerr=part["sem"], fmt="o-", label=layer)
        ax.set(title=direction, xscale="log", xlabel="Applied L2", ylabel="Fixed-baseline beard area change (pp)")
        ax.axhline(0, color="grey")
        ax.legend()
    pd.DataFrame(rows).to_csv(OUT / "matched_mid0_4228.csv", index=False)
    figure.savefig(OUT / "matched_mid0_4228.png", dpi=180)


if __name__ == "__main__":
    main()
