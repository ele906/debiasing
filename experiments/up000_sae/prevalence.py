"""Plot positive-token prevalence by group and per-image prevalence distributions."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parent
OUTPUT = RUN / "feature_prevalence"


def main():
    data = pd.read_csv(RUN / "discovery/spatial_by_image.csv")
    OUTPUT.mkdir(exist_ok=True)
    layers = ["up.0.0", "mid.0"]
    group_rates = data.groupby(["layer", "feature", "group"], as_index=False).agg(
        mean_active_fraction=("active_fraction", "mean"),
        image_activation_rate=("active_fraction", lambda x: x.gt(0).mean()),
        mean_activation=("mean", "mean"), images=("seed", "size"),
    )
    group_rates.to_csv(OUTPUT / "group_prevalence.csv", index=False)
    for layer in layers:
        ranking_path = (RUN / "discovery/top10.csv" if layer == "up.0.0" else
                        RUN.parents[1] / "layers/mid0_stability/finetuned_ranking/ranked_candidates.csv")
        ranking = pd.read_csv(ranking_path).head(10)
        for entry in ranking.itertuples():
            part = data[(data.layer == layer) & (data.feature == entry.feature)]
            rates = group_rates[(group_rates.layer == layer) & (group_rates.feature == entry.feature)]
            rates = rates.sort_values(["mean_active_fraction", "group"])
            figure, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
            axes[0].barh(rates.group.str.replace("_", " "), 100 * rates.mean_active_fraction,
                         color=np.where(rates.group == entry.peak_group, "#bd5636", "#426986"))
            axes[0].set(xlabel="Mean positive spatial tokens (%)", title="Prevalence across all 12 prompt groups")
            peak = part[part.group == entry.peak_group]
            others = part[part.group != entry.peak_group]
            maximum = max(100 * part.active_fraction.max(), 1)
            bins = np.unique(np.r_[0, 100 / 512, np.linspace(100 / 512, maximum + .01, 13)])
            for subset, label, color in ((peak, entry.peak_group.replace("_", " "), "#bd5636"),
                                         (others, "Other 11 prompt groups", "#426986")):
                axes[1].hist(100 * subset.active_fraction, bins=bins, histtype="step", linewidth=2,
                             weights=np.full(len(subset), 100 / len(subset)), color=color, label=label)
            axes[1].set(xlabel="Positive spatial tokens per image (%)", ylabel="Images in subset (%)",
                        title="Per-image prevalence distribution")
            axes[1].legend()
            figure.suptitle(f"{layer} · feature {entry.feature} · disparity rank {entry.Index + 1}")
            figure.savefig(OUTPUT / f"{layer}_feature_{entry.feature}.png", dpi=130)
            plt.close(figure)


if __name__ == "__main__":
    main()
