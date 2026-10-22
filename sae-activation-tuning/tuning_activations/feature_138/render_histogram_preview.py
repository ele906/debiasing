"""Render the feature 138 multiplier preview from saved activation maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "run_biased_002"
OUTPUT = ROOT / "feature_138_histogram_preview_stacked.png"
PREVIEW_ALPHAS = (-5, 5, 10)
GROUP_LABELS = {
    "white_woman": "White Woman",
    "middle_eastern_man": "Middle Eastern Man",
    "middle_eastern_woman": "Middle Eastern Woman",
    "indian_man": "Indian Man",
    "indian_woman": "Indian Woman",
    "hispanic_woman": "Hispanic Woman",
}
GROUP_COLORS = {
    "white_woman": "#8bb8e8",
    "middle_eastern_man": "#a95000",
    "middle_eastern_woman": "#f3ad62",
    "indian_man": "#5b3b86",
    "indian_woman": "#b39ddb",
    "hispanic_woman": "#c9a88e",
}


def main():
    maps = np.load(RUN / "activation_maps.npz")
    keys = [str(key) for key in maps["map_keys"]]
    groups = list(dict.fromkeys(key.split("|")[0] for key in keys))
    multipliers = [int(value) for value in maps["multipliers"]]
    alpha_indices = [multipliers.index(alpha) for alpha in PREVIEW_ALPHAS]
    scaled = maps["scaled_coefficient_maps"]
    pre = maps["pre_maps"]

    fig, (ax_count, ax_hist) = plt.subplots(
        2, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [1, 1.25]}
    )
    fig.suptitle(
        "Feature 138 · candidate multipliers −5, +5, +10\n"
        "Top: raw image counts; bottom: signed scaled-coefficient frequency",
        fontsize=18,
        y=0.98,
    )

    bottoms = np.zeros(len(PREVIEW_ALPHAS), dtype=np.int64)
    centers = np.asarray(PREVIEW_ALPHAS, dtype=float)
    for group in groups:
        if group not in GROUP_LABELS:
            continue
        indices = [i for i, key in enumerate(keys) if key.split("|")[0] == group]
        values = np.asarray([
            np.count_nonzero(np.any(scaled[indices, alpha_index] != 0, axis=(1, 2)))
            for alpha_index in alpha_indices
        ])
        if not np.any(values):
            continue
        ax_count.bar(
            centers,
            values,
            width=3.5,
            bottom=bottoms,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
        bottoms += values
    ax_count.set_title("Images with a nonzero scaled feature by multiplier")
    ax_count.set_ylabel("Images with nonzero coefficient (count)")
    ax_count.set_xlabel("Multiplier α (0 omitted)")
    ax_count.set_xticks((-10, -5, 0, 5, 10))
    ax_count.set_xlim(-13, 13)
    ax_count.axvline(0, color="black", linewidth=1)
    ax_count.grid(axis="y", alpha=0.25)
    ax_count.legend(loc="lower left", ncol=3, framealpha=0.85)

    bins = np.linspace(-550, 550, 51)
    hist_values, hist_weights, hist_colors, hist_labels = [], [], [], []
    hist_total = sum(
        np.count_nonzero((pre[[i for i, key in enumerate(keys)
                               if key.split("|")[0] == group], None] > 0))
        * len(alpha_indices)
        for group in groups if group in GROUP_LABELS
    )
    for group in groups:
        if group not in GROUP_LABELS:
            continue
        indices = [i for i, key in enumerate(keys) if key.split("|")[0] == group]
        # Keep coefficients at positions active before intervention.
        active_mask = np.broadcast_to(pre[indices, None] > 0,
                                      (len(indices), len(alpha_indices), *pre.shape[1:]))
        active_values = scaled[indices][:, alpha_indices][active_mask]
        if active_values.size == 0:
            continue
        hist_values.append(active_values)
        hist_weights.append(np.full(active_values.shape, 100.0 / hist_total))
        hist_colors.append(GROUP_COLORS[group])
        hist_labels.append(GROUP_LABELS[group])
    ax_hist.hist(
        hist_values,
        bins=bins,
        weights=hist_weights,
        histtype="stepfilled",
        stacked=True,
        color=hist_colors,
        alpha=0.78,
        label=hist_labels,
    )
    ax_hist.set_title("Signed scaled coefficients at active positions")
    ax_hist.set_xlabel("Scaled coefficient (α × pre-intervention coefficient)")
    ax_hist.set_ylabel("Frequency among active positions (%)")
    ax_hist.set_xlim(-550, 550)
    ax_hist.axvline(0, color="black", linewidth=1)
    ax_hist.grid(alpha=0.25)
    ax_hist.legend(loc="upper center", ncol=4, framealpha=0.85)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
