"""Render activation histograms for the visually selected multiplier candidates.

The candidate list is read from REASONABLE_MULTIPLIERS.md. These plots describe
coefficient activity and are not safety bounds or image quality evaluations.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "REASONABLE_MULTIPLIERS.md"
GROUP_COLORS = {
    "white_woman": "#8bb8e8",
    "middle_eastern_man": "#a95000",
    "middle_eastern_woman": "#f3ad62",
    "indian_man": "#5b3b86",
    "indian_woman": "#b39ddb",
    "hispanic_woman": "#c9a88e",
    "east_asian_man": "#6795a8",
    "black_man": "#c46b50",
    "east_asian_woman": "#8baf70",
    "hispanic_man": "#bd7ca5",
    "black_woman": "#6476a8",
    "indian_woman": "#b39ddb",
    "white_man": "#7f9c61",
}


def candidates_from_reference():
    candidates = {}
    for line in REFERENCE.read_text().splitlines():
        if not line.startswith("|") or "run_biased" not in line:
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        feature = int(columns[0])
        path_match = re.search(r"`(feature_\d+/[^`]+)`", columns[1])
        if path_match is None:
            continue
        candidate_text = columns[2]
        alphas = (None if candidate_text == "None identified" else
                  [int(value) for value in re.findall(r"[+-]\d+", candidate_text)])
        candidates[feature] = ((ROOT / path_match.group(1)).parent, alphas)
    return candidates


def display_group(group):
    return " ".join(word.capitalize() for word in group.split("_"))


def render(feature, run, alphas):
    maps = np.load(run / "activation_maps.npz")
    keys = [str(key) for key in maps["map_keys"]]
    groups = list(dict.fromkeys(key.split("|")[0] for key in keys))
    multipliers = [int(value) for value in maps["multipliers"]]
    alphas = multipliers if alphas is None else [alpha for alpha in alphas if alpha in multipliers]
    if not alphas:
        print(f"Skip feature {feature}: candidate multipliers absent from {run}")
        return
    alpha_indices = [multipliers.index(alpha) for alpha in alphas]
    scaled = maps["scaled_coefficient_maps"]
    pre = maps["pre_maps"]

    fig, (ax_count, ax_hist) = plt.subplots(
        2, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [1, 1.25]}
    )
    alpha_label = ", ".join(f"{alpha:+g}" for alpha in alphas)
    fig.suptitle(
        f"Feature {feature} · candidate multipliers {alpha_label}\n"
        "Top: raw image counts; bottom: signed scaled-coefficient frequency",
        fontsize=18,
        y=0.98,
    )

    bottoms = np.zeros(len(alphas), dtype=np.int64)
    centers = np.asarray(alphas, dtype=float)
    for group in groups:
        indices = [i for i, key in enumerate(keys) if key.split("|")[0] == group]
        values = np.asarray([
            np.count_nonzero(np.any(scaled[indices, ai] != 0, axis=(1, 2)))
            for ai in alpha_indices
        ])
        if not np.any(values):
            continue
        color = GROUP_COLORS.get(group, plt.get_cmap("tab20")(groups.index(group) % 20))
        ax_count.bar(centers, values, width=max(1, min(3.5, (max(alphas)-min(alphas))/len(alphas)*.6)) if len(alphas)>1 else 1,
                     bottom=bottoms, color=color, label=display_group(group))
        bottoms += values
    ax_count.set_title("Images with a nonzero scaled feature by multiplier (group colors labeled below)")
    ax_count.set_ylabel("Images with nonzero coefficient (count)")
    ax_count.set_xlabel("Multiplier α")
    ax_count.set_xticks(alphas)
    if len(alphas) > 1:
        pad = max(2, (max(alphas) - min(alphas)) * 0.08)
        ax_count.set_xlim(min(alphas) - pad, max(alphas) + pad)
    ax_count.axvline(0, color="black", linewidth=1)
    ax_count.grid(axis="y", alpha=0.25)

    bins = np.linspace(-550, 550, 51)
    hist_alpha_indices = [index for alpha, index in zip(alphas, alpha_indices) if alpha != 0]
    hist_values, hist_weights, hist_colors, hist_labels = [], [], [], []
    hist_total = sum(
        np.count_nonzero(pre[[i for i, key in enumerate(keys) if key.split("|")[0] == group]] > 0)
        * len(hist_alpha_indices)
        for group in groups
    )
    for group in groups if hist_alpha_indices and hist_total else []:
        indices = [i for i, key in enumerate(keys) if key.split("|")[0] == group]
        active_mask = np.broadcast_to(
            pre[indices, None] > 0,
            (len(indices), len(hist_alpha_indices), *pre.shape[1:]),
        )
        active_values = scaled[indices][:, hist_alpha_indices][active_mask]
        if active_values.size == 0:
            continue
        hist_values.append(active_values)
        hist_weights.append(np.full(active_values.shape, 100.0 / hist_total))
        hist_colors.append(GROUP_COLORS.get(group, plt.get_cmap("tab20")(groups.index(group) % 20)))
        hist_labels.append(display_group(group))
    if hist_values:
        ax_hist.hist(hist_values, bins=bins, weights=hist_weights, histtype="stepfilled",
                     stacked=True, color=hist_colors, alpha=0.78, label=hist_labels)
    ax_hist.set_title("Signed scaled coefficients at active positions")
    ax_hist.set_xlabel("Scaled coefficient (α × pre-intervention coefficient)")
    ax_hist.set_ylabel("Frequency among active positions (%)")
    ax_hist.set_xlim(-550, 550)
    ax_hist.axvline(0, color="black", linewidth=1)
    ax_hist.grid(alpha=0.25)
    if hist_labels:
        ax_hist.legend(loc="upper center", ncol=4, framealpha=0.85)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    position = ax_count.get_position()
    width = position.width * (2 / 3)
    ax_count.set_position((position.x0 + (position.width - width) / 2,
                           position.y0, width, position.height))
    output = ROOT / f"feature_{feature}" / f"feature_{feature}_candidate_histogram.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    for feature, (run, alphas) in candidates_from_reference().items():
        render(feature, run, alphas)


if __name__ == "__main__":
    main()
