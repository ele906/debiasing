"""Render mean-activation charts in the historical one-versus-many style."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parent
GROUPS = ("black", "white", "east_asian", "middle_eastern", "indian", "hispanic")
COLORS = ("#a72725", "#e79a94", "#32659b", "#93bfe4", "#3a8052", "#98d2a4",
          "#a45200", "#efaf69", "#593983", "#b6a1d9", "#856043", "#c8aa90")


def main():
    output = RUN / "one_vs_many"
    output.mkdir(exist_ok=True)
    labels = [f"{race.replace('_', ' ').title()}\n{gender}" for race in GROUPS
              for gender in ("men", "women")]
    group_keys = [f"{race}_{gender}" for race in GROUPS for gender in ("man", "woman")]
    with np.load(RUN / "discovery/group_activations.npz") as archive:
        order = [list(archive["groups"]).index(group) for group in group_keys]
        means = archive["means"][order]
    with (RUN / "discovery/top10.csv").open(newline="") as source:
        ranking = list(csv.DictReader(source))
    for row in ranking:
        feature = int(row["feature"])
        values = means[:, feature]
        peak = int(values.argmax())
        other_mean = np.delete(values, peak).mean()
        if group_keys[peak] != row["peak_group"] or not np.isclose(other_mean, float(row["other_mean"])):
            raise ValueError(f"Group activation data disagree with ranking for feature {feature}")
        peak_label = labels[peak].replace("\n", " ")
        fig, ax = plt.subplots(figsize=(15, 7), layout="constrained")
        ax.bar(labels, values, color=COLORS, edgecolor="white", linewidth=0.6)
        ax.set_title(f"up.0.0 SAE feature {feature}\n"
                     f"Peak • {peak_label} ({values[peak]:.3f})   |   Other 11 groups average {other_mean:.3f}",
                     fontsize=18)
        ax.set_ylabel("Mean SAE activation across spatial grid", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", color="#e2e2e2")
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        fig.savefig(output / f"feature_{feature}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
