"""Render the mid.0-style attenuation report from saved up.0.0 sweep shards."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

RUN = Path(__file__).resolve().parent
OUT = RUN / "attenuation"
ALPHAS = (-7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7)
SEED = 30
TILE = 216
VIEWS = ("plain_grids", "signed_change_heatmaps", "post_activation_heatmaps",
         "intervention_footprint_heatmaps")


def overlay(image, values, signed=False):
    magnitude = np.abs(values)
    scale = float(magnitude.max())
    if scale == 0:
        return image
    normalized = magnitude / scale
    if signed:
        warm = plt.get_cmap("autumn_r")(0.85 * normalized)
        cool = plt.get_cmap("winter_r")(0.78 * normalized)
        cool[..., 0] = 0.58 * normalized**3
        colors = np.where((values < 0)[..., None], cool, warm)
    else:
        colors = plt.get_cmap("jet")(normalized)
    heat = Image.fromarray((colors[..., :3] * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.NEAREST)
    alpha = Image.fromarray(((magnitude > 0) * 142).astype(np.uint8)).resize(
        image.size, Image.Resampling.NEAREST)
    return Image.composite(heat, image, alpha)


def make_grid(path, panels):
    step = TILE + 10
    row_height = TILE + 30
    sheet = Image.new("RGB", (4 * step, 3 * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(panels):
        row, column = divmod(index, 4)
        x, y = column * step, row * row_height
        draw.text((x + 4, y + 4), label, fill="#142539")
        sheet.paste(image, (x + 4, y + 20))
    path.parent.mkdir(exist_ok=True, parents=True)
    sheet.save(path, quality=87, optimize=True)


def plot_curves(path, metrics):
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, layout="constrained")
    for ax, (group, part) in zip(axes.flat, metrics.groupby("group")):
        for value, label in (("post_mean", "Re-encoded"), ("scaled_mean", "Intended signed")):
            ax.errorbar(part.alpha, part[f"{value}_mean"], yerr=part[f"{value}_sem"],
                        fmt="o-", markersize=3, label=label)
        ax.axhline(0, color="#bbb", linewidth=0.7)
        ax.set_title(group.replace("_", " "))
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("Intervention multiplier")
    fig.supylabel("Mean feature activation across spatial grid")
    fig.savefig(path, dpi=135)
    plt.close(fig)


def render(feature, peak):
    folder = OUT / f"feature_{feature}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(RUN / "full" / f"feature_{feature}" / "manifest.csv")
    subset = manifest[manifest.alpha.isin(ALPHAS)].sort_values(["group", "seed", "alpha"])
    if len(subset) != 60 * len(ALPHAS) or subset.duplicated(["group", "seed", "alpha"]).any():
        raise ValueError(f"Incomplete attenuation data for feature {feature}")
    records = []
    for (group, seed), rows in subset.groupby(["group", "seed"]):
        panels = {view: [] for view in VIEWS}
        for row in rows.itertuples():
            path = RUN / row.image
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                pre, post, applied = (data[key] for key in ("pre", "post", "applied_l2"))
            scaled = row.alpha * pre
            records.append(dict(group=group, seed=seed, alpha=row.alpha, feature=feature,
                                pre_mean=float(pre.mean()), post_mean=float(post.mean()),
                                scaled_mean=float(scaled.mean()), applied_l2=row.delta_fro_applied,
                                pixel_mae=row.pixel_mae_vs_clean, zero_activation=row.zero_activation))
            if seed != SEED:
                continue
            with Image.open(path) as image:
                tile = image.convert("RGB").resize((TILE, TILE), Image.Resampling.LANCZOS)
            label = f"alpha {row.alpha:+d} | L2 {row.delta_fro_applied:.1f}"
            panels["plain_grids"].append((label, tile))
            panels["signed_change_heatmaps"].append((label, overlay(tile, scaled, signed=True)))
            panels["post_activation_heatmaps"].append((label, overlay(tile, post)))
            panels["intervention_footprint_heatmaps"].append((label, overlay(tile, applied)))
        if seed == SEED:
            for view, edits in panels.items():
                original = edits[ALPHAS.index(1)][1]
                make_grid(folder / view / f"{group}_seed{seed}.jpg", [("Original", original), *edits])

    metrics = pd.DataFrame(records)
    metrics.to_csv(folder / "metrics.csv", index=False)
    columns = ["pre_mean", "post_mean", "scaled_mean", "applied_l2", "pixel_mae"]
    summary = metrics.groupby(["group", "alpha"])[columns].agg(["mean", "sem"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(folder / "mean_metrics_by_group_alpha.csv", index=False)
    plot_curves(folder / "activation_curves.png", summary)
    lines = [f"# up.0.0 feature {feature}", "",
             f"Peak discovery prompt group · {peak.replace('_', ' ')}", "",
             "[All 660 conditions](metrics.csv) · [Five-seed means and SEM](mean_metrics_by_group_alpha.csv)",
             "", f"![Signed coefficients for the peak group](signed_change_heatmaps/{peak}_seed{SEED}.jpg)",
             "", "![Activation curves](activation_curves.png)", "",
             "| Prompt group | Plain | Signed coefficients | Post activation | Applied L2 footprint |",
             "| --- | --- | --- | --- | --- |"]
    for group in sorted(metrics.group.unique()):
        links = [f"[{label}]({view}/{group}_seed{SEED}.jpg)" for label, view in
                 zip(("Plain", "Signed", "Post", "Footprint"), VIEWS)]
        lines.append("| " + " | ".join([group.replace("_", " "), *links]) + " |")
    (folder / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ranking = pd.read_csv(RUN / "discovery/top10.csv")
    for row in ranking.itertuples():
        render(row.feature, row.peak_group)
    root = ["# up.0.0 attenuation reports", "",
            "All ten independently ranked features use the saved evaluation sweep. Each feature report covers "
            "12 prompt groups, seeds 30–34, and multipliers −7, −5, −3, −2, −1, 0, +1, +2, +3, +5, and +7. "
            "The visual grids use seed 30 for every group so the same portraits can be compared across features.", "",
            "The 3 × 4 grid shows the full requested range in ascending order. The first tile shows the clean baseline. "
            "The signed view uses α times the original coefficient, with warm colors for positive values "
            "and cool colors for negative values. At α=0 the signed map is zero and transparent. "
            "The post-activation view shows the feature after re-encoding the edited residual. "
            "The footprint view shows the actual per-cell perturbation L2 after the generation dtype cast. "
            "Each tile is color-scaled independently. Its caption gives the full-tensor L2 for dose comparisons.", "",
            "The curves show mean intended and re-encoded activation with standard errors across five seeds. "
            "All-zero overlays indicate a zero coefficient or perturbation for that panel.", "",
            "| Rank | Feature report | Peak group | Example signed grid |",
            "| ---: | --- | --- | --- |"]
    for rank, row in enumerate(ranking.itertuples(), 1):
        folder = f"feature_{row.feature}"
        root.append(f"| {rank} | [{row.feature}]({folder}/README.md) | {row.peak_group.replace('_', ' ')} | "
                    f"[View]({folder}/signed_change_heatmaps/{row.peak_group}_seed{SEED}.jpg) |")
    (OUT / "README.md").write_text("\n".join(root) + "\n")


if __name__ == "__main__":
    main()
