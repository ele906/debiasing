"""Summarize the matched full sweeps on absolute and normalized applied doses."""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = RUN / "full/evaluation"


def main():
    frame = pd.read_csv(OUT / "per_image.csv")
    keys = ["layer", "feature", "group", "seed"]
    if frame.duplicated(keys + ["alpha"]).any() or set(frame.groupby("layer").size()) != {17400}:
        raise ValueError("Expected exactly 17,400 unique conditions per layer")
    frame["rms"] = frame.l2 / np.sqrt(1280 * 16 * 16)
    frame["face_localization"] = frame.inside_face_mae - frame.outside_face_mae
    metrics = ["detected", "image_mae", "inside_face_mae", "outside_face_mae", "face_localization"]
    active = frame[frame.active & frame.baseline_detected & (frame.l2 > 0)].copy()
    active["direction"] = np.where(active.alpha < 1, "decrease", "increase")
    table = []
    for axis in ("l2", "rms", "relative_l2"):
        for direction, part in active.groupby("direction"):
            bounds = part.groupby("layer")[axis].agg(["min", "max"])
            low, high = bounds["min"].max(), bounds["max"].min()
            part = part[part[axis].between(low, high)].copy()
            part["bin"] = pd.cut(part[axis], np.geomspace(low, high, 9), labels=False, include_lowest=True)
            for _, cell in part.groupby("bin"):
                pair_data = cell.groupby(["group", "seed", "layer"])[metrics + [axis]].mean().unstack("layer").dropna()
                if pair_data.empty:
                    continue
                for metric in metrics:
                    seed = pair_data[metric].groupby("seed").mean()
                    effect = seed["up.0.0"] - seed["mid.0"]
                    table.append(dict(axis=axis, direction=direction, dose_center=cell[axis].median(),
                                      dose_min=cell[axis].min(), dose_max=cell[axis].max(), metric=metric,
                                      mid0=seed["mid.0"].mean(), up000=seed["up.0.0"].mean(),
                                      difference=effect.mean(), seed_sem=effect.sem(),
                                      seed_count=len(seed), pair_count=len(pair_data)))
    curves = pd.DataFrame(table)
    curves.to_csv(OUT / "dose_matched_curves.csv", index=False)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, axis in enumerate(("l2", "relative_l2")):
        for row, metric in enumerate(("detected", "face_localization")):
            plot = axes[row, col]
            for direction, part in curves[(curves.axis == axis) & (curves.metric == metric)].groupby("direction"):
                plot.errorbar(part.dose_center, part.difference, yerr=part.seed_sem, fmt="o-", label=direction)
            plot.axhline(0, color="grey")
            plot.set(xscale="log", xlabel=axis, ylabel=f"up.0.0 − mid.0\n{metric.replace('_', ' ')}")
            plot.legend()
    figure.savefig(OUT / "dose_matched_curves.png", dpi=180)
    failures = active[~active.detected].groupby(keys + ["direction"], as_index=False).agg(
        first_failed_l2=("l2", "min"), first_failed_relative_l2=("relative_l2", "min"))
    trials = active.groupby(keys + ["direction"], as_index=False).agg(max_tested_l2=("l2", "max"))
    trials.merge(failures, how="left").to_csv(OUT / "sampled_detection_failures.csv", index=False)

    hist = ROOT / "experiments/mid0_sae_selectivity/face_distortion_consensus.csv"
    review = pd.read_csv(hist).rename(columns={"prompt_group": "group"})
    historical = frame[frame.layer == "mid.0"].merge(review, on=["feature", "group", "seed", "alpha"])
    if not historical.empty:
        historical.groupby(["detected", "majority_rating"]).size().rename("cases").to_csv(OUT / "human_review_counts.csv")

    spatial_path = RUN / "discovery/spatial_by_image.csv"
    if spatial_path.is_file():
        spatial = pd.read_csv(spatial_path)
        spatial["face_concentration"] = spatial.face_activation_fraction - spatial.face_area_fraction
        feature_stats = spatial.groupby(["layer", "feature"])[
            ["active_fraction", "mean", "face_activation_fraction", "face_area_fraction", "face_concentration"]
        ].mean()
        feature_stats.to_csv(OUT / "discovery_spatial_by_feature.csv")
        spatial.groupby("layer").residual_rms.agg(["mean", "median", "std"]).to_csv(OUT / "residual_scale.csv")

    for layer, part in frame.groupby("layer"):
        edited = part[part.alpha != 1]
        changed = edited[edited.l2 > 0]
        print(layer, "conditions", len(part), "active edits", len(changed), "control detection",
              part[part.alpha == 1].detected.mean(), "active-edit detection", changed.detected.mean(),
              "median active L2", changed.l2.median(), "median active relative L2", changed.relative_l2.median())


if __name__ == "__main__":
    main()
