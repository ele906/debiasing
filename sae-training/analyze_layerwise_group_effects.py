"""Held-out, seed-matched group comparison for the layerwise SAE capture."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAYERS = [
    "unet_down_blocks_2_attentions_1",
    "unet_mid_block_attentions_0",
    "unet_up_blocks_0_attentions_0",
    "unet_up_blocks_0_attentions_1",
]
RACES = ["black", "east asian", "hispanic", "indian", "middle eastern", "white"]
GENDERS = ["man", "woman"]


def ci95(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    # Resample seed-level paired contrasts, preserving the comparison unit.
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (float("nan"), float("nan"))
    draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    matrix = np.zeros((len(LAYERS), 12), dtype=float)
    feat_labels = np.empty((len(LAYERS), 12), dtype=object)
    rng = np.random.default_rng(73421)

    for li, layer in enumerate(LAYERS):
        path = args.input / f"layer_{layer}.npz"
        data = np.load(path, allow_pickle=False)
        x = data["X_face_mean"]
        race = np.char.lower(data["race_prompt"].astype(str))
        gender = np.char.lower(data["gender_prompt"].astype(str))
        seeds = data["seed"].astype(int)
        discovery = np.isin(seeds, np.arange(100, 150))
        heldout = np.isin(seeds, np.arange(150, 200))
        if not discovery.any() or not heldout.any():
            raise ValueError(f"Missing discovery or heldout samples in {path}")
        if x.shape[0] != len(seeds):
            raise ValueError(f"Sample count mismatch in {path}")

        for gi, (target_race, target_gender) in enumerate(
                (r, g) for g in GENDERS for r in RACES):
            cell = (race == target_race) & (gender == target_gender)
            other = (gender == target_gender) & (race != target_race)
            # Use only seed IDs represented by target and all five reference races.
            d_ids = sorted(set(seeds[discovery & cell]) & set(seeds[discovery & other]))
            h_ids = sorted(set(seeds[heldout & cell]) & set(seeds[heldout & other]))
            if len(d_ids) < 10 or len(h_ids) < 10:
                raise ValueError(f"Too few paired seeds for {target_race} {target_gender}: "
                                 f"{len(d_ids)} discovery, {len(h_ids)} heldout")

            def contrasts(seed_ids):
                out = []
                for seed in seed_ids:
                    target_rows = np.flatnonzero(cell & (seeds == seed))
                    other_rows = np.flatnonzero(other & (seeds == seed))
                    if len(target_rows) != 1 or len(other_rows) != 5:
                        continue
                    out.append(x[target_rows[0]] - np.nanmean(x[other_rows], axis=0))
                return np.asarray(out)

            d = contrasts(d_ids)
            h = contrasts(h_ids)
            if d.shape[0] < 10 or h.shape[0] < 10:
                raise ValueError(f"Incomplete seed matches for {target_race} {target_gender}")
            scores = np.nanmean(d, axis=0)
            feature = int(np.nanargmax(scores))
            values = h[:, feature]
            values = values[np.isfinite(values)]
            mean = float(np.mean(values))
            scale_values = x[heldout & (gender == target_gender), feature]
            scale_values = scale_values[np.isfinite(scale_values)]
            scale = float(np.std(scale_values, ddof=1))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid heldout activation scale for {layer}, "
                                 f"{target_race} {target_gender}, feature {feature}")
            std_effect = mean / scale
            lo, hi = ci95(values, rng)
            matrix[li, gi] = std_effect
            feat_labels[li, gi] = str(feature)
            rows.append({
                "layer": layer, "target_group": f"{target_race} {target_gender}",
                "feature_id_layer_specific": feature,
                "discovery_mean_contrast": float(scores[feature]),
                "heldout_mean_contrast": mean, "heldout_ci95_low": float(lo),
                "heldout_ci95_high": float(hi), "heldout_positive_seed_fraction":
                    float(np.mean(values > 0)), "discovery_paired_seeds":
                    int(np.isfinite(d[:, feature]).sum()),
                "heldout_paired_seeds": len(values),
                "heldout_feature_activation_sd": scale,
                "heldout_standardized_contrast": std_effect,
                "heldout_standardized_ci95_low": float(lo / scale),
                "heldout_standardized_ci95_high": float(hi / scale),
            })

    out_csv = args.output / "layerwise_group_effects.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(16, 7.5), constrained_layout=True)
    lim = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap="jet", aspect="auto", vmin=-lim, vmax=lim)
    groups = [f"{r}\n{g}" for g in GENDERS for r in RACES]
    ax.set_xticks(np.arange(12), groups, rotation=0)
    ax.set_yticks(np.arange(len(LAYERS)), [
        "Down block 2, attention 1", "Middle block, attention 0",
        "Up block 0, attention 0", "Up block 0, attention 1"])
    ax.set_xlabel("Target prompt group (contrasted with the other five groups of same gender)")
    ax.set_ylabel("Diffusion-model layer SAE")
    ax.set_title("Held-out standardized group contrast of discovery-selected features\n"
                 "Color = target minus same-seed, same-gender reference mean, in feature SDs")
    for i in range(len(LAYERS)):
        for j in range(12):
            ax.text(j, i, f"{matrix[i,j]:+.2f} SD\nF{feat_labels[i,j]}",
                    ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax, label="Held-out contrast / held-out feature activation SD")
    fig.savefig(args.output / "layerwise_group_effects_jet.png", dpi=200)
    plt.close(fig)

    report = [
        "# Layerwise SAE group-comparison pilot",
        "",
        "## Design",
        "",
        "The same 1,200 SDXL Turbo portraits (six race prompt categories × two gender prompt categories × seeds 100–199) were encoded with four pretrained, layer-specific SAEs. For each target group and gender, discovery seeds 100–149 selected the single SAE feature with the largest mean face-box activation contrast against the other five race-prompt groups of the same gender. Held-out seeds 150–199 evaluated that fixed feature. Confidence intervals resample paired seed-level contrasts.",
        "",
        "The heatmap uses a JET palette. X is the prompt group; Y is the model layer/SAE. Cell text gives the standardized held-out contrast (raw target-minus-reference contrast divided by the held-out SD of that layer's selected feature across same-gender samples) and that layer's feature ID. Feature IDs are specific to each SAE and must not be compared across layers as if they had the same meaning. Standardization makes the displayed contrast units more comparable across layers, but the independently trained SAEs may still encode different concepts. Two samples lacked usable face boxes and are excluded from the relevant seed-level contrast when they occur.",
        "",
        "## Interpretation limits",
        "",
        "This is a pilot association screen, not evidence that an SAE feature represents race, skin color, jewelry, or another human attribute. The generated prompt groups may differ in correlated visual details; automatic face boxes and face-region mean pooling can miss small/local attributes. A feature chosen as the strongest discovery contrast may be noisy, so interpret its held-out effect and interval, not the discovery score alone. The 95% intervals are unadjusted for the 48 layer/group comparisons. The SAEs in this pilot are pretrained for each layer; this does not yet test fine-tuning an SAE on this project’s face dataset.",
        "",
        "## Files",
        "",
        "- `layerwise_group_effects.csv`: selected feature and held-out statistics for each layer/group cell.",
        "- `layerwise_group_effects_jet.png`: held-out comparison plot.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {out_csv} and layerwise_group_effects_jet.png")


if __name__ == "__main__":
    main()
