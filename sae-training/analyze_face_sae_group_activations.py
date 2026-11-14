"""Rank face-SAE features associated with each prompt group using paired seeds."""
import argparse
import csv
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "sae-training/runs/face_sae_020/group_analysis_001")
    args = parser.parse_args()
    run = args.run.resolve()
    data = np.load(run / "face_sae_activations.npz", allow_pickle=False)
    ids = data["sample_id"].astype(str)
    races = data["race"].astype(str)
    genders = data["gender"].astype(str)
    seeds = data["seed"].astype(int)
    usable = data["face_box_usable"].astype(bool)
    x = data["X_face_mean"].astype(np.float64)
    categories = sorted(set(zip(races, genders)))
    seed_values = sorted(set(seeds.tolist()))
    by_key = {(race, gender, int(seed)): i for i, (race, gender, seed) in enumerate(zip(races, genders, seeds))}
    feature_count = x.shape[1]
    all_rows, top_rows = [], []

    for race, gender in categories:
        other_categories = [(r, g) for r, g in categories if g == gender and r != race]
        for split, split_seeds in (("discovery", [s for s in seed_values if s < 250]),
                                   ("heldout", [s for s in seed_values if s >= 250])):
            paired_effects = []
            for seed in split_seeds:
                own = by_key.get((race, gender, seed))
                others = [by_key[(r, g, seed)] for r, g in other_categories
                          if (r, g, seed) in by_key and usable[by_key[(r, g, seed)]]]
                if own is None or not usable[own] or not others:
                    continue
                paired_effects.append(x[own] - x[others].mean(axis=0))
            effects = np.stack(paired_effects)
            mean_delta = effects.mean(axis=0)
            se = effects.std(axis=0, ddof=1) / np.sqrt(len(effects))
            low, high = mean_delta - 1.96 * se, mean_delta + 1.96 * se
            sd = effects.std(axis=0, ddof=1)
            d = np.divide(mean_delta, sd, out=np.zeros_like(mean_delta), where=sd > 1e-12)
            order = np.argsort(mean_delta)[::-1]
            for rank, feature in enumerate(order[:20], 1):
                top_rows.append({
                    "race_prompt": race, "gender_prompt": gender, "split": split, "rank": rank,
                    "feature": int(feature), "mean_face_activation_delta": float(mean_delta[feature]),
                    "ci95_low": float(low[feature]), "ci95_high": float(high[feature]),
                    "paired_seed_count": len(effects), "cohens_d": float(d[feature]),
                })
            for feature in range(feature_count):
                all_rows.append({
                    "race_prompt": race, "gender_prompt": gender, "split": split,
                    "feature": feature, "mean_face_activation_delta": float(mean_delta[feature]),
                    "ci95_low": float(low[feature]), "ci95_high": float(high[feature]),
                    "paired_seed_count": len(effects), "cohens_d": float(d[feature]),
                })

    write_csv(run / "group_effects_top20.csv", top_rows)
    write_csv(run / "group_effects_all_features.csv", all_rows)

    report = [
        "# Face-SAE group activation scan",
        "",
        "This exploratory scan encodes the same 1,200 held-out SDXL Turbo residuals with the",
        "saved face-region SAE (selected epoch 20). Seeds 200–249 select candidate features;",
        "seeds 250–299 provide a held-out check. For each prompt group, it compares the",
        "mean code inside the automatic face box with the seed-matched average for the other",
        "five groups of the same gender. Positive delta means higher face-box activation for",
        "that prompt group. The groups are prompt labels, not independently verified identities.",
        "",
        f"Samples with usable face boxes: {int(usable.sum())}/{len(ids)}; feature width: {feature_count}.",
        "The reported 95% intervals are unadjusted across 5,120 scanned features; top-ranked",
        "features are candidates for visual review, not confirmed concepts.",
        "",
        "## Highest Indian-woman-associated features",
        "",
        "| Rank | Feature | Discovery delta [95% CI] | Held-out delta [95% CI] | Discovery d |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    indian_discovery = [r for r in top_rows if r["race_prompt"] == "indian" and r["gender_prompt"] == "woman" and r["split"] == "discovery"]
    indian_heldout = {r["feature"]: r for r in all_rows
                      if r["race_prompt"] == "indian" and r["gender_prompt"] == "woman" and r["split"] == "heldout"}
    for row in indian_discovery[:10]:
        test = indian_heldout[row["feature"]]
        report.append(f"| {row['rank']} | {row['feature']} | {row['mean_face_activation_delta']:.4f} "
                      f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                      f"{test['mean_face_activation_delta']:.4f} [{test['ci95_low']:.4f}, {test['ci95_high']:.4f}] | "
                      f"{row['cohens_d']:.3f} |")
    report.extend([
        "",
        "See `group_effects_top20.csv` for the top candidates per prompt group and",
        "`group_effects_all_features.csv` for all tested features. Spatial overlays are still",
        "needed to tell whether a candidate follows a forehead mark, earrings, skin texture,",
        "lighting, pose, or another group-correlated detail.",
    ])
    (run / "GROUP_ACTIVATION_REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Wrote group activation rankings to {run}")


if __name__ == "__main__":
    main()
