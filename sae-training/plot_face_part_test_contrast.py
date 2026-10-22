"""Plot held-out spatial activation contrast for validation-selected face-part features.

The feature IDs are fixed by the validation selection file. This script only
measures them on the independent test split. Contrast is the mean SAE
activation inside a part box minus its reference area, macro-averaged across
race/gender categories. Error bars are stratified image-bootstrap 95% CIs.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from SAE import SparseAutoencoder
from fine_tune_face_sae import face_boxes, read_csv
from render_all_face_parts_over_time import PARTS, category, load_boxes, masks_for


DEFAULT_RUN = ROOT / "sae-training/runs/face_parts_sae_001"
DEFAULT_DATA = DEFAULT_RUN / "independent_test_seed200_299"
DEFAULT_SELECTION = DEFAULT_RUN / "individual_part_epoch_grids/validation_selected_features.csv"
DEFAULT_OUTPUT = DEFAULT_DATA / "test_contrast"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="SAE run directory")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Independent test data directory")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION,
                        help="Validation-selected part/feature mapping")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    return args


def read_selection(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {row["part"]: int(row["feature"]) for row in rows}
    if set(selected) != set(PARTS):
        raise ValueError(f"Expected validation-selected features for {PARTS}; found {sorted(selected)}")
    return selected


def collect_test_contrasts(model, rows, boxes, face_map, residual_dir, features, device):
    by_part_category = {part: defaultdict(list) for part in PARTS}
    examples = {part: 0 for part in PARTS}
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            sample_id = row["sample_id"]
            residual_path = residual_dir / f"{sample_id}.npy"
            if sample_id not in face_map or not residual_path.is_file():
                continue
            residual_np = np.load(residual_path, allow_pickle=False)
            residual = torch.from_numpy(residual_np.astype(np.float32, copy=False)).to(device)
            codes = model.encode(residual)
            masks = masks_for(sample_id, residual.shape, boxes, face_map, device)
            group = category(row)
            for part in PARTS:
                region, reference = masks[part]
                if region is None:
                    continue
                feature = features[part]
                feature_map = codes[..., feature]
                contrast = (feature_map[region].mean() - feature_map[reference].mean()).item()
                by_part_category[part][group].append(contrast)
                examples[part] += 1
            del residual, codes
            if index % 100 == 0:
                print(f"Processed {index}/{len(rows)} test images", flush=True)
    return by_part_category, examples


def summarize(by_part_category, features, examples, bootstrap_samples, seed):
    summaries = []
    category_rows = []
    for part_index, part in enumerate(PARTS):
        groups = by_part_category[part]
        group_means = {name: float(np.mean(values)) for name, values in groups.items()}
        if not group_means:
            raise ValueError(f"No usable test annotations for {part}")
        values_by_group = {name: np.asarray(values, dtype=np.float64) for name, values in groups.items()}
        estimate = float(np.mean(list(group_means.values())))
        rng = np.random.default_rng(seed + part_index)
        draws = np.empty(bootstrap_samples, dtype=np.float64)
        names = sorted(values_by_group)
        for draw_index in range(bootstrap_samples):
            draws[draw_index] = np.mean([
                rng.choice(values_by_group[name], size=len(values_by_group[name]), replace=True).mean()
                for name in names
            ])
        low, high = np.quantile(draws, [0.025, 0.975])
        summaries.append({
            "part": part,
            "feature": features[part],
            "test_macro_contrast": estimate,
            "test_bootstrap_ci_low": float(low),
            "test_bootstrap_ci_high": float(high),
            "test_images": examples[part],
            "test_categories": len(group_means),
            "reference": "outside_face" if part == "face" else ("outside_image" if part == "ears" else "rest_of_face"),
        })
        for name in sorted(group_means):
            category_rows.append({
                "part": part,
                "feature": features[part],
                "category": name,
                "test_images": len(values_by_group[name]),
                "mean_test_contrast": group_means[name],
            })
    return summaries, category_rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path, summaries):
    labels = [f"{row['part'].title()}  ·  f{row['feature']}  ·  n={row['test_images']}"
              for row in summaries]
    y = np.arange(len(summaries))
    means = np.asarray([row["test_macro_contrast"] for row in summaries])
    lows = np.asarray([row["test_bootstrap_ci_low"] for row in summaries])
    highs = np.asarray([row["test_bootstrap_ci_high"] for row in summaries])
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    fig, ax = plt.subplots(figsize=(10.2, 5.7), constrained_layout=True)
    for index, (mean, low, high) in enumerate(zip(means, lows, highs)):
        ax.errorbar(mean, y[index], xerr=[[mean - low], [high - mean]], fmt="o",
                    color=colors[index], ecolor=colors[index], capsize=5,
                    markersize=8, linewidth=2)
    ax.axvline(0, color="#333333", linewidth=1.3, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test contrast (SAE activation units): mean in part box − reference area")
    ax.set_title("Independent test-set spatial contrast\nValidation-selected feature per part · 95% stratified bootstrap CI")
    ax.grid(axis="x", color="#cccccc", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run, data, output = args.run.resolve(), args.data.resolve(), args.output.resolve()
    manifest_path = data / "manifest.csv"
    boxes_path = data / "annotations/bounding_boxes.csv"
    residual_dir = data / "residuals"
    checkpoint = run / "checkpoints/epoch_020"
    if not checkpoint.is_dir():
        checkpoint = run / "sae"
    rows = [row for row in read_csv(manifest_path) if row["split"] == "test"]
    if not rows:
        raise ValueError(f"No test rows found in {manifest_path}")
    features = read_selection(args.selection)
    boxes = load_boxes(boxes_path)
    face_map = face_boxes(boxes_path)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(f"Loading epoch-20 SAE from {checkpoint} on {device}", flush=True)
    model = SparseAutoencoder.load_from_disk(str(checkpoint)).float().to(device).eval()
    by_part_category, examples = collect_test_contrasts(
        model, rows, boxes, face_map, residual_dir, features, device
    )
    summaries, category_rows = summarize(
        by_part_category, features, examples, args.bootstrap_samples, args.seed
    )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "test_contrast_summary.csv", summaries)
    write_csv(output / "test_contrast_by_category.csv", category_rows)
    make_plot(output / "test_contrast.png", summaries)
    metadata = {
        "split": "independent test, seeds 200-299",
        "checkpoint": str(checkpoint),
        "feature_selection": str(args.selection),
        "feature_ids": features,
        "contrast": "per-image mean SAE activation in annotated part cells minus reference cells; average images within race/gender category, then average category means equally",
        "references": {part: ("outside_face" if part == "face" else ("outside_image" if part == "ears" else "rest_of_face")) for part in PARTS},
        "bootstrap": {"method": "stratified image bootstrap within available race/gender categories; macro-average category contrast", "samples": args.bootstrap_samples, "seed": args.seed, "interval": "percentile 95%"},
        "box_filter": "same masks as validation feature selection: part boxes above 12% image area excluded; at least two part and reference cells required; facial boxes unreviewed automatic annotations",
        "test_images_with_usable_boxes": examples,
    }
    (output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved test contrast plot and metrics to {output}", flush=True)
    for row in summaries:
        print(f"{row['part']} f{row['feature']}: {row['test_macro_contrast']:.3f} "
              f"[{row['test_bootstrap_ci_low']:.3f}, {row['test_bootstrap_ci_high']:.3f}], "
              f"n={row['test_images']} images / {row['test_categories']} categories", flush=True)


if __name__ == "__main__":
    main()
