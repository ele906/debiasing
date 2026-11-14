"""Compare validation-selected SAE features across face regions on validation and test data."""

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
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from SAE import SparseAutoencoder
from fine_tune_face_sae import face_boxes, read_csv
from render_all_face_parts_over_time import PARTS, category, load_boxes, masks_for

RUN = ROOT / "sae-training/runs/face_parts_sae_001"
VALIDATION_DATA = ROOT / "backprop"
VALIDATION_BOXES = ROOT / "concept/category_analysis_full/bounding_boxes.csv"
TEST_DATA = RUN / "independent_test_seed200_299"
TEST_BOXES = TEST_DATA / "annotations/bounding_boxes.csv"
DEFAULT_OUTPUT = TEST_DATA / "test_contrast/feature_cross_contrast"
BOOTSTRAP_SAMPLES = 2000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--validation-data", type=Path, default=VALIDATION_DATA)
    parser.add_argument("--validation-boxes", type=Path, default=VALIDATION_BOXES)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--test-boxes", type=Path, default=TEST_BOXES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    return args


def load_selected_features(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    selected_for = defaultdict(list)
    selection_order = []
    for row in rows:
        feature = int(row["feature"])
        selected_for[feature].append(row["part"])
        if feature not in selection_order:
            selection_order.append(feature)
    return selection_order, selected_for


def collect_split(data_root, boxes_path, split_name, model, features, device):
    rows = [row for row in read_csv(data_root / "manifest.csv") if row["split"] == split_name]
    boxes = load_boxes(boxes_path)
    face_map = face_boxes(boxes_path)
    residual_dir = data_root / "residuals" if split_name == "test" else ROOT / "sae-training/residuals"
    category_values = {feature: {part: defaultdict(list) for part in PARTS} for feature in features}
    counts = {part: 0 for part in PARTS}
    usable_ids = set()
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            sample_id = row["sample_id"]
            residual_path = residual_dir / f"{sample_id}.npy"
            if sample_id not in face_map or not residual_path.is_file():
                continue
            residual_np = np.load(residual_path, mmap_mode="r", allow_pickle=False)
            residual = torch.from_numpy(np.array(residual_np, dtype=np.float32, copy=True)).to(device)
            codes = model.encode(residual)
            masks = masks_for(sample_id, residual.shape, boxes, face_map, device)
            group = category(row)
            usable_ids.add(sample_id)
            for part in PARTS:
                region, reference = masks[part]
                if region is None:
                    continue
                counts[part] += 1
                part_codes = codes[region]
                reference_codes = codes[reference]
                for feature in features:
                    value = (part_codes[:, feature].mean() - reference_codes[:, feature].mean()).item()
                    category_values[feature][part][group].append(value)
            del residual, codes
            if index % 100 == 0:
                print(f"{split_name}: processed {index}/{len(rows)} images", flush=True)
    return category_values, counts, len(usable_ids)


def macro_mean(groups):
    means = {name: float(np.mean(values)) for name, values in groups.items() if values}
    return float(np.mean(list(means.values()))) if means else float("nan"), means


def summarize_split(split, category_values, features, counts, bootstrap_samples, seed):
    summary = {}
    detail_rows = []
    for feature_index, feature in enumerate(features):
        summary[feature] = {}
        for part_index, part in enumerate(PARTS):
            groups = category_values[feature][part]
            mean, category_means = macro_mean(groups)
            if not category_means:
                raise ValueError(f"No usable {split} annotations for {part}")
            lows = highs = float("nan")
            if split == "test":
                arrays = {name: np.asarray(values, dtype=np.float64)
                          for name, values in groups.items() if values}
                names = sorted(arrays)
                rng = np.random.default_rng(seed + feature_index * len(PARTS) + part_index)
                draws = np.empty(bootstrap_samples, dtype=np.float64)
                for draw in range(bootstrap_samples):
                    draws[draw] = np.mean([
                        rng.choice(arrays[name], size=len(arrays[name]), replace=True).mean()
                        for name in names
                    ])
                lows, highs = np.quantile(draws, [0.025, 0.975])
            summary[feature][part] = {"mean": mean, "low": float(lows), "high": float(highs),
                                      "n": counts[part], "categories": len(category_means)}
            detail_rows.append({
                "split": split,
                "feature": feature,
                "region": part,
                "contrast_macro_category_mean": mean,
                "bootstrap_ci_low": float(lows),
                "bootstrap_ci_high": float(highs),
                "images_with_usable_box": counts[part],
                "categories_with_usable_box": len(category_means),
                "reference": "outside_face" if part == "face" else ("outside_image" if part == "ears" else "rest_of_face"),
            })
    return summary, detail_rows


def draw_heatmap(ax, summary, features, selected_for, split, norm, cmap):
    matrix = np.asarray([[summary[feature][part]["mean"] for part in PARTS]
                         for feature in features], dtype=np.float64)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(PARTS)), [part.title() for part in PARTS])
    row_names = [f"f{feature} (selected for {' / '.join(selected_for[feature])})"
                 for feature in features]
    ax.set_yticks(np.arange(len(features)), row_names)
    ax.set_title("Validation" if split == "validation" else "Independent test", fontsize=14, pad=12)
    ax.set_xlabel("Region being evaluated")
    ax.tick_params(axis="both", labelsize=10)
    for row, feature in enumerate(features):
        for col, part in enumerate(PARTS):
            entry = summary[feature][part]
            if split == "test":
                label = f"{entry['mean']:.1f}\n[{entry['low']:.1f}, {entry['high']:.1f}]\nn={entry['n']}"
            else:
                label = f"{entry['mean']:.1f}\nn={entry['n']}"
            color = "white" if abs(entry["mean"]) > norm.vmax * 0.56 else "black"
            ax.text(col, row, label, ha="center", va="center", color=color, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    args = parse_args()
    run = args.run.resolve()
    features, selected_for = load_selected_features(
        run / "individual_part_epoch_grids/validation_selected_features.csv"
    )
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = run / "checkpoints/epoch_020"
    print(f"Loading epoch-20 SAE from {checkpoint} on {device}; features={features}", flush=True)
    model = SparseAutoencoder.load_from_disk(str(checkpoint)).float().to(device).eval()
    validation_data, validation_counts, validation_samples = collect_split(
        args.validation_data.resolve(), args.validation_boxes.resolve(), "validation",
        model, features, device
    )
    test_data, test_counts, test_samples = collect_split(
        args.test_data.resolve(), args.test_boxes.resolve(), "test", model, features, device
    )
    validation, validation_rows = summarize_split(
        "validation", validation_data, features, validation_counts,
        args.bootstrap_samples, args.seed
    )
    test, test_rows = summarize_split(
        "test", test_data, features, test_counts, args.bootstrap_samples, args.seed + 1000
    )
    all_values = [entry["mean"] for summary in (validation, test)
                  for feature in features for entry in summary[feature].values()]
    scale = max(abs(float(value)) for value in all_values if np.isfinite(value))
    scale = max(scale, 1e-8)
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    cmap = "coolwarm"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), sharey=True, constrained_layout=True)
    draw_heatmap(axes[0], validation, features, selected_for, "validation", norm, cmap)
    draw_heatmap(axes[1], test, features, selected_for, "test", norm, cmap)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=axes, shrink=0.86, pad=0.025)
    colorbar.set_label("Mean activation in region − reference (SAE units)")
    fig.suptitle("Cross-region spatial contrast for validation-selected SAE features\n"
                 "Cells show macro-category means; test cells also show 95% stratified bootstrap intervals",
                 fontsize=15)
    fig.savefig(output / "feature_cross_contrast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    write_rows = validation_rows + test_rows
    with (output / "feature_cross_contrast.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(write_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(write_rows)
    metadata = {
        "checkpoint": str(checkpoint),
        "features": {str(feature): {"selected_for": selected_for[feature]} for feature in features},
        "validation_split": "backprop manifest rows labeled validation",
        "test_split": "independent_test_seed200_299, seeds 200-299",
        "metric": "per-image mean activation in region minus reference area; images averaged within race/gender categories, then category means equally averaged",
        "references": {part: "outside_face" if part == "face" else "outside_image" if part == "ears" else "rest_of_face" for part in PARTS},
        "shared_color_scale_sae_activation_units": [-scale, scale],
        "validation_usable_images_by_region": validation_counts,
        "test_usable_images_by_region": test_counts,
        "validation_samples_with_any_usable_box": validation_samples,
        "test_samples_with_any_usable_box": test_samples,
        "test_ci": {"method": "stratified image bootstrap within available categories, then equal category macro-average", "samples": args.bootstrap_samples, "interval": "percentile 95%"},
        "annotation_note": "Region boxes are automatic and have not been reviewed by a person.",
    }
    (output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved cross-region validation/test plot to {output / 'feature_cross_contrast.png'}", flush=True)
    print(f"Validation usable counts: {validation_counts}", flush=True)
    print(f"Test usable counts: {test_counts}", flush=True)


if __name__ == "__main__":
    main()
