"""Plot training and held-out reconstruction metrics for a face SAE run."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RUN = Path("sae-training/runs/face_sae_001")
AREAS = ("face_mse", "outside_mse", "all_mse")
AREA_LABELS = ("Inside face box", "Outside box", "Whole image")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN,
                        help="Training run directory containing the metric JSON files")
    parser.add_argument("--output", type=Path,
                        help="Directory for plots and CSV (default: the run directory)")
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text())


def write_history_csv(history, path):
    if not history:
        raise ValueError("Training history is empty")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_training_history(history, test_results, output_path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].plot(
        epochs,
        [row["train_loss_mean"] for row in history],
        marker="o",
        color="#356c9a",
    )
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Mean loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        epochs,
        [row["face_mse"] for row in history],
        marker="o",
        label="Inside face box",
        color="#b24848",
    )
    axes[1].plot(
        epochs,
        [row["outside_mse"] for row in history],
        marker="o",
        label="Outside box",
        color="#53866b",
    )
    axes[1].set(title="Validation reconstruction", xlabel="Epoch", ylabel="MSE")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    x = np.arange(len(AREAS))
    width = 0.36
    original = test_results["original_sae"]
    tuned = test_results["face_fine_tuned_sae"]
    axes[2].bar(
        x - width / 2,
        [original[area] for area in AREAS],
        width,
        label="Original SAE",
        color="#8798a8",
    )
    axes[2].bar(
        x + width / 2,
        [tuned[area] for area in AREAS],
        width,
        label="Face fine-tuned",
        color="#d07b61",
    )
    axes[2].set(
        title="Held-out test reconstruction",
        ylabel="MSE",
        xticks=x,
        xticklabels=AREA_LABELS,
    )
    axes[2].tick_params(axis="x", labelrotation=18)
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Face-region SAE fine-tuning · exploratory automatic-box pilot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(run_dir, output_dir, test_results):
    settings = read_json(run_dir / "settings.json")
    original = test_results["original_sae"]
    tuned = test_results["face_fine_tuned_sae"]
    face_gain = 100 * (original["face_mse"] - tuned["face_mse"]) / original["face_mse"]
    outside_gain = (
        100 * (original["outside_mse"] - tuned["outside_mse"])
        / original["outside_mse"]
    )
    image_count = read_json(run_dir / "heldout_reconstruction.json")["test"]["images"]

    lines = [
        "# Face-region SAE fine-tuning",
        "",
        "The existing Top-K SAE was initialized from its pretrained checkpoint and fine-tuned on activation grid cells whose centers fall inside Florence-2 object-detection boxes labeled `human face`. The generator and original checkpoint were not updated. Phrase grounding was not used.",
        "",
        "## Results",
        "",
        f"The best checkpoint was epoch {settings['best_validation_epoch']}, selected by validation face-box reconstruction MSE.",
        "",
        f"On the {image_count}-image test split, face-box MSE changed from {original['face_mse']:.3f} to {tuned['face_mse']:.3f} ({face_gain:.1f}% lower). Outside-box MSE changed from {original['outside_mse']:.3f} to {tuned['outside_mse']:.3f} ({outside_gain:.1f}% lower). Whole-grid MSE changed from {original['all_mse']:.3f} to {tuned['all_mse']:.3f}.",
        "",
        "The SAE reconstructs the held-out activations better inside the boxes, and also improves reconstruction outside them. This is not evidence by itself that individual features became face-specific or that changing them will edit faces.",
        "",
        "## Heatmaps and feature ranking",
        "",
        "The top features are ranked by mean activation inside minus outside the automatic boxes on validation images. The heatmap contact sheet shows those features on one held-out test portrait from each prompt group. Cyan outlines mark the same automatic box; red overlays show feature activation, normalized per map.",
        "",
        "![Training curves and held-out metrics](training_and_test_metrics.png)",
        "",
        f"![Held-out feature maps]({(run_dir / 'heatmaps/heldout_face_feature_maps.jpg').resolve()})",
        "",
        "See `face_localized_features.csv`, `training_history.json`, `heldout_reconstruction.json`, and `settings.json` for the measurements and provenance.",
        "",
        "## Limitations",
        "",
        "All boxes are automatic, unreviewed labels. Their median area is about 30% of the image, so the mask often includes more than the face. The detector boxes focus the training samples spatially, but the SAE received no labels for eyes, ears, hair, or skin tone individually. The heatmaps and feature ranking are exploratory and should be visually reviewed before feature interventions.",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    run_dir = args.run.resolve()
    output_dir = (args.output or run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = read_json(run_dir / "training_history.json")
    results = read_json(run_dir / "heldout_reconstruction.json")
    write_history_csv(history, output_dir / "training_history.csv")
    plot_training_history(
        history,
        results["test"],
        output_dir / "training_and_test_metrics.png",
    )
    write_report(run_dir, output_dir, results["test"])
    print(f"Wrote plots, training_history.csv, and REPORT.md to {output_dir}")


if __name__ == "__main__":
    main()
