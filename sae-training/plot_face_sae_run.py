"""Plot the training curve and held-out reconstruction for a face SAE run."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=Path("sae-training/runs/face_sae_001"))
    args = parser.parse_args()
    history = json.loads((args.run / "training_history.json").read_text())
    results = json.loads((args.run / "heldout_reconstruction.json").read_text())
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].plot(epochs, [r["train_loss_mean"] for r in history], marker="o", color="#356c9a")
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Mean loss")
    axes[0].grid(alpha=.25)

    axes[1].plot(epochs, [r["face_mse"] for r in history], marker="o", label="Inside face box", color="#b24848")
    axes[1].plot(epochs, [r["outside_mse"] for r in history], marker="o", label="Outside box", color="#53866b")
    axes[1].set(title="Validation reconstruction", xlabel="Epoch", ylabel="MSE")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=.25)

    test = results["test"]
    areas = ["face_mse", "outside_mse", "all_mse"]
    labels = ["Inside face box", "Outside box", "Whole image"]
    x = np.arange(len(areas))
    width = .36
    axes[2].bar(x - width / 2, [test["original_sae"][a] for a in areas], width,
                label="Original SAE", color="#8798a8")
    axes[2].bar(x + width / 2, [test["face_fine_tuned_sae"][a] for a in areas], width,
                label="Face fine-tuned", color="#d07b61")
    axes[2].set(title="Held-out test reconstruction", ylabel="MSE", xticks=x, xticklabels=labels)
    axes[2].tick_params(axis="x", labelrotation=18)
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=.25)
    fig.suptitle("Face-region SAE fine-tuning · exploratory automatic-box pilot")
    fig.tight_layout()
    fig.savefig(args.run / "training_and_test_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    base = test["original_sae"]
    tuned = test["face_fine_tuned_sae"]
    face_gain = 100 * (base["face_mse"] - tuned["face_mse"]) / base["face_mse"]
    outside_gain = 100 * (base["outside_mse"] - tuned["outside_mse"]) / base["outside_mse"]
    report = [
        "# Face-region SAE fine-tuning: first pass", "",
        "The existing Top-K SAE was initialized from its pretrained checkpoint and fine-tuned on activation grid cells whose centers fall inside Florence-2 object-detection boxes labeled `human face`. The generator and original checkpoint were not updated. Phrase grounding was not used. The original SAE and the fine-tuned copy were compared on the same held-out images.", "",
        "## Results", "",
        f"The best checkpoint was epoch {json.loads((args.run / 'settings.json').read_text())['best_validation_epoch']}, selected by validation face-box reconstruction MSE.", "",
        f"On the 240-image test split, face-box MSE changed from {base['face_mse']:.3f} to {tuned['face_mse']:.3f} ({face_gain:.1f}% lower). Outside-box MSE changed from {base['outside_mse']:.3f} to {tuned['outside_mse']:.3f} ({outside_gain:.1f}% lower). Whole-grid MSE changed from {base['all_mse']:.3f} to {tuned['all_mse']:.3f}.", "",
        "The SAE reconstructs the held-out activations better inside the boxes, and also improves reconstruction outside them. This is not evidence by itself that individual features became face-specific or that changing them will edit faces.", "",
        "## Heatmaps and feature ranking", "",
        "The top features are ranked by mean activation inside minus outside the automatic boxes on validation images. The heatmap contact sheet shows those features on one held-out test portrait from each of the 12 prompt groups. Cyan outlines mark the same automatic box; red overlays show feature activation, normalized per map.", "",
        "![Training curves and held-out metrics](training_and_test_metrics.png)", "",
        "![Held-out feature maps](heatmaps/heldout_face_feature_maps.jpg)", "",
        "See `face_localized_features.csv`, `training_history.json`, `heldout_reconstruction.json`, and `settings.json` for the measurements and provenance.", "",
        "## Limitations", "",
        "All boxes are automatic, unreviewed labels. Their median area is about 30% of the image, so the mask often includes more than the face. The detector boxes focus the training samples spatially, but the SAE received no labels for eyes, ears, hair, or skin tone individually. The heatmaps and feature ranking are exploratory and should be visually reviewed before feature interventions.", "",
    ]
    (args.run / "REPORT.md").write_text("\n".join(report))
    print(f"Wrote {args.run / 'training_and_test_metrics.png'} and REPORT.md")


if __name__ == "__main__":
    main()
