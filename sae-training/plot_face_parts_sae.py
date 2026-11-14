"""Plot per-epoch training and validation metrics for a face-parts SAE run."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REGIONS = ("eyes", "eyebrows", "nose", "ears")
COLORS = {
    "eyes": "#386cb0",
    "eyebrows": "#f0027f",
    "nose": "#7fc97f",
    "ears": "#fdc086",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run folder containing training_history.json")
    parser.add_argument("--output", type=Path, help="Output PNG path; defaults to the run folder")
    return parser.parse_args()


def load_history(path):
    history = json.loads(path.read_text())
    if not history:
        raise ValueError(f"Training history is empty: {path}")
    return history


def write_csv(history, path):
    columns = ["epoch", "train_loss_mean", "macro_region_mse"]
    columns += [f"{region}_{metric}" for region in REGIONS
                for metric in ("pooled_mse", "macro_category_mse")]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            flat = {
                "epoch": row["epoch"],
                "train_loss_mean": row["train_loss_mean"],
                "macro_region_mse": row["validation"]["macro_region_mse"],
            }
            for region in REGIONS:
                metrics = row["validation"].get(region, {})
                for name in ("pooled_mse", "macro_category_mse"):
                    flat[f"{region}_{name}"] = metrics.get(name)
            writer.writerow(flat)


def plot_history(history, path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, [row["train_loss_mean"] for row in history],
                 marker="o", color="#4c78a8")
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Mean loss")
    axes[0].grid(alpha=0.25)

    for region in REGIONS:
        values = [row["validation"][region]["pooled_mse"] for row in history]
        axes[1].plot(epochs, values, marker="o", markersize=3,
                     label=region.title(), color=COLORS[region])
    axes[1].set(title="Validation reconstruction by region",
                xlabel="Epoch", ylabel="Pooled MSE")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    values = [row["validation"]["macro_region_mse"] for row in history]
    axes[2].plot(epochs, values, marker="o", color="#e45756")
    axes[2].set(title="Validation selection metric", xlabel="Epoch",
                ylabel="Mean of region MSEs")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Face-parts SAE fine-tuning across epochs")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run = args.run.resolve()
    output = (args.output or run / "training_curves.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    history = load_history(run / "training_history.json")
    write_csv(history, output.with_name("training_history.csv"))
    plot_history(history, output)
    print(f"Wrote {output} and {output.with_name('training_history.csv')}")


if __name__ == "__main__":
    main()
