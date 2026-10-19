"""Render feature activation overlays from a completed face SAE run."""
import argparse
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from SAE import SparseAutoencoder
from fine_tune_face_sae import DEFAULT_BOXES, DEFAULT_MANIFEST, DEFAULT_RESIDUALS
from fine_tune_face_sae import face_boxes, read_csv, render_test_heatmaps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed SAE run with sae/ and feature ranking CSV")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--output", type=Path, help="Output directory (default: RUN/feature_maps_jet)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run.resolve()
    output_dir = (args.output or run_dir / "feature_maps_jet").resolve()
    if output_dir.exists():
        raise ValueError(f"Output exists; choose a new --output: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = SparseAutoencoder.load_from_disk(str(run_dir / "sae"))
    sae = sae.float().to(device).eval()
    boxes = face_boxes(args.boxes)
    test_rows = [
        row for row in read_csv(args.manifest)
        if row["split"] == "test"
        and row["sample_id"] in boxes
        and (args.residuals / f"{row['sample_id']}.npy").is_file()
    ]
    with (run_dir / "face_localized_features.csv").open(newline="") as ranking_file:
        ranked = list(csv.DictReader(ranking_file))

    output_dir.mkdir(parents=True)
    render_test_heatmaps(
        sae,
        ranked,
        test_rows,
        boxes,
        args.residuals,
        output_dir,
        device,
    )
    print(f"Saved jet pixel overlays to {output_dir / 'heatmaps'}")


if __name__ == "__main__":
    main()
