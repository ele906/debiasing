#!/usr/bin/env python3
"""Prepare a small Feature 4228 image sample for manual beard annotation."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path("/n/fs/goose")
SOURCE = ROOT / "layers/runs/mid0_sae_face_001/interventions/global_seed_average_001/feature_4228/images"
DEFAULT_OUT = Path("/tmp/goose_beard_annotation_pilot")
GROUPS = (
    "black_man", "black_woman", "east_asian_man", "east_asian_woman",
    "hispanic_man", "hispanic_woman", "indian_man", "indian_woman",
    "middle_eastern_man", "middle_eastern_woman", "white_man", "white_woman",
)
DEFAULT_ALPHAS = (-7, 1, 7)


def alpha_code(alpha: int) -> str:
    return f"m{abs(alpha)}" if alpha < 0 else f"p{alpha}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[30])
    parser.add_argument("--alphas", nargs="+", type=int, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_dir = args.out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for group in args.groups:
        for seed in args.seeds:
            for alpha in args.alphas:
                source = SOURCE / group / f"seed{seed:03d}_alpha_{alpha_code(alpha)}.png"
                if not source.is_file():
                    raise FileNotFoundError(source)
                filename = f"{group}__seed{seed:03d}__alpha_{alpha_code(alpha)}.png"
                dest = image_dir / filename
                if dest.exists() and not args.overwrite:
                    raise FileExistsError(f"{dest} already exists; use --overwrite to replace it")
                shutil.copy2(source, dest)
                manifest_rows.append({
                    "image": f"images/{filename}",
                    "annotation": f"images/{Path(filename).stem}.json",
                    "mask": f"masks/{Path(filename).stem}_beard.png",
                    "category": group,
                    "seed": seed,
                    "alpha": alpha,
                    "label_status": "needs_annotation",
                    "source_relative_to_repo": str(source.relative_to(ROOT)),
                })

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    (args.out / "labels.txt").write_text("beard\n")
    (args.out / "ANNOTATION_GUIDE.md").write_text(
        "# Feature 4228 beard-labeling pilot\n\n"
        f"Prepared {len(manifest_rows)} images ({len(args.groups)} categories, "
        f"{len(args.seeds)} seed(s), alpha values {', '.join(f'{a:+d}' for a in args.alphas)}).\n\n"
        "## Labeling rule\n\n"
        "Draw a polygon around visible beard pixels on the cheeks, jaw, and chin. "
        "Exclude skin, scalp hair, clothing, and the separate mustache. If no beard "
        "is visible, save a reviewed annotation with no beard polygon. Use the same "
        "boundary rule at every alpha. The Face Landmarker can help locate the face, "
        "but the polygon edge must follow the visible beard.\n\n"
        "## LabelMe\n\n"
        "Install LabelMe in the environment where you can use a graphical desktop, "
        "then run from the repository root:\n\n"
        "```bash\n"
        "python -m pip install labelme\n"
        "```\n\n"
        "Open the copied pilot images with:\n\n"
        "```bash\n"
        f"labelme --nodata --labels {args.out / 'labels.txt'} {image_dir}\n"
        "```\n\n"
        "Save polygon shapes with the exact label `beard`; JSON files should be saved "
        "beside their matching images. After labeling, export binary masks with:\n\n"
        "```bash\n"
        f"python instance-segmentation/export_labelme_beard_masks.py --pilot {args.out}\n"
        "```\n\n"
        "These hand-labeled masks will be ground truth for a small transfer-learning "
        "pilot. The failed generic-hair and first SAM masks are not used as labels.\n"
    )
    print(f"Prepared {len(manifest_rows)} images at {args.out}")
    print(f"Images for annotation: {image_dir}")


if __name__ == "__main__":
    main()
