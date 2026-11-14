#!/usr/bin/env python3
"""Export LabelMe polygons named ``beard`` as binary masks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def rasterize(annotation: Path, output: Path) -> int:
    record = json.loads(annotation.read_text())
    width, height = int(record["imageWidth"]), int(record["imageHeight"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    n_polygons = 0
    for shape in record.get("shapes", []):
        if shape.get("label") != "beard":
            continue
        if shape.get("shape_type") != "polygon":
            raise ValueError(f"Expected polygon annotation for {annotation}")
        points = [(float(x), float(y)) for x, y in shape["points"]]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
            n_polygons += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)
    return n_polygons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("/tmp/goose_beard_annotation_pilot"))
    args = parser.parse_args()
    manifest_path = args.pilot / "manifest.csv"
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {manifest_path}")

    n_reviewed = 0
    for row in rows:
        annotation = args.pilot / row["annotation"]
        if not annotation.is_file():
            continue
        polygons = rasterize(annotation, args.pilot / row["mask"])
        row["label_status"] = "reviewed_positive" if polygons else "reviewed_negative"
        n_reviewed += 1

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported masks from {n_reviewed} reviewed annotations to {args.pilot / 'masks'}")


if __name__ == "__main__":
    main()
