"""Create a manifest for the balanced unbiased image collection."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = ROOT / "data/datasets/unbiased/images"
OUTPUT = Path(__file__).resolve().parent / "input.csv"


def main():
    rows = []
    for category_dir in sorted(path for path in IMAGE_ROOT.iterdir() if path.is_dir()):
        parts = category_dir.name.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in {"man", "woman"}:
            raise ValueError(f"Unexpected category folder: {category_dir.name}")
        race, gender = parts
        for image_path in sorted(category_dir.glob("*")):
            if not image_path.is_file():
                continue
            sample_id = f"{category_dir.name}_{image_path.stem}"
            rows.append({
                "sample_id": sample_id,
                "split": "unlabeled",
                "race_prompt": race.replace("_", " "),
                "gender_prompt": gender,
                "image": str(image_path.resolve()),
                "features": "",
            })
    if len(rows) != 600 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected 600 unique images; found {len(rows)}")
    counts = {}
    for row in rows:
        category = f"{row['race_prompt']}_{row['gender_prompt']}"
        counts[category] = counts.get(category, 0) + 1
    if len(counts) != 12 or set(counts.values()) != {50}:
        raise ValueError(f"Expected 50 images in each of 12 groups; got {counts}")
    with OUTPUT.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} images across {len(counts)} balanced groups to {OUTPUT}")


if __name__ == "__main__":
    main()
