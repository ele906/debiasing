"""
Scores FairFace's race/gender predictions against the prompt-implied labels
from scripts/generate_race_ground_truth.py's output (folder name encodes
the requested category). Reports per-category accuracy and a confusion
matrix -- this measures FairFace's accuracy in SDXL-turbo's synthetic image
domain, separate from scripts/audit_fairface_independent.py's check for
gaming under the ReNO reward-optimization loop.

Usage:
    python scripts/score_fairface_on_ground_truth.py \\
        --data_dir outputs/race_ground_truth \\
        --weights fairface_weights/res34_fair_align_multi_7_20190809.pt
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN
from PIL import Image

RACE_LABELS = [
    "White", "Black", "Latino_Hispanic", "East Asian",
    "Southeast Asian", "Indian", "Middle Eastern",
]
GENDER_LABELS = ["Male", "Female"]
RACE_SLUGS = {r.lower().replace(" ", "_"): r for r in RACE_LABELS}
GENDER_SLUGS = {g.lower(): g for g in GENDER_LABELS}


def parse_category_dir(name: str):
    """category_dir names look like '{race_slug}_{gender_slug}', e.g.
    'east_asian_female'. Gender slug is always the last token; race slug is
    everything before it."""
    for gender_slug, gender_label in GENDER_SLUGS.items():
        suffix = f"_{gender_slug}"
        if name.endswith(suffix):
            race_slug = name[: -len(suffix)]
            race_label = RACE_SLUGS.get(race_slug)
            if race_label is not None:
                return race_label, gender_label
    raise ValueError(f"could not parse category dir name: {name!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()
    out_csv = args.out_csv or os.path.join(args.data_dir, "ground_truth_scores.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model = model.to(device).eval()
    mtcnn = MTCNN(image_size=224, margin=int(224 * 0.25), post_process=False, device="cpu")

    rows = []
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true_race][pred_race]
    category_dirs = sorted(
        d
        for d in glob.glob(os.path.join(args.data_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d) != "plots"
    )
    if not category_dirs:
        raise SystemExit(f"no category dirs found under {args.data_dir}")

    for category_dir in category_dirs:
        true_race, true_gender = parse_category_dir(os.path.basename(category_dir))
        for path in sorted(glob.glob(os.path.join(category_dir, "seed*.png"))):
            img = Image.open(path).convert("RGB")
            boxes, _ = mtcnn.detect(img)
            if boxes is None or len(boxes) == 0:
                rows.append(
                    {"path": path, "true_race": true_race, "true_gender": true_gender,
                     "pred_race": "", "pred_gender": "", "race_correct": "", "gender_correct": ""}
                )
                continue
            w, h = img.size
            x1, y1, x2, y2 = boxes[0]
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), w), min(int(y2), h)
            face = img.crop((x1, y1, x2, y2)) if (x2 - x1 >= 8 and y2 - y1 >= 8) else img

            x = TF.to_tensor(face.resize((224, 224))).unsqueeze(0).to(device)
            x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            with torch.no_grad():
                logits = model(x)
            pred_race = RACE_LABELS[logits[0, :7].argmax().item()]
            pred_gender = GENDER_LABELS[logits[0, 7:9].argmax().item()]

            confusion[true_race][pred_race] += 1
            rows.append(
                {"path": path, "true_race": true_race, "true_gender": true_gender,
                 "pred_race": pred_race, "pred_gender": pred_gender,
                 "race_correct": pred_race == true_race,
                 "gender_correct": pred_gender == true_gender}
            )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")

    scored = [r for r in rows if r["pred_race"] != ""]
    n_no_face = len(rows) - len(scored)
    print(f"scored {len(scored)} images ({n_no_face} had no detected face)")

    print("\nPer-category accuracy:")
    for race in RACE_LABELS:
        race_rows = [r for r in scored if r["true_race"] == race]
        if not race_rows:
            continue
        race_acc = sum(1 for r in race_rows if r["race_correct"]) / len(race_rows)
        gender_acc = sum(1 for r in race_rows if r["gender_correct"]) / len(race_rows)
        print(f"  {race:<18} race_acc={race_acc:.1%}  gender_acc={gender_acc:.1%}  (n={len(race_rows)})")

    overall_race_acc = sum(1 for r in scored if r["race_correct"]) / len(scored)
    overall_gender_acc = sum(1 for r in scored if r["gender_correct"]) / len(scored)
    print(f"\nOverall: race_acc={overall_race_acc:.1%}  gender_acc={overall_gender_acc:.1%}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "true\\pred".ljust(18) + "".join(r[:6].ljust(8) for r in RACE_LABELS)
    print(header)
    for true_race in RACE_LABELS:
        row_total = sum(confusion[true_race].values()) or 1
        line = true_race.ljust(18)
        for pred_race in RACE_LABELS:
            pct = confusion[true_race][pred_race] / row_total
            line += f"{pct:.0%}".ljust(8)
        print(line)


if __name__ == "__main__":
    main()
