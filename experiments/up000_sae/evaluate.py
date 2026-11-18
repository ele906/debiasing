"""Evaluate matched mid.0 and up.0.0 samples with the archived face detector."""
import argparse
import csv
import hashlib
from pathlib import Path
from functools import lru_cache
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

sys.path.append(str(ROOT / "face-landmarker/.deps"))
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def pixels(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pilot", "full"])
    args = parser.parse_args()
    out = RUN / args.mode / "evaluation"
    out.mkdir(exist_ok=True, parents=True)
    features = json.loads((RUN / "discovery/top10.json").read_text())["features"]
    features = features[:3] if args.mode == "pilot" else features
    files = [RUN / args.mode / f"feature_{feature}/manifest.csv" for feature in features]
    rows = [row for path in files for row in read_csv(path)]
    expected = len(features) * 60 * (7 if args.mode == "pilot" else 29)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} up.0.0 rows, found {len(rows)}")
    mid = read_csv(ROOT / "layers/mid0_stability/top10_results/per_image.csv")
    selected_alphas = {float(row["alpha"]) for row in rows}
    mid = [row for row in mid if float(row["alpha"]) in selected_alphas]
    model = ROOT / "face-landmarker/models/face_landmarker.task"
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)), num_faces=1,
        min_face_detection_confidence=.5, min_face_presence_confidence=.5,
        min_tracking_confidence=.5,
    )
    (out / "settings.json").write_text(json.dumps(dict(mediapipe=mp.__version__, model_sha256=digest(model),
                                                      confidence=.5, baseline_mask="convex hull of landmarks"), indent=2))
    evaluated = []
    controls = {(r["feature"], r["group"], r["seed"]): r["image"] for r in mid
                if float(r["alpha"]) == 1}
    with vision.FaceLandmarker.create_from_options(opts) as detector:
        def landmarks(rgb):
            return detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).face_landmarks

        @lru_cache(maxsize=120)
        def baseline(path):
            rgb = pixels(path)
            faces = landmarks(rgb)
            if not faces:
                return rgb, None
            from scipy.spatial import ConvexHull, QhullError
            height, width = rgb.shape[:2]
            points = np.array([(p.x * width, p.y * height) for p in faces[0]])
            try:
                hull = ConvexHull(points)
            except QhullError:
                return rgb, None
            mask = Image.new("1", (width, height))
            ImageDraw.Draw(mask).polygon([tuple(points[index]) for index in hull.vertices], fill=1)
            return rgb, np.asarray(mask, dtype=bool)

        for layer, conditions in (("up.0.0", rows), ("mid.0", mid)):
            for row in conditions:
                key = tuple(row[k] for k in ("feature", "group", "seed"))
                path = (RUN / row["image"]) if layer == "up.0.0" else ROOT / row["image"]
                base = path.with_name("baseline.png") if layer == "up.0.0" else ROOT / controls[key]
                reference, mask = baseline(base)
                image = pixels(path)
                diff = np.abs(image.astype(np.float32) - reference).mean(axis=2)
                evaluated.append(dict(layer=layer, feature=key[0], group=key[1], seed=int(key[2]),
                                      alpha=float(row["alpha"]), image=str(path),
                                      detected=bool(landmarks(image)), baseline_detected=mask is not None,
                                      active=(row["zero_activation"] == "False" if layer == "up.0.0"
                                              else row["initially_active"] == "True"),
                                      l2=float(row["delta_fro_applied"]),
                                      relative_l2=float(row["relative_delta_residual_fro"]),
                                      image_mae=float(diff.mean()),
                                      inside_face_mae=None if mask is None else float(diff[mask].mean()),
                                      outside_face_mae=None if mask is None else float(diff[~mask].mean())))
    write_csv(out / "per_image.csv", evaluated)

    import pandas as pd
    frame = pd.DataFrame(evaluated)
    metrics = ["detected", "image_mae", "inside_face_mae", "outside_face_mae"]
    for label in ("feature", "group"):
        keys = ["layer", label, "alpha"]
        seed_means = frame.groupby(keys + ["seed"])[metrics + ["l2", "relative_l2"]].mean()
        seed_means.groupby(keys).agg(["mean", "sem", "count"]).to_csv(out / f"by_{label}.csv")

    # Compare only common residual-relative doses, separately by intervention direction.
    from matplotlib import pyplot as plt

    active = frame[frame.active & frame.baseline_detected & (frame.l2 > 0)].copy()
    active["direction"] = np.where(active.alpha < 1, "decrease", "increase")
    comparison = []
    for direction, subset in active.groupby("direction"):
        bounds = subset.groupby("layer").relative_l2.agg(["min", "max"])
        low, high = bounds["min"].max(), bounds["max"].min()
        if low >= high or len(bounds) != 2:
            continue
        subset = subset[subset.relative_l2.between(low, high)].copy()
        subset["dose"] = pd.cut(subset.relative_l2, np.geomspace(low, high, 9), include_lowest=True)
        for dose, bin_rows in subset.groupby("dose", observed=True):
            # Give matched prompt/seed pairs equal weight within each dose bin.
            pairs = bin_rows.groupby(["group", "seed", "layer"])[metrics].mean().unstack("layer").dropna()
            if pairs.empty:
                continue
            for metric in metrics:
                changes = pairs[metric]["up.0.0"] - pairs[metric]["mid.0"]
                seed_means = changes.groupby("seed").mean()
                comparison.append(dict(direction=direction, dose=str(dose), metric=metric,
                                       up_minus_mid=seed_means.mean(), seed_sem=seed_means.sem(),
                                       pairs=len(changes), seeds=len(seed_means)))
    if comparison:
        write_csv(out / "matched_relative_l2.csv", comparison)

    for layer, part in frame.groupby("layer"):
        for key in ("feature", "group"):
            figure, axis = plt.subplots(figsize=(9, 5))
            for name, subset in part.groupby(key):
                rates = subset.groupby(["alpha", "seed"]).detected.mean().groupby("alpha").agg(["mean", "sem"])
                axis.errorbar(rates.index, rates["mean"], yerr=rates["sem"], label=str(name))
            axis.set(xlabel="Intervention multiplier", ylabel="MediaPipe detection rate", ylim=(-.05, 1.05))
            axis.legend(fontsize="small", ncol=2)
            figure.tight_layout()
            figure.savefig(out / f"{layer}_detection_by_{key}.png")
            plt.close(figure)

    gallery = out / "gallery"
    gallery.mkdir(exist_ok=True)
    for feature in features:
        subset = frame[(frame.layer == "up.0.0") & (frame.feature == str(feature))]
        edits = subset[subset.active & (subset.alpha != 1)].sort_values("l2")
        if edits.empty:
            continue
        dose = edits.iloc[len(edits) // 2]
        matched = subset[(subset.group == dose.group) & (subset.seed == dose.seed)].sort_values("alpha")
        examples = matched[matched.alpha.isin([matched.alpha.min(), 0, 1, dose.alpha, matched.alpha.max()])]
        sheet = Image.new("RGB", (192 * len(examples), 216), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(examples.itertuples()):
            image = Image.fromarray(pixels(Path(row.image))).resize((192, 192))
            sheet.paste(image, (192 * index, 24))
            draw.text((192 * index + 4, 4), f"alpha {row.alpha:g}   L2 {row.l2:.1f}", fill="black")
        sheet.save(gallery / f"feature_{feature}_{dose.group}_seed{dose.seed}.jpg")


if __name__ == "__main__":
    main()
