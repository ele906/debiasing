"""
Classify faces in the per-iteration images produced by
racial_composition_generate.py, and aggregate racial composition over the
50 optimization iterations for each prompt.

Uses FairFace (Karkkainen & Joo, 2019) for race classification -- the
standard model for this kind of demographic-audit study, trained on a
balanced-by-construction dataset with 7 race/ethnicity categories:
White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian,
Middle Eastern.

Requirements (not in this repo's environment.yml -- install separately):
    pip install facenet-pytorch

FairFace weights: download `fairface_res34_fair_align_multi_7_20190809.pt`
from the official release (https://github.com/dchen236/FairFace, in
the repo's `fair_face_models` release asset) and pass its path via
--fairface_weights. FairFace expects a face crop with 0.25 padding,
resized to 224x224, ImageNet-normalized -- MTCNN below approximates
that alignment.

Usage:
    python scripts/racial_composition_analyze.py \\
        --data_dir outputs/racial_composition \\
        --fairface_weights /path/to/fairface_res34_fair_align_multi_7_20190809.pt \\
        --out_csv outputs/racial_composition/composition.csv \\
        --out_plot_dir outputs/racial_composition/plots
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision
from PIL import Image

RACE_LABELS = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]


def build_fairface_model(weights_path: str, device: torch.device) -> nn.Module:
    """FairFace's public checkpoint is a resnet34 backbone with an 18-way
    head (7 race + 9 age + 2 gender logits, concatenated); we only read the
    first 7 (race) logits."""
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def get_face_cropper(device: torch.device):
    from facenet_pytorch import MTCNN

    return MTCNN(
        image_size=224, margin=int(224 * 0.25), post_process=False, device=device
    )


def classify_races(
    image_path: str,
    mtcnn,
    model: nn.Module,
    device: torch.device,
    normalize,
) -> list:
    """Returns a list of predicted race labels, one per detected face."""
    img = Image.open(image_path).convert("RGB")
    faces = mtcnn(img)
    if faces is None:
        return []
    if faces.dim() == 3:
        faces = faces.unsqueeze(0)
    faces = (faces / 255.0).to(device)
    faces = normalize(faces)
    with torch.no_grad():
        logits = model(faces)[:, :7]
        preds = logits.argmax(dim=1).cpu().tolist()
    return [RACE_LABELS[p] for p in preds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--fairface_weights", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_plot_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_fairface_model(args.fairface_weights, device)
    mtcnn = get_face_cropper(device)
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # counts[prompt_slug][iteration][race] = count
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    no_face_counts = defaultdict(lambda: defaultdict(int))

    prompt_dirs = sorted(
        d for d in glob.glob(os.path.join(args.data_dir, "*")) if os.path.isdir(d)
    )
    for prompt_dir in prompt_dirs:
        prompt_slug = os.path.basename(prompt_dir)
        seed_dirs = sorted(glob.glob(os.path.join(prompt_dir, "seed*")))
        for seed_dir in seed_dirs:
            iter_paths = glob.glob(os.path.join(seed_dir, "*.png"))
            for path in iter_paths:
                fname = os.path.basename(path)
                if fname in ("init_image.png", "best_image.png"):
                    continue
                iteration = int(os.path.splitext(fname)[0])
                races = classify_races(path, mtcnn, model, device, normalize)
                if not races:
                    no_face_counts[prompt_slug][iteration] += 1
                    continue
                for race in races:
                    counts[prompt_slug][iteration][race] += 1
        print(f"done: {prompt_slug}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "iteration", "race", "count", "pct_of_faces"])
        for prompt_slug, by_iter in counts.items():
            for iteration, by_race in sorted(by_iter.items()):
                total = sum(by_race.values())
                for race in RACE_LABELS:
                    n = by_race.get(race, 0)
                    writer.writerow(
                        [prompt_slug, iteration, race, n, n / total if total else 0.0]
                    )
    print(f"wrote {args.out_csv}")

    if args.out_plot_dir:
        plot_trajectories(counts, args.out_plot_dir)


def plot_trajectories(counts, out_dir: str):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    for prompt_slug, by_iter in counts.items():
        iterations = sorted(by_iter.keys())
        fig, ax = plt.subplots(figsize=(8, 5))
        for race in RACE_LABELS:
            pcts = []
            for it in iterations:
                total = sum(by_iter[it].values())
                pcts.append(by_iter[it].get(race, 0) / total if total else 0.0)
            ax.plot(iterations, pcts, label=race, marker="o", markersize=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fraction of detected faces")
        ax.set_title(f"Racial composition over iterations: {prompt_slug}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{prompt_slug}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
