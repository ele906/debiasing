"""
Same classification logic as scripts/analyze_racial_composition.ipynb, as a
plain script so it can be re-run for any racial_composition_* output dir
(baseline/fairness/debiased) to produce a stage-keyed composition.csv
(stage in {'init','0'..'49','best'}), for use with
scripts/plot_racial_composition_stages.py.

Usage:
    python scripts/racial_composition_analyze_by_stage.py \\
        --data_dir outputs/racial_composition_fairness \\
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

BATCH_SIZE = 32


def stage_sort_key(stage):
    if stage == "init":
        return -1
    if stage == "best":
        return 10**9
    return int(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_csv = os.path.join(data_dir, "composition_by_stage.csv")
    cache_csv = os.path.join(data_dir, "classify_cache.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()

    from facenet_pytorch import MTCNN

    # MTCNN's tiny-kernel convs hit a cuDNN "unable to find an engine" error on
    # some GPU/driver combos at small pyramid scales; it's cheap, so run it on CPU.
    mtcnn = MTCNN(image_size=224, margin=int(224 * 0.25), post_process=False, device="cpu")
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def classify_batch(paths):
        all_faces, owner = [], []
        for i, path in enumerate(paths):
            img = Image.open(path).convert("RGB")
            faces = mtcnn(img)
            if faces is None:
                continue
            if faces.dim() == 3:
                faces = faces.unsqueeze(0)
            all_faces.append(faces)
            owner.extend([i] * faces.shape[0])

        results = [[] for _ in paths]
        if not all_faces:
            return results

        batch = torch.cat(all_faces, dim=0)
        batch = (batch / 255.0).to(device)
        batch = normalize(batch)
        with torch.no_grad():
            logits = model(batch)[:, :7]
            preds = logits.argmax(dim=1).cpu().tolist()
        for idx, p in zip(owner, preds):
            results[idx].append(RACE_LABELS[p])
        return results

    def load_cache():
        cache = {}
        if os.path.exists(cache_csv):
            with open(cache_csv, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for path, races in reader:
                    cache[path] = races.split("|") if races else []
        return cache

    def append_cache(rows):
        write_header = not os.path.exists(cache_csv)
        with open(cache_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["path", "races"])
            for path, races in rows:
                writer.writerow([path, "|".join(races)])

    cache = load_cache()

    # counts[prompt_slug][stage][race] = count
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    no_face_counts = defaultdict(lambda: defaultdict(int))

    prompt_dirs = sorted(
        d for d in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isdir(d) and os.path.basename(d) != "plots"
    )
    for prompt_dir in prompt_dirs:
        prompt_slug = os.path.basename(prompt_dir)
        seed_dirs = sorted(glob.glob(os.path.join(prompt_dir, "seed*")))
        for seed_dir in seed_dirs:
            paths, stages = [], []
            for path in sorted(glob.glob(os.path.join(seed_dir, "*.png"))):
                fname = os.path.basename(path)
                if fname == "init_image.png":
                    stage = "init"
                elif fname == "best_image.png":
                    stage = "best"
                else:
                    stage = os.path.splitext(fname)[0]
                paths.append(path)
                stages.append(stage)

            to_classify = [p for p in paths if p not in cache]
            for i in range(0, len(to_classify), BATCH_SIZE):
                batch_paths = to_classify[i : i + BATCH_SIZE]
                batch_results = classify_batch(batch_paths)
                append_cache(zip(batch_paths, batch_results))
                cache.update(zip(batch_paths, batch_results))

            for path, stage in zip(paths, stages):
                races = cache[path]
                if not races:
                    no_face_counts[prompt_slug][stage] += 1
                    continue
                for race in races:
                    counts[prompt_slug][stage][race] += 1
        print(f"done: {prompt_slug}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "stage", "race", "count", "pct_of_faces"])
        for prompt_slug, by_stage in counts.items():
            for stage in sorted(by_stage.keys(), key=stage_sort_key):
                by_race = by_stage[stage]
                total = sum(by_race.values())
                for race in RACE_LABELS:
                    n = by_race.get(race, 0)
                    writer.writerow(
                        [prompt_slug, stage, race, n, n / total if total else 0.0]
                    )
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
