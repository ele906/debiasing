"""
Independent audit of FairFace's race/gender predictions on generated
best_images, using zero-shot CLIP (a different architecture, different
training data, never used as the optimization reward for race/gender) as a
cross-check.

This matters because FairFace is *both* the reward signal used to optimize
images (rewards/fairness.py) and, in scripts/racial_composition_analyze*.py,
the metric used to report whether optimization worked. A classifier judging
its own optimization target is circular: at fairness_weighting=15.0
(rewards/fairness.py docstring) the KL loss collapsed while an independent
visual check showed no real change in the depicted face, i.e. FairFace was
gamed with an imperceptible adversarial pattern the model itself couldn't
detect as fake. This script gives a second, architecturally independent
opinion on the same face crops so gaming at lower weightings can be caught
without eyeballing every image by hand.

Only audits the "best" stage image per seed (the actual reported result),
not every intermediate iteration.

Usage:
    python scripts/audit_fairface_independent.py \\
        --data_dir outputs/two_person_fairness_v2 \\
        --fairface_weights fairface_weights/res34_fair_align_multi_7_20190809.pt \\
        --cache_dir /n/fs/goose/el8403/hf-cache
"""
import argparse
import glob
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

RACE_LABELS = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]
GENDER_LABELS = ["Male", "Female"]

# Phrasing matched to FairFace's category names so the two classifiers are
# being asked "the same question," not different ones.
RACE_PROMPTS = {
    "White": "a photo of a white person's face",
    "Black": "a photo of a black person's face",
    "Latino_Hispanic": "a photo of a latino or hispanic person's face",
    "East Asian": "a photo of an east asian person's face",
    "Southeast Asian": "a photo of a southeast asian person's face",
    "Indian": "a photo of an indian person's face",
    "Middle Eastern": "a photo of a middle eastern person's face",
}
GENDER_PROMPTS = {
    "Male": "a photo of a man's face",
    "Female": "a photo of a woman's face",
}


def load_fairface(weights_path, device):
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.to(device).eval()


def classify_fairface(model, face_crop_pil, device):
    # 224x224, ImageNet-normalized, matching rewards/fairness.py preprocessing.
    x = TF.to_tensor(face_crop_pil.resize((224, 224))).unsqueeze(0).to(device)
    x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    with torch.no_grad():
        logits = model(x)
    race = RACE_LABELS[logits[0, :7].argmax().item()]
    gender = GENDER_LABELS[logits[0, 7:9].argmax().item()]
    return race, gender


def build_clip_audit(cache_dir, device):
    clip_model = CLIPModel.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
    ).to(device).eval()
    processor = CLIPProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
    )

    def embed_text(prompts_dict):
        labels = list(prompts_dict.keys())
        texts = [prompts_dict[k] for k in labels]
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)
        return labels, feats / feats.norm(dim=-1, keepdim=True)

    race_labels_order, race_text_feats = embed_text(RACE_PROMPTS)
    gender_labels_order, gender_text_feats = embed_text(GENDER_PROMPTS)

    def classify(face_crop_pil):
        inputs = processor(images=face_crop_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feats = clip_model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        race_sims = (img_feats @ race_text_feats.T)[0]
        gender_sims = (img_feats @ gender_text_feats.T)[0]
        race = race_labels_order[race_sims.argmax().item()]
        gender = gender_labels_order[gender_sims.argmax().item()]
        return race, gender

    return classify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--fairface_weights", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument(
        "--out_csv", type=str, default=None,
        help="defaults to {data_dir}/fairface_audit.csv",
    )
    args = parser.parse_args()
    out_csv = args.out_csv or os.path.join(args.data_dir, "fairface_audit.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fairface_model = load_fairface(args.fairface_weights, device)
    clip_classify = build_clip_audit(args.cache_dir, device)
    mtcnn = MTCNN(image_size=224, margin=int(224 * 0.25), post_process=False, device="cpu")

    rows = []
    best_images = sorted(glob.glob(os.path.join(args.data_dir, "*", "seed*", "best_image.png")))
    if not best_images:
        raise SystemExit(f"no best_image.png files found under {args.data_dir}")

    for path in best_images:
        seed_dir = os.path.dirname(path)
        prompt_slug = os.path.basename(os.path.dirname(seed_dir))
        seed = os.path.basename(seed_dir)

        img = Image.open(path).convert("RGB")
        boxes, _ = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            rows.append(
                {"path": path, "prompt": prompt_slug, "seed": seed, "face_idx": -1,
                 "fairface_race": "", "clip_race": "", "race_agree": "",
                 "fairface_gender": "", "clip_gender": "", "gender_agree": ""}
            )
            continue

        w, h = img.size
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), w), min(int(y2), h)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            face = img.crop((x1, y1, x2, y2))
            ff_race, ff_gender = classify_fairface(fairface_model, face, device)
            clip_race, clip_gender = clip_classify(face)
            rows.append(
                {
                    "path": path, "prompt": prompt_slug, "seed": seed, "face_idx": i,
                    "fairface_race": ff_race, "clip_race": clip_race,
                    "race_agree": ff_race == clip_race,
                    "fairface_gender": ff_gender, "clip_gender": clip_gender,
                    "gender_agree": ff_gender == clip_gender,
                }
            )

    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")

    faces_rows = [r for r in rows if r["face_idx"] != -1]
    n = len(faces_rows)
    n_no_face = len(rows) - n
    if n == 0:
        print("no faces detected in any image -- nothing to audit")
        return
    race_agree = sum(1 for r in faces_rows if r["race_agree"]) / n
    gender_agree = sum(1 for r in faces_rows if r["gender_agree"]) / n
    print(f"faces audited: {n} (images with no detected face: {n_no_face})")
    print(f"FairFace vs. CLIP zero-shot agreement -- race: {race_agree:.1%}, gender: {gender_agree:.1%}")
    print(
        "Low agreement suggests FairFace's predictions on these images don't "
        "reflect what an independent model sees -- inspect the disagreement "
        "rows (race_agree/gender_agree == False) by hand before trusting the "
        "composition numbers."
    )


if __name__ == "__main__":
    main()
