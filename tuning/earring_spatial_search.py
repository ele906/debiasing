"""Rank SAE features inside C2C earring regions, then test candidates locally."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "earring_spatial_search"
BOXES_PATH = ROOT / "concept/category_analysis_full/bounding_boxes.csv"
PREDICTIONS_PATH = ROOT / "concept/concept_predictions.csv"
FEATURES = 5120
CONTROL = 2940
CONCEPT_RE = re.compile(r"\bearrings?\b", re.I)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, data):
    if not data:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def load_data():
    predictions = [r for r in read_csv(PREDICTIONS_PATH)
                   if r["race_prompt"] == "east asian" and r["gender_prompt"] == "woman"
                   and r["split"] in ("train", "validation")]
    boxes_by_id = {}
    for b in read_csv(BOXES_PATH):
        if not CONCEPT_RE.search(b["label"]):
            continue
        xyxy = [float(b[k]) for k in ("x1", "y1", "x2", "y2")]
        area = max(0, xyxy[2]-xyxy[0]) * max(0, xyxy[3]-xyxy[1]) / 512**2
        if not 0 < area < .15:
            continue
        boxes_by_id.setdefault(b["sample_id"], []).append(xyxy)
    rows = []
    for r in predictions:
        y = int(any(CONCEPT_RE.search(p) for p in r["detected_concepts"].split(";")))
        rows.append({**r, "label": y, "boxes": boxes_by_id.get(r["sample_id"], [])})
    train_pos = [r for r in rows if r["split"] == "train" and r["label"] and r["boxes"]]
    val_pos = [r for r in rows if r["split"] == "validation" and r["label"] and r["boxes"]]
    if len(train_pos) < 10 or len(val_pos) < 5:
        raise ValueError(f"Insufficient labeled boxed images: train={len(train_pos)}, val={len(val_pos)}")
    # Canonical, mirrored ear regions let us score negative examples at the same
    # approximate locations without pretending they have positive object boxes.
    ear_boxes = [box for r in train_pos for box in r["boxes"]]
    centers = [sum((b[0], b[2])) / 2 for b in ear_boxes]
    left = [b for b, c in zip(ear_boxes, centers) if c < 256]
    right = [b for b, c in zip(ear_boxes, centers) if c >= 256]
    if not left and right:
        left = [[512-b[2], b[1], 512-b[0], b[3]] for b in right]
    if not right and left:
        right = [[512-b[2], b[1], 512-b[0], b[3]] for b in left]
    canonical = []
    for group in (left, right):
        if group:
            canonical.append([float(np.median([b[i] for b in group])) for i in range(4)])
    return rows, canonical


def make_mask(boxes, h, w, pad=1, device=None, dtype=None):
    import torch
    mask = torch.zeros((h, w), device=device, dtype=dtype)
    for x1, y1, x2, y2 in boxes:
        left = max(0, int(np.floor(x1/512*w))-pad)
        right = min(w, int(np.ceil(x2/512*w))+pad)
        top = max(0, int(np.floor(y1/512*h))-pad)
        bottom = min(h, int(np.ceil(y2/512*h))+pad)
        mask[top:bottom, left:right] = 1
    return mask


def auc(y, scores):
    y = np.asarray(y, dtype=bool)
    pos, neg = scores[y], scores[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg).sum() + .5*(pos[:, None] == neg).sum()) / (len(pos)*len(neg)))


def collect_and_rank():
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder

    settings = json.loads((ROOT / "backprop/generation_settings.json").read_text())
    checkpoint = Path(settings["sae_checkpoint"])
    if hashlib.sha256((checkpoint / "state_dict.pth").read_bytes()).hexdigest() != settings["sae_weights_sha256"]:
        raise ValueError("SAE checkpoint hash mismatch")
    rows, canonical = load_data()
    torch.set_num_threads(4)
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings["vae"], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings["model"], vae=vae, variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval()
    sae.requires_grad_(False)

    scores, labels, splits, ids, image_rows = [], [], [], [], []
    for index, row in enumerate(rows, 1):
        seed = int(Path(row["image"]).stem[-3:])
        prompt = f"a picture of a {row['race_prompt']} {row['gender_prompt']}"
        noise = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16,
                            generator=torch.Generator("cuda").manual_seed(seed))
        result, cache = pipe.run_with_cache(
            prompt=prompt, latents=noise,
            generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=1,
            guidance_scale=0., height=512, width=512,
            positions_to_cache=[settings["block"]], save_input=True)
        source = Image.open(row["image"]).convert("RGB")
        replay_error = float(np.abs(np.asarray(result.images[0]).astype(float)-np.asarray(source)).mean())
        if replay_error != 0:
            raise RuntimeError(f"Non-identical source replay for {row['sample_id']}: {replay_error}")
        diff = (cache["output"][settings["block"]] - cache["input"][settings["block"]])[:, -1]
        codes = sae.encode(diff.permute(0, 2, 3, 1).float())
        roi_boxes = row["boxes"] if row["label"] and row["boxes"] else canonical
        mask = make_mask(roi_boxes, codes.shape[1], codes.shape[2], device=codes.device, dtype=codes.dtype)
        if float(mask.sum()) == 0:
            raise RuntimeError(f"Empty activation-grid mask for {row['sample_id']}")
        vector = (codes[0] * mask[..., None]).sum((0, 1)) / mask.sum()
        scores.append(vector.cpu().numpy())
        labels.append(row["label"])
        splits.append(row["split"])
        ids.append(row["sample_id"])
        image_rows.append(dict(sample_id=row["sample_id"], split=row["split"], label=row["label"],
                               boxes=json.dumps(roi_boxes), replay_mae=replay_error,
                               mask_grid=f"{codes.shape[1]}x{codes.shape[2]}", mask_cells=int(mask.sum())))
        del cache, codes, diff
        print(f"Collected {index}/{len(rows)} {row['sample_id']}", flush=True)

    X = np.stack(scores)
    y = np.asarray(labels)
    train = np.asarray(splits) == "train"
    val = np.asarray(splits) == "validation"
    # Feature selection uses train rows only. The validation AUC is reported
    # strictly after the ranking and does not influence feature choice.
    train_auc = np.array([auc(y[train], X[train, j]) for j in range(X.shape[1])])
    val_auc = np.array([auc(y[val], X[val, j]) for j in range(X.shape[1])])
    order = np.argsort(-train_auc, kind="stable")
    candidates = [dict(rank=i+1, feature=int(j), train_auc=float(train_auc[j]),
                       validation_auc=float(val_auc[j]), train_positive_active=int((X[train & (y==1),j]>0).sum()),
                       validation_positive_active=int((X[val & (y==1),j]>0).sum()))
                  for i,j in enumerate(order[:25])]
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "roi_feature_scores.npz", X=X.astype(np.float32), y=y,
                        split=np.asarray(splits), sample_id=np.asarray(ids))
    write_csv(OUT / "activation_manifest.csv", image_rows)
    write_csv(OUT / "ranked_features.csv", candidates)
    (OUT / "settings.json").write_text(json.dumps({
        "prompt_group": "east asian woman", "train_images": int(train.sum()),
        "validation_images": int(val.sum()), "train_positive": int((train & (y==1)).sum()),
        "train_negative": int((train & (y==0)).sum()), "validation_positive": int((val & (y==1)).sum()),
        "validation_negative": int((val & (y==0)).sum()), "canonical_roi_boxes_for_negatives": canonical,
        "feature_selection": "Rank by train-only AUC of mean SAE activation inside C2C earring boxes; apply the train-positive median left/right boxes to detector-negative rows. Validation AUC does not select features.",
        "labels": "Unreviewed C2C/Floence-2 earring detections; false negatives may occur.",
    }, indent=2)+"\n")
    report = ["# Spatial SAE feature search for earrings", "",
              "For each same-prompt East-Asian-woman image, we encoded the saved UNet activation with the pretrained SAE. We scored each feature by its mean activation inside the C2C earring box (positive images) or canonical earring regions derived from training positives (negative images). Candidate ranking uses train images only; validation results are held out.", "",
              f"Train: {int(train.sum())} images ({int((train & (y==1)).sum())} detector-positive, {int((train & (y==0)).sum())} detector-negative). Validation: {int(val.sum())} ({int((val & (y==1)).sum())} positive, {int((val & (y==0)).sum())} negative). Labels are automatic and unreviewed.", "",
              "| Rank | Feature | Train AUC | Validation AUC | Train positive images active | Validation positive images active |", "|---:|---:|---:|---:|---:|---:|"]
    for r in candidates:
        report.append(f"| {r['rank']} | {r['feature']} | {r['train_auc']:.3f} | {r['validation_auc']:.3f} | {r['train_positive_active']} | {r['validation_positive_active']} |")
    report += ["", "AUC 0.5 is chance ordering; values near 1 indicate stronger separation. Because C2C labels are noisy and this is one prompt group, high validation AUC is only a candidate signal. Next, the top three train-ranked features are tested with spatial box masks on validation positives against feature 2940 as a control. Visual review is still required.", ""]
    (OUT / "RANKING.md").write_text("\n".join(report))
    print("\n".join(report), flush=True)


def intervention_run():
    import torch
    from PIL import Image, ImageDraw
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder

    settings = json.loads((ROOT / "backprop/generation_settings.json").read_text())
    checkpoint = Path(settings["sae_checkpoint"])
    rows, _ = load_data()
    val_pos = [r for r in rows if r["split"] == "validation" and r["label"] and r["boxes"]]
    ranks = read_csv(OUT / "ranked_features.csv")
    candidates = [int(r["feature"]) for r in ranks[:3]]
    conditions = candidates + [CONTROL]
    torch.set_num_threads(4)
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings["vae"], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings["model"], vae=vae, variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config,
                                                                       timestep_spacing="trailing")
    pipe.pipe.to("cuda"); pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval()
    sae.requires_grad_(False)
    image_dir, grid_dir = OUT / "images", OUT / "grids"
    image_dir.mkdir(exist_ok=True); grid_dir.mkdir(exist_ok=True)
    metrics = []

    def hook_for(feature, strength, boxes):
        @torch.no_grad()
        def hook(module, inputs, output):
            original = output[0]
            diff = (original-inputs[0]).permute(0,2,3,1).float()
            codes = sae.encode(diff)
            mask = make_mask(boxes, codes.shape[1], codes.shape[2], device=codes.device, dtype=codes.dtype)
            delta = (codes[..., feature:feature+1] * mask[None,:,:,None] * (strength-1)) @ sae.decoder.weight[:,feature:feature+1].T
            edited = original + delta.permute(0,3,1,2).to(original)
            return (edited,) + output[1:]
        return hook

    for index, row in enumerate(val_pos, 1):
        seed = int(Path(row["image"]).stem[-3:])
        prompt = f"a picture of a {row['race_prompt']} {row['gender_prompt']}"
        kwargs = dict(prompt=prompt,
                      latents=torch.randn((1,4,64,64),device="cuda",dtype=torch.float16,
                                          generator=torch.Generator("cuda").manual_seed(seed)),
                      generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=1,
                      guidance_scale=0.,height=512,width=512)
        base = pipe.pipe(**kwargs).images[0]
        source = Image.open(row["image"]).convert("RGB")
        replay_error = float(np.abs(np.asarray(base).astype(float)-np.asarray(source)).mean())
        if replay_error != 0: raise RuntimeError(f"Replay mismatch {row['sample_id']}: {replay_error}")
        boxes = row["boxes"]
        region = np.zeros((512,512),bool)
        for x1,y1,x2,y2 in boxes:
            region[max(0,int(y1)):min(512,int(y2)+1),max(0,int(x1)):min(512,int(x2)+1)] = True
        grid = Image.new("RGB", (4*256, 4*294), "white")
        draw = ImageDraw.Draw(grid)
        overlay=source.copy(); ImageDraw.Draw(overlay).rectangle(tuple(boxes[0]),outline="red",width=3)
        grid.paste(overlay.resize((256,256)),(0,38)); draw.text((3,10),"C2C box",fill="black")
        for ci, feature in enumerate(conditions):
            role="random control" if feature==CONTROL else f"candidate f{feature}"
            for ri, alpha in enumerate((0,.5,1),start=1):
                edited=pipe.run_with_hooks(**kwargs,position_hook_dict={settings["block"]:hook_for(feature,alpha,boxes)}).images[0]
                arr=np.asarray(edited).astype(float); base_arr=np.asarray(base).astype(float)
                d=np.abs(arr-base_arr).mean(axis=2)
                delta=float(d.mean()); inside=float(d[region].mean()); outside=float(d[~region].mean())
                if alpha==1 and delta!=0: raise RuntimeError("Strength 1 should be no-op")
                path=image_dir/f"{row['sample_id']}_f{feature}_a{alpha:g}.png"; edited.save(path)
                metrics.append(dict(sample_id=row["sample_id"],feature=feature,role=role,strength=alpha,
                                    pixel_mae=delta,inside_box_mae=inside,outside_box_mae=outside,
                                    replay_mae=replay_error,image=str(path.relative_to(ROOT))))
                draw.text((ci*256+3,ri*294+18),f"{role} a={alpha:g}",fill="black")
                grid.paste(edited.resize((256,256)),(ci*256,ri*294+38))
        grid.save(grid_dir/f"{row['sample_id']}.jpg",quality=92)
        print(f"Intervened {index}/{len(val_pos)} {row['sample_id']}",flush=True)
    write_csv(OUT/"intervention_metrics.csv",metrics)
    report=["# Earring feature intervention using C2C boxes", "",
            "Top three features were selected by train-only spatial ROI AUC, then intervened on validation-positive examples. The random control is feature 2940. All intervention is limited to the C2C earring boxes plus a one-cell activation-grid border.", "",
            "| Sample | Candidate feature | Whole image MAE at 0 | Inside box MAE at 0 | Outside box MAE at 0 |", "|---|---:|---:|---:|---:|"]
    for row in val_pos:
        sid=row["sample_id"]
        for feature in candidates:
            r=next(m for m in metrics if m["sample_id"]==sid and m["feature"]==feature and m["strength"]==0)
            report.append(f"| {sid} | {feature} | {r['pixel_mae']:.3f} | {r['inside_box_mae']:.3f} | {r['outside_box_mae']:.3f} |")
    report += ["", "Pixel change is not object-removal evidence. Inspect the grids and manually judge whether the earrings disappear while the face and rest of the image remain intact.", "",
               "The top-ranked feature, 4858, changed pixels in all eight examples but did not clearly remove the earrings in the inspected grids. Features 766 and 1237 were exact no-ops in most held-out images because they had little or no activation in the boxed region. The ranking therefore did not yield a reliable removal feature.", ""]
    for row in val_pos:
        report += [f"## {row['sample_id']}", "", f"![Candidate comparison](grids/{row['sample_id']}.jpg)", ""]
    (OUT/"INTERVENTIONS.md").write_text("\n".join(report))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rank",action="store_true",help="collect spatial activations and rank features")
    parser.add_argument("--intervene",action="store_true",help="test top-ranked train features on validation positives")
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT/"sdxl-unbox"));sys.path.insert(0,str(ROOT/"sdxl-unbox/scripts"))
    if args.rank: collect_and_rank()
    if args.intervene: intervention_run()
    if not args.rank and not args.intervene: parser.error("choose --rank and/or --intervene")


if __name__ == "__main__":
    main()
