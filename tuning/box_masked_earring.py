"""Test a C2C earring box as a spatial mask for SAE feature suppression."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "box_masked_earring"
FEATURE = 4858
CONTROL = 2940
BOX_PATTERN = re.compile(r"\bearrings?\b", re.I)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, data):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def prepare():
    source_manifest = rows(ROOT / "tuning/accessory_ablation/manifest.csv")
    c2c_boxes = rows(ROOT / "concept/category_analysis_full/bounding_boxes.csv")
    selected = []
    for sample in source_manifest:
        if sample["concept"] != "earrings" or sample["source_label"] != "detector_positive":
            continue
        boxes = []
        for box in c2c_boxes:
            if box["sample_id"] != sample["sample_id"] or not BOX_PATTERN.search(box["label"]):
                continue
            coords = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            area = max(0, coords[2]-coords[0]) * max(0, coords[3]-coords[1]) / (512*512)
            # Exclude broad generic boxes: they are not useful as a local mask.
            if 0 < area < .15:
                boxes.append({"label": box["label"], "xyxy": coords, "area_fraction": area})
        if not boxes:
            continue
        selected.append({"sample_id": sample["sample_id"], "split": sample["split"],
                         "race_prompt": sample["race_prompt"], "gender_prompt": sample["gender_prompt"],
                         "image": sample["image"], "source_label": sample["source_label"],
                         "feature": FEATURE, "c2c_boxes": boxes})
    if not selected:
        raise ValueError("No positive earring examples have a localized C2C box")
    OUT.mkdir(exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n")
    (OUT / "settings.json").write_text(json.dumps({
        "candidate_feature": FEATURE, "random_control_feature": CONTROL,
        "comparison": ["candidate_global", "candidate_inside_c2c_boxes", "control_inside_c2c_boxes"],
        "strengths": [0, 0.5, 1], "box_padding_activation_cells": 1,
        "selection": "Existing same-prompt validation positives with localized C2C earring boxes; boxes with >15% image area excluded.",
    }, indent=2) + "\n")
    print(f"Prepared {len(selected)} boxed earring examples")


def make_hook(sae, feature, strength, boxes=None, image_size=512, pad_cells=1):
    import torch

    @torch.no_grad()
    def hook(module, inputs, output):
        original = output[0]
        diff = (original - inputs[0]).permute(0, 2, 3, 1).float()
        codes = sae.encode(diff)
        mask = torch.ones(codes.shape[1:3], device=codes.device, dtype=codes.dtype)
        if boxes is not None:
            mask.zero_()
            height, width = codes.shape[1:3]
            for x1, y1, x2, y2 in boxes:
                # Boxes are expressed in the 512x512 C2C image coordinate system.
                left = int(np.floor(x1 / image_size * width))
                right = int(np.ceil(x2 / image_size * width))
                top = int(np.floor(y1 / image_size * height))
                bottom = int(np.ceil(y2 / image_size * height))
                left, right = max(0, left-pad_cells), min(width, right+pad_cells)
                top, bottom = max(0, top-pad_cells), min(height, bottom+pad_cells)
                mask[top:bottom, left:right] = 1
        delta = (codes[..., feature:feature+1] * mask[None, :, :, None] * (strength - 1)) @ sae.decoder.weight[:, feature:feature+1].T
        edited = original + delta.permute(0, 3, 1, 2).to(original)
        return (edited,) + output[1:]
    return hook


def run():
    import torch
    from PIL import Image, ImageDraw
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder

    settings = json.loads((ROOT / "backprop/generation_settings.json").read_text())
    checkpoint = Path(settings["sae_checkpoint"])
    if hashlib.sha256((checkpoint / "state_dict.pth").read_bytes()).hexdigest() != settings["sae_weights_sha256"]:
        raise ValueError("SAE checkpoint hash mismatch")
    manifest = json.loads((OUT / "manifest.json").read_text())
    out_images, out_grids, out_boxes = OUT / "images", OUT / "grids", OUT / "box_overlays"
    for d in (out_images, out_grids, out_boxes):
        d.mkdir(exist_ok=True)
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

    metrics = []
    for index, row in enumerate(manifest, 1):
        seed = int(Path(row["image"]).stem[-3:])
        prompt = f"a picture of a {row['race_prompt']} {row['gender_prompt']}"
        generator = torch.Generator("cuda").manual_seed(seed)
        kwargs = dict(prompt=prompt,
                      latents=torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16,
                                          generator=torch.Generator("cuda").manual_seed(seed)),
                      generator=generator, num_inference_steps=1, guidance_scale=0., height=512, width=512)
        base = pipe.pipe(**kwargs).images[0]
        source = Image.open(row["image"]).convert("RGB")
        replay_error = float(np.abs(np.asarray(base).astype(float)-np.asarray(source)).mean())
        if replay_error != 0:
            raise RuntimeError(f"Replay mismatch for {row['sample_id']}: {replay_error}")
        boxes = [b["xyxy"] for b in row["c2c_boxes"]]

        overlay = source.copy()
        draw = ImageDraw.Draw(overlay)
        region_mask = np.zeros((512, 512), dtype=bool)
        for b in boxes:
            draw.rectangle(tuple(b), outline="red", width=3)
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            region_mask[max(0, y1):min(512, y2+1), max(0, x1):min(512, x2+1)] = True
        overlay.save(out_boxes / f"{row['sample_id']}.png")

        conditions = [
            ("candidate_global", FEATURE, None),
            ("candidate_box", FEATURE, boxes),
            ("control_box", CONTROL, boxes),
        ]
        comparisons = []
        for condition, feature, region in conditions:
            tiles = []
            for alpha in (0, .5, 1):
                edit_hook = make_hook(sae, feature, alpha, region)
                result = pipe.run_with_hooks(**kwargs, position_hook_dict={settings["block"]: edit_hook}).images[0]
                abs_delta = np.abs(np.asarray(result).astype(float)-np.asarray(base).astype(float)).mean(axis=2)
                delta = float(abs_delta.mean())
                inside_mae = float(abs_delta[region_mask].mean())
                outside_mae = float(abs_delta[~region_mask].mean())
                if alpha == 1 and delta != 0:
                    raise RuntimeError("Strength 1 should be an exact no-op")
                path = out_images / f"{row['sample_id']}_{condition}_a{alpha:g}.png"
                result.save(path)
                tiles.append((f"{condition}, strength {alpha:g}", result))
                metrics.append({"sample_id": row["sample_id"], "condition": condition,
                                "feature": feature, "strength": alpha, "pixel_mae": delta,
                                "inside_box_mae": inside_mae, "outside_box_mae": outside_mae,
                                "source_replay_mae": replay_error,
                                "box_count": len(boxes), "boxes": json.dumps(boxes),
                                "image": str(path.relative_to(ROOT))})
            comparisons.append(tiles)
        label_width = 150
        sheet = Image.new("RGB", (label_width+3*256, 5*294), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 3), f"{row['sample_id']} | C2C boxes (red)", fill="black")
        draw.text((3, 20), "source", fill="black")
        sheet.paste(source.resize((256, 256)), (label_width, 38))
        draw.text((3, 294+20), "C2C boxes", fill="black")
        sheet.paste(overlay.resize((256, 256)), (label_width, 294+38))
        for ri, (label, tiles) in enumerate(zip(
                ("candidate_global", "candidate_box", "control_box"), comparisons), start=2):
            draw.text((3, ri*294+20), label, fill="black")
            for ci, (title, image) in enumerate(tiles):
                strength_label = title.rsplit("strength ", 1)[-1]
                draw.text((label_width+ci*256+3, ri*294+20), f"strength {strength_label}", fill="black")
                sheet.paste(image.resize((256, 256)), (label_width+ci*256, ri*294+38))
        sheet.save(out_grids / f"{row['sample_id']}.jpg", quality=92)
        print(f"{index}/{len(manifest)} {row['sample_id']} boxes={len(boxes)}", flush=True)

    write_csv(OUT / "metrics.csv", metrics)
    means = {}
    for condition in ("candidate_global", "candidate_box", "control_box"):
        for alpha in (0, .5, 1):
            sample_rows = [r for r in metrics if r["condition"] == condition and r["strength"] == alpha]
            means[(condition, alpha)] = {key: float(np.mean([r[key] for r in sample_rows]))
                                        for key in ("pixel_mae", "inside_box_mae", "outside_box_mae")}
    report = ["# C2C-box-masked SAE earring intervention", "",
              "C2C bounding boxes are mapped from 512x512 image coordinates to the SAE code's runtime spatial grid. The candidate SAE feature is reduced globally and inside only the boxed cells; a matched random feature is reduced inside the same boxes. A one-cell border is added to accommodate coarse activation resolution.", "",
              f"Candidate feature: {FEATURE}. Random control: {CONTROL}. {len(manifest)} same-prompt validation positives. Strength 1 is unchanged; 0 removes the feature contribution.", "",
              "## Mean image change", "",
              "MAE is the mean absolute RGB difference on a 0–255 scale. Inside/outside values measure the rendered image change relative to the C2C box; they are not object-presence scores.", "",
              "| Intervention | Strength | Whole image MAE | Inside box MAE | Outside box MAE |", "|---|---:|---:|---:|---:|"]
    for condition in ("candidate_global", "candidate_box", "control_box"):
        for alpha in (0, .5, 1):
            v = means[(condition, alpha)]
            report.append(f"| {condition} | {alpha:g} | {v['pixel_mae']:.3f} | {v['inside_box_mae']:.3f} | {v['outside_box_mae']:.3f} |")
    report += ["", "All source replays were pixel-identical, and strength 1 was an exact no-op. The candidate feature changed the images, but visual inspection showed the earrings remained. Global reduction altered more of the image; box masking reduced those side effects but did not make the candidate feature a reliable earring control.", "",
               "Inspect the red-box overlays and comparison grids below.", ""]
    for row in manifest:
        report += [f"## {row['sample_id']}", "",
                   f"![C2C earring boxes](box_overlays/{row['sample_id']}.png)", "",
                   f"![Intervention comparisons](grids/{row['sample_id']}.jpg)", ""]
    (OUT / "REPORT.md").write_text("\n".join(report))
    (OUT / "settings.json").write_text(json.dumps({
        "candidate_feature": FEATURE, "random_control_feature": CONTROL,
        "box_masked": True, "box_source": "concept/category_analysis_full/bounding_boxes.csv",
        "box_padding_activation_cells": 1, "strengths": [0, .5, 1], "samples": len(manifest),
        "interpretation": "Exploratory spatial intervention; C2C boxes and automatic labels require visual review.",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        sys.path.insert(0, str(ROOT / "sdxl-unbox"))
        sys.path.insert(0, str(ROOT / "sdxl-unbox/scripts"))
        run()
    if not args.prepare and not args.run:
        parser.error("choose --prepare and/or --run")


if __name__ == "__main__":
    main()
