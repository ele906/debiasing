"""Screen saved feature 138 multiplier images and mark passing tiles.

Florence-2 predictions are heuristic screening signals, not ground truth.
An image passes when Florence-2 returns a face detection and no configured
headwear/jewelry term appears in its caption or object labels. A multiplier is
accepted for the selected sample set only when its failure count is at most
--max-failures (default: zero).
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
MODEL = "microsoft/Florence-2-base-ft"
MODEL_REVISION = "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
MODEL_CACHE = PROJECT_ROOT / "concept/hf_cache"
FORBIDDEN_PATTERN = re.compile(
    r"\b(?:earrings?|ear\s*rings?|turbans?|head\s*scarfs?|head\s*scarves?|scarfs?|scarves?|"
    r"hijabs?|veils?|head\s*coverings?|head\s*wraps?|dupattas?)\b",
    re.IGNORECASE,
)
FACE_PATTERN = re.compile(r"\b(?:human\s+face|face)\b", re.IGNORECASE)
ALPHA_PATTERN = re.compile(r"^seed(?P<seed>\d+)_alpha_(?P<slug>[mp]\d+)\.png$")


def alpha_from_slug(slug):
    return -int(slug[1:]) if slug.startswith("m") else int(slug[1:])


def slug(alpha):
    alpha = int(alpha)
    return f"m{abs(alpha)}" if alpha < 0 else f"p{alpha}"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_florence(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, dynamic_module_utils
    from unittest.mock import patch
    import timm.models.layers as timm_model_layers

    # Florence-2's custom module imports the newer timm.layers path. The local
    # pinned environment has timm 0.6, which exposes the same symbols at this path.
    sys.modules.setdefault("timm.layers", timm_model_layers)

    snapshot = (MODEL_CACHE / ("models--" + MODEL.replace("/", "--")) /
                "snapshots" / MODEL_REVISION)
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned Florence-2 snapshot not found: {snapshot}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or run on a GPU node")
    dtype = torch.float16 if device == "cuda" else torch.float32
    common = dict(trust_remote_code=True, local_files_only=True)

    # Match the eager-attention compatibility workaround in concept/extract.py.
    original_get_imports = dynamic_module_utils.get_imports

    def eager_imports(filename):
        imports = original_get_imports(filename)
        if Path(filename).name == "modeling_florence2.py":
            imports = [name for name in imports if name != "flash_attn"]
        return imports

    with patch.object(dynamic_module_utils, "get_imports", eager_imports):
        processor = AutoProcessor.from_pretrained(str(snapshot), **common)
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot), torch_dtype=dtype, attn_implementation="eager", **common
        ).to(device).eval()
    return torch, processor, model, device, dtype


def infer(torch, processor, model, device, dtype, image, task):
    inputs = processor(text=task, images=image, return_tensors="pt").to(device, dtype)
    with torch.inference_mode():
        tokens = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=512, num_beams=3, do_sample=False,
        )
    raw = processor.batch_decode(tokens, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(raw, task=task, image_size=image.size)
    return parsed[task]


def detect_image(torch, processor, model, device, dtype, image_path):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    caption = infer(torch, processor, model, device, dtype, image, "<MORE_DETAILED_CAPTION>")
    objects = infer(torch, processor, model, device, dtype, image, "<OD>")
    labels = [str(label).strip() for label in objects.get("labels", [])]
    object_text = "; ".join(labels)
    caption_text = str(caption)
    detected_text = f"{caption_text}; {object_text}"
    has_face = any(FACE_PATTERN.search(label) for label in labels)
    forbidden = sorted({match.group(0).lower() for match in FORBIDDEN_PATTERN.finditer(detected_text)})
    reasons = []
    if not has_face:
        reasons.append("no_face_detection")
    if forbidden:
        reasons.append("forbidden_attribute:" + ",".join(forbidden))
    return {
        "image_sha256": digest,
        "caption": caption_text,
        "object_labels": labels,
        "face_detected": has_face,
        "forbidden_attributes": forbidden,
        "image_passes": not reasons,
        "rejection_reasons": reasons,
    }


def build_grid(run, group, seed, alphas, detections, accepted_alphas, output):
    cell = 320
    header = 46
    columns = 5
    entries = [(None, run / "images" / group / "baseline" / f"seed{seed:03d}.png")]
    entries.extend((alpha, run / "images" / group /
                    f"seed{seed:03d}_alpha_{slug(alpha)}.png") for alpha in alphas)
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + header)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (alpha, path) in enumerate(entries):
        row, col = divmod(index, columns)
        x, y = col * cell, row * (cell + header)
        if alpha is None:
            label = "source baseline"
        else:
            decision = detections[(group, seed, alpha)]
            if alpha in accepted_alphas:
                label = f"alpha {alpha:+g} · ACCEPT"
            elif decision["image_passes"]:
                label = f"alpha {alpha:+g} · fails on other images"
            else:
                reason = ", ".join(decision["rejection_reasons"])
                label = f"alpha {alpha:+g} · reject: {reason}"
        draw.text((x + 5, y + 5), label[:48], fill="black")
        with Image.open(path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((cell - 8, cell - header - 8))
            sheet.paste(tile, (x + (cell - tile.width) // 2, y + header))
        if alpha in accepted_alphas:
            draw.rectangle((x + 2, y + header - 2,
                            x + cell - 3, y + cell + header - 3),
                           outline="#e00000", width=5)
    sheet.save(output, quality=92, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "run_biased_002",
                        help="completed feature 138 sweep directory")
    parser.add_argument("--groups", nargs="+", default=["middle_eastern_woman"],
                        help="groups to screen; multiplier acceptance is across all selected images")
    parser.add_argument("--samples-per-group", type=int, default=20,
                        help="lowest seed images to screen per group")
    parser.add_argument("--alphas", nargs="+", type=int,
                        help="optional subset of saved multipliers to screen")
    parser.add_argument("--max-failures", type=int, default=0,
                        help="maximum failed images allowed for a multiplier to pass")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path,
                        help="output directory (defaults to RUN/candidate_screen)")
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else run / "candidate_screen")
    if args.samples_per_group < 1 or args.max_failures < 0:
        parser.error("samples-per-group must be positive and max-failures nonnegative")
    settings = json.loads((run / "settings.json").read_text())
    saved_alphas = [int(value) for value in settings["alphas"]]
    if args.alphas is None:
        alphas = saved_alphas
    else:
        alphas = list(dict.fromkeys(args.alphas))
        missing_alphas = sorted(set(alphas) - set(saved_alphas))
        if missing_alphas:
            parser.error(f"requested multipliers are not saved in this run: {missing_alphas}")
    samples = {}
    for group in args.groups:
        image_dir = run / "images" / group
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        found = {}
        for path in image_dir.glob("seed*_alpha_*.png"):
            match = ALPHA_PATTERN.match(path.name)
            if match:
                found.setdefault(int(match.group("seed")), {})[
                    alpha_from_slug(match.group("slug"))] = path
        seeds = sorted(found)[:args.samples_per_group]
        if len(seeds) < args.samples_per_group:
            raise ValueError(f"{group}: found {len(seeds)} seeds, expected {args.samples_per_group}")
        for seed in seeds:
            missing = set(alphas) - set(found[seed])
            if missing:
                raise FileNotFoundError(f"Missing {group} seed {seed} multipliers: {sorted(missing)}")
        samples[group] = {seed: found[seed] for seed in seeds}

    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "detections"
    cache_dir.mkdir(exist_ok=True)
    torch, processor, model, device, dtype = load_florence(args.device)
    detections = {}
    rows = []
    total = sum(len(seeds) * len(alphas) for seeds in samples.values())
    completed = 0
    for group, seeds in samples.items():
        for seed, alpha_paths in seeds.items():
            for alpha in alphas:
                image_path = alpha_paths[alpha]
                cache_path = cache_dir / f"{group}_seed{seed:03d}_alpha_{slug(alpha)}.json"
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if cache_path.exists():
                    cached = json.loads(cache_path.read_text())
                    result = cached.get("result") if cached.get("image_sha256") == digest else None
                else:
                    result = None
                if result is None:
                    result = detect_image(torch, processor, model, device, dtype, image_path)
                    cache_path.write_text(json.dumps({"image_sha256": digest, "result": result}, indent=2) + "\n")
                detections[(group, seed, alpha)] = result
                rows.append({
                    "group": group, "seed": seed, "alpha": alpha,
                    "image": str(image_path.relative_to(run)),
                    "face_detected": result["face_detected"],
                    "forbidden_attributes": "; ".join(result["forbidden_attributes"]),
                    "image_passes": result["image_passes"],
                    "rejection_reasons": "; ".join(result["rejection_reasons"]),
                    "caption": result["caption"],
                    "object_labels": "; ".join(result["object_labels"]),
                })
                completed += 1
                print(f"Screened {completed}/{total}: {group} seed={seed} alpha={alpha:+g}", flush=True)

    summary = []
    accepted_alphas = set()
    for alpha in alphas:
        alpha_rows = [row for row in rows if int(row["alpha"]) == alpha]
        failures = sum(not row["image_passes"] for row in alpha_rows)
        accepted = failures <= args.max_failures
        if accepted:
            accepted_alphas.add(alpha)
        summary.append({
            "alpha": alpha,
            "images_screened": len(alpha_rows),
            "images_passing": len(alpha_rows) - failures,
            "images_rejected": failures,
            "max_failures_allowed": args.max_failures,
            "accepted": accepted,
        })

    write_csv(output / "candidate_screen.csv", rows)
    write_csv(output / "candidate_summary.csv", summary)
    grids = output / "grids"
    grids.mkdir(exist_ok=True)
    for group, seeds in samples.items():
        for seed in seeds:
            build_grid(run, group, seed, alphas, detections, accepted_alphas,
                       grids / f"{group}_seed{seed:03d}.jpg")

    result = {
        "detector": MODEL,
        "detector_revision": MODEL_REVISION,
        "criterion": "Image passes if Florence-2 <OD> detects a human face and neither its detailed caption nor object labels mention a configured forbidden attribute.",
        "forbidden_attributes": ["earrings", "turban", "headscarf/scarf", "hijab", "veil", "head covering/wrap", "dupatta"],
        "groups": args.groups,
        "samples_per_group": args.samples_per_group,
        "alphas_screened": alphas,
        "max_failures_allowed_per_multiplier": args.max_failures,
        "accepted_multipliers": sorted(accepted_alphas),
        "interpretation": "Automated screening only. Florence-2 outputs are unverified predictions; a pass is not a quality or safety guarantee. Every alpha should receive human review before operational use.",
    }
    (output / "settings.json").write_text(json.dumps(result, indent=2) + "\n")
    report = [
        "# Feature 138 multiplier screen", "",
        "This screen uses the pinned local Florence-2 base fine-tuned checkpoint. An image passes when the object-detection task returns a face label and neither the model's detailed caption nor object labels mention an excluded attribute. A multiplier passes when its rejected-image count is at most the configured failure allowance.",
        "", "Florence-2 predictions are heuristic and can miss or hallucinate details. A pass is a screening result, not proof of image quality. Red borders in the grids mark multipliers that passed across every selected image (or within the configured failure allowance).", "",
        f"Groups: {', '.join(args.groups)}", f"Images per group: {args.samples_per_group}",
        f"Maximum failures per multiplier: {args.max_failures}",
        f"Accepted multipliers: {', '.join(f'{a:+g}' for a in sorted(accepted_alphas)) if accepted_alphas else 'none'}", "",
        "| Alpha | Images screened | Pass | Rejected | Accepted |", "|---:|---:|---:|---:|:---:|",
    ]
    report.extend(
        f"| {row['alpha']:+g} | {row['images_screened']} | {row['images_passing']} | {row['images_rejected']} | {'yes' if row['accepted'] else 'no'} |"
        for row in summary
    )
    report += ["", "## Grids", ""]
    for group, seeds in samples.items():
        report += [f"### {group}", ""]
        for seed in seeds:
            report.append(f"![{group} seed {seed} multiplier screen](grids/{group}_seed{seed:03d}.jpg)")
            report.append("")
    (output / "REPORT.md").write_text("\n".join(report))
    print(f"Accepted multipliers: {sorted(accepted_alphas)}", flush=True)
    print(f"Wrote candidate screen to {output}", flush=True)


if __name__ == "__main__":
    main()
