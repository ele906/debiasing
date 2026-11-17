#!/usr/bin/env python3
"""Run the FaceParsing repository's class-14 Beard head on saved pilot images.

This inference-only script does not generate or modify source images. It writes
binary Beard-class masks, overlays, and per-image class-area summaries.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn


DEFAULT_REPO = Path("/n/fs/goose/face-parsing-beard-class")
DEFAULT_INPUT = Path(
    "/n/fs/goose/instance-segmentation/results/feature_4228/"
    "beard_annotation_pilot/images"
)
DEFAULT_OUTPUT = Path(
    "/n/fs/goose/experiments/mid0_sae_selectivity/faceparser_beard_pilot"
)
DINO_NAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DECODER_NAME = "decoder.ckpt"
BEARD_CLASS_ID = 14  # FaceParsing README's published FaceSynthetics class map.
FACE_CLASS_IDS = tuple(range(1, 15))
PILOT_STEMS = (
    "indian_man__seed030__alpha_m7",
    "indian_man__seed030__alpha_p1",
    "indian_man__seed030__alpha_p7",
    "middle_eastern_man__seed030__alpha_m7",
    "middle_eastern_man__seed030__alpha_p1",
    "middle_eastern_man__seed030__alpha_p7",
)


def load_model(repo: Path, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(repo))

    def import_file(module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # Import these two modules directly to avoid importing the repository's
    # Lightning training wrapper, which is not needed for inference.
    encoder_module = import_file("faceparser_encoder", repo / "src/models/encoder.py")
    decoder_module = import_file("faceparser_decoder", repo / "src/models/decoder.py")
    Decoder = decoder_module.Decoder

    # The published decoder checkpoint contains the trained VGG encoder weights.
    # Avoid an unrelated ImageNet download while constructing the module; the
    # checkpoint weights are loaded immediately below.
    original_vgg = encoder_module.VGG19

    class VGG19FromCheckpoint(original_vgg):
        def __init__(self, pretrained: bool = False) -> None:
            super().__init__(pretrained=False)

    encoder_module.VGG19 = VGG19FromCheckpoint

    dino_path = repo / "checkpoints" / DINO_NAME
    decoder_path = repo / "experiments" / "checkpoints" / DECODER_NAME
    if not dino_path.is_file() or dino_path.stat().st_size < 100_000_000:
        raise FileNotFoundError(f"Missing or incomplete DINOv3 checkpoint: {dino_path}")
    if not decoder_path.is_file() or decoder_path.stat().st_size < 100_000_000:
        raise FileNotFoundError(f"Missing or incomplete parser decoder checkpoint: {decoder_path}")

    dino = torch.hub.load(
        str(repo / "src" / "models"),
        "dinov3_vitb16",
        source="local",
        weights=str(dino_path),
    )
    dino.eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)

    class InferenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder_module.Encoder(dinov3_weights=dino, device=device)
            self.decoder = Decoder(num_classes=19)

        def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
            return self.decoder(self.encoder(batch))

    model = InferenceModel()
    checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    # The parser decoder file intentionally omits DINOv3 weights; these are
    # loaded independently above from Meta's checkpoint.
    state = {key: value for key, value in state.items() if "encoder.dinov3" not in key}
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_dino = [key for key in incompatible.missing_keys if "encoder.dinov3" not in key]
    if unexpected or missing_non_dino:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"Unexpected keys={unexpected[:10]}, missing non-DINO keys={missing_non_dino[:10]}"
        )
    model.to(device).eval()
    print(
        f"Loaded decoder checkpoint; omitted DINO keys={len(incompatible.missing_keys) - len(missing_non_dino)}"
    )
    return model


def preprocess(image: Image.Image, device: torch.device) -> torch.Tensor:
    resized = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)


def make_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGB").resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR)
    rgb = np.asarray(base).copy()
    tint = np.zeros_like(rgb)
    tint[:] = (255, 40, 180)
    selected = mask == BEARD_CLASS_ID
    rgb[selected] = (0.45 * rgb[selected] + 0.55 * tint[selected]).astype(np.uint8)
    return Image.fromarray(rgb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    paths = [args.input_dir / f"{stem}.png" for stem in PILOT_STEMS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing expected saved pilot images:\n" + "\n".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    overlays_dir = args.output_dir / "overlays"
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    model = load_model(args.repo, device)
    rows: list[dict[str, object]] = []
    contact_tiles: list[tuple[str, Image.Image]] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        with torch.inference_mode():
            output = model(preprocess(image, device))
            probabilities = output["probabilities"][0]
            mask = probabilities.argmax(dim=0).to("cpu").numpy().astype(np.uint8)

        class_counts = np.bincount(mask.ravel(), minlength=19)
        beard_pixels = int(class_counts[BEARD_CLASS_ID])
        face_pixels = int(class_counts[list(FACE_CLASS_IDS)].sum())
        stem = path.stem
        Image.fromarray((mask == BEARD_CLASS_ID).astype(np.uint8) * 255).save(
            masks_dir / f"{stem}_class14_beard.png"
        )
        overlay = make_overlay(image, mask)
        overlay.save(overlays_dir / f"{stem}_class14_beard_overlay.png")

        row: dict[str, object] = {
            "image": path.name,
            "prompt_group": "middle_eastern_man" if stem.startswith("middle_eastern_man") else "indian_man",
            "seed": 30,
            "multiplier_label": stem.rsplit("_", 1)[-1],
            "beard_class_id": BEARD_CLASS_ID,
            "beard_pixels": beard_pixels,
            "image_pixels": int(mask.size),
            "beard_fraction_image": beard_pixels / int(mask.size),
            "face_classes_1_to_14_pixels": face_pixels,
            "beard_fraction_face_classes_1_to_14": beard_pixels / face_pixels if face_pixels else float("nan"),
        }
        row.update({f"class_{i}_pixels": int(n) for i, n in enumerate(class_counts)})
        rows.append(row)
        contact_tiles.append((stem, overlay))
        print(
            f"{path.name}: Beard pixels={beard_pixels}/{mask.size} "
            f"({100 * beard_pixels / mask.size:.2f}% of image)"
        )

    csv_path = args.output_dir / "beard_class_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # A compact 3-column by 2-row sheet, retaining the per-image overlays too.
    tile_w, tile_h, title_h = 512, 512, 34
    sheet = Image.new("RGB", (3 * tile_w, 2 * (tile_h + title_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (stem, overlay) in enumerate(contact_tiles):
        x = (i % 3) * tile_w
        y = (i // 3) * (tile_h + title_h)
        draw.text((x + 8, y + 8), stem, fill="black", font=ImageFont.load_default())
        sheet.paste(overlay, (x, y + title_h))
    sheet.save(args.output_dir / "beard_class_contact_sheet.jpg", quality=95)

    (args.output_dir / "README.txt").write_text(
        "Beard-class face-parsing pilot\n"
        "================================\n\n"
        "Model: jseobyun/FaceParsing, decoder.ckpt plus the supplied official DINOv3 ViT-B/16 LVD-1689M backbone.\n"
        "Class selection: class index 14, as listed as Beard in the parser README.\n"
        "Inputs: the six already-saved feature-4228 pilot images for Indian-man and Middle-Eastern-man prompt groups, seed 30, multipliers -7, +1, +7. No diffusion generation was run.\n"
        "Outputs: masks/ contains binary class-14 masks; overlays/ contains magenta overlays; beard_class_metrics.csv contains per-image pixel counts and all-class counts; beard_class_contact_sheet.jpg shows the six overlays.\n\n"
        "Caution: these are unvalidated model predictions, not ground-truth beard annotations. The upstream repository's README names class 14 Beard, but its visualization helper contains a conflicting class list; inspect the saved overlays before interpreting area changes.\n"
    )
    print(f"Saved pilot results to {args.output_dir}")


if __name__ == "__main__":
    main()
