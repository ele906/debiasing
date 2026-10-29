"""Render the frozen layerwise selections as JET maps and signed feature sweeps.

Defaults reproduce the 48 cells in results/layerwise_group_effects_jet.png.
Multiplier 1 is unchanged; 0 removes a contribution; negatives reverse it.
No model, SAE, or feature selection is trained on the displayed held-out seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sae-training"))
sys.path.insert(0, str(ROOT / "sdxl-unbox"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps

LAYERS = {
    "unet_down_blocks_2_attentions_1": "unet.down_blocks.2.attentions.1",
    "unet_mid_block_attentions_0": "unet.mid_block.attentions.0",
    "unet_up_blocks_0_attentions_0": "unet.up_blocks.0.attentions.0",
    "unet_up_blocks_0_attentions_1": "unet.up_blocks.0.attentions.1",
}
SHORT = dict(zip(LAYERS, ["down.2.1", "mid.0", "up.0.0", "up.0.1"]))
DEFAULT_STRENGTHS = [-5, -3, -2, -1, 0, 1, 2, 3, 5]


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def slug(group):
    return group.replace(" ", "_")


def get_font(size=17):
    for path in ["/usr/share/fonts/dejavu-sans/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def selected_rows(run, top_k):
    frozen = read_csv(run / "results/layerwise_group_effects.csv")
    result = []
    for row in frozen:
        result.append({**row, "rank": 1})
    if top_k == 1:
        return result
    # Optional extension: use exactly the source plot's raw face-mean contrast,
    # paired by seed and same gender, without selecting on held-out responses.
    result = []
    for layer in LAYERS:
        with np.load(run / f"layer_{layer}.npz", allow_pickle=False) as data:
            x = data["X_face_mean"]
            races, genders, seeds = data["race_prompt"], data["gender_prompt"], data["seed"]
            for row in (r for r in frozen if r["layer"] == layer):
                race, gender = row["target_group"].rsplit(" ", 1)
                contrasts = []
                for seed in range(100, 150):
                    target = np.flatnonzero((races == race) & (genders == gender) & (seeds == seed))
                    other = np.flatnonzero((races != race) & (genders == gender) & (seeds == seed))
                    if len(target) == 1 and len(other) == 5:
                        contrasts.append(x[target[0]] - np.nanmean(x[other], axis=0))
                scores = np.nanmean(contrasts, axis=0)
                ranked = np.argsort(-np.where(np.isfinite(scores), scores, -np.inf))[:top_k]
                if int(ranked[0]) != int(row["feature_id_layer_specific"]):
                    raise ValueError(f"Discovery ranking differs from frozen plot: {layer}, {race} {gender}")
                for rank, feature in enumerate(ranked, 1):
                    result.append({"layer": layer, "target_group": row["target_group"],
                                   "feature_id_layer_specific": int(feature), "rank": rank,
                                   "discovery_mean_contrast": float(scores[feature])})
    return result


def write_ranked_feature_table(output, selections, top_k):
    """Make a concise GitHub-readable source table for the discovery ranking."""
    lines = [
        "# Top discovery features by layer and prompt group",
        "",
        f"This table lists the top {top_k} features selected using **only discovery seeds 100–149**.",
        "The later signed image sweeps use held-out seed(s), so held-out images do not choose these IDs.",
        "",
        "`contrast` is the mean feature activation in the target prompt group minus the",
        "same-seed average over the other five race prompts of the same gender, within the",
        "face box. It is a raw SAE-activation difference, not a semantic label, fairness",
        "score, or image-quality score.",
        "",
    ]
    for layer in LAYERS:
        rows = [row for row in selections if row["layer"] == layer]
        lines.extend([f"## {SHORT[layer]} (`{LAYERS[layer]}`)", "",
                      "| Prompt group | Rank 1 | Rank 2 | Rank 3 |", "|---|---:|---:|---:|"])
        groups = list(dict.fromkeys(row["target_group"] for row in rows))
        for group in groups:
            ranked = sorted((row for row in rows if row["target_group"] == group),
                            key=lambda row: int(row["rank"]))
            cells = []
            for row in ranked:
                cells.append(f"F{int(row['feature_id_layer_specific'])} ({float(row['discovery_mean_contrast']):+.3f})")
            cells += ["—"] * (3 - len(cells))
            lines.append(f"| {group} | " + " | ".join(cells[:3]) + " |")
        lines.extend([""])
    lines.extend([
        "## How to use the galleries",
        "",
        "Each feature has an `activation_jet.jpg`, `signed_sweep.jpg`, and",
        "`signed_coefficient_jet.jpg`. The sweep uses multipliers −5, −3, −2, −1,",
        "0, 1, 2, 3, 5. `s = 1` is unchanged; `s = 0` removes the original feature",
        "contribution; negative values reverse it. JET color runs blue → cyan → green →",
        "yellow → orange → red. Read `REPORT.md` for the intervention equation and",
        "interpretation limits.",
        "",
    ])
    (output / "TOP_FEATURES_BY_LAYER.md").write_text("\n".join(lines))


def signed_hook(sae, feature, strength, stats):
    import torch

    @torch.no_grad()
    def hook(_module, inputs, output):
        original = output[0]
        residual = (original - inputs[0]).permute(0, 2, 3, 1).float()
        codes = sae.encode(residual)[..., feature]
        delta = ((strength - 1) * codes[..., None] * sae.decoder.weight[:, feature]).permute(0, 3, 1, 2)
        if not torch.isfinite(delta).all():
            raise ValueError("Nonfinite intervention delta")
        stats.append({"delta_rms": delta.square().mean().sqrt().item(),
                      "delta_max_abs": delta.abs().max().item()})
        if strength == 1:
            return output
        return (original + delta.to(original.dtype),) + output[1:]
    return hook


def jet_overlay(image, values, scale, *, signed=False, bound=5.0):
    """Nearest-neighbor tiles; inactive coefficients transparent; shared scale."""
    normalized = np.clip(values / max(scale, 1e-8), -bound if signed else 0, bound if signed else 1)
    color_position = (normalized / bound + 1) / 2 if signed else normalized
    colors = np.uint8(colormaps["jet"](color_position)[..., :3] * 255)
    color_image = Image.fromarray(colors).resize(image.size, Image.Resampling.NEAREST)
    # Constant blend opacity for active cells makes equal colors mean equal coefficients.
    alpha = Image.fromarray(np.uint8((np.abs(values) > 0) * 150)).resize(image.size, Image.Resampling.NEAREST)
    return Image.composite(color_image, image, alpha)


def heat_image(values, scale):
    colors = np.uint8(colormaps["jet"](np.clip(values / max(scale, 1e-8), 0, 1))[..., :3] * 255)
    return Image.fromarray(colors).resize((512, 512), Image.Resampling.NEAREST)


def contact_sheet(title, rows, output, subtitle="", tile=184):
    columns = max(len(items) for _, items in rows)
    left, top, label = 86, 92, 42
    sheet = Image.new("RGB", (left + columns * tile, top + len(rows) * (tile + label) + 28), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill="#13283d", font=get_font(22))
    draw.text((12, 44), subtitle, fill="#42556a", font=get_font(14))
    for i, (row_label, items) in enumerate(rows):
        y = top + i * (tile + label)
        draw.text((8, y + tile // 2), row_label, fill="#13283d", font=get_font(14))
        for j, (caption, image) in enumerate(items):
            x = left + j * tile
            draw.text((x + 4, y + 5), caption, fill="#13283d", font=get_font(14))
            rendered = image.resize((tile - 6, tile - 6), Image.Resampling.LANCZOS)
            sheet.paste(rendered, (x + 3, y + label))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=89, optimize=True)


def render_galleries(output, selections, examples, maps, strengths, seeds):
    scale = {}
    for key, array in maps.items():
        layer, group, feature, seed = key
        scale[(layer, feature)] = max(scale.get((layer, feature), 0), float(array.max()))
    scale_rows = []
    for (layer, feature), maximum in scale.items():
        scale_rows.append({"layer": layer, "feature": feature, "baseline_max_across_displayed_samples": maximum,
                           "signed_color_bound_multiplier": max(abs(s) for s in strengths)})
    write_csv(output / "overlay_scales.csv", scale_rows)
    groups = list(dict.fromkeys(s["target_group"] for s in selections))
    index = ["# Layerwise signed feature sweeps and JET overlays", "",
             "These visualize the exact layer/group selections in [the source plot](../layerwise_group_effects_jet.png).",
             "", f"Held-out seeds: {', '.join(map(str, seeds))}. Multipliers: {', '.join(f'{s:g}' for s in strengths)}.",
             "", "**1 = original, 0 = remove, negative = reverse the selected contribution.** Each edit uses `output + (s - 1) * z_f * decoder_direction_f` and preserves reconstruction error.",
             "", "The original-code overlay is nonnegative. Signed overlays show `s * original z_f` over edited images, not re-encoded activations or pixel-level causal attribution. All maps are coarse spatial SAE cells resized with nearest-neighbor (pixelated) rendering; zero cells are transparent.",
             "", "Color scale is shared across displayed seeds/groups for each layer-specific feature. It is not shared across different features or layers. See `overlay_scales.csv`. The signed palette spans minus to plus the largest displayed multiplier times that feature scale.",
             "", "The source experiment ranks **one feature per layer/group cell** (48 selections). The script also supports `--top-k` to expand the discovery ranking. Top-K SAE sparsity remains 10; it is different from the number of ranked features visualized.",
             "", "Prompt labels are generation conditions, not inferred human identity. These are exploratory visual interventions; no bias reduction or image-quality improvement is established.", "",
             "See [top discovery features by layer and prompt group](TOP_FEATURES_BY_LAYER.md).",
             "",
             "![Baseline JET overview](layerwise_feature_activation_overview_jet.jpg)", ""]
    for layer in LAYERS:
        if not any(s["layer"] == layer for s in selections):
            continue
        layer_index = [f"# {SHORT[layer]} · layer-specific feature sweeps", "",
                       "[Back to experiment index](../README.md)", "",
                       "Rows use the same held-out seeds in every panel. Multipliers are signed feature coefficients, not additive constants.", ""]
        for selection in (s for s in selections if s["layer"] == layer):
            group, f = selection["target_group"], int(selection["feature_id_layer_specific"])
            directory = output / layer / f"{slug(group)}_feature{f}"
            directory.mkdir(parents=True, exist_ok=True)
            base_rows, sweep_rows, signed_rows = [], [], []
            for seed in seeds:
                original = examples[(group, seed)]
                values = maps[(layer, group, f, seed)]
                maximum = scale[(layer, f)]
                overlay = jet_overlay(original, values, maximum)
                base_rows.append((f"seed {seed}", [("original", original), ("activation JET", heat_image(values, maximum)), ("JET overlay", overlay)]))
                panels = [("original", original), ("baseline JET", overlay)]
                signed_panels = []
                for strength in strengths:
                    filename = output / "raw/images" / layer / slug(group) / f"seed{seed}_feature{f}_s{strength:g}.png"
                    with Image.open(filename) as source:
                        edited = source.convert("RGB")
                    panels.append((f"s = {strength:g}", edited))
                    signed_panels.append((f"s = {strength:g}", jet_overlay(edited, strength * values, maximum,
                        signed=True, bound=max(abs(s) for s in strengths))))
                sweep_rows.append((f"seed {seed}", panels))
                signed_rows.append((f"seed {seed}", signed_panels))
            title = f"{SHORT[layer]} | {group} | feature {f} (rank {selection['rank']})"
            contact_sheet(title, base_rows, directory / "activation_jet.jpg",
                          "Original nonnegative feature activation | shared feature scale across displayed examples", tile=240)
            contact_sheet(title, sweep_rows, directory / "signed_sweep.jpg",
                          "Signed multiplier sweep | s=1 unchanged; s=0 removes; negative values reverse the contribution")
            contact_sheet(title, signed_rows, directory / "signed_coefficient_jet.jpg",
                          "JET of applied s * ORIGINAL activation over edited image | fixed signed color scale | not re-encoded activity")
            relative = directory.relative_to(output / layer)
            layer_index.extend([f"## {group} · F{f} · rank {selection['rank']}", "",
                f"![Baseline activation]({relative}/activation_jet.jpg)", "",
                f"![Signed image sweep]({relative}/signed_sweep.jpg)", "",
                f"![Signed coefficient overlay]({relative}/signed_coefficient_jet.jpg)", ""])
        (output / layer / "README.md").write_text("\n".join(layer_index))
        index.extend([f"- [{SHORT[layer]}: all selected features and signed sweeps]({layer}/README.md)"])
    # Match the 4x12 organization of the source figure using the first fixed evaluation seed.
    tile, row_height, left, top = 190, 286, 115, 82
    overview = Image.new("RGB", (left + len(groups)*tile, top + len(LAYERS)*row_height), "white")
    draw = ImageDraw.Draw(overview)
    draw.text((12, 10), f"Layerwise selected-feature JET overlays | held-out seed {seeds[0]}", fill="#13283d", font=get_font(24))
    draw.text((12, 44), "Same cells as the source contrast plot. Spatial activation overlay, not a semantic label or a causal pixel attribution.", fill="#42556a", font=get_font(17))
    for i, layer in enumerate(LAYERS):
        y = top + i*row_height
        draw.text((8, y+120), SHORT[layer], fill="#13283d", font=get_font(17))
        for j, group in enumerate(groups):
            match = next((s for s in selections if s["layer"] == layer and s["target_group"] == group and s["rank"] == 1), None)
            if match is None:
                continue
            f = int(match["feature_id_layer_specific"])
            x = left + j*tile
            draw.text((x+3, y+3), group.replace(" ", "\n", 1), fill="#13283d", font=get_font(14))
            draw.text((x+3, y+43), f"F{f}", fill="#087e8b", font=get_font(17))
            image = jet_overlay(examples[(group,seeds[0])], maps[(layer,group,f,seeds[0])], scale[(layer,f)])
            overview.paste(image.resize((tile-8,tile-8),Image.Resampling.LANCZOS),(x+4,y+73))
    overview.save(output / "layerwise_feature_activation_overview_jet.jpg", quality=92, optimize=True)
    index.extend(["", "## Verification and data", "",
        "- `verification.json`: source hash, checkpoint hashes, exact replay/no-op checks, cached pooling agreement.",
        "- `selected_features.csv`: frozen source selections (or discovery-only top-k extension).",
        "- `intervention_metrics.csv`: signed multiplier and image/delta measurements for every edit.",
        "- `visual_review.csv`: blank covering/hair/clothing/face/artifact review fields.",
        "- `overlay_scales.csv`: numeric normalization for each layer-specific feature.",
        "", "Full-resolution PNGs and raw arrays are local under `raw/` (git-ignored); compact JPEG visualizations and numeric metrics are committed.", ""])
    (output / "README.md").write_text("\n".join(index))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "layers/runs/turbo_layer_depth_001")
    p.add_argument("--output", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[150, 151, 152])
    p.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    p.add_argument("--top-k", type=int, default=1, help="1 uses exactly the frozen figure; >1 expands discovery ranking")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "ReNO/hf_cache")
    args = p.parse_args()
    if not args.seeds or any(seed < 150 or seed > 199 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        p.error("Use distinct held-out seeds from 150 through 199")
    if not 1 <= args.top_k <= 5120 or len(set(args.strengths)) != len(args.strengths) or not np.isfinite(args.strengths).all():
        p.error("Invalid top-k or multipliers")
    if 1 not in args.strengths or max(abs(s) for s in args.strengths) == 0:
        p.error("Include strength 1 for the no-op control")
    run = args.run.resolve()
    output = (args.output or run / "results/feature_visuals_signed").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitignore").write_text("# Reproducible full-resolution PNGs and numeric spatial maps stay local.\nraw/\n")
    (output / "raw").mkdir()
    settings = json.loads((run / "settings.json").read_text())
    selections = selected_rows(run, args.top_k)
    if args.top_k == 1 and len(selections) != 48:
        raise ValueError("Expected 48 frozen layer/group selections")
    write_csv(output / "selected_features.csv", selections)
    write_ranked_feature_table(output, selections, args.top_k)
    manifest = read_csv(run / "manifest.csv")
    source_root = Path(settings["image_source"]).parent
    rows = [r for r in manifest if int(r["seed"]) in args.seeds]
    if len(rows) != 12 * len(args.seeds):
        raise ValueError("Missing balanced held-out samples")
    (output / "settings.json").write_text(json.dumps({
        "source_run": str(run), "source_plot": "../layerwise_group_effects_jet.png",
        "feature_selection": "frozen source CSV" if args.top_k == 1 else "discovery seeds 100-149 only",
        "top_k": args.top_k, "seeds": args.seeds, "strengths": args.strengths,
        "intervention": "output + (strength - 1) * original_SAE_activation * decoder_direction",
        "scope": "one layer and feature per generation; full spatial feature map (not face-masked)",
        "jet_overlay": "nearest-neighbor; zero coefficients transparent; feature-wise common scale",
        "labels": "generation-prompt conditions, not inferred human identities",
        "quantitative_outcome": "pixel changes are not image-quality or fairness measures",
    }, indent=2) + "\n")
    import torch
    from capture_layerwise_face_sae import BLOCKS, face_boxes, face_mask
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    load = dict(torch_dtype=torch.float16, cache_dir=str(args.cache_dir), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings["vae"], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings["model"], vae=vae, variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    saes, hashes = {}, {}
    for layer, block in LAYERS.items():
        path = ROOT / "sdxl-unbox/checkpoints" / BLOCKS[block]
        hashes[block] = digest(path / "state_dict.pth")
        if hashes[block] != settings["sae_weight_sha256"][block]:
            raise ValueError(f"SAE checkpoint differs from source run: {block}")
        saes[layer] = SparseAutoencoder.load_from_disk(str(path)).float().cuda().eval()
        saes[layer].requires_grad_(False)
    boxes = face_boxes(Path(settings["face_box_source"]))

    def generation_args(row):
        seed = int(row["seed"])
        noise_rng = torch.Generator("cuda").manual_seed(seed)
        scheduler_rng = torch.Generator("cuda").manual_seed(seed)
        latents = torch.randn((1, 4, 64, 64), generator=noise_rng, device="cuda", dtype=torch.float16)
        return dict(prompt=row["prompt"], latents=latents, generator=scheduler_rng,
                    num_inference_steps=settings["steps"], guidance_scale=settings["guidance_scale"],
                    height=settings["height"], width=settings["width"])

    cached = {}
    for layer in LAYERS:
        with np.load(run / f"layer_{layer}.npz", allow_pickle=False) as data:
            cached[layer] = {"ids": {sid: i for i,sid in enumerate(data["sample_id"])},
                             "face": data["X_face_mean"], "global": data["X_global_mean"]}
    examples, maps, replay_errors, pooling_errors = {}, {}, [], []
    # First collect all unmodified maps to set common per-feature color scales.
    for i,row in enumerate(rows,1):
        group,seed = f"{row['race_prompt']} {row['gender_prompt']}",int(row["seed"])
        result, cache = pipe.run_with_cache(**generation_args(row), positions_to_cache=list(LAYERS.values()), save_input=True)
        original = result.images[0]
        with Image.open(source_root / row["image"]) as source:
            saved = source.convert("RGB")
        error = float(np.abs(np.asarray(original).astype(float)-np.asarray(saved).astype(float)).max())
        replay_errors.append(error)
        if error != 0:
            raise ValueError(f"Source image replay differs: {row['sample_id']}, max pixel error {error}")
        examples[(group,seed)] = original
        original_path = output / "raw/originals" / f"{row['sample_id']}.png"
        original_path.parent.mkdir(parents=True,exist_ok=True)
        original.save(original_path)
        for layer,block in LAYERS.items():
            residual = (cache["output"][block]-cache["input"][block])[:, -1].permute(0,2,3,1).float()
            z = saes[layer].encode(residual)[0]
            mask = torch.from_numpy(face_mask(boxes.get(row["sample_id"],[]),*z.shape[:2])).cuda()
            sample_index = cached[layer]["ids"][row["sample_id"]]
            for selection in (s for s in selections if s["layer"]==layer and s["target_group"]==group):
                f = int(selection["feature_id_layer_specific"])
                global_mean = z[...,f].mean().item()
                pool_error = abs(global_mean-float(cached[layer]["global"][sample_index,f]))
                if mask.any():
                    pool_error = max(pool_error,abs(z[...,f][mask].mean().item()-float(cached[layer]["face"][sample_index,f])))
                pooling_errors.append(pool_error)
                if pool_error > 1e-4:
                    raise ValueError(f"Source pooling mismatch: {layer}, {row['sample_id']}, F{f}: {pool_error}")
                array = z[...,f].cpu().numpy()
                maps[(layer,group,f,seed)] = array
                map_path = output / "raw/maps" / layer / f"{row['sample_id']}_feature{f}.npy"
                map_path.parent.mkdir(parents=True,exist_ok=True)
                np.save(map_path,array,allow_pickle=False)
            del z,residual
        del cache
        print(f"Replay and map checks {i}/{len(rows)}: {row['sample_id']}",flush=True)
    del cached
    metrics, review = [], []
    for i,selection in enumerate(selections,1):
        layer,group,f = selection["layer"],selection["target_group"],int(selection["feature_id_layer_specific"])
        for row in (r for r in rows if f"{r['race_prompt']} {r['gender_prompt']}"==group):
            seed=int(row["seed"])
            original=examples[(group,seed)]
            directory=output/"raw/images"/layer/slug(group)
            directory.mkdir(parents=True,exist_ok=True)
            values=maps[(layer,group,f,seed)]
            for strength in args.strengths:
                stats=[]
                image=pipe.run_with_hooks(**generation_args(row),position_hook_dict={LAYERS[layer]:signed_hook(saes[layer],f,strength,stats)}).images[0]
                if len(stats)!=1:
                    raise ValueError(f"Expected one hook call, got {len(stats)}")
                pixel_difference=np.abs(np.asarray(image).astype(float)-np.asarray(original).astype(float))
                pixel_error=float(pixel_difference.mean())
                if strength==1 and pixel_difference.max()!=0:
                    raise ValueError("No-op identity check failed")
                if values.max()==0 and pixel_difference.max()!=0:
                    raise ValueError("Inactive feature changed image")
                path=directory/f"seed{seed}_feature{f}_s{strength:g}.png"
                image.save(path)
                metrics.append(dict(layer=layer,target_group=group,feature=f,rank=selection["rank"],seed=seed,
                    multiplier=strength,pixel_mean_absolute_change=pixel_error,pixel_max_change=float(pixel_difference.max()),
                    baseline_feature_max=float(values.max()),baseline_active_fraction=float((values>0).mean()),
                    delta_rms=stats[0]["delta_rms"],delta_max_abs=stats[0]["delta_max_abs"],
                    source_replay_max_pixel_error=0,image=str(path.relative_to(output))))
                review.append(dict(layer=layer,target_group=group,feature=f,seed=seed,multiplier=strength,
                    unwanted_covering="",hair_change="",clothing_change="",face_detail_change="",artifacts="",preference="",notes=""))
        write_csv(output/"intervention_metrics.csv",metrics)
        print(f"Signed sweep {i}/{len(selections)}: {SHORT[layer]} | {group} | F{f}",flush=True)
    write_csv(output/"visual_review.csv",review)
    render_galleries(output,selections,examples,maps,args.strengths,args.seeds)
    verification={"source_selection_csv_sha256":digest(run/"results/layerwise_group_effects.csv"),
        "source_settings_sha256":digest(run/"settings.json"),"sae_weight_sha256":hashes,
        "replay_samples":len(replay_errors),"max_source_replay_pixel_error":max(replay_errors),
        "max_cached_pooling_error":max(pooling_errors),"layer_group_selections":len(selections),
        "edits":len(metrics),"no_op_images":sum(m["multiplier"]==1 for m in metrics),
        "max_no_op_pixel_error":max(m["pixel_max_change"] for m in metrics if m["multiplier"]==1),
        "seeds":args.seeds,"multipliers":args.strengths,
        "device":torch.cuda.get_device_name(),"torch_version":torch.__version__}
    (output/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    print(json.dumps(verification,indent=2),flush=True)


if __name__=="__main__":
    main()
