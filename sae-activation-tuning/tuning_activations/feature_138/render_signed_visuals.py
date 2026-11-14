"""Render signed feature coefficient overlays and summaries from a saved run."""

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ.get("SAE_FEATURE_RUN", ROOT / "run_biased_002")).expanduser().resolve()
FACTORS = []
SEEDS = []
CONDITIONS = []


def slug(value):
    return f"m{abs(value)}" if value < 0 else f"p{value}"


def signed_rgb(values, negative_limit, positive_limit):
    """Negative: green to blue to purple; positive: yellow to orange to red."""
    x = np.asarray(values, dtype=np.float32)
    magnitude = np.zeros(x.shape, dtype=np.float32)
    rgb = np.zeros((*x.shape, 3), dtype=np.float32)
    neg = x < 0
    pos = x > 0
    magnitude[neg] = np.clip(-x[neg] / max(negative_limit, 1e-8), 0, 1)
    magnitude[pos] = np.clip(x[pos] / max(positive_limit, 1e-8), 0, 1)
    # Negative values move from green near zero to purple at high magnitude.
    green = np.array([0.05, 0.72, 0.18], dtype=np.float32)
    blue = np.array([0.08, 0.28, 0.92], dtype=np.float32)
    purple = np.array([0.63, 0.05, 0.78], dtype=np.float32)
    yellow = np.array([1.0, 0.95, 0.05], dtype=np.float32)
    orange = np.array([1.0, 0.42, 0.0], dtype=np.float32)
    red = np.array([0.95, 0.0, 0.02], dtype=np.float32)
    t = magnitude[..., None]
    neg_rgb = np.where(t <= 0.55,
                       green + (blue - green) * (t / 0.55),
                       blue + (purple - blue) * ((t - 0.55) / 0.45))
    pos_rgb = np.where(t <= 0.5,
                       yellow + (orange - yellow) * (t / 0.5),
                       orange + (red - orange) * ((t - 0.5) / 0.5))
    rgb[neg] = neg_rgb[neg]
    rgb[pos] = pos_rgb[pos]
    return (rgb * 255).astype(np.uint8), magnitude


def save_overlay(image_path, values, neg_limit, pos_limit, output_path):
    base = Image.open(image_path).convert("RGB")
    colors, magnitude = signed_rgb(values, neg_limit, pos_limit)
    # Use a gentle nonlinear ramp to make moderate values easier to see while
    # preserving transparency at zero and leaving the source image visible.
    alpha = np.clip(np.power(magnitude, 0.65) * 210, 0, 210).astype(np.uint8)
    layer = np.dstack((colors, alpha))
    heat = Image.fromarray(layer, "RGBA").resize(base.size, Image.Resampling.NEAREST)
    Image.alpha_composite(base.convert("RGBA"), heat).convert("RGB").save(output_path, quality=93, optimize=True)


def limits_for(values):
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    negative = np.abs(finite[finite < 0])
    positive = finite[finite > 0]
    return (max(float(np.percentile(negative, 99.5)) if negative.size else 0, 1e-6),
            max(float(np.percentile(positive, 99.5)) if positive.size else 0, 1e-6))


def get_image_path(group, seed, alpha=None):
    if alpha is None:
        stem = RUN / "images" / group / "baseline" / f"seed{seed:03d}"
        legacy = RUN / "images" / group / "baseline" / f"seed{seed}"
        if legacy.with_suffix(".jpg").exists():
            return legacy.with_suffix(".jpg")
    else:
        stem = RUN / "images" / group / f"seed{seed:03d}_alpha_{slug(alpha)}"
        legacy = RUN / "images" / group / f"seed{seed}_alpha_{slug(alpha)}.jpg"
        if legacy.exists():
            return legacy
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image for group={group}, seed={seed}, alpha={alpha}")


def main():
    global FACTORS, SEEDS, CONDITIONS
    maps = np.load(RUN / "activation_maps.npz")
    map_keys = [str(k) for k in maps["map_keys"]]
    factors = [int(a) for a in maps["multipliers"]]
    FACTORS = factors
    CONDITIONS = list(dict.fromkeys(k.split("|")[0] for k in map_keys))
    seeds_by_group = {group: [int(k.split("|")[1]) for k in map_keys if k.split("|")[0] == group]
                      for group in CONDITIONS}
    signed = maps["scaled_coefficient_maps"]
    allvals = signed[np.isfinite(signed)]
    neg_limit = max(float(np.percentile(np.abs(allvals[allvals < 0]), 99.5)) if np.any(allvals < 0) else 0, 1e-6)
    pos_limit = max(float(np.percentile(allvals[allvals > 0], 99.5)) if np.any(allvals > 0) else 0, 1e-6)

    for cond_index, condition in enumerate(CONDITIONS):
        for seed in seeds_by_group[condition]:
            map_index = map_keys.index(f"{condition}|{seed}")
            out_dir = RUN / condition / "signed_coefficient_overlays"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Include original as the first image, then all alphas in ascending order.
            image_paths = [get_image_path(condition, seed)]
            overlay_maps = [maps["pre_maps"][map_index]]
            for alpha_index, alpha in enumerate(FACTORS):
                image_paths.append(get_image_path(condition, seed, alpha))
                # Alpha zero removes the coefficient. Show its saved baseline map
                # on that edited image so the suppressed feature location is visible.
                overlay_maps.append(maps["pre_maps"][map_index] if alpha == 0
                                    else signed[map_index, alpha_index])
            tiles = []
            for idx, (image_path, values) in enumerate(zip(image_paths, overlay_maps)):
                temp = out_dir / f".tile_{condition}_{seed}_{idx}.jpg"
                tile_neg_limit, tile_pos_limit = limits_for(values)
                save_overlay(image_path, values, tile_neg_limit, tile_pos_limit, temp)
                tiles.append(("original" if idx == 0 else f"alpha {FACTORS[idx - 1]:+g}", temp))
            cell, top = 384, 30
            plain_dir = RUN / condition / "seed_sheets"
            plain_dir.mkdir(parents=True, exist_ok=True)
            plain_sheet = Image.new("RGB", (5 * cell, 4 * (cell + top)), "white")
            plain_draw = ImageDraw.Draw(plain_sheet)
            for i, (image_path, label) in enumerate(zip(image_paths,
                    ["original"] + [f"alpha {a:+g}" for a in FACTORS])):
                row, col = divmod(i, 5)
                x, y = col * cell, row * (cell + top)
                plain_draw.text((x + 6, y + 5), label, fill="black")
                tile = Image.open(image_path).convert("RGB")
                tile.thumbnail((cell - 10, cell - top - 8))
                plain_sheet.paste(tile, (x + (cell - tile.width) // 2, y + top))
            plain_sheet.save(plain_dir / f"{condition}_seed{seed}.jpg", quality=93, optimize=True)

            sheet = Image.new("RGB", (5 * cell, 4 * (cell + top)), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (label, tile_path) in enumerate(tiles):
                row, col = divmod(i, 5)
                x, y = col * cell, row * (cell + top)
                draw.text((x + 5, y + 4), label, fill="black")
                tile = Image.open(tile_path).convert("RGB")
                tile.thumbnail((cell - 8, cell - top - 6))
                sheet.paste(tile, (x + (cell - tile.width) // 2, y + top))
                tile_path.unlink()
            sheet.save(out_dir / f"{condition}_seed{seed}.jpg", quality=93, optimize=True)

            # Normalize maps per tile so the original and near-zero scales remain
            # legible alongside large alpha values. Alpha zero uses the baseline
            # map to mark the feature location that was suppressed.
            for folder, array_key in (
                ("post_activation_overlays", "post_reencoded_maps"),
                ("intervention_footprint_overlays", "intervention_footprint_maps"),
            ):
                array = maps[array_key]
                nonnegative_dir = RUN / condition / folder
                nonnegative_dir.mkdir(parents=True, exist_ok=True)
                tiles = []
                original_values = maps["pre_maps"][map_index]
                temp = nonnegative_dir / f".tile_{condition}_{seed}_original.jpg"
                _, vmax = limits_for(original_values)
                save_overlay(image_paths[0], original_values, vmax, vmax, temp)
                tiles.append(("original", temp))
                for alpha_index, alpha in enumerate(FACTORS):
                    temp = nonnegative_dir / f".tile_{condition}_{seed}_{alpha_index}.jpg"
                    values = (original_values if alpha == 0 and array_key == "post_reencoded_maps"
                              else array[map_index, alpha_index])
                    _, tile_vmax = limits_for(values)
                    save_overlay(image_paths[alpha_index + 1], values, tile_vmax, tile_vmax, temp)
                    tiles.append((f"alpha {alpha:+g}", temp))
                sheet = Image.new("RGB", (5 * cell, 4 * (cell + top)), "white")
                draw = ImageDraw.Draw(sheet)
                for i, (label, tile_path) in enumerate(tiles):
                    row, col = divmod(i, 5)
                    x, y = col * cell, row * (cell + top)
                    draw.text((x + 5, y + 4), label, fill="black")
                    tile = Image.open(tile_path).convert("RGB")
                    tile.thumbnail((cell - 8, cell - top - 6))
                    sheet.paste(tile, (x + (cell - tile.width) // 2, y + top))
                    tile_path.unlink()
                sheet.save(nonnegative_dir / f"{condition}_seed{seed}.jpg", quality=93, optimize=True)

        rows = list(csv.DictReader((RUN / "metrics.csv").open()))
        group_field = "group" if "group" in rows[0] else "condition"
        means = []
        for alpha in FACTORS:
            subset = [r for r in rows if r[group_field] == condition and int(r["alpha"]) == alpha]
            means.append({
                "mean_scaled_coefficient": float(np.mean([float(r["mean_scaled_coefficient"]) for r in subset])),
                "mean_abs_scaled_coefficient": float(np.mean([float(r["mean_abs_scaled_coefficient"]) for r in subset])),
                "mean_abs_intervention_coefficient_change": float(np.mean([float(r["mean_abs_intervention_coefficient_change"]) for r in subset])),
                "post_reencoded_mean_activation": float(np.mean([float(r["post_reencoded_mean_activation"]) for r in subset])),
                "pixel_mae_vs_original": float(np.mean([float(r["pixel_mae_vs_original"]) for r in subset])),
            })

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 1, figsize=(17, 17), constrained_layout=True)
        x = np.arange(len(FACTORS))
        signed_values = np.asarray([r["mean_scaled_coefficient"] for r in means])
        color_values, mags = signed_rgb(signed_values, neg_limit, pos_limit)
        axes[0].bar(x, signed_values, color=color_values / 255.0, edgecolor="white", linewidth=0.5)
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_title("Mean signed scaled coefficient (negative: green-blue-purple; positive: yellow-orange-red)")
        specs = [
            ("mean_abs_scaled_coefficient", "Mean absolute scaled coefficient"),
            ("mean_abs_intervention_coefficient_change", "Mean absolute coefficient change applied"),
            ("post_reencoded_mean_activation", "Mean post-intervention re-encoded activation"),
            ("pixel_mae_vs_original", "Mean RGB pixel MAE vs. original (alpha +1)"),
        ]
        pos_cols, _ = signed_rgb(np.asarray([1.0]), neg_limit, pos_limit)
        # Magnitude bars use the positive yellow-to-red scale.
        for axis, (key, title) in zip(axes[1:], specs):
            values = np.asarray([r[key] for r in means])
            denom = max(float(values.max()), 1e-8)
            cols, _ = signed_rgb(values / denom * pos_limit, neg_limit, pos_limit)
            axis.bar(x, values, color=cols / 255.0, edgecolor="white", linewidth=0.5)
            axis.set_title(title)
        for ax in axes:
            ax.set_xticks(x, [f"{a:+g}" for a in FACTORS], rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        fig.suptitle(f"Feature 138 — {condition.replace('_', ' ').title()}\nNegative: green to blue to purple; positive: yellow to orange to red; zero transparent; translucent overlays")
        graph_dir = RUN / condition / "activation_graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(graph_dir / f"{condition}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    settings_path = RUN / "settings.json"
    import json
    settings = json.loads(settings_path.read_text())
    settings["heatmaps"] = ("Signed scaled SAE feature coefficients: exact zero transparent; negative values transition green near zero through blue to purple at high magnitude; "
                            "positive values transition yellow near zero through orange to red at high magnitude. Overlay opacity scales with magnitude and remains translucent. "
                            "Post-re-encoded SAE activations remain nonnegative.")
    settings["signed_heatmap_scale"] = {"negative_abs_p99_5": neg_limit, "positive_p99_5": pos_limit}
    settings["overlay_visibility"] = ("Each heatmap tile uses its own 99.5th percentile color scale for visibility. Original tiles show pre-intervention activations. "
                                      "Alpha zero tiles show the pre-intervention map over the edited image to mark the suppressed feature location; "
                                      "the actual post-re-encoded activation at alpha zero is zero.")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Rendered signed overlays; negative p99.5={neg_limit:.4f}, positive p99.5={pos_limit:.4f}")


if __name__ == "__main__":
    main()
