"""Exploratory paired FaceParsing analysis for the visually screened feature 3742."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
ARCHIVE = ROOT / "experiments/mid0_sae_selectivity/beard_interventions_20261117_020843_219166"
sys.path.insert(0, str(ARCHIVE / "config/code"))
from infer import predict, helpers
from analyze import interval

FEATURE = 3742


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    output = RUN / "beard"
    masks = output / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    rows = read(RUN / f"full/feature_{FEATURE}/manifest.csv")
    model_settings = json.loads((ARCHIVE / "config/experiment_config.json").read_text())
    for path, expected in zip(model_settings["checkpoint_paths"], model_settings["checkpoint_hashes"]):
        if digest(Path(path)) != expected:
            raise ValueError(f"Parser checkpoint hash changed: {path}")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    model = helpers.load_model(Path(model_settings["parser_repo"]), device)
    predictions = {}
    for row in rows:
        source = RUN / row["image"]
        key = row["image_sha256"]
        if key in predictions:
            continue
        if digest(source) != key:
            raise ValueError(f"Image hash changed: {source}")
        path = masks / f"{key}.png"
        if path.exists():
            with Image.open(path) as image:
                mask = np.array(image)
        else:
            with Image.open(source) as image:
                mask, _, _ = predict(model, image.convert("RGB"), device)
            Image.fromarray(mask).save(path.with_suffix(".tmp.png"))
            path.with_suffix(".tmp.png").replace(path)
        predictions[key] = (int((mask == 14).sum()), int(np.isin(mask, [*range(1, 12), 14]).sum()))
    controls = {(r["group"], r["seed"]): r for r in rows if r["alpha"] == "1"}
    paired = []
    for row in rows:
        control = controls[row["group"], row["seed"]]
        beard, face = predictions[row["image_sha256"]]
        base_beard, base_face = predictions[control["image_sha256"]]
        fraction = base_beard / base_face if base_face else None
        paired.append(dict(group=row["group"], seed=int(row["seed"]), alpha=int(row["alpha"]),
                           applied_l2=float(row["delta_fro_applied"]), beard=beard, face=face,
                           baseline_beard=base_beard, baseline_face=base_face,
                           baseline_bearded=fraction is not None and fraction >= .01,
                           delta_current=100 * (beard / face - fraction) if face and base_face else None,
                           delta_fixed=100 * (beard - base_beard) / base_face if base_face else None,
                           delta_image=100 * (beard - base_beard) / 512**2,
                           control=control["image"], edit=row["image"], mask=str((masks / (row["image_sha256"] + ".png")).relative_to(RUN))))
    write(output / "paired.csv", paired)
    (output / "settings.json").write_text(json.dumps(dict(feature=FEATURE, checkpoint_hashes=model_settings["checkpoint_hashes"],
                                                         unique_images=len(predictions), groups=12, seeds=list(range(30,35)),
                                                         baseline_bearded_threshold=.01), indent=2))
    from matplotlib import pyplot as plt

    summaries = []
    metrics = ["delta_current", "delta_fixed", "delta_image"]
    for group in ["all", *sorted({r["group"] for r in paired})]:
        group_rows = [r for r in paired if group == "all" or r["group"] == group]
        for alpha in sorted({r["alpha"] for r in paired}):
            for subset in ("all", "baseline_bearded"):
                for metric in metrics:
                    selected = [r for r in group_rows if r["alpha"] == alpha and r[metric] is not None
                                and (subset == "all" or r["baseline_bearded"])]
                    seeds = sorted({r["seed"] for r in selected})
                    means = [np.mean([r[metric] for r in selected if r["seed"] == seed]) for seed in seeds]
                    lo, hi = interval(means)
                    summaries.append(dict(group=group, alpha=alpha, subset=subset, metric=metric,
                                          mean=np.mean(means) if means else None, ci_low=lo, ci_high=hi,
                                          seeds=len(seeds), conditions=len(selected)))
    write(output / "paired_summary.csv", summaries)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric in zip(axes[:2], metrics):
        for subset in ("all", "baseline_bearded"):
            part = [r for r in summaries if r["group"] == "all" and r["subset"] == subset and r["metric"] == metric]
            ax.plot([r["alpha"] for r in part], [r["mean"] for r in part], label=subset)
            ax.fill_between([r["alpha"] for r in part],
                            [np.nan if r["ci_low"] is None else r["ci_low"] for r in part],
                            [np.nan if r["ci_high"] is None else r["ci_high"] for r in part], alpha=.15)
        ax.set(xlabel="Multiplier", ylabel="Beard area change (pp)", title=metric.replace("_", " "))
        ax.legend()
    for direction, sign in (("decrease", -1), ("increase", 1)):
        part = [r for r in paired if (r["alpha"] - 1) * sign > 0 and r["delta_fixed"] is not None]
        axes[2].scatter([r["applied_l2"] for r in part], [r["delta_fixed"] for r in part], s=7, alpha=.2, label=direction)
    axes[2].set(xlabel="Applied L2", ylabel="Fixed-baseline beard area change (pp)")
    axes[2].legend()
    figure.savefig(output / "beard_response.png", dpi=180)
    example = next(r for r in paired if r["group"] == "white_man" and r["seed"] == 33 and r["alpha"] == 100)
    gallery = Image.new("RGB", (768, 256))
    for index, key in enumerate(("control", "edit", "mask")):
        with Image.open(RUN / example[key]) as image:
            if key == "mask":
                image = Image.fromarray((np.asarray(image) == 14).astype(np.uint8) * 255)
            gallery.paste(image.convert("RGB").resize((256, 256)), (256 * index, 0))
    gallery.save(output / "baseline_edit_mask.png")


if __name__ == "__main__":
    main()
