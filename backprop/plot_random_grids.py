"""Save reproducible random 5-by-5 Matplotlib grids for every prompt category."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--data", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "random_grids")
    args = parser.parse_args()
    settings = json.loads((args.data / "generation_settings.json").read_text())
    rng = np.random.default_rng(args.seed)
    selections = {}
    jobs = []
    for gender in settings["genders"]:
        for race in settings["races"]:
            category = f"{race.replace(' ', '_')}_{gender}"
            images = sorted((args.data / "images" / category).glob("seed*.png"))
            if len(images) < 25:
                raise ValueError(f"{category}: need 25 images, found {len(images)}")
            positions = rng.choice(len(images), size=25, replace=False)
            chosen = [images[int(i)] for i in positions]
            prompt = settings["prompt_template"].format(race=race, gender=gender)
            selections[category] = dict(prompt=prompt,
                indices=[int(p.stem.removeprefix("seed")) for p in chosen],
                images=[str(p.relative_to(args.data)) for p in chosen])
            jobs.append((category, prompt, chosen))
    args.output.mkdir(parents=True, exist_ok=True)
    for category, prompt, chosen in jobs:
        fig, axes = plt.subplots(5, 5, figsize=(15, 16), facecolor="white")
        fig.suptitle(f"{prompt}\n25 random samples", fontsize=20)
        for ax, path in zip(axes.flat, chosen):
            with Image.open(path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"Index {int(path.stem.removeprefix('seed'))}", fontsize=10)
            ax.axis("off")
        fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.93,
                            wspace=.035, hspace=.12)
        output = args.output / f"{category}.png"
        fig.savefig(output, dpi=150, facecolor="white")
        plt.close(fig)
        print(output, flush=True)
    (args.output / "sampled_indices.json").write_text(
        json.dumps(dict(random_seed=args.seed, sampling="25 per category without replacement",
                        categories=selections), indent=2) + "\n")


if __name__ == "__main__":
    main()
