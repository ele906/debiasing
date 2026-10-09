"""Replay discovery images and overlay maps from the existing pretrained SAE.

Adapted from baseline/sae_top10_heatmaps.ipynb. Does not train or suppress features.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from generate_images import HERE, ROOT, BLOCK, CHECKPOINT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE)
    p.add_argument('--analysis', type=Path, default=HERE / 'sae_analysis')
    p.add_argument('--output', type=Path, default=HERE / 'sae_heatmaps')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--examples-per-cell', type=int, default=3)
    args = p.parse_args()
    if args.top_k < 1 or args.examples_per_cell < 1:
        p.error('top-k and examples-per-cell must be positive')
    if args.output.exists():
        raise ValueError('Choose a new output to preserve earlier maps')
    settings = json.loads((args.data / 'generation_settings.json').read_text())
    if hashlib.sha256((CHECKPOINT / 'state_dict.pth').read_bytes()).hexdigest() != settings['sae_weights_sha256']:
        raise ValueError('SAE changed since generation')
    with (args.data / 'manifest.csv').open() as f:
        manifest = list(csv.DictReader(f))
    with (args.analysis / 'top_features_by_race_gender.csv').open() as f:
        rankings = list(csv.DictReader(f))
    import torch
    from PIL import Image, ImageDraw
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / 'ReNO/hf_cache'), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings['vae'], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings['model'], vae=vae, variant='fp16', **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config, timestep_spacing='trailing')
    pipe.pipe.to('cuda'); pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(CHECKPOINT)).float().cuda().eval()
    args.output.mkdir(parents=True)
    metrics, report = [], ['# SAE top-feature heatmaps', '',
        'Discovery examples only. No training or suppression. Labels come from prompts.', '',
        'Each overlay is normalized to its own map maximum; compare CSV values, not overlay brightness.', '']
    for gender in settings['genders']:
        for race in settings['races']:
            features = [int(r['feature']) for r in rankings if r['race_prompt'] == race and
                        r['gender_prompt'] == gender and int(r['rank']) <= args.top_k]
            examples = [r for r in manifest if r['race_prompt'] == race and r['gender_prompt'] == gender
                        and r['split'] == 'train'][:args.examples_per_cell]
            group = race.replace(' ', '_') + '_' + gender
            directory = args.output / group; directory.mkdir()
            grid = Image.new('RGB', ((len(features) + 1) * 192, len(examples) * 218), 'white')
            draw = ImageDraw.Draw(grid)
            for row_idx, row in enumerate(examples):
                seed = int(row['seed'])
                noise = torch.randn((1, 4, 64, 64), device='cuda', dtype=torch.float16,
                                    generator=torch.Generator('cuda').manual_seed(seed))
                with torch.no_grad():
                    result, cache = pipe.run_with_cache(prompt=row['prompt'], latents=noise,
                        generator=torch.Generator('cuda').manual_seed(seed), num_inference_steps=1,
                        guidance_scale=0., height=512, width=512, positions_to_cache=[BLOCK], save_input=True)
                    diff = (cache['output'][BLOCK] - cache['input'][BLOCK])[:, -1].permute(0, 2, 3, 1).float()
                    codes = sae.encode(diff)
                replay = result.images[0]
                with Image.open(args.data / row['image']) as im:
                    original = im.convert('RGB')
                error = float(np.abs(np.asarray(original).astype(float) - np.asarray(replay)).mean())
                if error != 0:
                    raise RuntimeError(f'{row["sample_id"]}: replay differs from saved image (MAE {error}); cannot overlay reliably')
                tiles = [('original', original)]
                for feature in features:
                    values = codes[0, ..., feature].cpu().numpy()
                    np.save(directory / f'seed{seed:03d}_feature{feature}.npy', values)
                    heat = Image.fromarray(np.uint8(np.clip(values / max(float(values.max()), 1e-6), 0, 1) * 255)).resize(original.size)
                    overlay = Image.composite(Image.new('RGB', original.size, 'red'), original, heat.point(lambda v: int(v * .55)))
                    overlay.save(directory / f'seed{seed:03d}_feature{feature}.png')
                    metrics.append(dict(sample_id=row['sample_id'], race_prompt=race, gender_prompt=gender,
                        feature=feature, mean=float(values.mean()), maximum=float(values.max()),
                        spatial_active_fraction=float((values > 0).mean()), replay_pixel_mae=error))
                    tiles.append((f'feature {feature}', overlay))
                for col, (label, image) in enumerate(tiles):
                    x, y = col * 192, row_idx * 218
                    grid.paste(image.resize((192, 192)), (x, y + 26))
                    draw.text((x + 3, y + 4), f'seed {seed}: {label}', fill='black')
                del cache, codes, diff
            grid.save(directory / 'comparison.jpg')
            report += [f'## {race} {gender}', '', f'![top features]({group}/comparison.jpg)', '']
            print(f'Finished {group}: {features}', flush=True)
    with (args.output / 'activation_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    (args.output / 'REPORT.md').write_text('\n'.join(report))
    print('Heatmaps complete; no training performed.', flush=True)


if __name__ == '__main__':
    main()
