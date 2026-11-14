"""FUTURE experiment: sweep fixed SAE strengths on held-out generated images.

Manual backprop can suggest candidate IDs via train_probe.py; this script
intervenes in the diffusion generator. It does not train or optimize strengths.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from generate_images import HERE, ROOT, BLOCK, CHECKPOINT

sys.path.insert(0, str(ROOT / 'sdxl-unbox/scripts'))
from sae_experiment import intervention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE)
    p.add_argument('--output', type=Path, default=HERE / 'suppression')
    p.add_argument('--features', type=int, nargs='+', required=True,
                   help='Use IDs selected on train/validation data; IDs are checkpoint-specific')
    p.add_argument('--strengths', type=float, nargs='+', default=[0, .5, 1, 2])
    p.add_argument('--seeds-per-cell', type=int, default=3)
    args = p.parse_args()
    if args.seeds_per_cell < 1 or not all(np.isfinite(args.strengths)):
        p.error('Need positive seeds-per-cell and finite strengths')
    if args.output.exists():
        raise ValueError('Use a new output directory to preserve earlier experiments')
    if 1 not in args.strengths:
        p.error('Include strength 1 to test exact no-op behavior')
    import torch
    from PIL import Image, ImageDraw
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder

    settings = json.loads((args.data / 'generation_settings.json').read_text())
    if settings['block'] != BLOCK or settings['sae_checkpoint'] != str(CHECKPOINT):
        raise ValueError('Generation SAE/block does not match this intervention')
    import hashlib
    if hashlib.sha256((CHECKPOINT / 'state_dict.pth').read_bytes()).hexdigest() != settings['sae_weights_sha256']:
        raise ValueError('SAE weights changed since collection')
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / 'ReNO/hf_cache'), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings['vae'], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings['model'], vae=vae, variant='fp16', **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config, timestep_spacing='trailing')
    pipe.pipe.to('cuda')
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(CHECKPOINT)).float().cuda().eval()
    if any(j < 0 or j >= sae.n_dirs for j in args.features):
        raise ValueError('Feature outside SAE dictionary')
    with (args.data / 'manifest.csv').open() as f:
        rows = list(csv.DictReader(f))
    selected = []
    for g in settings['genders']:
        for r in settings['races']:
            selected.extend([row for row in rows if row['split'] == 'test' and row['race_prompt'] == r
                             and row['gender_prompt'] == g][:args.seeds_per_cell])
    if not selected:
        raise ValueError('No test images in manifest')
    args.output.mkdir(parents=True)
    (args.output / 'settings.json').write_text(json.dumps(dict(features=args.features,
        strengths=args.strengths, seeds_per_cell=args.seeds_per_cell, data=str(args.data),
        interpretation='Exploratory fixed edits; need human review to assess outcome'), indent=2))
    metrics, review = [], []

    def gen_args(row):
        seed = int(row['seed'])
        latents = torch.randn((1, 4, 64, 64), generator=torch.Generator('cuda').manual_seed(seed),
                              device='cuda', dtype=torch.float16)
        return dict(prompt=row['prompt'], latents=latents, generator=torch.Generator('cuda').manual_seed(seed),
                    num_inference_steps=1, guidance_scale=0., height=512, width=512)

    for row in selected:
        directory = args.output / row['sample_id']
        directory.mkdir()
        replay = pipe.pipe(**gen_args(row)).images[0]
        replay.save(directory / 'replay.png')
        with Image.open(args.data / row['image']) as saved:
            replay_error = float(np.abs(np.asarray(saved.convert('RGB')).astype(float) - np.asarray(replay)).mean())
        for feature in args.features:
            tiles = [('replay', replay)]
            for strength in args.strengths:
                edited = pipe.run_with_hooks(**gen_args(row), position_hook_dict={BLOCK: intervention(sae, [feature], strength)}).images[0]
                change = float(np.abs(np.asarray(edited).astype(float) - np.asarray(replay)).mean())
                if strength == 1 and change != 0:
                    raise RuntimeError('Strength 1 failed exact pixel identity')
                path = directory / f'feature{feature}_strength{strength:g}.png'
                edited.save(path)
                metrics.append(dict(sample_id=row['sample_id'], feature=feature, strength=strength,
                    pixel_change=change, saved_baseline_replay_error=replay_error))
                review.append(dict(**metrics[-1], race_prompt=row['race_prompt'], gender_prompt=row['gender_prompt'],
                    original=str((args.data / row['image']).resolve()), edited=str(path.resolve()),
                    head_covering='', hair_damage='', artifacts='', prompt_match='', notes=''))
                tiles.append((f'strength {strength:g}', edited))
            grid = Image.new('RGB', (192 * len(tiles), 216), 'white')
            draw = ImageDraw.Draw(grid)
            for i, (name, img) in enumerate(tiles):
                grid.paste(img.resize((192, 192)), (i * 192, 24))
                draw.text((i * 192 + 3, 4), name, fill='black')
            grid.save(directory / f'feature{feature}_comparison.jpg')
        print(f'Finished {row["sample_id"]}', flush=True)
    for name, values in [('metrics.csv', metrics), ('human_review.csv', review)]:
        with (args.output / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)


if __name__ == '__main__':
    main()
