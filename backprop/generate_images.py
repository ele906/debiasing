"""Resume simple-prompt SDXL Turbo generation; cache pretrained SAE features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'sdxl-unbox'))
RACES = ['black', 'white', 'east asian', 'middle eastern', 'indian', 'hispanic']
GENDERS = ['man', 'woman']
BLOCK = 'unet.down_blocks.2.attentions.1'
CHECKPOINT = ROOT / 'sdxl-unbox/checkpoints' / f'{BLOCK}_k10_hidden5120_auxk256_bs4096_lr0.0001/final'
TEMPLATE = 'a picture of a {race} {gender}'


def records(count):
    import numpy as np
    # Hold out whole seeds across all prompt groups, not individual spatial tokens.
    order = np.random.default_rng(20261009).permutation(count)
    ntrain, nval = int(count * .6), int(count * .2)
    split = {int(s): 'train' if i < ntrain else 'validation' if i < ntrain + nval else 'test'
             for i, s in enumerate(order)}
    return [dict(sample_id=f'{race.replace(" ", "_")}_{gender}_seed{seed:03d}',
                 race_prompt=race, gender_prompt=gender, seed=seed, split=split[seed],
                 prompt=TEMPLATE.format(race=race, gender=gender),
                 image=f'images/{race.replace(" ", "_")}_{gender}/seed{seed:03d}.png',
                 features=f'features/{race.replace(" ", "_")}_{gender}/seed{seed:03d}.npz')
            for gender in GENDERS for race in RACES for seed in range(count)]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=100, help='Images per race/gender cell')
    p.add_argument('--output', type=Path, default=HERE)
    args = p.parse_args()
    if args.count < 5:
        p.error('count must be at least 5 for train/validation/test splits')
    import numpy as np
    import torch
    import diffusers
    from PIL import Image, ImageDraw
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(prompt_template=TEMPLATE, races=RACES, genders=GENDERS,
                    images_per_cell=args.count, model='stabilityai/sdxl-turbo',
                    vae='madebyollin/sdxl-vae-fp16-fix', block=BLOCK,
                    sae_checkpoint=str(CHECKPOINT),
                    sae_weights_sha256=hashlib.sha256((CHECKPOINT / 'state_dict.pth').read_bytes()).hexdigest(),
                    steps=1, guidance_scale=0.0, height=512, width=512, dtype='float16',
                    scheduler='EulerAncestralDiscreteScheduler', timestep_spacing='trailing',
                    rng='Independent CUDA generators, each seeded with sample seed, for initial noise and scheduler',
                    pooling=['spatial_mean', 'spatial_max', 'spatial_active_fraction'],
                    split='60/20/20 by seed, shared across all 12 groups; RNG 20261009',
                    torch_version=torch.__version__, diffusers_version=diffusers.__version__,
                    labels='Generation prompt categories, not racial identities inferred from images')
    settings_path = out / 'generation_settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Existing generation settings differ; use a new output directory')
    settings_path.write_text(json.dumps(settings, indent=2))
    rows = records(args.count)
    write_csv(out / 'manifest.csv', rows)
    annotations = out / 'annotations.csv'
    if not annotations.exists():
        write_csv(annotations, [dict(**r, head_covering='', hair_damage='', artifacts='', notes='') for r in rows])

    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / 'ReNO/hf_cache'), local_files_only=True)
    print('Loading cached models; no downloads or training', flush=True)
    vae = AutoencoderKL.from_pretrained(settings['vae'], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings['model'], vae=vae, variant='fp16', **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config, timestep_spacing='trailing')
    pipe.pipe.to('cuda')
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(CHECKPOINT)).float().cuda().eval()
    sae.requires_grad_(False)
    start = time.monotonic()
    generated = 0
    for i, r in enumerate(rows):
        imgpath, featpath = out / r['image'], out / r['features']
        if imgpath.exists() and featpath.exists():
            continue
        noise_rng = torch.Generator('cuda').manual_seed(r['seed'])
        scheduler_rng = torch.Generator('cuda').manual_seed(r['seed'])
        latents = torch.randn((1, 4, 64, 64), generator=noise_rng, device='cuda', dtype=torch.float16)
        with torch.no_grad():
            result, cache = pipe.run_with_cache(prompt=r['prompt'], latents=latents, generator=scheduler_rng,
                num_inference_steps=1, guidance_scale=0.0, height=512, width=512,
                positions_to_cache=[BLOCK], save_input=True)
            residual = (cache['output'][BLOCK] - cache['input'][BLOCK])[:, -1].permute(0, 2, 3, 1).float()
            z = sae.encode(residual)
            mean = z.mean(dim=(0, 1, 2)).cpu().numpy()
            maximum = z.amax(dim=(0, 1, 2)).cpu().numpy()
            active = (z > 0).float().mean(dim=(0, 1, 2)).cpu().numpy()
        imgpath.parent.mkdir(parents=True, exist_ok=True)
        featpath.parent.mkdir(parents=True, exist_ok=True)
        # Feature save follows image save: interruption is recoverable by deterministic replay.
        result.images[0].save(imgpath)
        np.savez_compressed(featpath, mean=mean, maximum=maximum, active_fraction=active)
        del cache, residual, z
        generated += 1
        if generated == 1 or generated % 20 == 0:
            print(f'{i + 1}/{len(rows)} complete; {r["race_prompt"]} {r["gender_prompt"]}; '
                  f'{(time.monotonic() - start) / generated:.2f}s/new image', flush=True)
    means, maxima, fractions = [], [], []
    for r in rows:
        with np.load(out / r['features']) as f:
            means.append(f['mean']); maxima.append(f['maximum']); fractions.append(f['active_fraction'])
    np.savez_compressed(out / 'sae_features.npz', X_mean=np.stack(means), X_max=np.stack(maxima),
        X_active_fraction=np.stack(fractions), sample_id=np.array([r['sample_id'] for r in rows]),
        y_race=np.array([RACES.index(r['race_prompt']) for r in rows]),
        y_gender=np.array([GENDERS.index(r['gender_prompt']) for r in rows]),
        seed=np.array([r['seed'] for r in rows]), split=np.array([r['split'] for r in rows]),
        races=np.array(RACES), genders=np.array(GENDERS))
    for gender in GENDERS:
        for race in RACES:
            group = race.replace(' ', '_') + '_' + gender
            group_rows = [r for r in rows if r['race_prompt'] == race and r['gender_prompt'] == gender]
            grid = Image.new('RGB', (10 * 128, ((args.count + 9) // 10) * 148), 'white')
            draw = ImageDraw.Draw(grid)
            for i, r in enumerate(group_rows):
                x, y = (i % 10) * 128, (i // 10) * 148
                with Image.open(out / r['image']) as img:
                    grid.paste(img.resize((128, 128)), (x, y + 20))
                draw.text((x + 3, y + 3), f'seed {r["seed"]} {r["split"]}', fill='black')
            (out / 'contact_sheets').mkdir(exist_ok=True)
            grid.save(out / 'contact_sheets' / f'{group}.jpg')
    summary = dict(total_images=len(rows), per_cell=args.count, feature_dimensions=int(means[0].shape[0]),
                   training_run=False, complete=True)
    (out / 'generation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
