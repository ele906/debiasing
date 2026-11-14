"""Isolated mid.0 experiment. Existing generation and SAE code remain unchanged."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
BLOCK = 'unet.mid_block.attentions.0'
MULTIPLIERS = [-7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7]
CHECKPOINT = ROOT / 'fine-tuned-model/mid0_sae_001/sae'
EXPECTED_HASH = '8a2ba905e231a4d0276490b9825a08052655f464e06d55cb5192014758029c82'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def events(run):
    path = run / 'logs/events.jsonl'
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed write is retained as evidence; subsequent writes start a new line.
            continue
    return result


def record(run, row):
    with (run / 'logs/events.jsonl').open('a') as f:
        f.write('\n' + json.dumps(row) + '\n')
        f.flush()
        os.fsync(f.fileno())


def inventory(run):
    files = [dict(path=str(p.relative_to(run)), bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('output_manifest.json', 'run.lock')]
    atomic_json(run / 'output_manifest.json', files)


def index(run):
    latest = {}
    for e in events(run):
        latest[e['image_id']] = e
    csv_write(run / 'results/all_generations.csv', list(latest.values()))
    return latest


def prepare(args):
    if digest(CHECKPOINT / 'state_dict.pth') != EXPECTED_HASH:
        raise ValueError('Selected SAE hash mismatch')
    with args.manifest.open() as f:
        rows = [r for r in csv.DictReader(f) if int(r['seed']) in args.seeds and
                (not args.groups or r['race_prompt'].replace(' ', '_') + '_' + r['gender_prompt'] in args.groups)]
    if not rows:
        raise ValueError('No matching prompts/seeds')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    run = Path(__file__).parent / ('run_' + stamp)
    run.mkdir(exist_ok=False)
    for d in ('config', 'images', 'activations/raw', 'activations/modified', 'evaluations', 'results', 'figures', 'logs'):
        (run / d).mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT / 'backprop/generation_settings.json').read_text())
    cfg = dict(run_id=run.name, layer=BLOCK, sae_checkpoint=str(CHECKPOINT.resolve()),
               sae_sha256=EXPECTED_HASH, model_checkpoint=settings['model'], features=args.features,
               feature_selection=args.feature_selection, prompts=rows, multipliers=MULTIPLIERS,
               additive_quantiles=[0.25, 0.5], additive_scales=[0.1, 0.5], matched_rms_quantiles=[0.25, 0.5],
               full_tensors=args.full_tensors, settings=settings,
               intervention='residual-preserving decoder-direction delta; no clipping or re-topk',
               generator_updated=False, sae_updated=True)
    atomic_json(run / 'config/experiment_config.json', cfg)
    (run/'config/source').mkdir()
    import shutil
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source,run/'config/source'/source.name)
    atomic_json(run / 'config/prompts.json', rows)
    csv_write(run / 'config/feature_list.csv', [dict(feature_id=f, methodology=args.feature_selection, checkpoint_sha256=EXPECTED_HASH) for f in args.features])
    files = [CHECKPOINT / 'config.json', CHECKPOINT / 'state_dict.pth', ROOT / 'backprop/generation_settings.json', run/'config/source/run.py', args.manifest, ROOT/'sdxl-unbox/SAE/sae.py', ROOT/'sdxl-unbox/SDLens/hooked_sd_pipeline.py', ROOT/'FairFace/scripts/audit_fairface_independent.py']
    # Include all local diffusion snapshot files, including resolved symlink targets.
    for model in ('stabilityai--sdxl-turbo', 'madebyollin--sdxl-vae-fp16-fix'):
        files.extend(p for p in (ROOT / 'ReNO/hf_cache' / ('models--' + model) / 'snapshots').glob('**/*') if p.is_file())
    atomic_json(run / 'config/checkpoint_manifest.json', [dict(path=str(p), resolved_path=str(p.resolve()), bytes=p.stat().st_size, sha256=digest(p)) for p in files])
    env = platform.platform() + '\n' + sys.version + '\n'
    for command in (['git', 'rev-parse', 'HEAD'], ['git', 'status', '--porcelain'], [sys.executable, '-m', 'pip', 'freeze']):
        env += '\n' + repr(command) + '\n' + subprocess.run(command, cwd=ROOT, capture_output=True, text=True).stdout
    (run / 'config/environment.txt').write_text(env)
    (run/'README.md').write_text('Reproduce/resume from /n/fs/goose:\n\n' +
        'PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps ' + sys.executable +
        ' experiments/mid0_sae_selectivity/run.py --run ' + str(run) + '\n\n' +
        'Configuration: config/experiment_config.json. Original PNGs: images/. Latents: activations/raw/*.npz. ' +
        'NPZ keys original_N, modified_N, latent_change_N, post_reencoded_N contain full spatial selected-feature arrays at hook call N; ' +
        'full_original_N and full_modified_N are optional complete dictionaries. timestep_N stores scheduler timestep. ' +
        'Signed modified latents are decoded linearly without clipping or reapplying Top-K. ' +
        'Post-reencoded values are a different quantity. logs/events.jsonl preserves all attempts; results/all_generations.csv is the latest condition index. ' +
        'output_manifest.json inventories artifacts. See ../README.md for analysis commands and limitations.\n')
    architecture = json.loads((CHECKPOINT / 'config.json').read_text())
    conditions = len(rows) * len(args.features) * 17
    raw_bytes = conditions * settings['steps'] * 16 * 16 * (architecture['n_dirs_local'] * 2 if args.full_tensors else 4) * 4
    atomic_json(run / 'config/storage_estimate.json', dict(estimated_conditions=conditions, activation_bytes_uncompressed=raw_bytes,
                png_upper_estimate_bytes=conditions * settings['height'] * settings['width'] * 4,
                free_bytes=__import__('shutil').disk_usage(run).free,
                note='16x16 mid.0 grid at 512px; actual size recorded per hook; excludes post-reencoded full tensors'))
    print(run, flush=True)
    return run


def execute(run):
    import numpy as np
    import torch
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    sys.path.insert(0, str(ROOT / 'sdxl-unbox'))
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder
    cfg = json.loads((run / 'config/experiment_config.json').read_text())
    for entry in json.loads((run / 'config/checkpoint_manifest.json').read_text()):
        if digest(entry['path']) != entry['sha256']:
            archived = run/'config/source'/Path(entry['path']).name
            if Path(entry['path']).name != 'run.py' or not archived.exists() or digest(archived) != entry['sha256']:
                raise ValueError('Input file changed: ' + entry['path'])
    estimate = json.loads((run/'config/storage_estimate.json').read_text())
    if __import__('shutil').disk_usage(run).free < estimate['activation_bytes_uncompressed'] + estimate['png_upper_estimate_bytes']:
        raise OSError('Insufficient disk space for conservative experiment estimate')
    if digest(Path(cfg['sae_checkpoint']) / 'state_dict.pth') != cfg['sae_sha256']:
        raise ValueError('Checkpoint changed since preparation')
    settings = cfg['settings']
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / 'ReNO/hf_cache'), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings['vae'], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(cfg['model_checkpoint'], vae=vae, variant='fp16', **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config, timestep_spacing='trailing')
    pipe.pipe.to('cuda')
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(cfg['sae_checkpoint']).float().cuda().eval()
    sae.requires_grad_(False)
    atomic_json(run / 'config/runtime.json', dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
        scheduler=dict(pipe.pipe.scheduler.config), sae=json.loads((CHECKPOINT / 'config.json').read_text()), precision='diffusion fp16; SAE fp32'))
    done = index(run)

    def generate(prompt, feature, method, strength):
        condition = dict(prompt_id=prompt['sample_id'], feature_id=feature, intervention_type=method, strength=strength)
        image_id = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[:24]
        if image_id in done and done[image_id]['generation_status'] == 'complete':
            old = done[image_id]
            for field, hashfield in [('image_path','image_sha256'), ('activation_path','activation_sha256')]:
                if digest(run / old[field]) != old[hashfield]:
                    raise ValueError('Completed artifact missing or changed: ' + old[field])
            return old
        # Every retry is a new attempt. Orphaned images/tensors are never overwritten.
        attempt = sum(e['image_id'] == image_id and e['generation_status'] == 'started' for e in events(run))
        stem = image_id + '_attempt' + str(attempt)
        image = run / 'images' / method / ('feature_' + str(feature)) / (stem + '.png')
        image.parent.mkdir(parents=True, exist_ok=True)
        activation = run / 'activations/raw' / (stem + '.npz')
        row = dict(run_id=run.name, image_id=image_id, image_path=str(image.relative_to(run)),
            activation_path=str(activation.relative_to(run)), timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            model_checkpoint=cfg['model_checkpoint'], sae_checkpoint=cfg['sae_checkpoint'], layer=BLOCK,
            feature_id=feature, prompt_id=prompt['sample_id'], prompt_text=prompt['prompt'], seed=int(prompt['seed']),
            group=prompt['race_prompt'] + ' ' + prompt['gender_prompt'], intervention_type=method,
            multiplier=strength if method=='multiplicative' else '', additive_offset=strength if method in ('additive','matched_additive') else '',
            negative_multiplier_ood=method=='multiplicative' and strength<0, generation_status='started', attempt=attempt, original_activation_mean=None, original_activation_max=None,
            original_activation_nonzero_fraction=None, modified_activation_mean=None, perturbation_norm=None, denoising_timesteps=[])
        record(run, row)
        captured = dict(feature_id=np.asarray(feature),layer=np.asarray(BLOCK))
        norms = []

        @torch.no_grad()
        def hook(module, inputs, output):
            original = output[0]
            residual = (original-inputs[0]).permute(0,2,3,1).float()
            codes = sae.encode(residual)
            z = codes[...,feature:feature+1]
            change = z*(strength-1.0) if method=='multiplicative' else torch.full_like(z, strength if method in ('additive','matched_additive') else 0.)
            modified = codes.clone()
            modified[...,feature:feature+1] = z+change
            delta = change @ sae.decoder.weight[:,feature:feature+1].T
            edited = original + delta.permute(0,3,1,2).to(original)
            if method=='reconstruction_only':
                edited = inputs[0] + (sae.decoder(codes)+sae.pre_bias).permute(0,3,1,2).to(original)
            post = sae.encode((edited-inputs[0]).permute(0,2,3,1).float())
            i = len(norms)
            values = dict(original=z, modified=z+change, post_reencoded=post[...,feature:feature+1], latent_change=change)
            if cfg['full_tensors']:
                values.update(full_original=codes, full_modified=modified)
            for name, value in values.items():
                captured[f'{name}_{i}'] = value.cpu().numpy()
            captured[f'timestep_{i}'] = np.asarray(pipe.pipe.scheduler.timesteps[i].item())
            norms.append(float((edited-original).float().norm().item()))
            # Flush after every hook so interrupted generations retain captured tensors.
            temporary = activation.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f, **captured)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, activation)
            return output if method == 'baseline' else (edited,) + output[1:]

        try:
            rng = torch.Generator('cuda').manual_seed(row['seed'])
            latents = torch.randn((1,4,settings['height']//8,settings['width']//8), generator=rng,device='cuda',dtype=torch.float16)
            result = pipe.run_with_hooks(prompt=row['prompt_text'], latents=latents,
                generator=torch.Generator('cuda').manual_seed(row['seed']), num_inference_steps=settings['steps'],
                guidance_scale=settings['guidance_scale'],height=settings['height'],width=settings['width'],
                position_hook_dict={BLOCK:hook})
            result.images[0].save(image, format='PNG')
            a = np.concatenate([captured[f'original_{i}'].ravel() for i in range(len(norms))])
            b = np.concatenate([captured[f'modified_{i}'].ravel() for i in range(len(norms))])
            row.update(original_activation_mean=float(a.mean()), original_activation_max=float(a.max()),
                original_activation_nonzero_fraction=float(np.count_nonzero(a)/a.size), original_activation_zero=bool(not np.any(a)),
                modified_activation_mean=float(b.mean()), modified_activation_max=float(b.max()),
                modified_activation_nonzero_fraction=float(np.count_nonzero(b)/b.size),
                modified_activation_negative_fraction=float(np.mean(b<0)),
                modified_activation_zero_fraction=float(np.mean(b==0)),
                perturbation_norm=float(np.linalg.norm(b-a)), perturbation_rms=float(np.sqrt(np.mean((b-a)**2))),
                decoded_perturbation_norm=float(np.linalg.norm(norms)),denoising_timesteps=[float(captured[f'timestep_{i}']) for i in range(len(norms))],
                generation_status='complete', image_sha256=digest(image), activation_sha256=digest(activation))
        except BaseException as exc:
            row.update(generation_status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed', error=traceback.format_exc())
            record(run,row)
            index(run)
            raise
        record(run,row)
        done[image_id] = row
        index(run)
        print(image_id, method, strength, 'complete', flush=True)
        return row

    for feature in cfg['features']:
        controls = [generate(p,feature,'baseline',0.) for p in cfg['prompts']]
        positives = []
        rms = []
        for r in controls:
            with np.load(run/r['activation_path']) as a:
                z = np.concatenate([a[k].ravel() for k in a.files if k.startswith('original_')])
            positives.extend(z[z>0].tolist())
            rms.append(float(np.sqrt(np.mean(z*z))))
        if not positives:
            raise ValueError(f'Feature {feature} inactive in all calibration contexts; offsets cannot be calibrated. Expand prompts/seeds.')
        offsets = [float(np.quantile(positives,q))*scale for q,scale in zip(cfg['additive_quantiles'], cfg.get('additive_scales',[1.,1.]))]
        targets = [float(np.quantile([r for r in rms if r>0],q)) for q in cfg['matched_rms_quantiles']]
        atomic_json(run / f'config/calibration_feature_{feature}.json', dict(offsets=offsets, matched_targets=targets,
            source_image_ids=[r['image_id'] for r in controls], rule='scaled positive-value quartiles (scales in experiment config);  matched offsets are common latent RMS targets (constant offset RMS=abs(beta))'))
        for p in cfg['prompts']:
            generate(p,feature,'reconstruction_only',0.)
            for alpha in MULTIPLIERS:
                generate(p,feature,'multiplicative',alpha)
            for beta in offsets:
                generate(p,feature,'additive',beta)
            for beta in targets:
                generate(p,feature,'matched_additive',beta)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'backprop/manifest.csv')
    p.add_argument('--run', type=Path, help='Resume exactly this existing run')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--features', nargs='+', type=int)
    p.add_argument('--feature-selection', help='Exact feature-selection provenance for the selected fine-tuned SAE')
    p.add_argument('--seeds', nargs='+', type=int, default=[0,1])
    p.add_argument('--groups', nargs='+', default=['black_man','white_man'])
    p.add_argument('--full-tensors', action='store_true')
    args = p.parse_args()
    if not args.run and (not args.features or not args.feature_selection):
        p.error('New runs require --features and --feature-selection; pretrained rankings are incompatible')
    if args.features and any(not 0<=f<5120 for f in args.features):
        p.error('Feature IDs must be in [0,5119]')
    run = args.run.resolve() if args.run else prepare(args)
    with (run / 'logs/run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.prepare_only:
            try:
                execute(run)
            except BaseException:
                (run/'logs/execution_failure.txt').write_text(traceback.format_exc())
                raise
            finally:
                index(run)
                inventory(run)

if __name__=='__main__':
    main()
