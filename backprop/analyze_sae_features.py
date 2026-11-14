"""Adapted from sdxl-unbox/scripts/sae_bias_experiment.py: cached SAE rankings.

Use discovery seeds for ranking, validation seeds for confirmation, and leave
test seeds untouched. No SAE or classifier training is performed.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE)
    p.add_argument('--output', type=Path, default=HERE / 'sae_analysis')
    p.add_argument('--pooling', choices=['mean', 'max'], default='mean')
    p.add_argument('--top-k', type=int, default=20)
    args = p.parse_args()
    if args.top_k < 1:
        p.error('top-k must be positive')
    if args.output.exists():
        raise ValueError('Use a new output directory to preserve previous analysis')
    with np.load(args.data / 'sae_features.npz', allow_pickle=False) as f:
        X = f[f'X_{args.pooling}'].astype(float)
        fractions = f['X_active_fraction'].copy()
        race, gender, split = f['y_race'].copy(), f['y_gender'].copy(), f['split'].copy()
        races, genders = f['races'].tolist(), f['genders'].tolist()
        ids = f['sample_id'].copy()
    if X.shape[0] != len(ids) or not np.isfinite(X).all():
        raise ValueError('Feature data are malformed/nonfinite')
    train, val = split == 'train', split == 'validation'
    scale = X[train].std(0).clip(1e-6)
    cell_rows, race_rows, prevalence = [], [], []
    for r, race_name in enumerate(races):
        train_contrasts, val_contrasts = [], []
        for g, gender_name in enumerate(genders):
            a, b = train & (race == r) & (gender == g), train & (race != r) & (gender == g)
            av, bv = val & (race == r) & (gender == g), val & (race != r) & (gender == g)
            if not all(mask.any() for mask in (a, b, av, bv)):
                raise ValueError('Every cell needs discovery and validation examples')
            contrast = (X[a].mean(0) - X[b].mean(0)) / scale
            held = (X[av].mean(0) - X[bv].mean(0)) / scale
            train_contrasts.append(contrast); val_contrasts.append(held)
            # Positive contrasts find features present more strongly in the target
            # group, rather than features chiefly absent from that group.
            candidates = np.flatnonzero(contrast > 0)
            top = candidates[np.argsort(-contrast[candidates])][:args.top_k]
            for rank, j in enumerate(top, 1):
                cell_rows.append(dict(race_prompt=race_name, gender_prompt=gender_name, rank=rank,
                    feature=int(j), discovery_contrast=float(contrast[j]), validation_contrast=float(held[j]),
                    discovery_mean=float(X[a, j].mean()), validation_mean=float(X[av, j].mean()),
                    discovery_spatial_active_fraction=float(fractions[a, j].mean()),
                    validation_spatial_active_fraction=float(fractions[av, j].mean()),
                    validation_same_sign=bool(held[j] > 0)))
            # Active fraction is positions active / all spatial positions, not
            # the old notebooks' active-position count divided only by images.
            for j in sorted(set(top.tolist() + [138, 527, 3530])):
                prevalence.append(dict(race_prompt=race_name, gender_prompt=gender_name, feature=int(j),
                    discovery_image_active_fraction=float((X[a, j] > 0).mean()),
                    discovery_spatial_active_fraction=float(fractions[a, j].mean()),
                    validation_image_active_fraction=float((X[av, j] > 0).mean()),
                    validation_spatial_active_fraction=float(fractions[av, j].mean())))
        score, held = np.mean(train_contrasts, 0), np.mean(val_contrasts, 0)
        candidates = np.flatnonzero(score > 0)
        for rank, j in enumerate(candidates[np.argsort(-score[candidates])][:args.top_k], 1):
            race_rows.append(dict(race_prompt=race_name, rank=rank, feature=int(j),
                discovery_contrast=float(score[j]), validation_contrast=float(held[j]),
                discovery_gender_consistent=bool(np.all(np.array(train_contrasts)[:, j] > 0)),
                validation_gender_consistent=bool(np.all(np.array(val_contrasts)[:, j] > 0))))
    args.output.mkdir(parents=True)
    write_csv(args.output / 'top_features_by_race_gender.csv', cell_rows)
    write_csv(args.output / 'top_features_by_race.csv', race_rows)
    write_csv(args.output / 'feature_prevalence.csv', prevalence)
    summary = dict(images=len(X), dimensions=X.shape[1], discovery=int(train.sum()),
        validation=int(val.sum()), reserved_test=int((split == 'test').sum()), pooling=args.pooling,
        method='Discovery-standardized within-gender one-race-vs-other-races mean contrast',
        trained=False, interpretation='Prompt-group associations; not racism, covering labels, or causal effects',
        source='Adapted from ../sdxl-unbox/scripts/sae_bias_experiment.py')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# SAE features on simple-prompt images', '',
        'No training performed. Feature IDs refer to the existing pretrained down.2.1 SAE.', '',
        'Ranked using 720 discovery images; validation uses 240 separate images. The 240 test images are untouched.', '',
        'These features distinguish generation-prompt groups. They do not yet identify head coverings or harmful bias.', '',
        '| Prompt group | Top feature | Discovery contrast | Validation contrast |',
        '|---|---:|---:|---:|']
    for row in cell_rows:
        if row['rank'] == 1:
            lines.append(f'| {row["race_prompt"]} {row["gender_prompt"]} | {row["feature"]} | '
                         f'{row["discovery_contrast"]:.3f} | {row["validation_contrast"]:.3f} |')
    lines += ['', '[Full per-group rankings](top_features_by_race_gender.csv)', '',
        '[Gender-averaged race rankings](top_features_by_race.csv)', '',
        '[Feature prevalence and active fractions](feature_prevalence.csv)', '',
        'Run `sbatch backprop/sae_heatmaps.slurm` to replay discovery examples and visualize top-feature maps.', '',
        'Before suppressing a feature, inspect examples where it fires on both covered and uncovered heads.']
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary), flush=True)
    for row in cell_rows:
        if row['rank'] == 1:
            print(f'{row["race_prompt"]} {row["gender_prompt"]}: feature {row["feature"]}, '
                  f'discovery {row["discovery_contrast"]:.3f}, validation {row["validation_contrast"]:.3f}')


if __name__ == '__main__':
    main()
