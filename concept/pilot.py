"""Prepare concept review, then rank SAE associations from reviewed labels."""
import argparse
import csv
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'backprop'


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, lineterminator='\n', fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def prepare(args):
    rows = read_csv(args.data / 'manifest.csv')
    selected, counts = [], {}
    for row in rows:
        key = (row['race_prompt'], row['gender_prompt'], row['split'])
        if row['split'] == 'test' or counts.get(key, 0) >= args.per_cell:
            continue
        image = (args.data / row['image']).resolve()
        features = (args.data / row['features']).resolve()
        if not image.is_file() or not features.is_file():
            raise ValueError(f'Missing image/features: {row["sample_id"]}')
        counts[key] = counts.get(key, 0) + 1
        selected.append(dict(row, image=str(image), features=str(features),
                             concepts='', harmful_depiction='', harm_reason='', reviewer='', notes=''))
    if not selected:
        raise ValueError('No discovery/validation samples')
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'review.csv', selected, list(selected[0]))
    cards = []
    for r in selected:
        esc = lambda s: html.escape(str(s), quote=True)
        # Relative image links work when the workspace is served or opened locally.
        import os
        src = os.path.relpath(r['image'], args.output)
        cards.append(f'<article><img loading="lazy" src="{esc(src)}"><p>{esc(r["sample_id"])}<br>{esc(r["prompt"])}<br>{esc(r["split"])}</p></article>')
    (args.output / 'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Concept review pilot</title><style>body{font:16px sans-serif;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}img{width:100%}article{border:1px solid #ccc;padding:8px}</style><h1>Concept review pilot</h1><p>Record visible concepts in review.csv, separated by semicolons. Harm judgments need a separate rationale. Prompt labels are not inferred identities.</p><main>' + ''.join(cards) + '</main>')
    settings = json.loads((args.data / 'generation_settings.json').read_text())
    (args.output / 'provenance.json').write_text(json.dumps(dict(source=str(args.data.resolve()), generation=settings, samples=len(selected), per_cell_per_split=args.per_cell, test_used=False), indent=2))
    print(f'Prepared {len(selected)} images: {args.output / "gallery.html"}')


def rank(args):
    import numpy as np
    rows = read_csv(args.review)
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs')
    for r in rows:
        if r['split'] not in ('train', 'validation'):
            raise ValueError('Rank only discovery/validation samples; reserve test')
    # Blank concept lists mean unreviewed, not concept absence.
    rows = [r for r in rows if r['concepts'].strip()]
    if not rows:
        raise ValueError('No reviewed concepts yet. Use NONE for a reviewed empty list.')
    labels = [{c.strip().lower() for c in r['concepts'].split(';') if c.strip() and c.strip().lower() != 'none'} for r in rows]
    vectors = []
    for r in rows:
        with np.load(r['features'], allow_pickle=False) as f:
            vectors.append(f[args.pooling])
    x = np.stack(vectors)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite pooled feature vectors')
    concepts = sorted(set().union(*labels))
    results, frequencies = [], []
    for concept in concepts:
        y = np.array([concept in c for c in labels])
        effects = {}
        for split in ('train', 'validation'):
            groups = sorted({(r['race_prompt'], r['gender_prompt']) for r in rows if r['split'] == split})
            contrasts = []
            for race, gender in groups:
                mask = np.array([r['split'] == split and r['race_prompt'] == race and r['gender_prompt'] == gender for r in rows])
                pos, neg = mask & y, mask & ~y
                frequencies.append(dict(concept=concept, split=split, race_prompt=race, gender_prompt=gender, reviewed=int(mask.sum()), present=int(pos.sum())))
                if pos.sum() >= args.min_count and neg.sum() >= args.min_count:
                    contrasts.append(x[pos].mean(0) - x[neg].mean(0))
            effects[split] = (np.mean(contrasts, axis=0) if contrasts else None, len(contrasts))
        train, ntrain = effects['train']
        val, nval = effects['validation']
        if train is None:
            print(f'{concept}: no within-group positive/negative controls; no ranking')
            continue
        for order, feature in enumerate(np.argsort(-train)[:args.top_k], 1):
            results.append(dict(concept=concept, rank=order, feature=int(feature), train_mean_difference=float(train[feature]), validation_mean_difference='' if val is None else float(val[feature]), train_matched_groups=ntrain, validation_matched_groups=nval))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'candidates.csv', results, ['concept','rank','feature','train_mean_difference','validation_mean_difference','train_matched_groups','validation_matched_groups'])
    write_csv(args.output / 'frequencies.csv', frequencies, ['concept','split','race_prompt','gender_prompt','reviewed','present'])
    (args.output / 'settings.json').write_text(json.dumps(dict(review=str(args.review.resolve()), pooling=args.pooling, min_count=args.min_count, top_k=args.top_k, reviewed_samples=len(rows), interpretation='Within-prompt-group associations, not semantic or causal validation'), indent=2))
    print(f'Saved {len(results)} candidate associations to {args.output}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--data', type=Path, default=DATA)
    q.add_argument('--output', type=Path, default=HERE / 'pilot')
    q.add_argument('--per-cell', type=int, default=3)
    q.set_defaults(run=prepare)
    q = sub.add_parser('rank')
    q.add_argument('--review', type=Path, default=HERE / 'pilot/review.csv')
    q.add_argument('--output', type=Path, default=HERE / 'pilot/rankings')
    q.add_argument('--pooling', choices=['mean', 'maximum', 'active_fraction'], default='mean')
    q.add_argument('--min-count', type=int, default=2)
    q.add_argument('--top-k', type=int, default=10)
    q.set_defaults(run=rank)
    args = p.parse_args()
    for name in ('per_cell', 'min_count', 'top_k'):
        if hasattr(args, name) and getattr(args, name) < 1:
            p.error(f'{name} must be positive')
    args.run(args)


if __name__ == '__main__':
    main()
