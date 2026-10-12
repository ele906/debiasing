"""Exploratory detector-label/SAE agreement; no SAE training or intervention."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
# Explicit phrase normalization, not human labels. Only detector labels are used.
PATTERNS = {
    'head_covering': r'\b(head\s*scarf|head\s*covering|headdress|turban|hat|headband)\b',
    'turban': r'\bturban\b',
    'headscarf': r'\b(head\s*scarf|headdress)\b',
    'earrings': r'\bearrings?\b',
    'necklace': r'\bnecklace\b',
    'beard': r'\b(beard|goatee)\b',
    'afro': r'\bafro\b',
    'kimono': r'\bkimono\b',
    'shirt': r'\bshirt\b',
    'tank_top': r'\btank top\b',
    'smile': r'\bsmile\b',
    'flower': r'\bflowers?\b',
    'red_dot': r'\bred dot\b',
}


def auc(y, score):
    """Pairwise ROC AUC with ties worth one half; undefined for one class."""
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))


def matched_difference(x, y, rows, split):
    contrasts = []
    for group in sorted({(r['race_prompt'], r['gender_prompt']) for r in rows}):
        mask = np.array([r['split'] == split and (r['race_prompt'], r['gender_prompt']) == group for r in rows])
        pos, neg = mask & y, mask & ~y
        if pos.any() and neg.any():
            contrasts.append(x[pos].mean(0) - x[neg].mean(0))
    return (np.mean(contrasts, axis=0) if contrasts else None), len(contrasts)


def write(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, lineterminator='\n', fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, default=HERE / 'concept_predictions.csv')
    p.add_argument('--data', type=Path, default=HERE.parent / 'backprop')
    p.add_argument('--output', type=Path, default=HERE / 'sae_comparison')
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()
    if args.top_k < 1:
        p.error('top-k must be positive')
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    with args.predictions.open() as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Need unique nonempty rows')
    if any(r['split'] not in ['train','validation'] for r in rows):
        raise ValueError('Test examples must remain reserved')
    settings = json.loads((args.data / 'generation_settings.json').read_text())
    checkpoint = Path(settings['sae_checkpoint']) / 'state_dict.pth'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != settings['sae_weights_sha256']:
        raise ValueError('Checkpoint changed since activations were collected')
    # Join by sample ID, never by row position or guessed feature path.
    with np.load(args.data / 'sae_features.npz', allow_pickle=False) as f:
        lookup = {str(s): i for i, s in enumerate(f['sample_id'])}
        indices = [lookup[r['sample_id']] for r in rows]
        x = f['X_mean'][indices].astype(float)
        if any(str(f['split'][idx]) != r['split'] for idx, r in zip(indices, rows)):
            raise ValueError('Split mismatch')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features')
    train = np.array([r['split']=='train' for r in rows])
    val = ~train
    sd = x[train].std(0)
    sd[sd < 1e-8] = 1
    summary, candidates, labels = [], [], []
    for concept, pattern in PATTERNS.items():
        y = np.array([any(re.search(pattern, phrase.lower()) for phrase in r['detected_concepts'].split(';')) for r in rows])
        for r, present in zip(rows, y):
            labels.append(dict(sample_id=r['sample_id'], split=r['split'], concept=concept, detector_present=int(present)))
        counts = dict(train_positive=int((train & y).sum()), train_negative=int((train & ~y).sum()), validation_positive=int((val & y).sum()), validation_negative=int((val & ~y).sum()))
        within, ng = matched_difference(x, y, rows, 'train')
        vwithin, nvg = matched_difference(x, y, rows, 'validation')
        if not counts['train_positive'] or not counts['train_negative']:
            summary.append(dict(concept=concept, **counts, status='no discovery class contrast', selection='', feature='', train_auc='', validation_auc='', train_matched_groups=ng, validation_matched_groups=nvg, validation_within_difference=''))
            continue
        global_diff = x[train & y].mean(0) - x[train & ~y].mean(0)
        selection = 'within_prompt_groups' if within is not None else 'global_confounded'
        effect = within if within is not None else global_diff
        ranked = np.argsort(-(effect / sd), kind='stable')[:args.top_k]
        for rank, feature in enumerate(ranked,1):
            a, va = auc(y[train], x[train,feature]), auc(y[val],x[val,feature])
            vdiff = None if vwithin is None else float(vwithin[feature])
            record=dict(concept=concept, rank=rank, feature=int(feature), selection=selection, train_standardized_difference=float(effect[feature]/sd[feature]), train_auc=a, validation_auc='' if va is None else va, train_matched_groups=ng, validation_matched_groups=nvg, validation_within_difference='' if vdiff is None else vdiff)
            candidates.append(record)
            if rank == 1:
                status = 'within-group controls available; inspect validation effect' if ng and nvg else 'insufficient within-group validation controls'
                summary.append(dict(concept=concept, **counts, status=status, **{k:record[k] for k in ['selection','feature','train_auc','validation_auc','train_matched_groups','validation_matched_groups','validation_within_difference']}))
    args.output.mkdir(parents=True)
    write(args.output/'summary.csv',summary,list(summary[0]))
    write(args.output/'candidates.csv',candidates,list(candidates[0]) if candidates else ['concept','feature'])
    write(args.output/'detector_targets.csv',labels,list(labels[0]))
    (args.output/'settings.json').write_text(json.dumps(dict(predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),patterns=PATTERNS,sae_checkpoint=settings['sae_checkpoint'],sae_weights_sha256=settings['sae_weights_sha256'],block=settings['block'],pooling='spatial mean',samples=len(rows),train=int(train.sum()),validation=int(val.sum()),top_k=args.top_k,selection='Train only; within-group standardized mean differences when possible, otherwise flagged global contrast',interpretation='Unreviewed detector-label agreement; not causal validation or evidence of reduced racism'),indent=2))
    report=['# SAE agreement with extracted concepts','','Existing pretrained SAE; no training or suppression. Targets are unreviewed Florence-2 detector phrases, normalized by explicit patterns. Detector absence may be a missed detection.','',f'{len(rows)} images: {train.sum()} train, {val.sum()} validation; test untouched.','', '| Concept | Feature | Train AUC | Validation AUC | Matched groups train/val | Status |','|---|---:|---:|---:|---:|---|']
    for r in summary:
        fmt=lambda v: '' if v=='' else f'{v:.3f}'
        report.append(f'| {r["concept"]} | {r["feature"]} | {fmt(r["train_auc"])} | {fmt(r["validation_auc"])} | {r["train_matched_groups"]}/{r["validation_matched_groups"]} | {r["status"]} |')
    report += ['', 'AUC 0.5 means chance ordering, 1 means perfect separation of detector positives/negatives. Validation labels never choose the feature. Group controls can contain only one positive/negative image in this small pilot; these are exploratory results with no significance claim. High global AUC can reflect prompt-group identity rather than the named concept. Missing matched validation groups prevents distinguishing those explanations. Features can mix concepts. No concept has been causally established.', '', 'Next: inspect spatial SAE maps, review/correct detector labels, expand within-group positive/negative controls, then test interventions and random-feature controls. Clothing and hairstyles are not themselves harm targets.']
    (args.output/'REPORT.md').write_text('\n'.join(report))
    print('\n'.join(report),flush=True)


if __name__=='__main__':
    main()
