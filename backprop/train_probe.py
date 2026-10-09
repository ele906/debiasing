"""FUTURE training entry point: manual backprop on human-labelled head coverings."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy_backprop import (init_model, inference, loss_crossentropy, calc_gradient,
                            update_weights, positive_logit_gradient, save_model)

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE)
    p.add_argument('--output', type=Path, default=HERE / 'trained_probe')
    p.add_argument('--pooling', choices=['mean', 'max'], default='mean')
    p.add_argument('--hidden-size', type=int, default=0, help='0 gives interpretable logistic regression')
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--learning-rate', type=float, default=.01)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    args = p.parse_args()
    if args.iterations < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        p.error('Iterations, batch size and learning rate must be positive')
    with (args.data / 'annotations.csv').open() as f:
        rows = {r['sample_id']: r for r in csv.DictReader(f)}
    with np.load(args.data / 'sae_features.npz', allow_pickle=False) as f:
        X, splits = f[f'X_{args.pooling}'].astype(np.float64), f['split'].copy()
        ids, race, gender = f['sample_id'].copy(), f['y_race'].copy(), f['y_gender'].copy()
    # Blank/uncertain labels are missing, never negative examples.
    labels = []
    for sid in ids:
        value = rows[str(sid)]['head_covering'].strip().lower()
        if value not in ('', 'uncertain', '0', '1'):
            raise ValueError(f'{sid}: head_covering must be 0, 1, uncertain, or blank')
        labels.append(int(value) if value in ('0', '1') else -1)
    y = np.array(labels)
    train = np.flatnonzero((splits == 'train') & (y >= 0))
    val = np.flatnonzero((splits == 'validation') & (y >= 0))
    test = np.flatnonzero((splits == 'test') & (y >= 0))
    if len(np.unique(y[train])) != 2 or len(np.unique(y[val])) != 2:
        raise ValueError('Label both covered and uncovered examples in train AND validation before training')
    if args.output.exists():
        raise ValueError('Output already exists; choose a new --output to preserve earlier runs')
    # Normalization and feature selection never see validation/test statistics.
    mean, std = X[train].mean(0), X[train].std(0).clip(1e-6)
    Z = ((X - mean) / std).T
    model = init_model(X.shape[1], 2, args.hidden_size)
    rng = np.random.default_rng(0)
    best, best_loss, history = None, float('inf'), []
    for step in range(args.iterations):
        batch = rng.choice(train, size=min(args.batch_size, len(train)), replace=False)
        logits, cache = inference(model, Z[:, batch])
        loss, dv = loss_crossentropy(logits, y[batch])
        grads, _ = calc_gradient(model, cache, dv)
        update_weights(model, grads, args.learning_rate, args.weight_decay)
        if (step + 1) % 25 == 0 or step + 1 == args.iterations:
            v_logits, _ = inference(model, Z[:, val])
            vloss, _ = loss_crossentropy(v_logits, y[val])
            row = dict(iteration=step + 1, batch_loss=loss, validation_loss=vloss,
                       validation_accuracy=float((v_logits.argmax(0) == y[val]).mean()))
            history.append(row)
            if vloss < best_loss:
                best, best_loss = copy.deepcopy(model), vloss
            print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True)
    settings_hash = hashlib.sha256((args.data / 'generation_settings.json').read_bytes()).hexdigest()
    metadata = dict(target='head_covering', positive_class=1, pooling=args.pooling,
                    generation_settings_sha256=settings_hash, selection='validation cross entropy',
                    interpretation='Attribute probe, not a fairness score or a causal feature label')
    save_model(args.output / 'model.npz', best, mean, std, metadata)
    (args.output / 'training_history.json').write_text(json.dumps(history, indent=2))
    report = {}
    for split, idx in [('validation', val), ('test', test)]:
        if not len(idx):
            report[split] = {'labelled_samples': 0}; continue
        logits, _ = inference(best, Z[:, idx])
        pred = logits.argmax(0)
        report[split] = dict(labelled_samples=len(idx), accuracy=float((pred == y[idx]).mean()),
                            confusion=np.array([[np.sum((y[idx] == a) & (pred == b)) for b in (0, 1)]
                                                for a in (0, 1)]).tolist())
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2))
    # Equal-weight within-cell covered-vs-uncovered contrasts reduce prompt-group confounding.
    contrasts = []
    positive_idx = []
    for r in range(6):
        for g in range(2):
            a = train[(race[train] == r) & (gender[train] == g) & (y[train] == 1)]
            b = train[(race[train] == r) & (gender[train] == g) & (y[train] == 0)]
            if len(a) >= 2 and len(b) >= 2:
                contrasts.append((X[a].mean(0) - X[b].mean(0)) / std)
                positive_idx.extend(a.tolist())
    if not contrasts:
        (args.output / 'feature_selection_status.txt').write_text(
            'No candidate ranking: need >=2 covered and >=2 uncovered TRAIN examples within at least one race/gender cell.\n')
        return
    contrast = np.stack(contrasts).mean(0)
    gradient = positive_logit_gradient(best, Z[:, positive_idx]) / std[:, None]
    local_margin_contribution = (gradient * X[positive_idx].T).mean(1)
    candidates = np.flatnonzero((contrast > 0) & (local_margin_contribution > 0))
    candidates = candidates[np.argsort(-local_margin_contribution[candidates])][:50]
    with (args.output / 'candidate_features.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['feature', 'within_cell_contrast', 'local_margin_contribution', 'matched_cells'])
        writer.writeheader()
        writer.writerows(dict(feature=int(j), within_cell_contrast=float(contrast[j]),
            local_margin_contribution=float(local_margin_contribution[j]), matched_cells=len(contrasts)) for j in candidates)


if __name__ == '__main__':
    main()
