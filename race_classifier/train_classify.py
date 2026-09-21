"""
Train/eval step only: loads the cached feature matrix from generate_data.py and fits
a race classifier against it. No image generation, no GPU required — runs in seconds
on CPU, so iterate here freely without re-running generate_data.py.

Input:  race_classifier/race_classify_features.npz
Output: race_classifier/race_classify_results.json + printed report
"""
import json
import numpy as np
import torch

FEATURES_NPZ = '/n/fs/goose/race_classifier/race_classify_features.npz'
OUT_JSON = '/n/fs/goose/race_classifier/race_classify_results.json'

TEST_FRAC = 0.2
EPOCHS = 300
LR = 0.05
WEIGHT_DECAY = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = np.load(FEATURES_NPZ, allow_pickle=True)
X = data['X']
y_race = data['y_race']
y_gender = data['y_gender']
RACES = data['races'].tolist()
GENDERS = data['genders'].tolist()
n_dirs = X.shape[1]
print(f"Loaded X {X.shape} from {FEATURES_NPZ}", flush=True)

# --- stratified train/test split (per race x gender cell) ---
train_idx, test_idx = [], []
rng2 = np.random.default_rng(0)
for race_i in range(len(RACES)):
    for gender_i in range(len(GENDERS)):
        idx = np.where((y_race == race_i) & (y_gender == gender_i))[0]
        rng2.shuffle(idx)
        n_test = max(1, int(len(idx) * TEST_FRAC))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

# --- put data into train/test arrays ---
train_idx = np.array(train_idx)
test_idx = np.array(test_idx)

Xtr, Xte = X[train_idx], X[test_idx]
ytr, yte = y_race[train_idx], y_race[test_idx]

# normalize data
mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
Xtr_n = (Xtr - mu) / sigma
Xte_n = (Xte - mu) / sigma

Xtr_t = torch.tensor(Xtr_n, dtype=torch.float32, device=DEVICE)
ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
Xte_t = torch.tensor(Xte_n, dtype=torch.float32, device=DEVICE)
yte_t = torch.tensor(yte, dtype=torch.long, device=DEVICE)

# parameters to train: W (weight matrix), b (bias)
n_classes = len(RACES)
W = torch.zeros(n_dirs, n_classes, device=DEVICE, requires_grad=True)
b = torch.zeros(n_classes, device=DEVICE, requires_grad=True)
opt = torch.optim.Adam([W, b], lr=LR, weight_decay=WEIGHT_DECAY)

# train loop
print("Training logistic regression (race, 6-class)...", flush=True)
for epoch in range(EPOCHS):
    opt.zero_grad()
    logits = Xtr_t @ W + b
    loss = torch.nn.functional.cross_entropy(logits, ytr_t)  # negative log-softmax for the correct class
    loss.backward()
    opt.step()
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        with torch.no_grad():
            train_acc = (logits.argmax(-1) == ytr_t).float().mean().item()
        print(f"  epoch {epoch}: loss={loss.item():.4f} train_acc={train_acc:.3f}", flush=True)

with torch.no_grad():
    test_logits = Xte_t @ W + b
    test_pred = test_logits.argmax(-1)
    test_acc = (test_pred == yte_t).float().mean().item()

    n = n_classes
    confusion = np.zeros((n, n), dtype=int)
    for true_i, pred_i in zip(yte_t.cpu().numpy(), test_pred.cpu().numpy()):
        confusion[true_i, pred_i] += 1

    # nearest-centroid baseline for comparison
    centroids = torch.stack([Xtr_t[ytr_t == c].mean(0) for c in range(n_classes)])
    dists = torch.cdist(Xte_t, centroids)
    nc_pred = dists.argmin(-1)
    nc_acc = (nc_pred == yte_t).float().mean().item()

    # chance baseline
    chance_acc = 1.0 / n_classes

    # top-weighted features per class (which SAE dims the classifier relies on)
    top_feats_per_class = {}
    for c in range(n_classes):
        w_c = W[:, c].detach().cpu().numpy()
        top_idx = np.argsort(-np.abs(w_c))[:10]
        top_feats_per_class[RACES[c]] = [(int(i), float(w_c[i])) for i in top_idx]

print("\n=== RESULTS ===")
print(f"n_train={len(train_idx)} n_test={len(test_idx)}")
print(f"Chance accuracy: {chance_acc:.3f}")
print(f"Nearest-centroid accuracy: {nc_acc:.3f}")
print(f"Logistic regression test accuracy: {test_acc:.3f}")
print("\nConfusion matrix (rows=true, cols=pred), order:", RACES)
print(confusion)
print("\nTop-weighted SAE feature dims per race (logistic regression):")
for race, feats in top_feats_per_class.items():
    print(f"  {race}: {feats}")

results = {
    "n_train": len(train_idx),
    "n_test": len(test_idx),
    "chance_accuracy": chance_acc,
    "nearest_centroid_accuracy": nc_acc,
    "logreg_test_accuracy": test_acc,
    "races": RACES,
    "confusion_matrix": confusion.tolist(),
    "top_feats_per_class": top_feats_per_class,
}
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {OUT_JSON}")


"""
1. opt.zero_grad() — Clearing old memoryBefore calculating anything new, this clears out the gradients from the previous loop iteration. If you don't do this, PyTorch will accidentally add the new gradients to the old ones (accumulating them).
2. logits = Xtr_t @ W + b — The Forward PassThe model takes the training features (Xtr_t), multiplies them by the weights matrix (W), and adds the bias (b). This generates raw prediction scores (logits) for each class.
3. loss = torch.nn.functional.cross_entropy(...) — Measuring ErrorThe code calculates Cross-Entropy Loss, which measures how far off the predictions (logits) are from the actual true labels (ytr_t). A high loss means the model made poor predictions.
4. loss.backward() — Calculating the GradientsThis is the Backpropagation step.It traces backward through the mathematical computation graph starting from your loss value.It computes the derivative of the loss with respect to every parameter that requires a gradient (in this case, your weights W and bias b).It stores these calculated values inside W.grad and b.grad.Crucial note: At this exact moment, the weights have not changed yet. The model has only calculated how it needs to change them.
5. opt.step() — Updating the WeightsThis is where the actual modification of the model happens. The optimizer (opt) looks inside W.grad and b.grad and uses its internal formula (like standard SGD or Adam) to adjust the actual values of W and b to lower the loss.
"""