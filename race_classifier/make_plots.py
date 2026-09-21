"""
Plotting step: reads race_classify_features.npz + race_classify_results.json
(both produced by generate_data.py / train_classify.py) and renders PNGs into
race_classifier/plots/. No GPU, no regeneration — pure analysis on cached data.

Outputs (race_classifier/plots/):
  01_confusion_matrix.png       - logistic regression confusion matrix, as a heatmap
  02_pca_scatter.png            - 2D PCA of the 600 pooled activation vectors, colored by race
  03_per_race_accuracy.png      - logreg vs nearest-centroid accuracy, per race
  04_confound_feature_dists.png - box plots of features 138/3279/2182/4511 activation, by race
  05_top_weight_bars.png        - top-10 |weight| per race from the logistic regression
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FEATURES_NPZ = '/n/fs/goose/race_classifier/race_classify_features.npz'
RESULTS_JSON = '/n/fs/goose/race_classifier/race_classify_results.json'
PLOT_DIR = '/n/fs/goose/race_classifier/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# --- dataviz skill categorical palette (fixed hue order, light mode) ---
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
SEQUENTIAL_BLUE = ["#eaf2fc", "#bdd7f5", "#7fb0ea", "#2a78d6", "#184f8f"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "font.size": 11,
})

data = np.load(FEATURES_NPZ, allow_pickle=True)
X = data['X']
y_race = data['y_race']
y_gender = data['y_gender']
RACES = data['races'].tolist()
GENDERS = data['genders'].tolist()

with open(RESULTS_JSON) as f:
    results = json.load(f)

RACE_COLOR = {race: CATEGORICAL[i % len(CATEGORICAL)] for i, race in enumerate(RACES)}


# ---------------------------------------------------------------- 1. confusion matrix
def plot_confusion_matrix():
    cm = np.array(results["confusion_matrix"])
    races = results["races"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    vmax = cm.max()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(races)), races, rotation=40, ha="right")
    ax.set_yticks(range(len(races)), races)
    ax.set_xlabel("Predicted race")
    ax.set_ylabel("True race")
    ax.set_title(f"Confusion matrix — logistic regression\n(test accuracy: {results['logreg_test_accuracy']:.0%})")
    for i in range(len(races)):
        for j in range(len(races)):
            v = cm[i, j]
            color = "white" if v > vmax * 0.6 else TEXT_PRIMARY
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "01_confusion_matrix.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. PCA scatter
def plot_pca_scatter():
    Xc = X - X.mean(0, keepdims=True)
    # SVD-based PCA (no sklearn dependency)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    var_explained = (S[:2] ** 2) / (S ** 2).sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, race in enumerate(RACES):
        mask = y_race == i
        ax.scatter(
            pcs[mask, 0], pcs[mask, 1],
            s=28, alpha=0.85, color=CATEGORICAL[i % len(CATEGORICAL)],
            label=race, edgecolors="white", linewidths=0.4,
        )
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")
    ax.set_title("PCA of pooled SAE activation vectors, colored by race")
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "02_pca_scatter.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. per-race accuracy (logreg vs nearest-centroid)
def plot_per_race_accuracy():
    # recompute per-race breakdown from the confusion matrix (logreg) — nearest-centroid
    # per-race isn't saved, so recompute both from scratch using the same split logic.
    import torch
    TEST_FRAC = 0.2
    rng2 = np.random.default_rng(0)
    train_idx, test_idx = [], []
    for race_i in range(len(RACES)):
        for gender_i in range(len(GENDERS)):
            idx = np.where((y_race == race_i) & (y_gender == gender_i))[0]
            rng2.shuffle(idx)
            n_test = max(1, int(len(idx) * TEST_FRAC))
            test_idx.extend(idx[:n_test].tolist())
            train_idx.extend(idx[n_test:].tolist())
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y_race[train_idx], y_race[test_idx]
    mu, sigma = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_n, Xte_n = (Xtr - mu) / sigma, (Xte - mu) / sigma

    centroids = np.stack([Xtr_n[ytr == c].mean(0) for c in range(len(RACES))])
    dists = np.linalg.norm(Xte_n[:, None, :] - centroids[None, :, :], axis=2)
    nc_pred = dists.argmin(1)

    cm = np.array(results["confusion_matrix"])
    logreg_per_race = cm.diagonal() / cm.sum(1)

    nc_per_race = []
    for c in range(len(RACES)):
        mask = yte == c
        nc_per_race.append((nc_pred[mask] == c).mean())

    x = np.arange(len(RACES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, logreg_per_race, width, label="Logistic regression", color=CATEGORICAL[0])
    ax.bar(x + width / 2, nc_per_race, width, label="Nearest centroid", color=CATEGORICAL[1])
    ax.axhline(1 / len(RACES), color=TEXT_SECONDARY, linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x, RACES, rotation=30, ha="right")
    ax.set_ylabel("Per-race test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-race accuracy: logistic regression vs. nearest-centroid")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "03_per_race_accuracy.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 4. confound feature distributions
def plot_confound_feature_dists():
    CONFOUND_FEATURES = [138, 3279, 2182, 4511]
    fig, axes = plt.subplots(1, len(CONFOUND_FEATURES), figsize=(16, 4.5), sharey=False)
    for ax, feat_idx in zip(axes, CONFOUND_FEATURES):
        vals_by_race = [X[y_race == i, feat_idx] for i in range(len(RACES))]
        bp = ax.boxplot(
            vals_by_race, patch_artist=True, showfliers=False,
            medianprops=dict(color=TEXT_PRIMARY, linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], CATEGORICAL):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(TEXT_SECONDARY)
        ax.set_xticks(range(1, len(RACES) + 1), RACES, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"feature {feat_idx}", fontsize=11)
        ax.set_ylabel("max-pooled activation" if feat_idx == CONFOUND_FEATURES[0] else "")
    fig.suptitle("Activation distributions of previously-flagged confound features, by race")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "04_confound_feature_dists.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 5. top-weight bars per race
def plot_top_weight_bars():
    top_feats = results["top_feats_per_class"]
    n_races = len(top_feats)
    fig, axes = plt.subplots(2, (n_races + 1) // 2, figsize=(16, 8), sharex=False)
    axes = axes.flatten()
    for ax, (race, feats) in zip(axes, top_feats.items()):
        feats_sorted = sorted(feats, key=lambda t: abs(t[1]), reverse=True)
        idxs = [f"f{i}" for i, _ in feats_sorted]
        weights = [w for _, w in feats_sorted]
        colors = [CATEGORICAL[2] if w > 0 else CATEGORICAL[7 % len(CATEGORICAL)] for w in weights]
        ax.barh(range(len(idxs)), weights, color=colors)
        ax.set_yticks(range(len(idxs)), idxs, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_title(race, fontsize=11)
    for ax in axes[n_races:]:
        ax.axis("off")
    fig.suptitle("Top-10 |weight| logistic-regression features per race\n(blue = positive weight, red = negative)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "05_top_weight_bars.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_pca_scatter()
    plot_per_race_accuracy()
    plot_confound_feature_dists()
    plot_top_weight_bars()
    print(f"Saved plots to {PLOT_DIR}")
