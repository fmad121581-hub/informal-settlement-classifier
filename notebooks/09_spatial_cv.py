# =============================================================================
# NOTEBOOK 09 — SPATIAL CROSS-VALIDATION
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
#
# PURPOSE
# -------
# Notebooks 04b/06 report cross-validation accuracy using ordinary random
# StratifiedKFold. For spatial data this can overstate real-world accuracy:
# adjacent wards tend to share urban morphology (a dense informal cluster
# spans several neighbouring wards), so a random split can put near-
# duplicate wards in both the train and test fold, letting the model
# "cheat" via spatial autocorrelation rather than learning a transferable
# informality signal.
#
# This script re-evaluates both the reference (12-feature, rule-circular)
# and decoupled (7-feature, rule-independent) models from notebook 06
# under THREE cross-validation schemes:
#
#   1. RANDOM      — ordinary StratifiedKFold (the NB04b/06 baseline)
#   2. THANA-GROUP — StratifiedGroupKFold grouped by NAME_3 (thana), the
#                    real administrative/spatial unit one level above
#                    ward. No ward from a given thana appears in both the
#                    train and test fold of the same split.
#   3. SPATIAL-BLOCK — StratifiedGroupKFold grouped by KMeans clusters of
#                    ward centroids (k=8 compact spatial blocks). Thana
#                    sizes are very unequal (Dhamrai has 17 labeled wards,
#                    many thanas have 1), so this gives a second, more
#                    size-balanced spatial-blocking scheme — standard
#                    practice in the spatial-CV literature (Roberts et al.
#                    2017) when administrative units are too uneven to
#                    block on directly.
#
# A real drop in accuracy from (1) to (2)/(3) is evidence of spatial
# leakage in the random-CV numbers; a small/no drop is evidence the
# reported accuracy is not just an artifact of spatial autocorrelation.
#
# OUTPUT
# ------
#   outputs/model/spatial_cv_report.txt
#   outputs/figures/09_spatial_cv_comparison.png
#   outputs/figures/09_spatial_blocks_map.png
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_SEED = 42
CV_FOLDS = 5
N_SPATIAL_BLOCKS = 8

FEATURES_EXT = "data/processed/ward_features_extended.csv"
LABELS_CSV   = "data/processed/ward_labels.csv"
WARDS_GPKG   = "data/processed/ward_labeled.gpkg"   # has geometry + label

OUT_REPORT = "outputs/model/spatial_cv_report.txt"
OUT_FIG    = "outputs/figures/09_spatial_cv_comparison.png"
OUT_MAP    = "outputs/figures/09_spatial_blocks_map.png"

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Same feature split as notebook 06
RULE_DERIVED = ["built_fraction", "ndvi_mean", "pop_mean", "lst_mean", "ndbi_mean"]

ALL_FEATURES = [
    "ndvi_mean", "savi_mean", "ndbi_mean", "lst_mean", "slope_mean",
    "pop_mean", "pop_std", "built_fraction",
    "s2_ndbi_mean", "s2_mndwi_mean", "osm_building_density", "osm_mean_building_area",
]
DECOUPLED_FEATURES = [c for c in ALL_FEATURES if c not in RULE_DERIVED]

print("="*70)
print("SPATIAL CROSS-VALIDATION EVALUATION")
print("="*70)

# =============================================================================
# LOAD + MERGE DATA (features + labels + geometry)
# =============================================================================

df_ext = pd.read_csv(FEATURES_EXT).drop(columns=["NAME_4", "NAME_3", "NAME_2"], errors="ignore")
gdf    = gpd.read_file(WARDS_GPKG)[["GID_4", "NAME_3", "label", "label_src", "geometry"]]

df = gdf.merge(df_ext, on="GID_4", how="left")
df_labeled = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)

print(f"\nLabeled wards: {len(df_labeled)} "
      f"(formal={ (df_labeled['label']==1).sum() }, informal={ (df_labeled['label']==0).sum() })")
print(f"Unique thanas among labeled wards: {df_labeled['NAME_3'].nunique()}")

# =============================================================================
# BUILD SPATIAL GROUPS
# =============================================================================

# --- Group A: thana (real administrative unit) ---
df_labeled["group_thana"] = df_labeled["NAME_3"].astype(str)

# --- Group B: KMeans spatial blocks on ward centroids ---
centroids = np.array([(geom.centroid.x, geom.centroid.y) for geom in df_labeled.geometry])
km = KMeans(n_clusters=N_SPATIAL_BLOCKS, random_state=RANDOM_SEED, n_init=10)
df_labeled["group_block"] = km.fit_predict(centroids)

print("\nThana-group sizes (labeled wards per thana):")
print(df_labeled["group_thana"].value_counts().describe())

print("\nSpatial-block sizes (labeled wards per KMeans block):")
print(df_labeled["group_block"].value_counts().sort_index())


def fill_nan(X):
    X = X.astype(float)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    return X


def make_model(y):
    n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)
    scale_w = n_neg / n_pos if n_pos > 0 else 1.0
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_w,
        eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0
    )


def evaluate(df_subset, feature_cols, group_col, scheme_name, model_name):
    X = fill_nan(df_subset[feature_cols].values)
    y = df_subset["label"].values
    groups = df_subset[group_col].values if group_col else None

    if scheme_name == "RANDOM":
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splitter = cv.split(X, y)
    else:
        n_groups = len(np.unique(groups))
        n_splits = min(CV_FOLDS, n_groups)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        splitter = cv.split(X, y, groups=groups)

    accs, f1s, aucs = [], [], []
    n_degenerate = 0
    for train_idx, test_idx in splitter:
        xgb = make_model(y[train_idx])
        xgb.fit(X[train_idx], y[train_idx])
        pred = xgb.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="weighted"))
        if len(np.unique(y[test_idx])) < 2:
            n_degenerate += 1  # fold's test set is single-class -> AUC undefined
        else:
            proba = xgb.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], proba))

    accs = np.array(accs)
    ci95 = 1.96 * accs.std() / np.sqrt(len(accs))
    auc_mean = np.mean(aucs) if aucs else np.nan

    label = f"{model_name} | {scheme_name}"
    print(f"\n{label}")
    print(f"  n_folds={len(accs)}  Accuracy={accs.mean():.4f}±{accs.std():.4f} "
          f"(95% CI ±{ci95:.4f})  F1={np.mean(f1s):.4f}  "
          f"AUC={auc_mean:.4f} (from {len(aucs)}/{len(accs)} folds; "
          f"{n_degenerate} fold(s) had a single-class test set)")

    return {
        "model": model_name, "scheme": scheme_name, "n_folds": len(accs),
        "acc_mean": accs.mean(), "acc_std": accs.std(), "acc_ci95": ci95,
        "f1_mean": np.mean(f1s), "auc_mean": auc_mean, "n_auc_folds": len(aucs),
        "n_degenerate": n_degenerate,
    }


# =============================================================================
# RUN ALL COMBINATIONS
# =============================================================================

results = []
for model_name, feats in [("Reference (12 feat, circular)", ALL_FEATURES),
                           ("Decoupled (7 feat)", DECOUPLED_FEATURES)]:
    results.append(evaluate(df_labeled, feats, None,         "RANDOM",        model_name))
    results.append(evaluate(df_labeled, feats, "group_thana", "THANA-GROUP",   model_name))
    results.append(evaluate(df_labeled, feats, "group_block", "SPATIAL-BLOCK", model_name))

res_df = pd.DataFrame(results)

# =============================================================================
# REPORT
# =============================================================================

with open(OUT_REPORT, "w") as f:
    f.write("="*70 + "\n")
    f.write("SPATIAL CROSS-VALIDATION EVALUATION\n")
    f.write("="*70 + "\n\n")
    f.write(f"Labeled wards: {len(df_labeled)}  "
            f"(formal={(df_labeled['label']==1).sum()}, informal={(df_labeled['label']==0).sum()})\n")
    f.write(f"Thana groups: {df_labeled['NAME_3'].nunique()}  |  Spatial blocks (KMeans): {N_SPATIAL_BLOCKS}\n\n")
    f.write(f"{'Model':35s} {'Scheme':15s} {'Folds':>6s} {'Accuracy':>18s} {'F1':>8s} {'AUC':>18s}\n")
    f.write("-"*105 + "\n")
    for r in results:
        auc_str = f"{r['auc_mean']:.4f} ({r['n_auc_folds']}/{r['n_folds']} folds)"
        f.write(f"{r['model']:35s} {r['scheme']:15s} {r['n_folds']:6d} "
                f"{r['acc_mean']:.4f} +/- {r['acc_ci95']:.4f}   "
                f"{r['f1_mean']:.4f}  {auc_str:>18s}\n")
    f.write("\n")
    f.write("Interpretation:\n")
    for model_name in res_df["model"].unique():
        sub = res_df[res_df["model"] == model_name].set_index("scheme")
        drop_thana = sub.loc["RANDOM", "acc_mean"] - sub.loc["THANA-GROUP", "acc_mean"]
        drop_block = sub.loc["RANDOM", "acc_mean"] - sub.loc["SPATIAL-BLOCK", "acc_mean"]
        f.write(f"  {model_name}: random->thana-group drop = {drop_thana:+.4f}, "
                f"random->spatial-block drop = {drop_block:+.4f}\n")

print(f"\nReport written -> {OUT_REPORT}")

# =============================================================================
# COMPARISON FIGURE (grouped bar chart)
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
schemes = ["RANDOM", "THANA-GROUP", "SPATIAL-BLOCK"]
models  = res_df["model"].unique()
x = np.arange(len(schemes))
width = 0.35
colors = {"Reference (12 feat, circular)": "#94a3b8", "Decoupled (7 feat)": "#dc2626"}

for i, model_name in enumerate(models):
    sub = res_df[res_df["model"] == model_name].set_index("scheme").loc[schemes]
    ax.bar(x + (i - 0.5) * width, sub["acc_mean"], width,
           yerr=sub["acc_ci95"], capsize=5, label=model_name,
           color=colors.get(model_name))
    for xi, v in zip(x + (i - 0.5) * width, sub["acc_mean"]):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(["Random\nStratifiedKFold", "Thana-grouped\n(StratifiedGroupKFold)",
                     "Spatial block\n(KMeans, k=8)"])
ax.set_ylabel("CV Accuracy")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_title("CV accuracy under random vs spatially-grouped folds")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
print(f"Figure written -> {OUT_FIG}")

# =============================================================================
# SPATIAL BLOCKS MAP (for appendix / transparency)
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 10))
df_labeled.plot(ax=ax, column="group_block", categorical=True, cmap="tab10",
                 edgecolor="white", linewidth=0.4, legend=True,
                 legend_kwds={"title": "KMeans spatial block", "loc": "lower right", "fontsize": 8})
ax.set_title(f"Spatial blocks used for grouped CV (k={N_SPATIAL_BLOCKS})\n"
             "Dhaka Metropolitan Region · EPSG:32646", fontsize=12, fontweight="bold")
ax.set_xlabel("Easting (m)", fontsize=9)
ax.set_ylabel("Northing (m)", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_MAP, dpi=150, bbox_inches="tight")
print(f"Figure written -> {OUT_MAP}")

print("\n" + "="*70)
print("DONE")
print("="*70)
