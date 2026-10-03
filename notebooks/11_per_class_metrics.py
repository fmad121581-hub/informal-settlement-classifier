# =============================================================================
# NOTEBOOK 11 — PER-CLASS METRICS & CONFUSION MATRIX
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
#
# PURPOSE
# -------
# NB04b/06/09 report accuracy, weighted-F1 and ROC-AUC. These can look
# good even when the minority/under-represented class (formal, 109 vs 61
# here -- actually formal is the MAJORITY; informal is the smaller class)
# is poorly characterised. This notebook builds OUT-OF-FOLD predictions
# (every ward predicted exactly once, by a model that never saw it in
# training) for the reference and decoupled models under both RANDOM and
# THANA-GROUPED cross-validation, then reports:
#   - Confusion matrix
#   - Per-class precision / recall / F1
#   - Macro-F1, balanced accuracy, Matthews correlation coefficient (MCC)
# MCC and balanced accuracy are included because they do not inflate
# under class imbalance the way plain accuracy can.
#
# OUTPUT
# ------
#   outputs/model/per_class_metrics_report.txt
#   outputs/figures/11_confusion_matrices.png
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              f1_score, balanced_accuracy_score, matthews_corrcoef,
                              accuracy_score)
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

RANDOM_SEED = 42
CV_FOLDS = 5

FEATURES_EXT = "data/processed/ward_features_extended.csv"
WARDS_GPKG   = "data/processed/ward_labeled.gpkg"

OUT_REPORT = "outputs/model/per_class_metrics_report.txt"
OUT_FIG    = "outputs/figures/11_confusion_matrices.png"

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

RULE_DERIVED = ["built_fraction", "ndvi_mean", "pop_mean", "lst_mean", "ndbi_mean"]
ALL_FEATURES = [
    "ndvi_mean", "savi_mean", "ndbi_mean", "lst_mean", "slope_mean",
    "pop_mean", "pop_std", "built_fraction",
    "s2_ndbi_mean", "s2_mndwi_mean", "osm_building_density", "osm_mean_building_area",
]
DECOUPLED_FEATURES = [c for c in ALL_FEATURES if c not in RULE_DERIVED]
# label: 1 = formal, 0 = informal
CLASS_NAMES = ["Informal (0)", "Formal (1)"]

print("="*70)
print("PER-CLASS METRICS & CONFUSION MATRIX")
print("="*70)

# =============================================================================
# LOAD
# =============================================================================

df_ext = pd.read_csv(FEATURES_EXT).drop(columns=["NAME_4", "NAME_3", "NAME_2"], errors="ignore")
gdf = gpd.read_file(WARDS_GPKG)[["GID_4", "NAME_3", "label", "label_src", "geometry"]]
df = gdf.merge(df_ext, on="GID_4", how="left")
df_labeled = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)

df_labeled["group_thana"] = df_labeled["NAME_3"].astype(str)
centroids = np.array([(g.centroid.x, g.centroid.y) for g in df_labeled.geometry])
km = KMeans(n_clusters=8, random_state=RANDOM_SEED, n_init=10)
df_labeled["group_block"] = km.fit_predict(centroids)

print(f"\nLabeled wards: {len(df_labeled)} "
      f"(formal={(df_labeled['label']==1).sum()}, informal={(df_labeled['label']==0).sum()})")


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


def out_of_fold_predictions(df_subset, feature_cols, scheme):
    X = fill_nan(df_subset[feature_cols].values)
    y = df_subset["label"].values
    oof_pred = np.full(len(y), -1)

    if scheme == "RANDOM":
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splits = cv.split(X, y)
    elif scheme == "THANA-GROUP":
        groups = df_subset["group_thana"].values
        n_splits = min(CV_FOLDS, len(np.unique(groups)))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        splits = cv.split(X, y, groups=groups)
    else:
        raise ValueError(scheme)

    for train_idx, test_idx in splits:
        m = make_model(y[train_idx])
        m.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = m.predict(X[test_idx])

    return y, oof_pred


def report_block(df_subset, feature_cols, scheme, model_name):
    y, pred = out_of_fold_predictions(df_subset, feature_cols, scheme)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    macro_f1 = f1_score(y, pred, average="macro")
    bal_acc = balanced_accuracy_score(y, pred)
    mcc = matthews_corrcoef(y, pred)
    acc = accuracy_score(y, pred)

    lines = []
    lines.append(f"{model_name} | {scheme}")
    lines.append(f"  Out-of-fold accuracy : {acc:.4f}")
    lines.append(f"  Balanced accuracy    : {bal_acc:.4f}")
    lines.append(f"  Macro-F1             : {macro_f1:.4f}")
    lines.append(f"  MCC                  : {mcc:.4f}")
    lines.append(f"  Confusion matrix (rows=true, cols=pred) [Informal, Formal]:")
    lines.append(f"    {cm[0].tolist()}")
    lines.append(f"    {cm[1].tolist()}")
    for i, cname in enumerate(CLASS_NAMES):
        lines.append(f"  {cname:15s} precision={prec[i]:.4f}  recall={rec[i]:.4f}  "
                      f"f1={f1[i]:.4f}  support={support[i]}")
    print("\n" + "\n".join(lines))

    return {"model": model_name, "scheme": scheme, "cm": cm, "prec": prec, "rec": rec,
            "f1": f1, "support": support, "macro_f1": macro_f1, "bal_acc": bal_acc,
            "mcc": mcc, "acc": acc}


configs = [
    ("Reference (12 feat, circular)", ALL_FEATURES),
    ("Decoupled (7 feat)", DECOUPLED_FEATURES),
]
schemes = ["RANDOM", "THANA-GROUP"]

results = []
for model_name, feats in configs:
    for scheme in schemes:
        results.append(report_block(df_labeled, feats, scheme, model_name))

# =============================================================================
# REPORT
# =============================================================================

with open(OUT_REPORT, "w") as f:
    f.write("="*70 + "\n")
    f.write("PER-CLASS METRICS & CONFUSION MATRIX (out-of-fold predictions)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Labeled wards: {len(df_labeled)}  "
            f"(formal={(df_labeled['label']==1).sum()}, informal={(df_labeled['label']==0).sum()})\n\n")
    for r in results:
        f.write(f"{r['model']} | {r['scheme']}\n")
        f.write(f"  Accuracy={r['acc']:.4f}  Balanced accuracy={r['bal_acc']:.4f}  "
                f"Macro-F1={r['macro_f1']:.4f}  MCC={r['mcc']:.4f}\n")
        f.write(f"  Confusion matrix [rows=true, cols=pred; order=Informal,Formal]:\n")
        f.write(f"    {r['cm'][0].tolist()}\n")
        f.write(f"    {r['cm'][1].tolist()}\n")
        for i, cname in enumerate(CLASS_NAMES):
            f.write(f"  {cname:15s} precision={r['prec'][i]:.4f}  recall={r['rec'][i]:.4f}  "
                    f"f1={r['f1'][i]:.4f}  support={r['support'][i]}\n")
        f.write("\n")

print(f"\nReport written -> {OUT_REPORT}")

# =============================================================================
# FIGURE — 2x2 confusion matrix grid
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, r in zip(axes.flatten(), results):
    cm = r["cm"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{r['model']}\n{r['scheme']}  (Bal.Acc={r['bal_acc']:.3f}, MCC={r['mcc']:.3f})",
                 fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
print(f"Figure written -> {OUT_FIG}")

print("\n" + "="*70)
print("DONE")
print("="*70)
