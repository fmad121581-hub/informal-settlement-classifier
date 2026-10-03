# =============================================================================
# NOTEBOOK 10 — FEATURE ABLATION STUDY
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
#
# PURPOSE
# -------
# Notebook 06 reports one number for the "decoupled" (7-feature) model.
# A reviewer's natural question: which features are actually driving that
# number, and is the gain just "feature accumulation"? This notebook adds
# feature groups one at a time -- staying ENTIRELY within the rule-
# independent (non-circular) feature set until the final step -- and
# reports both CV accuracy (random StratifiedKFold, as in NB06) and
# external validation (Spearman r vs EO4SD 2017, as in NB07/08) at each
# stage. The rule-circular reference model (all 12 features, including
# the 5 used to build the rule-based labels) is included ONLY as a final
# row for context, clearly flagged as circular.
#
# FEATURE GROUPS (cumulative, all rule-independent until M5)
#   M1  Spectral/terrain core      : savi_mean, slope_mean
#   M2  + population structure     : + pop_std
#   M3  + Sentinel-2 indices       : + s2_ndbi_mean, s2_mndwi_mean
#   M4  + OSM building morphology  : + osm_building_density, osm_mean_building_area
#        (M4 = the full 7-feature "decoupled" model from NB06/08)
#   M5  Reference (CIRCULAR)       : full 12-feature set incl. rule-derived vars
#
# OUTPUT
# ------
#   outputs/model/feature_ablation_report.txt
#   outputs/figures/10_feature_ablation.png
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

RANDOM_SEED = 42
CV_FOLDS = 5

FEATURES_EXT = "data/processed/ward_features_extended.csv"
LABELS_CSV   = "data/processed/ward_labels.csv"
EO4SD_CSV    = "data/processed/ward_eo4sd_comparison.csv"

OUT_REPORT = "outputs/model/feature_ablation_report.txt"
OUT_FIG    = "outputs/figures/10_feature_ablation.png"

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

GROUPS = [
    ("M1", "Spectral/terrain core",         ["savi_mean", "slope_mean"]),
    ("M2", "+ population structure",        ["pop_std"]),
    ("M3", "+ Sentinel-2 indices",          ["s2_ndbi_mean", "s2_mndwi_mean"]),
    ("M4", "+ OSM building morphology",     ["osm_building_density", "osm_mean_building_area"]),
]
REFERENCE_FEATURES = [
    "ndvi_mean", "savi_mean", "ndbi_mean", "lst_mean", "slope_mean",
    "pop_mean", "pop_std", "built_fraction",
    "s2_ndbi_mean", "s2_mndwi_mean", "osm_building_density", "osm_mean_building_area",
]

print("="*70)
print("FEATURE ABLATION STUDY")
print("="*70)

# =============================================================================
# LOAD
# =============================================================================

df_ext = pd.read_csv(FEATURES_EXT)
df_lab = pd.read_csv(LABELS_CSV)
df_eo4sd = pd.read_csv(EO4SD_CSV)

df = df_ext.merge(df_lab[["GID_4", "label", "label_src"]], on="GID_4", how="left")
df["label"] = df["label"].fillna(-1).astype(int)
df_labeled = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)

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


def cv_accuracy(df_subset, feature_cols):
    X = fill_nan(df_subset[feature_cols].values)
    y = df_subset["label"].values
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    accs, f1s = [], []
    for train_idx, test_idx in cv.split(X, y):
        m = make_model(y[train_idx])
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="weighted"))
    accs = np.array(accs)
    ci95 = 1.96 * accs.std() / np.sqrt(len(accs))
    return accs.mean(), ci95, np.mean(f1s)


def external_correlation(df_subset, feature_cols, df_all, df_eo4sd):
    X = fill_nan(df_subset[feature_cols].values)
    y = df_subset["label"].values
    model = make_model(y)
    model.fit(X, y)
    X_all = fill_nan(df_all[feature_cols].values)
    prob_informal = 1 - model.predict_proba(X_all)[:, 1]
    pred_df = df_all[["GID_4"]].copy()
    pred_df["prob_informal"] = prob_informal
    merged = pred_df.merge(df_eo4sd[["GID_4", "eo4sd_informal_fraction"]], on="GID_4", how="left")
    valid = merged.dropna(subset=["prob_informal", "eo4sd_informal_fraction"])
    r_s, p_s = stats.spearmanr(valid["prob_informal"], valid["eo4sd_informal_fraction"])
    return r_s, p_s, len(valid)


# =============================================================================
# RUN ABLATION
# =============================================================================

results = []
cumulative_features = []
for code, desc, feats in GROUPS:
    cumulative_features += feats
    acc, ci95, f1 = cv_accuracy(df_labeled, cumulative_features)
    r_s, p_s, n_ext = external_correlation(df_labeled, cumulative_features, df, df_eo4sd)
    print(f"\n{code} — {desc}  (features so far: {cumulative_features})")
    print(f"  CV Accuracy = {acc:.4f} ± {ci95:.4f}   F1 = {f1:.4f}")
    print(f"  External Spearman r = {r_s:.4f} (p={p_s:.4g}, n={n_ext})")
    results.append({"code": code, "desc": desc, "n_features": len(cumulative_features),
                     "features": list(cumulative_features),
                     "acc": acc, "ci95": ci95, "f1": f1, "ext_r": r_s, "ext_p": p_s})

# Reference / circular model — final context row
acc, ci95, f1 = cv_accuracy(df_labeled, REFERENCE_FEATURES)
r_s, p_s, n_ext = external_correlation(df_labeled, REFERENCE_FEATURES, df, df_eo4sd)
print(f"\nM5 — Reference (CIRCULAR, incl. rule-derived features)  "
      f"(features: {REFERENCE_FEATURES})")
print(f"  CV Accuracy = {acc:.4f} ± {ci95:.4f}   F1 = {f1:.4f}")
print(f"  External Spearman r = {r_s:.4f} (p={p_s:.4g}, n={n_ext})")
results.append({"code": "M5", "desc": "Reference (CIRCULAR, incl. rule-derived)",
                 "n_features": len(REFERENCE_FEATURES), "features": list(REFERENCE_FEATURES),
                 "acc": acc, "ci95": ci95, "f1": f1, "ext_r": r_s, "ext_p": p_s})

# =============================================================================
# REPORT
# =============================================================================

with open(OUT_REPORT, "w") as f:
    f.write("="*70 + "\n")
    f.write("FEATURE ABLATION STUDY\n")
    f.write("="*70 + "\n\n")
    f.write(f"Labeled wards: {len(df_labeled)}  "
            f"(formal={(df_labeled['label']==1).sum()}, informal={(df_labeled['label']==0).sum()})\n")
    f.write("M1-M4 are cumulative and entirely rule-independent (non-circular).\n")
    f.write("M5 (reference) includes the 5 rule-derived variables and is circular --\n")
    f.write("shown only for context, not as a fair comparison point.\n\n")
    f.write(f"{'Model':5s} {'Description':30s} {'#Feat':>6s} {'CV Accuracy':>18s} {'F1':>8s} {'External r':>14s}\n")
    f.write("-"*95 + "\n")
    for r in results:
        f.write(f"{r['code']:5s} {r['desc']:30s} {r['n_features']:6d} "
                f"{r['acc']:.4f} +/- {r['ci95']:.4f}   {r['f1']:.4f}  "
                f"{r['ext_r']:.4f} (p={r['ext_p']:.2g})\n")
    f.write("\nCumulative feature lists:\n")
    for r in results:
        f.write(f"  {r['code']}: {r['features']}\n")

print(f"\nReport written -> {OUT_REPORT}")

# =============================================================================
# FIGURE — dual-axis: CV accuracy (bars) + external r (line)
# =============================================================================

fig, ax1 = plt.subplots(figsize=(10, 6))
codes = [r["code"] for r in results]
accs  = [r["acc"] for r in results]
cis   = [r["ci95"] for r in results]
rs    = [r["ext_r"] for r in results]
colors = ["#60a5fa"]*4 + ["#94a3b8"]  # M1-M4 blue, M5 grey (circular/context)

bars = ax1.bar(codes, accs, yerr=cis, capsize=5, color=colors)
ax1.set_ylabel("CV Accuracy", color="#1d4ed8")
ax1.set_ylim(0, 1.05)
ax1.tick_params(axis="y", labelcolor="#1d4ed8")
for b, v in zip(bars, accs):
    ax1.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.3f}", ha="center", fontsize=9)

ax2 = ax1.twinx()
ax2.plot(codes, rs, color="#dc2626", marker="o", linewidth=2, label="External Spearman r (vs EO4SD)")
ax2.set_ylabel("External Spearman r (vs EO4SD 2017)", color="#dc2626")
ax2.set_ylim(0, 1.0)
ax2.tick_params(axis="y", labelcolor="#dc2626")
for xi, v in zip(codes, rs):
    ax2.text(xi, v+0.03, f"{v:.3f}", ha="center", fontsize=9, color="#dc2626")

ax1.set_xticklabels([f"{r['code']}\n{r['desc']}" for r in results], fontsize=8)
ax1.set_title("Feature ablation: CV accuracy and external validation by feature group\n"
              "(M1-M4 rule-independent; M5 circular reference shown for context only)")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
print(f"Figure written -> {OUT_FIG}")

print("\n" + "="*70)
print("DONE")
print("="*70)
