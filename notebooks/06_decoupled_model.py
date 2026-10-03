# =============================================================================
# NOTEBOOK 06 — DECOUPLED / HONEST MODEL
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
#
# PURPOSE
# -------
# The rule-based labels (03_labeling.py) were generated from thresholds on
# built_fraction, ndvi_mean, pop_mean, lst_mean. The extended model
# (04b_retrain.py) then uses those SAME variables (plus ndbi_mean = -ndvi)
# as features. For rule-labeled wards this is circular: the model can
# largely recover the label by re-deriving the threshold rule, not by
# learning a genuine informality signal.
#
# This script re-evaluates honesty in two ways:
#
#   TEST A — "Decoupled features": retrain using ONLY features that were
#            NOT used in the rule-based labeling (savi_mean, slope_mean,
#            pop_std, s2_ndbi_mean, s2_mndwi_mean, osm_building_density,
#            osm_mean_building_area), on the full labeled set (163 wards).
#            This shows what the model can learn when it can't just
#            re-derive the rule.
#
#   TEST B — "Anchor-only": restrict to the 49 ground-truth anchor wards
#            (thana:informal_anchor / thana:formal_anchor), whose labels
#            were NOT rule-derived at all, using the FULL 12-feature set.
#            This is a smaller but fully independent test of whether the
#            features predict genuinely-sourced labels.
#
#   TEST C — Decoupled features + anchor-only labels (strictest test).
#
# OUTPUT
# ------
#   outputs/model/decoupled_model_report.txt
#   outputs/figures/06_decoupled_comparison.png
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

RANDOM_SEED = 42
CV_FOLDS = 5

FEATURES_EXT = "data/processed/ward_features_extended.csv"
LABELS_CSV   = "data/processed/ward_labels.csv"

OUT_REPORT = "outputs/model/decoupled_model_report.txt"
OUT_FIG    = "outputs/figures/06_decoupled_comparison.png"

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Variables used directly in the rule-based labeling thresholds (03_labeling.py)
RULE_DERIVED = ["built_fraction", "ndvi_mean", "pop_mean", "lst_mean", "ndbi_mean"]
# ndbi_mean is included because README states it = -NDVI (same information as ndvi_mean)

ALL_FEATURES = [
    "ndvi_mean", "savi_mean", "ndbi_mean", "lst_mean", "slope_mean",
    "pop_mean", "pop_std", "built_fraction",
    "s2_ndbi_mean", "s2_mndwi_mean", "osm_building_density", "osm_mean_building_area",
]

DECOUPLED_FEATURES = [c for c in ALL_FEATURES if c not in RULE_DERIVED]

print("="*70)
print("DECOUPLED / HONEST MODEL EVALUATION")
print("="*70)
print(f"\nRule-derived features EXCLUDED from decoupled tests: {RULE_DERIVED}")
print(f"Decoupled feature set ({len(DECOUPLED_FEATURES)}): {DECOUPLED_FEATURES}")

# =============================================================================
# LOAD DATA
# =============================================================================

df_ext = pd.read_csv(FEATURES_EXT)
df_lab = pd.read_csv(LABELS_CSV)

df = df_ext.merge(df_lab[["GID_4", "label", "label_src"]], on="GID_4", how="left")
df["label"] = df["label"].fillna(-1).astype(int)

df_labeled = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
df_anchor  = df_labeled[df_labeled["label_src"].str.startswith("thana:")].copy().reset_index(drop=True)

print(f"\nFull labeled set   : {len(df_labeled)} wards "
      f"(formal={ (df_labeled['label']==1).sum() }, informal={ (df_labeled['label']==0).sum() })")
print(f"Anchor-only subset : {len(df_anchor)} wards "
      f"(formal={ (df_anchor['label']==1).sum() }, informal={ (df_anchor['label']==0).sum() })")
print(df_labeled.groupby("label_src").size())


def fill_nan(X):
    X = X.astype(float)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    return X


def evaluate(df_subset, feature_cols, name, min_cv_class_count=2):
    X = fill_nan(df_subset[feature_cols].values)
    y = df_subset["label"].values
    n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)

    result = {"name": name, "n": len(y), "n_formal": n_pos, "n_informal": n_neg,
              "features": feature_cols}

    print("\n" + "-"*70)
    print(f"{name}  (n={len(y)}, formal={n_pos}, informal={n_neg}, features={len(feature_cols)})")
    print("-"*70)

    if min(n_pos, n_neg) < CV_FOLDS:
        print(f"  Too few samples in minority class for {CV_FOLDS}-fold CV — skipping CV.")
        result["cv_f1_mean"] = None
        result["cv_f1_std"]  = None
    else:
        scale_w = n_neg / n_pos if n_pos > 0 else 1.0
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_w,
            eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0
        )
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        cv_f1  = cross_val_score(xgb, X, y, cv=cv, scoring="f1_weighted")
        cv_acc = cross_val_score(xgb, X, y, cv=cv, scoring="accuracy")
        cv_auc = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc")

        # 95% CI via normal approx on the CV fold distribution
        ci95 = 1.96 * cv_acc.std() / np.sqrt(len(cv_acc))

        print(f"  CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}  "
              f"(95% CI ≈ [{cv_acc.mean()-ci95:.4f}, {cv_acc.mean()+ci95:.4f}])")
        print(f"  CV F1       : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        print(f"  CV AUC      : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

        result.update({
            "cv_acc_mean": cv_acc.mean(), "cv_acc_std": cv_acc.std(), "cv_acc_ci95": ci95,
            "cv_f1_mean": cv_f1.mean(), "cv_f1_std": cv_f1.std(),
            "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
        })

        # Fit on full subset to get feature importance
        xgb.fit(X, y)
        importances = dict(zip(feature_cols, xgb.feature_importances_))
        result["importances"] = importances
        print("  Feature importances:")
        for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
            print(f"    {k:28s} {v:.4f}")

    return result


results = []
results.append(evaluate(df_labeled, ALL_FEATURES,       "TEST 0 (reference) — original 12 features, all labeled wards"))
results.append(evaluate(df_labeled, DECOUPLED_FEATURES, "TEST A — decoupled features, all labeled wards"))
results.append(evaluate(df_anchor,  ALL_FEATURES,       "TEST B — original 12 features, anchor-only labels"))
results.append(evaluate(df_anchor,  DECOUPLED_FEATURES, "TEST C — decoupled features, anchor-only labels (strictest)"))

# =============================================================================
# REPORT
# =============================================================================

with open(OUT_REPORT, "w") as f:
    f.write("="*70 + "\n")
    f.write("DECOUPLED / HONEST MODEL EVALUATION\n")
    f.write("="*70 + "\n\n")
    f.write(f"Rule-derived features excluded in decoupled tests: {RULE_DERIVED}\n")
    f.write(f"Decoupled feature set ({len(DECOUPLED_FEATURES)}): {DECOUPLED_FEATURES}\n\n")
    for r in results:
        f.write("-"*70 + "\n")
        f.write(f"{r['name']}\n")
        f.write(f"n={r['n']}  formal={r['n_formal']}  informal={r['n_informal']}\n")
        f.write(f"features={r['features']}\n")
        if r.get("cv_acc_mean") is not None:
            f.write(f"CV Accuracy : {r['cv_acc_mean']:.4f} +/- {r['cv_acc_std']:.4f}  "
                    f"(95% CI +/- {r['cv_acc_ci95']:.4f})\n")
            f.write(f"CV F1       : {r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f}\n")
            f.write(f"CV AUC      : {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f}\n")
            f.write("Feature importances:\n")
            for k, v in sorted(r["importances"].items(), key=lambda kv: -kv[1]):
                f.write(f"  {k:28s} {v:.4f}\n")
        else:
            f.write("CV skipped — insufficient minority-class samples.\n")
        f.write("\n")

print(f"\nReport written -> {OUT_REPORT}")

# =============================================================================
# COMPARISON FIGURE
# =============================================================================

valid = [r for r in results if r.get("cv_acc_mean") is not None]
fig, ax = plt.subplots(figsize=(9, 5.5))
names = [r["name"].split(" — ")[0] for r in valid]
means = [r["cv_acc_mean"] for r in valid]
errs  = [r["cv_acc_ci95"] for r in valid]
colors = ["#94a3b8", "#f97316", "#94a3b8", "#dc2626"]
bars = ax.bar(names, means, yerr=errs, capsize=6, color=colors[:len(valid)])
ax.set_ylabel("CV Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Accuracy: original (circular) vs decoupled feature sets")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (balanced)")
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width()/2, m + 0.03, f"{m:.3f}", ha="center", fontsize=10)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
print(f"Figure written -> {OUT_FIG}")

print("\n" + "="*70)
print("DONE")
print("="*70)
