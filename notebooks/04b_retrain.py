# =============================================================================
# NOTEBOOK 04b — MODEL RETRAINING WITH EXTENDED FEATURES
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
# Author  : (your name)
# Date    : 2024
#
# PURPOSE
# -------
# Retrain XGBoost, Random Forest and SVM using the extended 12-feature set
# (original 8 + S2 NDBI + S2 MNDWI + OSM building density + OSM mean area)
# and compare accuracy against the baseline 8-feature model from NB 04.
#
# INPUT
# -----
#   data/processed/ward_features_extended.csv   ← 12 features, 203 wards
#   data/processed/ward_labels.csv              ← 163 labeled wards
#
# OUTPUT
# ------
#   outputs/model/best_model_extended.joblib
#   outputs/model/scaler_extended.joblib
#   outputs/model/model_report_extended.txt
#   outputs/figures/04b_confusion_matrices.png
#   outputs/figures/04b_feature_importance.png
#   outputs/figures/04b_baseline_vs_extended.png
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

FEATURES_EXT  = "data/processed/ward_features_extended.csv"
LABELS_CSV    = "data/processed/ward_labels.csv"

OUT_MODEL     = "outputs/model/best_model_extended.joblib"
OUT_SCALER    = "outputs/model/scaler_extended.joblib"
OUT_REPORT    = "outputs/model/model_report_extended.txt"
OUT_CM        = "outputs/figures/04b_confusion_matrices.png"
OUT_FI        = "outputs/figures/04b_feature_importance.png"
OUT_COMPARE   = "outputs/figures/04b_baseline_vs_extended.png"

os.makedirs("outputs/model",   exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

RANDOM_SEED  = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

FEATURE_COLS = [
    "ndvi_mean",
    "savi_mean",
    "ndbi_mean",
    "lst_mean",
    "slope_mean",
    "pop_mean",
    "pop_std",
    "built_fraction",
    "s2_ndbi_mean",
    "s2_mndwi_mean",
    "osm_building_density",
    "osm_mean_building_area",
]

FEATURE_LABELS = {
    "ndvi_mean":             "NDVI (Landsat)",
    "savi_mean":             "SAVI (Landsat)",
    "ndbi_mean":             "NDBI proxy (Landsat)",
    "lst_mean":              "LST °C",
    "slope_mean":            "Slope °",
    "pop_mean":              "Pop density",
    "pop_std":               "Pop std dev",
    "built_fraction":        "Built-up fraction",
    "s2_ndbi_mean":          "S2 True NDBI ★",
    "s2_mndwi_mean":         "S2 MNDWI ★",
    "osm_building_density":  "OSM Build. density ★",
    "osm_mean_building_area":"OSM Mean bldg area ★",
}

# Baseline scores from NB 04 — for comparison
BASELINE = {
    "Random Forest": {"cv_f1": 0.9017, "cv_std": 0.0409, "auc": 0.9577},
    "XGBoost":       {"cv_f1": 0.9444, "cv_std": 0.0237, "auc": 0.9654},
    "SVM":           {"cv_f1": 0.9439, "cv_std": 0.0542, "auc": 0.9923},
}


# =============================================================================
# 1.  LOAD AND MERGE DATA
# =============================================================================

print("="*60)
print("STEP 1 — Load extended features and labels")
print("="*60)

df_ext    = pd.read_csv(FEATURES_EXT)
df_labels = pd.read_csv(LABELS_CSV)

print(f"  Extended features : {df_ext.shape}")
print(f"  Labels            : {df_labels.shape}")

# Merge on GID_4
df = df_ext.merge(
    df_labels[["GID_4", "label", "label_src"]],
    on="GID_4", how="left"
)
df["label"] = df["label"].fillna(-1).astype(int)

# Keep only labeled wards
df_labeled = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
print(f"\n  Labeled wards     : {len(df_labeled)}")
print(f"  Formal   (1)      : {(df_labeled['label']==1).sum()}")
print(f"  Informal (0)      : {(df_labeled['label']==0).sum()}")

# Check all feature columns exist
missing_cols = [c for c in FEATURE_COLS if c not in df_labeled.columns]
if missing_cols:
    print(f"\n  WARNING: Missing columns: {missing_cols}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df_labeled.columns]
    print(f"  Using {len(FEATURE_COLS)} available features")

print(f"\n  Features used: {FEATURE_COLS}")

X = df_labeled[FEATURE_COLS].values
y = df_labeled["label"].values

# Fill any NaN
for j in range(X.shape[1]):
    mask = np.isnan(X[:, j])
    if mask.any():
        X[mask, j] = np.nanmedian(X[:, j])

print(f"\n  Feature matrix X : {X.shape}")
print(f"  Class balance    : {np.bincount(y)} (informal=0, formal=1)")


# =============================================================================
# 2.  TRAIN / TEST SPLIT + SCALING
# =============================================================================

print("\n" + "="*60)
print("STEP 2 — Train/test split and feature scaling")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

print(f"  Train : {X_train.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]} samples")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

dump(scaler, OUT_SCALER)
print(f"  Scaler saved → {OUT_SCALER}")


# =============================================================================
# 3.  TRAIN ALL THREE MODELS
# =============================================================================

print("\n" + "="*60)
print("STEP 3 — Train Random Forest")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2, max_features="sqrt",
    class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_acc   = accuracy_score(y_test, rf_pred)
rf_f1    = f1_score(y_test, rf_pred, average="weighted")
rf_auc   = roc_auc_score(y_test, rf_proba)
print(f"  Acc={rf_acc:.4f}  F1={rf_f1:.4f}  AUC={rf_auc:.4f}")

print("\n" + "="*60)
print("STEP 4 — Train XGBoost")
print("="*60)

scale_w = np.sum(y_train==0) / np.sum(y_train==1)
xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_w,
    use_label_encoder=False, eval_metric="logloss",
    random_state=RANDOM_SEED, verbosity=0
)
xgb.fit(X_train, y_train)
xgb_pred  = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_acc   = accuracy_score(y_test, xgb_pred)
xgb_f1    = f1_score(y_test, xgb_pred, average="weighted")
xgb_auc   = roc_auc_score(y_test, xgb_proba)
print(f"  Acc={xgb_acc:.4f}  F1={xgb_f1:.4f}  AUC={xgb_auc:.4f}")

print("\n" + "="*60)
print("STEP 5 — Train SVM")
print("="*60)

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(C=10, kernel="rbf", gamma="scale",
                class_weight="balanced", probability=True,
                random_state=RANDOM_SEED))
])
svm_pipe.fit(X_train, y_train)
svm_pred  = svm_pipe.predict(X_test)
svm_proba = svm_pipe.predict_proba(X_test)[:, 1]
svm_acc   = accuracy_score(y_test, svm_pred)
svm_f1    = f1_score(y_test, svm_pred, average="weighted")
svm_auc   = roc_auc_score(y_test, svm_proba)
print(f"  Acc={svm_acc:.4f}  F1={svm_f1:.4f}  AUC={svm_auc:.4f}")


# =============================================================================
# 4.  CROSS-VALIDATION
# =============================================================================

print("\n" + "="*60)
print("STEP 6 — 5-fold cross-validation")
print("="*60)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                     random_state=RANDOM_SEED)

rf_cv  = cross_val_score(rf,       X, y, cv=cv, scoring="f1_weighted")
xgb_cv = cross_val_score(xgb,      X, y, cv=cv, scoring="f1_weighted")
svm_cv = cross_val_score(svm_pipe, X, y, cv=cv, scoring="f1_weighted")

print(f"  Random Forest CV F1 : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"  XGBoost       CV F1 : {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
print(f"  SVM           CV F1 : {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")


# =============================================================================
# 5.  SELECT BEST MODEL
# =============================================================================

print("\n" + "="*60)
print("STEP 7 — Select best model")
print("="*60)

results = {
    "Random Forest": {"model": rf,       "acc": rf_acc,  "f1": rf_f1,
                      "auc": rf_auc,  "cv_f1": rf_cv.mean(),
                      "cv_std": rf_cv.std(), "pred": rf_pred,
                      "proba": rf_proba},
    "XGBoost":       {"model": xgb,      "acc": xgb_acc, "f1": xgb_f1,
                      "auc": xgb_auc, "cv_f1": xgb_cv.mean(),
                      "cv_std": xgb_cv.std(), "pred": xgb_pred,
                      "proba": xgb_proba},
    "SVM":           {"model": svm_pipe, "acc": svm_acc, "f1": svm_f1,
                      "auc": svm_auc, "cv_f1": svm_cv.mean(),
                      "cv_std": svm_cv.std(), "pred": svm_pred,
                      "proba": svm_proba},
}

print(f"\n  {'Model':15s}  {'Acc':>6}  {'F1':>6}  "
      f"{'AUC':>6}  {'CV F1':>8}  {'CV Std':>7}")
print("  " + "-"*55)
for name, r in results.items():
    baseline_f1 = BASELINE[name]["cv_f1"]
    diff = r["cv_f1"] - baseline_f1
    arrow = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "→")
    print(f"  {name:15s}  {r['acc']:6.4f}  {r['f1']:6.4f}  "
          f"{r['auc']:6.4f}  {r['cv_f1']:8.4f}  "
          f"{r['cv_std']:7.4f}  {arrow}{abs(diff):.4f} vs baseline")

best_name = max(results, key=lambda k: results[k]["cv_f1"])
best      = results[best_name]
print(f"\n  BEST MODEL: {best_name}  (CV F1 = {best['cv_f1']:.4f})")

dump(best["model"], OUT_MODEL)
print(f"  Saved → {OUT_MODEL}")


# =============================================================================
# 6.  PLOTS
# =============================================================================

print("\n" + "="*60)
print("STEP 8 — Generate plots")
print("="*60)

# ── Confusion matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Confusion Matrices — Extended Feature Set (12 features)",
             fontsize=13, fontweight="bold")

for ax, (name, r) in zip(axes, results.items()):
    cm   = confusion_matrix(y_test, r["pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Informal", "Formal"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAcc={r['acc']:.3f}  F1={r['f1']:.3f}",
                 fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_CM, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_CM}")

# ── Feature importance ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Feature Importance — Extended Model",
             fontsize=13, fontweight="bold")

feat_names = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]

for ax, model, title, color in [
    (axes[0], rf,  "Random Forest", "#42A5F5"),
    (axes[1], xgb, "XGBoost",       "#66BB6A"),
]:
    imp   = model.feature_importances_
    order = np.argsort(imp)
    bars  = ax.barh([feat_names[i] for i in order],
                    imp[order], color=color)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.axvline(x=1/len(FEATURE_COLS), color="red",
               linestyle="--", alpha=0.5, label="Equal weight")
    ax.legend(fontsize=8)

    # Highlight new features
    for bar, idx in zip(bars, order):
        if "★" in feat_names[idx]:
            bar.set_edgecolor("red")
            bar.set_linewidth(2)

plt.tight_layout()
plt.savefig(OUT_FI, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_FI}")

# ── Baseline vs extended comparison ──────────────────────────────────────────
model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Baseline (8 features) vs Extended (12 features)",
             fontsize=13, fontweight="bold")

for ax, metric, title in [
    (axes[0], "cv_f1", "Cross-Validation F1"),
    (axes[1], "auc",   "ROC-AUC"),
]:
    baseline_vals = [BASELINE[m][metric] for m in model_names]
    extended_vals = [results[m][metric]  for m in model_names]

    b1 = ax.bar(x - width/2, baseline_vals, width,
                label="Baseline (8 features)", color="#90CAF9", alpha=0.9)
    b2 = ax.bar(x + width/2, extended_vals, width,
                label="Extended (12 features)", color="#42A5F5", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=10)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="navy")

plt.tight_layout()
plt.savefig(OUT_COMPARE, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_COMPARE}")


# =============================================================================
# 7.  WRITE REPORT
# =============================================================================

print("\n" + "="*60)
print("STEP 9 — Write model report")
print("="*60)

lines = [
    "="*60,
    "EXTENDED MODEL REPORT — 12 FEATURES",
    "Dhaka Informal Settlement Classifier",
    "="*60,
    "",
    f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}",
    f"Training samples : {len(y_train)}",
    f"Test samples     : {len(y_test)}",
    "",
    "="*60,
    "MODEL COMPARISON — EXTENDED vs BASELINE",
    "="*60,
    f"{'Model':15s}  {'Acc':>6}  {'F1':>6}  {'AUC':>6}  "
    f"{'CV F1':>8}  {'vs Baseline':>12}",
    "-"*60,
]

for name, r in results.items():
    diff = r["cv_f1"] - BASELINE[name]["cv_f1"]
    sign = "+" if diff >= 0 else ""
    lines.append(
        f"{name:15s}  {r['acc']:6.4f}  {r['f1']:6.4f}  "
        f"{r['auc']:6.4f}  {r['cv_f1']:8.4f}  "
        f"{sign}{diff:.4f}"
    )

lines += [
    "",
    f"BEST MODEL: {best_name}",
    f"  CV F1  : {best['cv_f1']:.4f} ± {best['cv_std']:.4f}",
    f"  Acc    : {best['acc']:.4f}",
    f"  AUC    : {best['auc']:.4f}",
    "",
    "="*60,
    "CLASSIFICATION REPORT",
    "="*60,
    classification_report(y_test, best["pred"],
                          target_names=["Informal", "Formal"]),
    "",
    "="*60,
    "FEATURE IMPORTANCE (XGBoost)",
    "="*60,
]

for feat, imp in sorted(zip(FEATURE_COLS, xgb.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    marker = " ★ NEW" if feat in ["s2_ndbi_mean", "s2_mndwi_mean",
                                    "osm_building_density",
                                    "osm_mean_building_area"] else ""
    lines.append(f"  {FEATURE_LABELS.get(feat, feat):30s}: "
                 f"{imp:.4f}{marker}")

report = "\n".join(lines)
with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\n  [SAVED] {OUT_REPORT}")


# =============================================================================
# 8.  FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("NOTEBOOK 04b COMPLETE")
print("="*60)

print(f"\n  Best model    : {best_name}")
print(f"  CV F1         : {best['cv_f1']:.4f} ± {best['cv_std']:.4f}")
print(f"  Test accuracy : {best['acc']:.4f}")
print(f"  ROC-AUC       : {best['auc']:.4f}")

print("\n  Improvement over baseline:")
for name, r in results.items():
    diff = r["cv_f1"] - BASELINE[name]["cv_f1"]
    sign = "+" if diff >= 0 else ""
    print(f"    {name:15s}: {sign}{diff:.4f} CV F1")

print("\n  Output files:")
for label, path in [
    ("Model",    OUT_MODEL),
    ("Scaler",   OUT_SCALER),
    ("Report",   OUT_REPORT),
    ("CM plot",  OUT_CM),
    ("FI plot",  OUT_FI),
    ("Compare",  OUT_COMPARE),
]:
    status = "✓" if os.path.exists(path) else "✗"
    print(f"    {status}  {label:8s}  →  {path}")

print("\nNext step: run Notebook 05b to generate updated prediction maps.")
print("="*60)
