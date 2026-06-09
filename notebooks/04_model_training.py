# =============================================================================
# NOTEBOOK 04 — MODEL TRAINING
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
# Author  : (your name)
# Date    : 2024
#
# PURPOSE
# -------
# Train three classifiers on the labeled ward feature table:
#   1. Random Forest
#   2. XGBoost
#   3. Support Vector Machine (SVM)
#
# Then evaluate all three and save the best model for prediction in NB 05.
#
# WORKFLOW
# --------
#   1. Load labeled features (163 wards)
#   2. Preprocessing — scale features, stratified 80/20 split
#   3. Train Random Forest
#   4. Train XGBoost
#   5. Train SVM
#   6. Compare all three — accuracy, F1, confusion matrix
#   7. Feature importance plot (RF + XGBoost)
#   8. Cross-validation (5-fold) for robust accuracy estimate
#   9. Save best model + scaler to outputs/model/
#
# INPUT
# -----
#   data/processed/ward_labels.csv
#
# OUTPUT
# ------
#   outputs/model/best_model.joblib      ← best trained model
#   outputs/model/scaler.joblib          ← fitted StandardScaler
#   outputs/model/model_report.txt       ← full evaluation report
#   outputs/figures/04_confusion_matrices.png
#   outputs/figures/04_feature_importance.png
#   outputs/figures/04_model_comparison.png
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from joblib import dump, load

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

# ── XGBoost ───────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

LABELS_CSV    = "data/processed/ward_labels.csv"
OUT_MODEL     = "outputs/model/best_model.joblib"
OUT_SCALER    = "outputs/model/scaler.joblib"
OUT_REPORT    = "outputs/model/model_report.txt"
OUT_CM        = "outputs/figures/04_confusion_matrices.png"
OUT_FI        = "outputs/figures/04_feature_importance.png"
OUT_COMPARE   = "outputs/figures/04_model_comparison.png"

os.makedirs("outputs/model",   exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

RANDOM_SEED   = 42
TEST_SIZE     = 0.20    # 80% train, 20% test
CV_FOLDS      = 5       # stratified k-fold cross-validation

FEATURE_COLS  = [
    "ndvi_mean",
    "savi_mean",
    "ndbi_mean",
    "lst_mean",
    "slope_mean",
    "pop_mean",
    "pop_std",
    "built_fraction",
]

FEATURE_LABELS = {
    "ndvi_mean":      "NDVI (mean)",
    "savi_mean":      "SAVI (mean)",
    "ndbi_mean":      "NDBI proxy",
    "lst_mean":       "LST °C",
    "slope_mean":     "Slope °",
    "pop_mean":       "Pop density",
    "pop_std":        "Pop std dev",
    "built_fraction": "Built-up fraction",
}


# =============================================================================
# 1.  LOAD AND PREPARE DATA
# =============================================================================

print("="*60)
print("STEP 1 — Load and prepare labeled data")
print("="*60)

# Load labeled ward features
df = pd.read_csv(LABELS_CSV)
print(f"  Total rows loaded : {len(df)}")
print(f"  Columns           : {list(df.columns)}")

# Keep only labeled wards (label 0 or 1)
# Unlabeled wards (label = -1) are excluded from training
df_labeled = df[df["label"].isin([0, 1])].copy()
df_labeled = df_labeled.reset_index(drop=True)

print(f"\n  Labeled wards     : {len(df_labeled)}")
print(f"  Formal   (1)      : {(df_labeled['label'] == 1).sum()}")
print(f"  Informal (0)      : {(df_labeled['label'] == 0).sum()}")

# ── Features and target ───────────────────────────────────────────────────────
X = df_labeled[FEATURE_COLS].values
y = df_labeled["label"].values

print(f"\n  Feature matrix X  : {X.shape}")
print(f"  Target vector y   : {y.shape}")
print(f"  Class balance     : {np.bincount(y)} (informal=0, formal=1)")

# Check for any remaining NaN values
nan_count = np.isnan(X).sum()
if nan_count > 0:
    print(f"  WARNING: {nan_count} NaN values found — filling with column median")
    for j in range(X.shape[1]):
        col_median = np.nanmedian(X[:, j])
        X[np.isnan(X[:, j]), j] = col_median
else:
    print("  No NaN values found — data is clean.")


# =============================================================================
# 2.  TRAIN / TEST SPLIT + SCALING
#
#     Stratified split ensures both classes are proportionally
#     represented in both train and test sets.
#
#     StandardScaler: subtracts mean, divides by std dev.
#     Required for SVM. Also helps RF and XGBoost slightly.
#     IMPORTANT: fit scaler on TRAIN set only, then apply to test set.
#     Fitting on the full dataset would cause data leakage.
# =============================================================================

print("\n" + "="*60)
print("STEP 2 — Train/test split and feature scaling")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
    stratify     = y          # maintain class proportions
)

print(f"  Train set : {X_train.shape[0]} samples "
      f"(informal={np.sum(y_train==0)}, formal={np.sum(y_train==1)})")
print(f"  Test set  : {X_test.shape[0]} samples "
      f"(informal={np.sum(y_test==0)}, formal={np.sum(y_test==1)})")

# Fit scaler on training data only
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Save scaler immediately — needed in NB 05 for prediction
dump(scaler, OUT_SCALER)
print(f"\n  Scaler saved to {OUT_SCALER}")


# =============================================================================
# 3.  TRAIN RANDOM FOREST
#
#     Random Forest builds many decision trees on random subsets of
#     data and features, then votes. It handles mixed-scale features
#     well and provides feature importances.
#
#     Key hyperparameters:
#       n_estimators : number of trees (more = better but slower)
#       max_depth    : None = fully grown trees
#       min_samples_leaf: minimum samples at each leaf (controls overfitting)
# =============================================================================

print("\n" + "="*60)
print("STEP 3 — Train Random Forest")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = None,
    min_samples_leaf = 2,
    max_features     = "sqrt",  # sqrt(n_features) features per split
    class_weight     = "balanced",  # handles class imbalance
    random_state     = RANDOM_SEED,
    n_jobs           = -1       # use all CPU cores
)

# Train on UNSCALED data — RF doesn't need scaling
rf_model.fit(X_train, y_train)

# Predictions
rf_pred  = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# Metrics
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1  = f1_score(y_test, rf_pred, average="weighted")
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"  Accuracy  : {rf_acc:.4f}")
print(f"  F1 score  : {rf_f1:.4f}")
print(f"  ROC-AUC   : {rf_auc:.4f}")
print(f"\n  Classification report:")
print(classification_report(y_test, rf_pred,
                             target_names=["Informal", "Formal"]))


# =============================================================================
# 4.  TRAIN XGBOOST
#
#     XGBoost builds trees sequentially where each tree corrects the
#     errors of the previous one (gradient boosting). Often the best
#     performer on tabular data.
#
#     Key hyperparameters:
#       n_estimators   : number of boosting rounds
#       max_depth      : depth of each tree (3-6 is typical)
#       learning_rate  : step size (lower = more trees needed)
#       subsample      : fraction of samples per tree (prevents overfitting)
#       scale_pos_weight: handles class imbalance = count(neg)/count(pos)
# =============================================================================

print("\n" + "="*60)
print("STEP 4 — Train XGBoost")
print("="*60)

# Class imbalance weight
neg_count  = np.sum(y_train == 0)
pos_count  = np.sum(y_train == 1)
scale_weight = neg_count / pos_count
print(f"  scale_pos_weight = {scale_weight:.3f} "
      f"(neg={neg_count}, pos={pos_count})")

xgb_model = XGBClassifier(
    n_estimators      = 300,
    max_depth         = 4,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_weight,
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = RANDOM_SEED,
    verbosity         = 0
)

# XGBoost also works on unscaled data
xgb_model.fit(X_train, y_train)

xgb_pred  = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1  = f1_score(y_test, xgb_pred, average="weighted")
xgb_auc = roc_auc_score(y_test, xgb_proba)

print(f"  Accuracy  : {xgb_acc:.4f}")
print(f"  F1 score  : {xgb_f1:.4f}")
print(f"  ROC-AUC   : {xgb_auc:.4f}")
print(f"\n  Classification report:")
print(classification_report(y_test, xgb_pred,
                             target_names=["Informal", "Formal"]))


# =============================================================================
# 5.  TRAIN SVM
#
#     Support Vector Machine finds the hyperplane that best separates
#     the two classes with maximum margin. With RBF kernel it can
#     handle non-linear boundaries.
#     MUST use scaled features.
#
#     Key hyperparameters:
#       C      : regularisation (higher = less regularisation)
#       kernel : "rbf" works well for most cases
#       gamma  : "scale" = 1/(n_features * X.var())
# =============================================================================

print("\n" + "="*60)
print("STEP 5 — Train SVM")
print("="*60)

svm_model = SVC(
    C            = 10,
    kernel       = "rbf",
    gamma        = "scale",
    class_weight = "balanced",
    probability  = True,    # needed for predict_proba and ROC-AUC
    random_state = RANDOM_SEED
)

# SVM uses SCALED data
svm_model.fit(X_train_sc, y_train)

svm_pred  = svm_model.predict(X_test_sc)
svm_proba = svm_model.predict_proba(X_test_sc)[:, 1]

svm_acc = accuracy_score(y_test, svm_pred)
svm_f1  = f1_score(y_test, svm_pred, average="weighted")
svm_auc = roc_auc_score(y_test, svm_proba)

print(f"  Accuracy  : {svm_acc:.4f}")
print(f"  F1 score  : {svm_f1:.4f}")
print(f"  ROC-AUC   : {svm_auc:.4f}")
print(f"\n  Classification report:")
print(classification_report(y_test, svm_pred,
                             target_names=["Informal", "Formal"]))


# =============================================================================
# 6.  CROSS-VALIDATION
#
#     With only 163 samples, a single 80/20 split can be noisy.
#     5-fold cross-validation gives a more reliable accuracy estimate
#     by training and testing on 5 different splits.
# =============================================================================

print("\n" + "="*60)
print("STEP 6 — 5-fold cross-validation")
print("="*60)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                     random_state=RANDOM_SEED)

# RF cross-val (unscaled)
rf_cv  = cross_val_score(rf_model,  X, y, cv=cv, scoring="f1_weighted")
# XGB cross-val (unscaled)
xgb_cv = cross_val_score(xgb_model, X, y, cv=cv, scoring="f1_weighted")
# SVM cross-val (scaled — use Pipeline to avoid leakage)
svm_pipe = Pipeline([("scaler", StandardScaler()),
                     ("svm",    SVC(C=10, kernel="rbf", gamma="scale",
                                   class_weight="balanced",
                                   probability=True,
                                   random_state=RANDOM_SEED))])
svm_cv = cross_val_score(svm_pipe, X, y, cv=cv, scoring="f1_weighted")

print(f"  Random Forest CV F1 : {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"  XGBoost       CV F1 : {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
print(f"  SVM           CV F1 : {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")


# =============================================================================
# 7.  SELECT BEST MODEL
# =============================================================================

print("\n" + "="*60)
print("STEP 7 — Select best model")
print("="*60)

# Rank by CV F1 score (most reliable metric with small dataset)
results = {
    "Random Forest": {
        "model":    rf_model,
        "acc":      rf_acc,
        "f1":       rf_f1,
        "auc":      rf_auc,
        "cv_f1":    rf_cv.mean(),
        "cv_std":   rf_cv.std(),
        "pred":     rf_pred,
        "proba":    rf_proba,
        "scaled":   False,
    },
    "XGBoost": {
        "model":    xgb_model,
        "acc":      xgb_acc,
        "f1":       xgb_f1,
        "auc":      xgb_auc,
        "cv_f1":    xgb_cv.mean(),
        "cv_std":   xgb_cv.std(),
        "pred":     xgb_pred,
        "proba":    xgb_proba,
        "scaled":   False,
    },
    "SVM": {
        "model":    svm_pipe,   # save full pipeline including scaler
        "acc":      svm_acc,
        "f1":       svm_f1,
        "auc":      svm_auc,
        "cv_f1":    svm_cv.mean(),
        "cv_std":   svm_cv.std(),
        "pred":     svm_pred,
        "proba":    svm_proba,
        "scaled":   True,
    },
}

print(f"\n  {'Model':15s}  {'Acc':>6}  {'F1':>6}  {'AUC':>6}  "
      f"{'CV F1':>8}  {'CV Std':>7}")
print("  " + "-"*55)
for name, r in results.items():
    print(f"  {name:15s}  {r['acc']:6.4f}  {r['f1']:6.4f}  "
          f"{r['auc']:6.4f}  {r['cv_f1']:8.4f}  {r['cv_std']:7.4f}")

# Pick model with highest CV F1
best_name = max(results, key=lambda k: results[k]["cv_f1"])
best      = results[best_name]
print(f"\n  BEST MODEL: {best_name}  (CV F1 = {best['cv_f1']:.4f})")

# Save best model
dump(best["model"], OUT_MODEL)
print(f"  Saved to   : {OUT_MODEL}")


# =============================================================================
# 8.  VISUALISATIONS
# =============================================================================

print("\n" + "="*60)
print("STEP 8 — Generate evaluation plots")
print("="*60)

# ── 8a. Confusion matrices for all three models ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Confusion Matrices — Test Set",
             fontsize=13, fontweight="bold")

for ax, (name, r) in zip(axes, results.items()):
    cm  = confusion_matrix(y_test, r["pred"])
    disp = ConfusionMatrixDisplay(cm,
                                  display_labels=["Informal", "Formal"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAcc={r['acc']:.3f}  F1={r['f1']:.3f}",
                 fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_CM, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_CM}")

# ── 8b. Feature importance — RF and XGBoost ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Feature Importance", fontsize=13, fontweight="bold")

feature_names = [FEATURE_LABELS[c] for c in FEATURE_COLS]

# Random Forest importances
rf_imp   = rf_model.feature_importances_
rf_order = np.argsort(rf_imp)
axes[0].barh([feature_names[i] for i in rf_order],
             rf_imp[rf_order], color="#42A5F5")
axes[0].set_title("Random Forest", fontweight="bold")
axes[0].set_xlabel("Mean decrease in impurity")
axes[0].axvline(x=1/len(FEATURE_COLS), color="red",
                linestyle="--", alpha=0.5, label="Equal importance")
axes[0].legend(fontsize=8)

# XGBoost importances
xgb_imp   = xgb_model.feature_importances_
xgb_order = np.argsort(xgb_imp)
axes[1].barh([feature_names[i] for i in xgb_order],
             xgb_imp[xgb_order], color="#66BB6A")
axes[1].set_title("XGBoost", fontweight="bold")
axes[1].set_xlabel("Feature importance score")

plt.tight_layout()
plt.savefig(OUT_FI, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_FI}")

# ── 8c. Model comparison bar chart ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")

model_names = list(results.keys())
metrics     = ["acc", "f1", "auc"]
titles      = ["Accuracy", "F1 Score (weighted)", "ROC-AUC"]
colors      = ["#42A5F5", "#66BB6A", "#EF5350"]

for ax, metric, title, color in zip(axes, metrics, titles, colors):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(model_names, vals, color=color, alpha=0.85,
                  edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=15)
    # Annotate bars with values
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_COMPARE, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_COMPARE}")


# =============================================================================
# 9.  WRITE MODEL REPORT
# =============================================================================

print("\n" + "="*60)
print("STEP 9 — Write model report")
print("="*60)

report_lines = [
    "="*60,
    "INFORMAL SETTLEMENT CLASSIFIER — MODEL REPORT",
    "Dhaka Metropolitan Region",
    "="*60,
    "",
    f"Training samples : {len(y_train)}",
    f"Test samples     : {len(y_test)}",
    f"Features used    : {', '.join(FEATURE_COLS)}",
    f"Random seed      : {RANDOM_SEED}",
    "",
    "="*60,
    "MODEL COMPARISON",
    "="*60,
    f"{'Model':15s}  {'Acc':>6}  {'F1':>6}  {'AUC':>6}  "
    f"{'CV F1':>8}  {'CV Std':>7}",
    "-"*55,
]

for name, r in results.items():
    report_lines.append(
        f"{name:15s}  {r['acc']:6.4f}  {r['f1']:6.4f}  "
        f"{r['auc']:6.4f}  {r['cv_f1']:8.4f}  {r['cv_std']:7.4f}"
    )

report_lines += [
    "",
    f"BEST MODEL: {best_name}",
    f"  Accuracy : {best['acc']:.4f}",
    f"  F1 Score : {best['f1']:.4f}",
    f"  ROC-AUC  : {best['auc']:.4f}",
    f"  CV F1    : {best['cv_f1']:.4f} ± {best['cv_std']:.4f}",
    "",
    "="*60,
    "CLASSIFICATION REPORT — BEST MODEL",
    "="*60,
    classification_report(y_test, best["pred"],
                          target_names=["Informal", "Formal"]),
    "",
    "="*60,
    "FEATURE IMPORTANCE (Random Forest)",
    "="*60,
]

rf_fi_sorted = sorted(zip(FEATURE_COLS, rf_model.feature_importances_),
                      key=lambda x: x[1], reverse=True)
for feat, imp in rf_fi_sorted:
    report_lines.append(f"  {FEATURE_LABELS[feat]:20s}: {imp:.4f}")

report_text = "\n".join(report_lines)

with open(OUT_REPORT, "w") as f:
    f.write(report_text)

print(report_text)
print(f"\n  [SAVED] {OUT_REPORT}")


# =============================================================================
# 10.  FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("NOTEBOOK 04 COMPLETE — Model training summary")
print("="*60)

print(f"\n  Best model        : {best_name}")
print(f"  CV F1 score       : {best['cv_f1']:.4f} ± {best['cv_std']:.4f}")
print(f"  Test accuracy     : {best['acc']:.4f}")
print(f"  Test ROC-AUC      : {best['auc']:.4f}")

print("\n  Output files:")
for label, path in [
    ("Best model",    OUT_MODEL),
    ("Scaler",        OUT_SCALER),
    ("Report",        OUT_REPORT),
    ("Conf matrices", OUT_CM),
    ("Feature imp",   OUT_FI),
    ("Comparison",    OUT_COMPARE),
]:
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"    {status}  {label:15s}  →  {path}")

print("\nReady for Notebook 05 (Prediction Mapping).")
print("="*60)
