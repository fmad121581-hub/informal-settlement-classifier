# =============================================================================
# NOTEBOOK 08 — DECOUPLED MODEL x EXTERNAL VALIDATION (bulletproofing)
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
#
# PURPOSE
# -------
# Notebook 06 showed the decoupled (rule-independent-feature) model beats
# a majority-class baseline on anchor-only labels. Notebook 07 showed the
# REFERENCE (rule-circular) model's probabilities correlate with the
# independent EO4SD 2017 informal-settlement layer.
#
# This notebook closes the loop: it checks whether the DECOUPLED model's
# predictions ALSO correlate with EO4SD. If they do, the paper's headline
# result is immune to the circularity critique end-to-end — both the
# accuracy number and the external-validation correlation survive with
# rule-derived features removed.
#
# Two decoupled models are trained (fit on ALL available data, not CV
# splits, since the goal here is generating a full 203-ward probability
# surface to correlate against EO4SD, not estimating generalization error
# — that was already done properly with CV in notebook 06):
#
#   MODEL A — decoupled features, trained on ALL 170 labeled wards
#   MODEL C — decoupled features, trained on ONLY the 63 anchor-only wards
#             (strictest — no rule-derived label OR rule-derived feature)
#
# OUTPUT
# ------
#   outputs/model/decoupled_external_validation_report.txt
#   outputs/figures/08_decoupled_external_validation.png
#   data/processed/ward_decoupled_eo4sd_comparison.csv
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from xgboost import XGBClassifier

RANDOM_SEED = 42

FEATURES_EXT = "data/processed/ward_features_extended.csv"
LABELS_CSV   = "data/processed/ward_labels.csv"
EO4SD_CSV    = "data/processed/ward_eo4sd_comparison.csv"

OUT_REPORT = "outputs/model/decoupled_external_validation_report.txt"
OUT_FIG    = "outputs/figures/08_decoupled_external_validation.png"
OUT_CSV    = "data/processed/ward_decoupled_eo4sd_comparison.csv"

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

RULE_DERIVED = ["built_fraction", "ndvi_mean", "pop_mean", "lst_mean", "ndbi_mean"]
ALL_FEATURES = [
    "ndvi_mean", "savi_mean", "ndbi_mean", "lst_mean", "slope_mean",
    "pop_mean", "pop_std", "built_fraction",
    "s2_ndbi_mean", "s2_mndwi_mean", "osm_building_density", "osm_mean_building_area",
]
DECOUPLED_FEATURES = [c for c in ALL_FEATURES if c not in RULE_DERIVED]

print("="*70)
print("DECOUPLED MODEL x EXTERNAL VALIDATION")
print("="*70)
print(f"Decoupled features ({len(DECOUPLED_FEATURES)}): {DECOUPLED_FEATURES}")

# =============================================================================
# LOAD
# =============================================================================

df_ext = pd.read_csv(FEATURES_EXT)
df_lab = pd.read_csv(LABELS_CSV)
df_eo4sd = pd.read_csv(EO4SD_CSV)  # from notebook 07: GID_4, eo4sd_informal_fraction, ...

df = df_ext.merge(df_lab[["GID_4", "label", "label_src"]], on="GID_4", how="left")
df["label"] = df["label"].fillna(-1).astype(int)

df_labeled = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)
df_anchor  = df_labeled[df_labeled["label_src"].str.startswith("thana:")].copy().reset_index(drop=True)

print(f"\nAll wards          : {len(df)}")
print(f"Labeled (all)       : {len(df_labeled)}  (formal={ (df_labeled['label']==1).sum() }, informal={ (df_labeled['label']==0).sum() })")
print(f"Anchor-only labeled : {len(df_anchor)}  (formal={ (df_anchor['label']==1).sum() }, informal={ (df_anchor['label']==0).sum() })")


def fill_nan(X):
    X = X.astype(float)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    return X


def train_and_predict_all(train_df, feature_cols, label_col="label", name=""):
    Xtr = fill_nan(train_df[feature_cols].values)
    ytr = train_df[label_col].values
    n_pos, n_neg = np.sum(ytr == 1), np.sum(ytr == 0)
    scale_w = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_w,
        eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0
    )
    model.fit(Xtr, ytr)

    X_all = fill_nan(df[feature_cols].values)
    prob_informal_all = 1 - model.predict_proba(X_all)[:, 1]  # class 1 = formal
    out = df[["GID_4", "NAME_4", "NAME_3"]].copy()
    out[f"prob_informal_{name}"] = prob_informal_all
    print(f"\n  Trained '{name}' on n={len(ytr)} (formal={n_pos}, informal={n_neg}), "
          f"predicted for all {len(out)} wards.")
    return out


print("\n" + "-"*70)
print("Training decoupled models on full ward set for EO4SD correlation")
print("-"*70)

predA = train_and_predict_all(df_labeled, DECOUPLED_FEATURES, name="decoupled_all170")
predC = train_and_predict_all(df_anchor,  DECOUPLED_FEATURES, name="decoupled_anchor63")

# reference (rule-inclusive) model, all labeled wards — for side-by-side comparison
predRef = train_and_predict_all(df_labeled, ALL_FEATURES, name="reference_all170")

merged = predA.merge(predC[["GID_4", "prob_informal_decoupled_anchor63"]], on="GID_4")
merged = merged.merge(predRef[["GID_4", "prob_informal_reference_all170"]], on="GID_4")
merged = merged.merge(df_eo4sd[["GID_4", "eo4sd_informal_fraction", "label", "label_src"]],
                       on="GID_4", how="left")
merged.to_csv(OUT_CSV, index=False)
print(f"\nComparison table saved -> {OUT_CSV}")

# =============================================================================
# CORRELATIONS
# =============================================================================

report_lines = ["="*70, "DECOUPLED MODEL x EXTERNAL VALIDATION (EO4SD 2017)", "="*70, ""]

for col, label in [
    ("prob_informal_reference_all170",  "Reference model (12 features, rule-circular)"),
    ("prob_informal_decoupled_all170",  "Decoupled model (7 features, all 170 labeled wards)"),
    ("prob_informal_decoupled_anchor63","Decoupled model (7 features, 63 anchor-only wards) -- STRICTEST"),
]:
    valid = merged[merged[col].notnull() & merged["eo4sd_informal_fraction"].notnull()]
    r_p, p_p = stats.pearsonr(valid[col], valid["eo4sd_informal_fraction"])
    r_s, p_s = stats.spearmanr(valid[col], valid["eo4sd_informal_fraction"])
    line1 = f"{label}"
    line2 = f"  n={len(valid)}  Pearson r={r_p:.4f} (p={p_p:.4g})   Spearman r={r_s:.4f} (p={p_s:.4g})"
    print(line1)
    print(line2)
    report_lines.append(line1)
    report_lines.append(line2)
    report_lines.append("")

with open(OUT_REPORT, "w") as f:
    f.write("\n".join(report_lines))
print(f"\nReport written -> {OUT_REPORT}")

# =============================================================================
# FIGURE
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
cols = ["prob_informal_reference_all170", "prob_informal_decoupled_all170", "prob_informal_decoupled_anchor63"]
titles = ["Reference model\n(12 features)", "Decoupled model\n(7 features, n=170)",
          "Decoupled model\n(7 features, anchor-only n=63)"]

for ax, col, title in zip(axes, cols, titles):
    valid = merged[merged[col].notnull() & merged["eo4sd_informal_fraction"].notnull()]
    r_s, p_s = stats.spearmanr(valid[col], valid["eo4sd_informal_fraction"])
    ax.scatter(valid[col], valid["eo4sd_informal_fraction"], alpha=0.45, s=16, color="#2563eb")
    ax.set_xlabel("Predicted P(informal)")
    ax.set_title(f"{title}\nSpearman r={r_s:.3f}, p={p_s:.2g}")
axes[0].set_ylabel("EO4SD 2017 informal-patch area fraction")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
print(f"Figure written -> {OUT_FIG}")

print("\n" + "="*70)
print("DONE")
print("="*70)
