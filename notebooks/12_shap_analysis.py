# =============================================================================
# NOTEBOOK 12 — SHAP FEATURE ATTRIBUTION
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
#
# PURPOSE
# -------
# NB06's "feature importances" are XGBoost gain-based importances, which
# are a reasonable global ranking but don't show direction of effect
# (does higher OSM building density push toward informal or formal?) or
# per-ward variation. SHAP values address both: a summary plot shows
# each feature's direction and magnitude of effect across all wards, in
# one figure, for the decoupled (rule-independent) model -- the model
# used for the paper's headline numbers.
#
# OUTPUT
# ------
#   outputs/figures/12_shap_summary.png
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

RANDOM_SEED = 42

FEATURES_EXT = "data/processed/ward_features_extended.csv"
LABELS_CSV   = "data/processed/ward_labels.csv"
OUT_FIG = "outputs/figures/12_shap_summary.png"

os.makedirs("outputs/figures", exist_ok=True)

RULE_DERIVED = ["built_fraction", "ndvi_mean", "pop_mean", "lst_mean", "ndbi_mean"]
ALL_FEATURES = [
    "ndvi_mean", "savi_mean", "ndbi_mean", "lst_mean", "slope_mean",
    "pop_mean", "pop_std", "built_fraction",
    "s2_ndbi_mean", "s2_mndwi_mean", "osm_building_density", "osm_mean_building_area",
]
DECOUPLED_FEATURES = [c for c in ALL_FEATURES if c not in RULE_DERIVED]

FEATURE_LABELS = {
    "savi_mean": "SAVI (mean)",
    "slope_mean": "Slope (mean, deg.)",
    "pop_std": "Population density (std. dev.)",
    "s2_ndbi_mean": "Sentinel-2 true NDBI",
    "s2_mndwi_mean": "Sentinel-2 MNDWI",
    "osm_building_density": "OSM building density",
    "osm_mean_building_area": "OSM mean building area",
}

print("="*70)
print("SHAP FEATURE ATTRIBUTION — DECOUPLED MODEL")
print("="*70)

df_ext = pd.read_csv(FEATURES_EXT)
df_lab = pd.read_csv(LABELS_CSV)
df = df_ext.merge(df_lab[["GID_4", "label", "label_src"]], on="GID_4", how="left")
df["label"] = df["label"].fillna(-1).astype(int)
df_labeled = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)


def fill_nan(X):
    X = X.astype(float)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    return X


X = fill_nan(df_labeled[DECOUPLED_FEATURES].values)
y = df_labeled["label"].values
n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)
scale_w = n_neg / n_pos if n_pos > 0 else 1.0

model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_w,
    eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0
)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print(f"\nSHAP values computed for n={len(y)} wards, {len(DECOUPLED_FEATURES)} features.")
print("Note: label=1 is Formal, so a POSITIVE SHAP value pushes toward Formal;")
print("a NEGATIVE SHAP value pushes toward Informal.")

display_names = [FEATURE_LABELS.get(f, f) for f in DECOUPLED_FEATURES]

plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X, feature_names=display_names, show=False)
plt.title("SHAP summary — decoupled model (7 rule-independent features)\n"
          "Positive = pushes toward Formal · Negative = pushes toward Informal", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"\nFigure written -> {OUT_FIG}")

# mean |SHAP| ranking, for the text
mean_abs = np.abs(shap_values).mean(axis=0)
order = np.argsort(-mean_abs)
print("\nMean |SHAP| ranking:")
for i in order:
    print(f"  {DECOUPLED_FEATURES[i]:28s} {mean_abs[i]:.4f}")

print("\n" + "="*70)
print("DONE")
print("="*70)
