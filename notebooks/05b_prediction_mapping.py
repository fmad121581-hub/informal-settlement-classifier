# =============================================================================
# NOTEBOOK 05b — UPDATED PREDICTION MAPPING
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
# Author  : (your name)
# Date    : 2024
#
# PURPOSE
# -------
# Generate final prediction maps using the extended 12-feature XGBoost model.
# Compares predictions against the baseline 8-feature model.
#
# INPUT
# -----
#   data/processed/ward_features_extended.csv
#   data/processed/ward_labels.csv
#   data/processed/ward_features.gpkg         ← geometries
#   outputs/model/best_model_extended.joblib
#   outputs/model/best_model.joblib           ← baseline for comparison
#
# OUTPUT
# ------
#   data/processed/ward_predictions_extended.csv
#   data/processed/ward_predictions_extended.gpkg
#   outputs/figures/05b_probability_map.png
#   outputs/figures/05b_dashboard.png
#   outputs/figures/05b_baseline_vs_extended_map.png
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from joblib import load


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

FEATURES_EXT   = "data/processed/ward_features_extended.csv"
FEATURES_BASE  = "data/processed/ward_features.csv"
LABELS_CSV     = "data/processed/ward_labels.csv"
WARDS_GPKG     = "data/processed/ward_features.gpkg"

MODEL_EXT      = "outputs/model/best_model_extended.joblib"
MODEL_BASE     = "outputs/model/best_model.joblib"

OUT_CSV        = "data/processed/ward_predictions_extended.csv"
OUT_GPKG       = "data/processed/ward_predictions_extended.gpkg"
OUT_PROB_MAP   = "outputs/figures/05b_probability_map.png"
OUT_DASHBOARD  = "outputs/figures/05b_dashboard.png"
OUT_COMPARE    = "outputs/figures/05b_baseline_vs_extended_map.png"

os.makedirs("data/processed",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

FEATURE_COLS_EXT = [
    "ndvi_mean", "savi_mean", "ndbi_mean", "lst_mean",
    "slope_mean", "pop_mean", "pop_std", "built_fraction",
    "s2_ndbi_mean", "s2_mndwi_mean",
    "osm_building_density", "osm_mean_building_area",
]

FEATURE_COLS_BASE = [
    "ndvi_mean", "savi_mean", "ndbi_mean", "lst_mean",
    "slope_mean", "pop_mean", "pop_std", "built_fraction",
]

THRESHOLD = 0.50


# =============================================================================
# 1.  LOAD DATA
# =============================================================================

print("="*60)
print("STEP 1 — Load data and models")
print("="*60)

df_ext   = pd.read_csv(FEATURES_EXT)
df_base  = pd.read_csv(FEATURES_BASE)
df_labels = pd.read_csv(LABELS_CSV)
gdf      = gpd.read_file(WARDS_GPKG)

print(f"  Extended features  : {df_ext.shape}")
print(f"  Baseline features  : {df_base.shape}")
print(f"  Ward geometries    : {len(gdf)}, CRS: {gdf.crs}")

model_ext  = load(MODEL_EXT)
model_base = load(MODEL_BASE)
print(f"  Extended model     : {type(model_ext).__name__}")
print(f"  Baseline model     : {type(model_base).__name__}")

# Merge labels
df_ext = df_ext.merge(
    df_labels[["GID_4", "label", "label_src"]],
    on="GID_4", how="left"
)
df_ext["label"]     = df_ext["label"].fillna(-1).astype(int)
df_ext["label_src"] = df_ext["label_src"].fillna("unlabeled")

df_base = df_base.merge(
    df_labels[["GID_4", "label"]],
    on="GID_4", how="left"
)
df_base["label"] = df_base["label"].fillna(-1).astype(int)


# =============================================================================
# 2.  PREPARE FEATURE MATRICES
# =============================================================================

print("\n" + "="*60)
print("STEP 2 — Prepare feature matrices")
print("="*60)

def prepare_X(df, feature_cols):
    X = df[feature_cols].values.astype(float)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])
    return X

X_ext  = prepare_X(df_ext,  FEATURE_COLS_EXT)
X_base = prepare_X(df_base, FEATURE_COLS_BASE)

print(f"  Extended  X : {X_ext.shape}")
print(f"  Baseline  X : {X_base.shape}")


# =============================================================================
# 3.  PREDICT WITH BOTH MODELS
# =============================================================================

print("\n" + "="*60)
print("STEP 3 — Generate predictions")
print("="*60)

# Extended model predictions
proba_ext     = model_ext.predict_proba(X_ext)
prob_inf_ext  = proba_ext[:, 0]   # P(informal)
pred_ext      = np.where(prob_inf_ext >= THRESHOLD, "Informal", "Formal")
conf_ext      = np.abs(prob_inf_ext - 0.5)

# Baseline model predictions
proba_base    = model_base.predict_proba(X_base)
prob_inf_base = proba_base[:, 0]
pred_base     = np.where(prob_inf_base >= THRESHOLD, "Informal", "Formal")

print(f"  Extended  — Informal: {(pred_ext=='Informal').sum()}  "
      f"Formal: {(pred_ext=='Formal').sum()}")
print(f"  Baseline  — Informal: {(pred_base=='Informal').sum()}  "
      f"Formal: {(pred_base=='Formal').sum()}")

# Agreement between models
agreement = (pred_ext == pred_base).sum()
print(f"  Models agree on    : {agreement}/203 wards ({agreement/203*100:.1f}%)")

# Risk tiers
def risk_tier(p):
    if p >= 0.75:   return "High informal risk"
    elif p >= 0.50: return "Moderate informal risk"
    elif p >= 0.25: return "Low informal risk"
    else:           return "Formal / planned"

df_ext["prob_informal"]  = prob_inf_ext
df_ext["prob_formal"]    = 1 - prob_inf_ext
df_ext["pred_label"]     = pred_ext
df_ext["confidence"]     = conf_ext
df_ext["risk_tier"]      = [risk_tier(p) for p in prob_inf_ext]
df_ext["prob_inf_base"]  = prob_inf_base
df_ext["pred_base"]      = pred_base
df_ext["model_agrees"]   = (pred_ext == pred_base)

print("\n  Risk tier distribution (extended model):")
print(df_ext["risk_tier"].value_counts().to_string())


# =============================================================================
# 4.  MERGE WITH GEOMETRIES
# =============================================================================

print("\n" + "="*60)
print("STEP 4 — Merge with ward geometries")
print("="*60)

pred_gdf = gdf.merge(df_ext, on="GID_4", how="left",
                     suffixes=("", "_drop"))
pred_gdf = pred_gdf[[c for c in pred_gdf.columns
                      if not c.endswith("_drop")]]

# Clean up duplicate name columns
for col in ["NAME_4", "NAME_3", "NAME_2"]:
    if f"{col}_x" in pred_gdf.columns:
        pred_gdf = pred_gdf.rename(columns={f"{col}_x": col})
    if f"{col}_y" in pred_gdf.columns:
        pred_gdf = pred_gdf.drop(columns=[f"{col}_y"])

print(f"  Prediction GDF : {pred_gdf.shape}, CRS: {pred_gdf.crs}")

print("\n  Top 10 most informal wards:")
top10 = pred_gdf.nlargest(10, "prob_informal")[
    ["NAME_4", "NAME_3", "prob_informal", "risk_tier"]
]
print(top10.to_string(index=False))

print("\n  Top 10 most formal wards:")
top10f = pred_gdf.nlargest(10, "prob_formal")[
    ["NAME_4", "NAME_3", "prob_formal", "risk_tier"]
]
print(top10f.to_string(index=False))


# =============================================================================
# 5.  SAVE OUTPUTS
# =============================================================================

print("\n" + "="*60)
print("STEP 5 — Save prediction outputs")
print("="*60)

id_cols   = ["GID_4", "NAME_4", "NAME_3", "NAME_2"]
pred_cols = ["label", "label_src", "prob_informal", "prob_formal",
             "pred_label", "confidence", "risk_tier",
             "prob_inf_base", "pred_base", "model_agrees"]
feat_cols = [c for c in FEATURE_COLS_EXT if c in pred_gdf.columns]

save_cols = [c for c in id_cols + pred_cols + feat_cols
             if c in pred_gdf.columns]

pred_gdf[save_cols].to_csv(OUT_CSV, index=False)
print(f"  [SAVED] {OUT_CSV}")

pred_gdf.to_file(OUT_GPKG, driver="GPKG")
print(f"  [SAVED] {OUT_GPKG}")


# =============================================================================
# 6.  MAIN PROBABILITY MAP
# =============================================================================

print("\n" + "="*60)
print("STEP 6 — Generate maps")
print("="*60)

def add_colorbar(fig, ax, cmap, vmin, vmax, label):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(label, fontsize=9)

# Main probability map
fig, ax = plt.subplots(figsize=(11, 13))
pred_gdf.plot(
    column="prob_informal", ax=ax,
    cmap="RdYlGn_r", vmin=0, vmax=1,
    edgecolor="white", linewidth=0.4, legend=False
)
add_colorbar(fig, ax, "RdYlGn_r", 0, 1,
             "Informality Probability\n(0=Formal · 1=Informal)")

ax.set_title(
    "Informal Settlement Probability by Ward\n"
    "Dhaka Metropolitan Region · Extended Model (12 features)",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Easting (m)", fontsize=9)
ax.set_ylabel("Northing (m)", fontsize=9)

# Annotate landmarks
landmarks = {
    "Gulshan":    "formal",
    "Uttara":     "formal",
    "Lalbagh":    "informal",
    "Hazaribagh": "informal",
    "Demra":      "informal",
}
for ward, wtype in landmarks.items():
    row = pred_gdf[pred_gdf["NAME_4"].str.contains(
        ward, case=False, na=False)]
    if len(row) > 0:
        c   = row.geometry.iloc[0].centroid
        col = "#B71C1C" if wtype == "informal" else "#1A237E"
        ax.annotate(ward, xy=(c.x, c.y), fontsize=7.5,
                    color=col, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=col, alpha=0.8))

plt.tight_layout()
plt.savefig(OUT_PROB_MAP, dpi=180, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_PROB_MAP}")


# =============================================================================
# 7.  FINAL DASHBOARD
# =============================================================================

fig = plt.figure(figsize=(20, 22))
gs  = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08)

# Panel A — probability
ax1 = fig.add_subplot(gs[0, 0])
pred_gdf.plot(column="prob_informal", ax=ax1,
              cmap="RdYlGn_r", vmin=0, vmax=1,
              edgecolor="white", linewidth=0.3, legend=False)
add_colorbar(fig, ax1, "RdYlGn_r", 0, 1, "Informality prob.")
ax1.set_title("A — Informality Probability\n(Extended model, 12 features)",
              fontweight="bold", fontsize=11)
ax1.set_xlabel("Easting (m)", fontsize=7)
ax1.set_ylabel("Northing (m)", fontsize=7)
ax1.tick_params(labelsize=7)

# Panel B — binary map
ax2 = fig.add_subplot(gs[0, 1])
color_dict = {"Informal": "#E53935", "Formal": "#1E88E5"}
pred_gdf["map_color"] = pred_gdf["pred_label"].map(color_dict)
pred_gdf.plot(ax=ax2, color=pred_gdf["map_color"],
              edgecolor="white", linewidth=0.3)
n_inf = (pred_gdf["pred_label"] == "Informal").sum()
n_for = (pred_gdf["pred_label"] == "Formal").sum()
ax2.legend(handles=[
    mpatches.Patch(color="#E53935", label=f"Informal (n={n_inf})"),
    mpatches.Patch(color="#1E88E5", label=f"Formal (n={n_for})"),
], loc="lower right", fontsize=10, framealpha=0.9)
ax2.set_title("B — Binary Classification",
              fontweight="bold", fontsize=11)
ax2.set_xlabel("Easting (m)", fontsize=7)
ax2.tick_params(labelsize=7)

# Panel C — model agreement map
ax3 = fig.add_subplot(gs[1, 0])
agree_colors = pred_gdf["model_agrees"].map(
    {True: "#43A047", False: "#E53935"}
)
pred_gdf.plot(ax=ax3, color=agree_colors,
              edgecolor="white", linewidth=0.3)
n_agree    = pred_gdf["model_agrees"].sum()
n_disagree = (~pred_gdf["model_agrees"]).sum()
ax3.legend(handles=[
    mpatches.Patch(color="#43A047",
                   label=f"Models agree (n={n_agree})"),
    mpatches.Patch(color="#E53935",
                   label=f"Models disagree (n={n_disagree})"),
], loc="lower right", fontsize=9, framealpha=0.9)
ax3.set_title("C — Baseline vs Extended Agreement",
              fontweight="bold", fontsize=11)
ax3.set_xlabel("Easting (m)", fontsize=7)
ax3.set_ylabel("Northing (m)", fontsize=7)
ax3.tick_params(labelsize=7)

# Panel D — risk tier bar chart
ax4 = fig.add_subplot(gs[1, 1])
tier_order  = ["High informal risk", "Moderate informal risk",
               "Low informal risk",  "Formal / planned"]
tier_colors = ["#B71C1C", "#EF5350", "#81C784", "#1E88E5"]
tier_counts = [(pred_gdf["risk_tier"] == t).sum() for t in tier_order]

bars = ax4.barh(tier_order, tier_counts, color=tier_colors,
                edgecolor="white", height=0.6)
for bar, count in zip(bars, tier_counts):
    ax4.text(bar.get_width() + 0.5,
             bar.get_y() + bar.get_height()/2,
             f"{count} wards", va="center",
             fontsize=10, fontweight="bold")

ax4.set_xlabel("Number of wards", fontsize=10)
ax4.set_title("D — Risk Tier Distribution",
              fontweight="bold", fontsize=11)
ax4.set_xlim(0, max(tier_counts) * 1.3)
ax4.grid(axis="x", linestyle="--", alpha=0.4)

# Model info box
info = (
    f"Extended Model: XGBoost\n"
    f"Features: 12 (8 original + 4 new)\n"
    f"CV F1: 0.9321 ± 0.0409\n"
    f"Test Accuracy: 90.91%\n"
    f"ROC-AUC: 0.9692\n"
    f"Training wards: 163\n"
    f"Predicted wards: 203\n"
    f"New features:\n"
    f"  S2 true NDBI, S2 MNDWI\n"
    f"  OSM building density\n"
    f"  OSM mean building area"
)
ax4.text(0.97, 0.05, info, transform=ax4.transAxes,
         fontsize=8.5, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                   ec="grey", alpha=0.9))

fig.suptitle(
    "Informal Settlement Classification — Dhaka Metropolitan Region\n"
    "Extended Model · Sentinel-2 + OSM + Landsat · XGBoost",
    fontsize=15, fontweight="bold", y=1.005
)

plt.savefig(OUT_DASHBOARD, dpi=180, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_DASHBOARD}")


# =============================================================================
# 8.  SIDE-BY-SIDE COMPARISON MAP
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(20, 13))
fig.suptitle(
    "Baseline (8 features) vs Extended (12 features)\n"
    "Informality Probability per Ward",
    fontsize=14, fontweight="bold"
)

for ax, col, title, subtitle in [
    (axes[0], "prob_inf_base",
     "Baseline Model",
     "8 features · CV F1=0.944 · Test Acc=84.9%"),
    (axes[1], "prob_informal",
     "Extended Model",
     "12 features · CV F1=0.932 · Test Acc=90.9%"),
]:
    pred_gdf.plot(column=col, ax=ax,
                  cmap="RdYlGn_r", vmin=0, vmax=1,
                  edgecolor="white", linewidth=0.3, legend=False)
    add_colorbar(fig, ax, "RdYlGn_r", 0, 1, "Informality prob.")
    ax.set_title(f"{title}\n{subtitle}",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Easting (m)", fontsize=8)
    ax.tick_params(labelsize=8)

axes[0].set_ylabel("Northing (m)", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_COMPARE, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_COMPARE}")


# =============================================================================
# 9.  FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("NOTEBOOK 05b COMPLETE")
print("="*60)

print(f"\n  Total wards predicted : {len(pred_gdf)}")
print(f"  Predicted informal    : {n_inf}")
print(f"  Predicted formal      : {n_for}")
print(f"  Models agree          : {n_agree}/203 ({n_agree/203*100:.1f}%)")

print(f"\n  Risk tier breakdown:")
for tier, count in zip(tier_order, tier_counts):
    pct = count / len(pred_gdf) * 100
    print(f"    {tier:30s}: {count:3d} ({pct:.1f}%)")

print("\n  Output files:")
for label, path in [
    ("Predictions CSV",  OUT_CSV),
    ("Predictions GPKG", OUT_GPKG),
    ("Probability map",  OUT_PROB_MAP),
    ("Dashboard",        OUT_DASHBOARD),
    ("Comparison map",   OUT_COMPARE),
]:
    status = "✓" if os.path.exists(path) else "✗"
    print(f"    {status}  {label:18s}  →  {path}")

print("\n" + "="*60)
print("FULL PROJECT COMPLETE")
print("="*60)
print("""
  Complete pipeline:
    NB 01  — Data preparation          ✓
    NB 02  — Feature extraction        ✓
    NB 02b — Extended features         ✓
    NB 03  — Labeling                  ✓
    NB 04  — Baseline model            ✓
    NB 04b — Extended model            ✓
    NB 05  — Baseline predictions      ✓
    NB 05b — Extended predictions      ✓

  Key outputs:
    outputs/figures/05b_dashboard.png        ← main deliverable
    outputs/figures/05b_baseline_vs_extended_map.png
    data/processed/ward_predictions_extended.gpkg  ← load in QGIS

  To update QGIS:
    Layer → Add Vector Layer
    → data/processed/ward_predictions_extended.gpkg
    → Style by prob_informal, Graduated, RdYlGn_r inverted
""")
print("="*60)
