# =============================================================================
# NOTEBOOK 05 — PREDICTION MAPPING
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
# Author  : (your name)
# Date    : 2024
#
# PURPOSE
# -------
# Use the trained XGBoost model to predict informal settlement probability
# for ALL 203 wards (including the 40 unlabeled ones) and produce the
# final choropleth map.
#
# OUTPUT MAPS
# -----------
#   Map 1 — Informality probability (0–1 continuous, all 203 wards)
#   Map 2 — Binary classification (formal / informal)
#   Map 3 — Confidence map (how certain the model is)
#   Map 4 — Ground truth labels vs predictions (validation overlay)
#
# INPUT FILES
# -----------
#   data/processed/ward_features.csv    ← features for ALL 203 wards
#   data/processed/ward_labels.csv      ← labels (163 labeled, 40 unlabeled)
#   data/processed/ward_features.gpkg   ← geometries for all 203 wards
#   outputs/model/best_model.joblib     ← trained XGBoost model
#   outputs/model/scaler.joblib         ← fitted StandardScaler
#
# OUTPUT FILES
# ------------
#   data/processed/ward_predictions.csv
#   data/processed/ward_predictions.gpkg
#   outputs/figures/05_informality_probability.png   ← MAIN MAP
#   outputs/figures/05_binary_classification.png
#   outputs/figures/05_confidence_map.png
#   outputs/figures/05_validation_overlay.png
#   outputs/figures/05_final_dashboard.png           ← all maps together
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
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from joblib import load


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

FEATURES_CSV   = "data/processed/ward_features.csv"
LABELS_CSV     = "data/processed/ward_labels.csv"
WARDS_GPKG     = "data/processed/ward_features.gpkg"
MODEL_PATH     = "outputs/model/best_model.joblib"
SCALER_PATH    = "outputs/model/scaler.joblib"

OUT_PRED_CSV   = "data/processed/ward_predictions.csv"
OUT_PRED_GPKG  = "data/processed/ward_predictions.gpkg"
OUT_PROB_MAP   = "outputs/figures/05_informality_probability.png"
OUT_BIN_MAP    = "outputs/figures/05_binary_classification.png"
OUT_CONF_MAP   = "outputs/figures/05_confidence_map.png"
OUT_VAL_MAP    = "outputs/figures/05_validation_overlay.png"
OUT_DASHBOARD  = "outputs/figures/05_final_dashboard.png"

os.makedirs("data/processed",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

FEATURE_COLS = [
    "ndvi_mean",
    "savi_mean",
    "ndbi_mean",
    "lst_mean",
    "slope_mean",
    "pop_mean",
    "pop_std",
    "built_fraction",
]

# Probability threshold for binary classification
# Wards with informal probability > THRESHOLD are classified as informal
THRESHOLD = 0.50


# =============================================================================
# 1.  LOAD ALL DATA
# =============================================================================

print("="*60)
print("STEP 1 — Load data, model, and scaler")
print("="*60)

# ── Features for ALL 203 wards ────────────────────────────────────────────────
features_df = pd.read_csv(FEATURES_CSV)
print(f"  Features CSV    : {features_df.shape} — {list(features_df.columns)}")

# ── Labels (163 labeled + 40 unlabeled) ───────────────────────────────────────
labels_df = pd.read_csv(LABELS_CSV)
print(f"  Labels CSV      : {labels_df.shape}")

# ── Ward geometries ───────────────────────────────────────────────────────────
gdf = gpd.read_file(WARDS_GPKG)
print(f"  Ward GeoPackage : {len(gdf)} features, CRS: {gdf.crs}")

# ── Trained model ─────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run NB 04 first.")
model = load(MODEL_PATH)
print(f"  Model loaded    : {MODEL_PATH}")
print(f"  Model type      : {type(model).__name__}")

# ── Scaler ────────────────────────────────────────────────────────────────────
scaler = load(SCALER_PATH)
print(f"  Scaler loaded   : {SCALER_PATH}")


# =============================================================================
# 2.  PREPARE FULL FEATURE MATRIX
#     We predict for ALL 203 wards, not just the 163 labeled ones.
#     The 40 unlabeled wards will get predictions for the first time here.
# =============================================================================

print("\n" + "="*60)
print("STEP 2 — Prepare full feature matrix (203 wards)")
print("="*60)

# Merge labels into features on GID_4
# Use how="left" so all 203 wards are kept even if label is missing
full_df = features_df.merge(
    labels_df[["GID_4", "label", "label_src"]],
    on  = "GID_4",
    how = "left"
)

# Fill unlabeled wards with -1
full_df["label"]     = full_df["label"].fillna(-1).astype(int)
full_df["label_src"] = full_df["label_src"].fillna("unlabeled")

print(f"  Full feature table : {full_df.shape}")
print(f"  Labeled (0 or 1)   : {(full_df['label'] != -1).sum()}")
print(f"  Unlabeled (-1)     : {(full_df['label'] == -1).sum()}")

# Extract feature matrix
X_all = full_df[FEATURE_COLS].values

# Check for NaN
nan_count = np.isnan(X_all).sum()
if nan_count > 0:
    print(f"  Filling {nan_count} NaN values with column median...")
    for j in range(X_all.shape[1]):
        col_median = np.nanmedian(X_all[:, j])
        X_all[np.isnan(X_all[:, j]), j] = col_median

print(f"  Feature matrix     : {X_all.shape} — no NaN values")


# =============================================================================
# 3.  PREDICT
#
#     XGBoost was saved directly (not in a Pipeline), so we pass raw
#     unscaled features. The scaler is only needed if SVM had won.
#
#     predict_proba returns [P(informal), P(formal)] for each ward.
#     We use P(informal) = column 0 as our informality score.
# =============================================================================

print("\n" + "="*60)
print("STEP 3 — Generate predictions for all 203 wards")
print("="*60)

# Get class probabilities
# Column 0 = P(informal=0), Column 1 = P(formal=1)
proba_all = model.predict_proba(X_all)

prob_informal = proba_all[:, 0]   # informality probability (our main output)
prob_formal   = proba_all[:, 1]   # formality probability

# Binary prediction using threshold
pred_binary = (prob_informal >= THRESHOLD).astype(int)
# 1 = informal predicted, 0 = formal predicted
# Note: we flip convention here so 1=informal for the map
pred_label = np.where(pred_binary == 1, "Informal", "Formal")

# Confidence = distance from 0.5 (how certain the model is)
# 0.0 = completely uncertain, 0.5 = completely certain
confidence = np.abs(prob_informal - 0.5)

print(f"  Predicted informal : {pred_binary.sum()} wards")
print(f"  Predicted formal   : {(pred_binary == 0).sum()} wards")
print(f"  Mean informality prob : {prob_informal.mean():.3f}")
print(f"  Mean confidence       : {confidence.mean():.3f}")

# ── Add predictions to dataframe ──────────────────────────────────────────────
full_df["prob_informal"] = prob_informal
full_df["prob_formal"]   = prob_formal
full_df["pred_label"]    = pred_label
full_df["confidence"]    = confidence

# Informality risk tier
def risk_tier(p):
    if p >= 0.75:   return "High informal risk"
    elif p >= 0.50: return "Moderate informal risk"
    elif p >= 0.25: return "Low informal risk"
    else:           return "Formal / planned"

full_df["risk_tier"] = full_df["prob_informal"].apply(risk_tier)

print("\n  Risk tier distribution:")
print(full_df["risk_tier"].value_counts().to_string())


# =============================================================================
# 4.  MERGE PREDICTIONS WITH GEOMETRIES
# =============================================================================

print("\n" + "="*60)
print("STEP 4 — Merge predictions with ward geometries")
print("="*60)

# Merge on GID_4
pred_gdf = gdf.merge(full_df, on="GID_4", how="left", suffixes=("", "_drop"))

# Drop duplicate columns
drop_cols = [c for c in pred_gdf.columns if c.endswith("_drop")]
pred_gdf  = pred_gdf.drop(columns=drop_cols)

# Clean up NAME columns if duplicated by merge
for col in ["NAME_4", "NAME_3", "NAME_2"]:
    if f"{col}_x" in pred_gdf.columns:
        pred_gdf = pred_gdf.rename(columns={f"{col}_x": col})
        if f"{col}_y" in pred_gdf.columns:
            pred_gdf = pred_gdf.drop(columns=[f"{col}_y"])

print(f"  Prediction GDF : {pred_gdf.shape}")
print(f"  CRS            : {pred_gdf.crs}")
print(f"  Columns        : {list(pred_gdf.columns)}")

# Preview top 10 most informal wards
print("\n  Top 10 most informal wards (by probability):")
top10 = pred_gdf.nlargest(10, "prob_informal")[
    ["NAME_4", "NAME_3", "prob_informal", "risk_tier", "label"]
]
print(top10.to_string(index=False))

# Preview top 10 most formal wards
print("\n  Top 10 most formal wards (by probability):")
top10f = pred_gdf.nlargest(10, "prob_formal")[
    ["NAME_4", "NAME_3", "prob_formal", "risk_tier", "label"]
]
print(top10f.to_string(index=False))


# =============================================================================
# 5.  SAVE PREDICTION TABLE
# =============================================================================

print("\n" + "="*60)
print("STEP 5 — Save prediction outputs")
print("="*60)

# CSV (no geometry)
save_cols = ["GID_4", "NAME_4", "NAME_3", "NAME_2",
             "label", "label_src",
             "prob_informal", "prob_formal",
             "pred_label", "confidence", "risk_tier"] + FEATURE_COLS

pred_gdf[save_cols].to_csv(OUT_PRED_CSV, index=False)
print(f"  [SAVED] {OUT_PRED_CSV}")

# GeoPackage (with geometry)
pred_gdf.to_file(OUT_PRED_GPKG, driver="GPKG")
print(f"  [SAVED] {OUT_PRED_GPKG}")


# =============================================================================
# 6.  MAP 1 — INFORMALITY PROBABILITY CHOROPLETH
#     This is the MAIN output of the entire project.
#     Each ward is shaded from white (formal) to dark red (very informal).
# =============================================================================

print("\n" + "="*60)
print("STEP 6 — Generate maps")
print("="*60)

def add_colorbar(fig, ax, cmap, vmin, vmax, label, orientation="vertical"):
    """Add a neat colorbar to an axis."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation=orientation,
                        shrink=0.7, pad=0.02)
    cbar.set_label(label, fontsize=9)
    return cbar

# ── Map 1: Informality probability ───────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(11, 13))

pred_gdf.plot(
    column    = "prob_informal",
    ax        = ax,
    cmap      = "RdYlGn_r",     # red = informal, green = formal
    vmin      = 0,
    vmax      = 1,
    edgecolor = "white",
    linewidth = 0.4,
    legend    = False,
)

add_colorbar(fig, ax, "RdYlGn_r", 0, 1,
             "Informality Probability\n(0 = Formal · 1 = Informal)")

ax.set_title(
    "Informal Settlement Probability by Ward\n"
    "Dhaka Metropolitan Region · Landsat 9 · Feb 2022",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Easting (m)", fontsize=9)
ax.set_ylabel("Northing (m)", fontsize=9)

# Annotate landmark wards
landmarks = {
    "Gulshan":    "formal",
    "Uttara":     "formal",
    "Lalbagh":    "informal",
    "Hazaribagh": "informal",
    "Demra":      "informal",
}
for ward_name, wtype in landmarks.items():
    row = pred_gdf[pred_gdf["NAME_4"].str.contains(
        ward_name, case=False, na=False)]
    if len(row) > 0:
        c   = row.geometry.iloc[0].centroid
        col = "#B71C1C" if wtype == "informal" else "#1A237E"
        ax.annotate(
            ward_name,
            xy     = (c.x, c.y),
            fontsize = 7.5,
            color  = col,
            fontweight = "bold",
            ha     = "center",
            bbox   = dict(boxstyle="round,pad=0.25", fc="white",
                          ec=col, alpha=0.8, linewidth=1)
        )

plt.tight_layout()
plt.savefig(OUT_PROB_MAP, dpi=180, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_PROB_MAP}")


# ── Map 2: Binary classification ─────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(11, 13))

color_dict = {"Informal": "#E53935", "Formal": "#1E88E5"}
pred_gdf["map_color"] = pred_gdf["pred_label"].map(color_dict)

pred_gdf.plot(
    ax        = ax,
    color     = pred_gdf["map_color"],
    edgecolor = "white",
    linewidth = 0.4,
)

n_inf = (pred_gdf["pred_label"] == "Informal").sum()
n_for = (pred_gdf["pred_label"] == "Formal").sum()
legend_handles = [
    mpatches.Patch(color="#E53935",
                   label=f"Informal settlement (n={n_inf})"),
    mpatches.Patch(color="#1E88E5",
                   label=f"Formal settlement (n={n_for})"),
]
ax.legend(handles=legend_handles, loc="lower right",
          fontsize=11, framealpha=0.9)

ax.set_title(
    f"Informal vs Formal Settlement Classification\n"
    f"Dhaka Metropolitan Region · Threshold = {THRESHOLD}",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Easting (m)", fontsize=9)
ax.set_ylabel("Northing (m)", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_BIN_MAP, dpi=180, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_BIN_MAP}")


# ── Map 3: Confidence map ─────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(11, 13))

pred_gdf.plot(
    column    = "confidence",
    ax        = ax,
    cmap      = "viridis",
    vmin      = 0,
    vmax      = 0.5,
    edgecolor = "white",
    linewidth = 0.4,
    legend    = False,
)

add_colorbar(fig, ax, "viridis", 0, 0.5,
             "Model Confidence\n(0 = uncertain · 0.5 = certain)")

ax.set_title(
    "Model Prediction Confidence by Ward\n"
    "Dhaka Metropolitan Region",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Easting (m)", fontsize=9)
ax.set_ylabel("Northing (m)", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_CONF_MAP, dpi=180, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_CONF_MAP}")


# ── Map 4: Ground truth vs prediction comparison ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 13))

# Left: ground truth labels
label_colors = {1: "#1E88E5", 0: "#E53935", -1: "#BDBDBD"}
pred_gdf["gt_color"] = pred_gdf["label"].map(label_colors)

pred_gdf.plot(ax=axes[0], color=pred_gdf["gt_color"],
              edgecolor="white", linewidth=0.4)
axes[0].set_title("Ground Truth Labels\n(from NB 03)",
                  fontsize=12, fontweight="bold")
axes[0].set_xlabel("Easting (m)", fontsize=8)
axes[0].set_ylabel("Northing (m)", fontsize=8)

gt_handles = [
    mpatches.Patch(color="#1E88E5", label="Formal (labeled)"),
    mpatches.Patch(color="#E53935", label="Informal (labeled)"),
    mpatches.Patch(color="#BDBDBD", label="Unlabeled"),
]
axes[0].legend(handles=gt_handles, loc="lower right",
               fontsize=9, framealpha=0.9)

# Right: model predictions
pred_gdf.plot(ax=axes[1], color=pred_gdf["map_color"],
              edgecolor="white", linewidth=0.4)
axes[1].set_title("Model Predictions (XGBoost)\n(all 203 wards)",
                  fontsize=12, fontweight="bold")
axes[1].set_xlabel("Easting (m)", fontsize=8)

pred_handles = [
    mpatches.Patch(color="#1E88E5", label=f"Formal (n={n_for})"),
    mpatches.Patch(color="#E53935", label=f"Informal (n={n_inf})"),
]
axes[1].legend(handles=pred_handles, loc="lower right",
               fontsize=9, framealpha=0.9)

fig.suptitle(
    "Ground Truth vs Model Predictions\nDhaka Metropolitan Region",
    fontsize=14, fontweight="bold", y=1.01
)

plt.tight_layout()
plt.savefig(OUT_VAL_MAP, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_VAL_MAP}")


# =============================================================================
# 7.  FINAL DASHBOARD
#     All four maps in one figure — the main deliverable for the project.
# =============================================================================

print("\n  Generating final dashboard ...")

fig = plt.figure(figsize=(20, 22))
gs  = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08)

# ── Panel 1: Probability ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
pred_gdf.plot(
    column="prob_informal", ax=ax1,
    cmap="RdYlGn_r", vmin=0, vmax=1,
    edgecolor="white", linewidth=0.3, legend=False
)
add_colorbar(fig, ax1, "RdYlGn_r", 0, 1, "Informality prob.")
ax1.set_title("A — Informality Probability", fontweight="bold", fontsize=11)
ax1.set_xlabel("Easting (m)", fontsize=7)
ax1.set_ylabel("Northing (m)", fontsize=7)
ax1.tick_params(labelsize=7)

# ── Panel 2: Binary ───────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
pred_gdf.plot(
    ax=ax2, color=pred_gdf["map_color"],
    edgecolor="white", linewidth=0.3
)
ax2.legend(handles=pred_handles, loc="lower right",
           fontsize=8, framealpha=0.9)
ax2.set_title("B — Binary Classification", fontweight="bold", fontsize=11)
ax2.set_xlabel("Easting (m)", fontsize=7)
ax2.tick_params(labelsize=7)

# ── Panel 3: Confidence ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
pred_gdf.plot(
    column="confidence", ax=ax3,
    cmap="viridis", vmin=0, vmax=0.5,
    edgecolor="white", linewidth=0.3, legend=False
)
add_colorbar(fig, ax3, "viridis", 0, 0.5, "Confidence")
ax3.set_title("C — Prediction Confidence", fontweight="bold", fontsize=11)
ax3.set_xlabel("Easting (m)", fontsize=7)
ax3.set_ylabel("Northing (m)", fontsize=7)
ax3.tick_params(labelsize=7)

# ── Panel 4: Risk tier bar chart ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

tier_order  = ["High informal risk", "Moderate informal risk",
               "Low informal risk",  "Formal / planned"]
tier_colors = ["#B71C1C", "#EF5350", "#81C784", "#1E88E5"]
tier_counts = [
    (pred_gdf["risk_tier"] == t).sum() for t in tier_order
]

bars = ax4.barh(tier_order, tier_counts, color=tier_colors,
                edgecolor="white", height=0.6)

# Annotate bars
for bar, count in zip(bars, tier_counts):
    ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
             f"{count} wards", va="center", fontsize=10, fontweight="bold")

ax4.set_xlabel("Number of wards", fontsize=10)
ax4.set_title("D — Risk Tier Distribution\n(all 203 wards)",
              fontweight="bold", fontsize=11)
ax4.set_xlim(0, max(tier_counts) * 1.25)
ax4.tick_params(labelsize=9)
ax4.grid(axis="x", linestyle="--", alpha=0.4)

# Add model info text box
info_text = (
    f"Model: XGBoost\n"
    f"CV F1: 0.9444 ± 0.0237\n"
    f"Test Acc: 84.85%\n"
    f"ROC-AUC: 0.9654\n"
    f"Features: 8\n"
    f"Training wards: 163\n"
    f"Predicted wards: 203"
)
ax4.text(0.97, 0.05, info_text,
         transform=ax4.transAxes,
         fontsize=8.5, verticalalignment="bottom",
         horizontalalignment="right",
         bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                   ec="grey", alpha=0.9))

fig.suptitle(
    "Informal Settlement Classification — Dhaka Metropolitan Region\n"
    "Landsat 9 OLI/TIRS · February 2022 · XGBoost Classifier",
    fontsize=15, fontweight="bold", y=1.005
)

plt.savefig(OUT_DASHBOARD, dpi=180, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_DASHBOARD}")


# =============================================================================
# 8.  FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("NOTEBOOK 05 COMPLETE — Prediction mapping summary")
print("="*60)

print(f"\n  Total wards predicted  : {len(pred_gdf)}")
print(f"  Predicted informal     : {n_inf}")
print(f"  Predicted formal       : {n_for}")

print(f"\n  Risk tier breakdown:")
for tier, count in zip(tier_order, tier_counts):
    pct = count / len(pred_gdf) * 100
    print(f"    {tier:30s}: {count:3d} wards ({pct:.1f}%)")

print("\n  Output files:")
for label, path in [
    ("Predictions CSV",  OUT_PRED_CSV),
    ("Predictions GPKG", OUT_PRED_GPKG),
    ("Probability map",  OUT_PROB_MAP),
    ("Binary map",       OUT_BIN_MAP),
    ("Confidence map",   OUT_CONF_MAP),
    ("Validation map",   OUT_VAL_MAP),
    ("Final dashboard",  OUT_DASHBOARD),
]:
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"    {status}  {label:18s}  →  {path}")

print("\n" + "="*60)
print("PROJECT COMPLETE")
print("="*60)
print("""
  Pipeline summary:
    NB 01 — Data preparation     ✓
    NB 02 — Feature extraction   ✓
    NB 03 — Labeling             ✓
    NB 04 — Model training       ✓
    NB 05 — Prediction mapping   ✓

  Key outputs:
    outputs/figures/05_final_dashboard.png   ← main deliverable
    outputs/model/best_model.joblib          ← saved model
    data/processed/ward_predictions.gpkg     ← load in QGIS

  To view in QGIS:
    1. Open QGIS
    2. Layer → Add Layer → Add Vector Layer
    3. Select data/processed/ward_predictions.gpkg
    4. Style by prob_informal column using Graduated renderer
    5. Use RdYlGn_r color ramp for publication-quality map
""")
print("="*60)
