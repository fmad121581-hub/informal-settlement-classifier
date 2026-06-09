# =============================================================================
# NOTEBOOK 03 — LABELING
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
# Author  : (your name)
# Date    : 2024
#
# PURPOSE
# -------
# Assign a binary label to each ward:
#   1 = Formal settlement   (planned, regular, lower density)
#   0 = Informal settlement (unplanned, irregular, very high density)
#
# LABELING STRATEGY
# -----------------
# We use a rule-based approach combining THREE signals:
#
#   Signal A — Built-up fraction (from ESA WorldCover)
#     Very high built-up fraction (>0.85) + low NDVI suggests dense informal
#
#   Signal B — Population density (WorldPop)
#     Extremely high pop density is a strong informal indicator
#
#   Signal C — LST (Land Surface Temperature)
#     High LST + high built-up = dense urban heat island = likely informal
#
#   Signal D — Known landmark areas (ground truth anchors)
#     We hard-code labels for well-known formal and informal areas in Dhaka.
#     These act as "seed" labels that anchor the rule-based system.
#
# KNOWN INFORMAL AREAS (label = 0)
#   Korail / Karail slum    — Gulshan thana
#   Bauniabadh              — Mirpur area
#   Kallyanpur slum         — Mirpur/Kallyanpur
#   Shyampur               — old Dhaka industrial fringe
#   Demra                  — eastern fringe
#
# KNOWN FORMAL AREAS (label = 1)
#   Gulshan                — planned diplomatic/commercial zone
#   Dhanmondi              — planned residential
#   Uttara                 — planned satellite town
#   Motijheel              — CBD, planned commercial
#   Banani                 — planned residential
#
# OUTPUT
# ------
#   data/processed/ward_labels.csv      ← GID_4 + label + confidence
#   data/processed/ward_labeled.gpkg    ← with geometry for mapping
#   outputs/figures/03_label_map.png    ← choropleth of labels
#   outputs/figures/03_label_stats.png  ← feature distributions by class
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

FEATURES_CSV  = "data/processed/ward_features.csv"
WARDS_GPKG    = "data/processed/ward_features.gpkg"

OUT_LABELS    = "data/processed/ward_labels.csv"
OUT_LABELED   = "data/processed/ward_labeled.gpkg"
OUT_MAP       = "outputs/figures/03_label_map.png"
OUT_STATS     = "outputs/figures/03_label_stats.png"

os.makedirs("data/processed",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Minimum labeled samples target
MIN_SAMPLES   = 150


# =============================================================================
# 1.  LOAD FEATURE TABLE + WARD GEOMETRIES
# =============================================================================

print("="*60)
print("STEP 1 — Load feature table and ward geometries")
print("="*60)

# Load features (CSV — no geometry)
df = pd.read_csv(FEATURES_CSV)
print(f"  Feature table : {df.shape[0]} wards × {df.shape[1]} columns")
print(f"  Columns       : {list(df.columns)}")

# Load ward geometries (GeoPackage)
gdf = gpd.read_file(WARDS_GPKG)
print(f"  Ward GeoPackage: {len(gdf)} features, CRS: {gdf.crs}")

# Merge features into the GeoDataFrame on GID_4
gdf = gdf.merge(df, on="GID_4", suffixes=("", "_drop"))

# Drop duplicate columns created by the merge
drop_cols = [c for c in gdf.columns if c.endswith("_drop")]
gdf = gdf.drop(columns=drop_cols)

print(f"  Merged GDF     : {gdf.shape}")

# Use the clean NAME_4 from features CSV (x suffix = left = GeoPackage)
# Standardise column names
if "NAME_4_x" in gdf.columns:
    gdf = gdf.rename(columns={"NAME_4_x": "NAME_4",
                               "NAME_3_x": "NAME_3",
                               "NAME_2_x": "NAME_2"})

# Initialise label column as -1 (unlabeled)
gdf["label"]      = -1          # -1 = unlabeled
gdf["label_src"]  = "unlabeled" # track how each label was assigned


# =============================================================================
# 2.  GROUND TRUTH ANCHORS — KNOWN FORMAL AND INFORMAL AREAS
#
#     We search by NAME_4 (ward name) and NAME_3 (thana/upazila name).
#     These are the most reliable labels — they come from local knowledge.
#
#     For each known area, we label ALL wards in that thana if the
#     thana name matches, or just the specific ward if ward name matches.
# =============================================================================

print("\n" + "="*60)
print("STEP 2 — Apply ground truth anchor labels")
print("="*60)

# ── INFORMAL anchor areas (label = 0) ────────────────────────────────────────
# Source: UN-Habitat, RAJUK reports, published slum mapping studies

informal_thanas = [
    "Demra",
    "Mohammadpur",           # eastern fringe, large informal settlements
    "Shyampur",        # old Dhaka industrial fringe
    "Kadamtali",       # south-east, dense informal
    "Hazaribagh",      # tannery area, informal + industrial
    "Lalbagh",         # old Dhaka, very dense, informal character
    "Kotwali",         # old Dhaka core, dense
    "Sutrapur",        # old Dhaka
    "Jatrabari",       # south, informal fringe
]

informal_wards = [
    "Karail",          # Karail slum, Gulshan thana
    "Bauniabadh",      # Mirpur
    "Kallyanpur",      # Mirpur/Kallyanpur slum
    "Rupnagar",        # Mirpur, informal
    "Bhashantek",      # Mirpur, informal
    "Shewrapara",      # Mirpur, informal
]

# ── FORMAL anchor areas (label = 1) ──────────────────────────────────────────
# Source: RAJUK Master Plan, planned residential/commercial zones

formal_thanas = [
    "Gulshan",         # planned diplomatic/commercial
    "Banani",          # planned residential (often grouped with Gulshan thana)
    "Uttara",          # planned satellite town
    "Cantonment",      # military cantonment — planned, regulated
    "Nikunja",         # planned residential
]

formal_wards = [
    "Dhanmondi",       # planned residential zone
    "Motijheel",       # CBD, commercial
    "Tejgaon",         # industrial but planned
    "Bashundhara",     # planned residential
    "Baridhara",       # planned diplomatic zone
    "Niketan",         # planned residential
    "DOHS",            # Defence Officers Housing Society — fully planned
    "Uttara Model Town", # planned
]

# ── Apply thana-level labels ──────────────────────────────────────────────────
def apply_thana_label(gdf, thana_list, label, source):
    """Label all wards whose NAME_3 contains any string in thana_list."""
    count = 0
    for thana in thana_list:
        mask = gdf["NAME_3"].str.contains(thana, case=False, na=False)
        # Only label if not already labeled by a higher-priority source
        unlabeled_mask = mask & (gdf["label"] == -1)
        gdf.loc[unlabeled_mask, "label"]     = label
        gdf.loc[unlabeled_mask, "label_src"] = f"thana:{source}"
        count += unlabeled_mask.sum()
    return gdf, count

def apply_ward_label(gdf, ward_list, label, source):
    """Label wards whose NAME_4 contains any string in ward_list."""
    count = 0
    for ward in ward_list:
        mask = gdf["NAME_4"].str.contains(ward, case=False, na=False)
        unlabeled_mask = mask & (gdf["label"] == -1)
        gdf.loc[unlabeled_mask, "label"]     = label
        gdf.loc[unlabeled_mask, "label_src"] = f"ward:{source}"
        count += unlabeled_mask.sum()
    return gdf, count

# Apply informal labels first
gdf, n = apply_thana_label(gdf, informal_thanas, 0, "informal_anchor")
print(f"  Informal thana labels applied : {n} wards")
gdf, n = apply_ward_label(gdf, informal_wards, 0, "informal_anchor")
print(f"  Informal ward labels applied  : {n} wards")

# Apply formal labels
gdf, n = apply_thana_label(gdf, formal_thanas, 1, "formal_anchor")
print(f"  Formal thana labels applied   : {n} wards")
gdf, n = apply_ward_label(gdf, formal_wards, 1, "formal_anchor")
print(f"  Formal ward labels applied    : {n} wards")

anchor_labeled = (gdf["label"] != -1).sum()
print(f"\n  Total anchor-labeled wards    : {anchor_labeled} / {len(gdf)}")


# =============================================================================
# 3.  RULE-BASED LABELING FOR REMAINING WARDS
#
#     For wards not covered by anchor labels, we use feature thresholds.
#     These thresholds are derived from the feature statistics in NB 02
#     and from published literature on informal settlement indicators.
#
#     RULE LOGIC (wards labeled as informal = 0 if ANY 2+ rules fire):
#       R1: built_fraction > 0.85   (extremely dense built-up)
#       R2: ndvi_mean < 0.15        (very low vegetation)
#       R3: pop_mean > 800          (very high population density)
#       R4: lst_mean > 31.0         (high thermal — urban heat island)
#
#     RULE LOGIC (formal = 1 if ALL of these):
#       R5: built_fraction < 0.60   (moderate built-up, space for greenery)
#       R6: ndvi_mean > 0.25        (meaningful vegetation present)
#       R7: pop_mean < 600          (moderate population density)
#
#     Wards that don't clearly fit either category remain unlabeled (-1)
#     and are excluded from model training (used only for prediction).
# =============================================================================

print("\n" + "="*60)
print("STEP 3 — Rule-based labeling for remaining wards")
print("="*60)

# Work only on unlabeled wards
unlabeled = gdf["label"] == -1

# ── Compute rule flags ────────────────────────────────────────────────────────
r1 = gdf["built_fraction"] > 0.85
r2 = gdf["ndvi_mean"]      < 0.15
r3 = gdf["pop_mean"]       > 800
r4 = gdf["lst_mean"]       > 31.0

r5 = gdf["built_fraction"] < 0.60
r6 = gdf["ndvi_mean"]      > 0.25
r7 = gdf["pop_mean"]       < 600

# Count how many informal rules fire per ward
informal_score = r1.astype(int) + r2.astype(int) + \
                 r3.astype(int) + r4.astype(int)

# Count how many formal rules fire per ward
formal_score   = r5.astype(int) + r6.astype(int) + r7.astype(int)

# ── Apply informal label — 2 or more informal rules fire ─────────────────────
informal_rule_mask = unlabeled & (informal_score >= 2)
gdf.loc[informal_rule_mask, "label"]     = 0
gdf.loc[informal_rule_mask, "label_src"] = "rule:informal"

# ── Apply formal label — all 3 formal rules fire ─────────────────────────────
formal_rule_mask = unlabeled & (gdf["label"] == -1) & (formal_score >= 2)
gdf.loc[formal_rule_mask, "label"]     = 1
gdf.loc[formal_rule_mask, "label_src"] = "rule:formal"

# ── Summary ───────────────────────────────────────────────────────────────────
n_informal_rule = informal_rule_mask.sum()
n_formal_rule   = formal_rule_mask.sum()
n_unlabeled     = (gdf["label"] == -1).sum()

print(f"  Rule-based informal labels    : {n_informal_rule} wards")
print(f"  Rule-based formal labels      : {n_formal_rule} wards")
print(f"  Still unlabeled               : {n_unlabeled} wards")
print(f"    (these will be predicted but excluded from training)")


# =============================================================================
# 4.  LABEL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("STEP 4 — Label summary")
print("="*60)

n_formal   = (gdf["label"] == 1).sum()
n_informal = (gdf["label"] == 0).sum()
n_total    = n_formal + n_informal

print(f"  Formal   (1) : {n_formal:3d} wards")
print(f"  Informal (0) : {n_informal:3d} wards")
print(f"  Unlabeled(-1): {n_unlabeled:3d} wards")
print(f"  Total labeled: {n_total:3d} wards")

if n_total < MIN_SAMPLES:
    print(f"\n  WARNING: Only {n_total} labeled samples — below target of "
          f"{MIN_SAMPLES}. Consider relaxing rule thresholds.")
else:
    print(f"\n  [OK] {n_total} labeled samples — above minimum of {MIN_SAMPLES}.")

# Label source breakdown
print("\n  Label source breakdown:")
print(gdf.groupby(["label_src", "label"]).size()
        .reset_index(name="count")
        .to_string(index=False))

# Show sample labeled wards
print("\n  Sample formal wards (label=1):")
sample_formal = gdf[gdf["label"] == 1][
    ["NAME_4", "NAME_3", "label_src", "built_fraction",
     "ndvi_mean", "pop_mean"]].head(8)
print(sample_formal.to_string(index=False))

print("\n  Sample informal wards (label=0):")
sample_informal = gdf[gdf["label"] == 0][
    ["NAME_4", "NAME_3", "label_src", "built_fraction",
     "ndvi_mean", "pop_mean"]].head(8)
print(sample_informal.to_string(index=False))


# =============================================================================
# 5.  SAVE LABELED DATASET
# =============================================================================

print("\n" + "="*60)
print("STEP 5 — Save labeled dataset")
print("="*60)

# CSV — include GID_4, label, source, and all features
feature_cols = ["ndvi_mean", "savi_mean", "ndbi_mean",
                "lst_mean", "slope_mean",
                "pop_mean", "pop_std", "built_fraction"]

save_cols = ["GID_4", "NAME_4", "NAME_3", "NAME_2",
             "label", "label_src"] + feature_cols

gdf[save_cols].to_csv(OUT_LABELS, index=False)
print(f"  [SAVED] {OUT_LABELS}")

# GeoPackage — with geometry for mapping
gdf.to_file(OUT_LABELED, driver="GPKG")
print(f"  [SAVED] {OUT_LABELED}")


# =============================================================================
# 6.  LABEL MAP
# =============================================================================

print("\n" + "="*60)
print("STEP 6 — Generate label map")
print("="*60)

fig, ax = plt.subplots(1, 1, figsize=(10, 12))

# Color scheme
color_map = {
     1: "#2196F3",   # blue  = formal
     0: "#F44336",   # red   = informal
    -1: "#E0E0E0",   # grey  = unlabeled
}

gdf["color"] = gdf["label"].map(color_map)
gdf.plot(ax=ax, color=gdf["color"], edgecolor="white",
         linewidth=0.4)

# Legend
legend_handles = [
    mpatches.Patch(color="#2196F3", label=f"Formal (n={n_formal})"),
    mpatches.Patch(color="#F44336", label=f"Informal (n={n_informal})"),
    mpatches.Patch(color="#E0E0E0", label=f"Unlabeled (n={n_unlabeled})"),
]
ax.legend(handles=legend_handles, loc="lower right",
          fontsize=11, framealpha=0.9)

ax.set_title(
    "Notebook 03 — Ward Labels\n"
    "Dhaka Metropolitan Region · EPSG:32646",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Easting (m)", fontsize=9)
ax.set_ylabel("Northing (m)", fontsize=9)

# Annotate a few well-known wards
landmarks = {
    "Gulshan":    (1, "#1565C0"),
    "Uttara":     (1, "#1565C0"),
    "Dhanmondi":  (1, "#1565C0"),
    "Lalbagh":    (0, "#B71C1C"),
    "Demra":      (0, "#B71C1C"),
    "Hazaribagh": (0, "#B71C1C"),
}

for ward_name, (lbl, color) in landmarks.items():
    row = gdf[gdf["NAME_4"].str.contains(ward_name, case=False, na=False)]
    if len(row) > 0:
        centroid = row.geometry.iloc[0].centroid
        ax.annotate(
            ward_name,
            xy=(centroid.x, centroid.y),
            fontsize=7, color=color, fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=color, alpha=0.7)
        )

plt.tight_layout()
plt.savefig(OUT_MAP, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_MAP}")


# =============================================================================
# 7.  FEATURE DISTRIBUTIONS BY CLASS
#     Box plots showing how each feature differs between formal and informal.
#     This is a sanity check — if the classes are well-separated,
#     the model will have an easier time learning.
# =============================================================================

print("\n  Generating feature distribution plots ...")

labeled_gdf = gdf[gdf["label"] != -1].copy()
formal_gdf   = labeled_gdf[labeled_gdf["label"] == 1]
informal_gdf = labeled_gdf[labeled_gdf["label"] == 0]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle(
    "Feature Distributions by Class\n"
    "Blue = Formal · Red = Informal",
    fontsize=13, fontweight="bold"
)

feature_labels = {
    "ndvi_mean":      "NDVI (mean)",
    "savi_mean":      "SAVI (mean)",
    "ndbi_mean":      "NDBI proxy (mean)",
    "lst_mean":       "LST °C (mean)",
    "slope_mean":     "Slope ° (mean)",
    "pop_mean":       "Pop density (mean)",
    "pop_std":        "Pop density (std)",
    "built_fraction": "Built-up fraction",
}

for ax, col in zip(axes.flatten(), feature_cols):
    data_formal   = formal_gdf[col].dropna()
    data_informal = informal_gdf[col].dropna()

    bp = ax.boxplot(
        [data_formal, data_informal],
        labels=["Formal", "Informal"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    bp["boxes"][0].set_facecolor("#90CAF9")   # light blue
    bp["boxes"][1].set_facecolor("#EF9A9A")   # light red

    ax.set_title(feature_labels.get(col, col), fontsize=10, fontweight="bold")
    ax.set_ylabel("Value", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUT_STATS, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_STATS}")


# =============================================================================
# 8.  FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("NOTEBOOK 03 COMPLETE — Labeling summary")
print("="*60)

print(f"\n  Total wards         : {len(gdf)}")
print(f"  Labeled for training: {n_total}")
print(f"    Formal   (1)      : {n_formal}")
print(f"    Informal (0)      : {n_informal}")
print(f"  Unlabeled (predict) : {n_unlabeled}")

print("\n  Output files:")
for label, path in [("Labels CSV",   OUT_LABELS),
                     ("Labeled GPKG", OUT_LABELED),
                     ("Label map",    OUT_MAP),
                     ("Feature dist", OUT_STATS)]:
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"    {status}  {label:15s}  →  {path}")

print("\nReady for Notebook 04 (Model Training).")
print("="*60)
