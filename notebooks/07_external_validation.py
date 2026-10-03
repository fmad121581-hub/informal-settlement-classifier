# =============================================================================
# NOTEBOOK 07 — EXTERNAL VALIDATION AGAINST EO4SD-URBAN (2017)
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
#
# PURPOSE
# -------
# Independent check: the ESA EO4SD-Urban Dhaka Informal Settlements 2017
# layer (World Bank Data Catalog, CC-BY 4.0) maps individual informal
# settlement patches from VHR imagery interpretation — a source that was
# NOT used anywhere in this project's labeling or feature engineering.
#
# For each ward we compute:
#   eo4sd_informal_fraction = (area of EO4SD informal patches in ward) / (ward area)
#
# and compare it against:
#   (a) this project's rule/anchor training label (0=informal, 1=formal)
#   (b) this project's model-predicted probability of informality
#
# OUTPUT
# ------
#   outputs/model/external_validation_report.txt
#   outputs/figures/07_external_validation.png
#   data/processed/ward_eo4sd_comparison.csv
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats

EO4SD_SHP     = "data/external/eo4sd/EO4SD_DHAKA_INFORMAL_2017.shp"
WARD_PRED     = "data/processed/ward_predictions_extended.gpkg"
LABELS_CSV    = "data/processed/ward_labels.csv"

OUT_REPORT = "outputs/model/external_validation_report.txt"
OUT_FIG    = "outputs/figures/07_external_validation.png"
OUT_CSV    = "data/processed/ward_eo4sd_comparison.csv"

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

print("="*70)
print("EXTERNAL VALIDATION — EO4SD-Urban Dhaka Informal Settlements 2017")
print("="*70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

eo4sd = gpd.read_file(EO4SD_SHP)
wards = gpd.read_file(WARD_PRED)
labels = pd.read_csv(LABELS_CSV)

print(f"\nEO4SD polygons : {len(eo4sd)}  CRS={eo4sd.crs}")
print(f"Wards          : {len(wards)}  CRS={wards.crs}")

if eo4sd.crs != wards.crs:
    print(f"Reprojecting EO4SD from {eo4sd.crs} to {wards.crs}")
    eo4sd = eo4sd.to_crs(wards.crs)

eo4sd = eo4sd[eo4sd.geometry.notnull() & eo4sd.geometry.is_valid].copy()
wards = wards[wards.geometry.notnull()].copy()
wards["ward_area_m2"] = wards.geometry.area

# =============================================================================
# 2. SPATIAL OVERLAY — EO4SD INFORMAL AREA PER WARD
# =============================================================================

print("\nRunning spatial overlay (this can take a minute)...")

overlay = gpd.overlay(
    eo4sd[["geometry"]].assign(eo4sd_id=range(len(eo4sd))),
    wards[["GID_4", "geometry"]],
    how="intersection"
)
overlay["overlap_area_m2"] = overlay.geometry.area

eo4sd_area_per_ward = overlay.groupby("GID_4")["overlap_area_m2"].sum().rename("eo4sd_informal_area_m2")

wards = wards.merge(eo4sd_area_per_ward, on="GID_4", how="left")
wards["eo4sd_informal_area_m2"] = wards["eo4sd_informal_area_m2"].fillna(0.0)
wards["eo4sd_informal_fraction"] = wards["eo4sd_informal_area_m2"] / wards["ward_area_m2"]

print(f"Wards with any EO4SD-mapped informal area : "
      f"{(wards['eo4sd_informal_area_m2'] > 0).sum()} / {len(wards)}")

# =============================================================================
# 3. MERGE WITH TRAINING LABELS AND MODEL PREDICTIONS
# =============================================================================

# ward_predictions_extended.gpkg already carries label/label_src from training;
# use those directly instead of merging (avoids _x/_y column collisions).
df = wards.copy()
if "label" not in df.columns or "label_src" not in df.columns:
    df = df.merge(labels[["GID_4", "label", "label_src"]], on="GID_4", how="left")

# prob_informal column name check
prob_col = None
for c in ["prob_informal", "informal_probability", "proba_informal", "prediction_proba"]:
    if c in df.columns:
        prob_col = c
        break

print(f"\nModel probability column found: {prob_col}")
print("Available columns with 'prob' or 'pred':",
      [c for c in df.columns if "prob" in c.lower() or "pred" in c.lower()])

comparison_cols = ["GID_4", "NAME_4", "NAME_3", "ward_area_m2",
                    "eo4sd_informal_area_m2", "eo4sd_informal_fraction",
                    "label", "label_src"]
if prob_col:
    comparison_cols.append(prob_col)
comparison_cols = [c for c in comparison_cols if c in df.columns]

out_df = df[comparison_cols].copy()
out_df.to_csv(OUT_CSV, index=False)
print(f"Comparison table saved -> {OUT_CSV}")

# =============================================================================
# 4. VALIDATION AGAINST TRAINING LABELS (rule/anchor)
# =============================================================================

report_lines = []
report_lines.append("="*70)
report_lines.append("EXTERNAL VALIDATION — EO4SD-Urban Dhaka Informal Settlements 2017")
report_lines.append("="*70)
report_lines.append(f"\nEO4SD polygons: {len(eo4sd)}")
report_lines.append(f"Wards with any EO4SD informal area: "
                     f"{(wards['eo4sd_informal_area_m2'] > 0).sum()} / {len(wards)}")

labeled = df[df["label"].isin([0, 1])].copy()
print(f"\nLabeled wards available for comparison: {len(labeled)}")

if len(labeled) > 0:
    informal_frac_by_label = labeled.groupby("label")["eo4sd_informal_fraction"].agg(
        ["mean", "median", "std", "count"])
    informal_frac_by_label.index = informal_frac_by_label.index.map({0: "informal(0)", 1: "formal(1)"})
    print("\nEO4SD informal fraction by training label:")
    print(informal_frac_by_label)
    report_lines.append("\nEO4SD informal fraction by training label (label 0=informal, 1=formal):")
    report_lines.append(informal_frac_by_label.to_string())

    # point-biserial correlation: label (0/1) vs eo4sd fraction
    # expect NEGATIVE correlation (label=1/formal should have LOW eo4sd fraction)
    r, p = stats.pointbiserialr(labeled["label"], labeled["eo4sd_informal_fraction"])
    print(f"\nPoint-biserial correlation (label vs EO4SD informal fraction): r={r:.4f}, p={p:.4g}")
    report_lines.append(f"\nPoint-biserial correlation (label vs EO4SD informal fraction): "
                         f"r={r:.4f}, p={p:.4g}")
    report_lines.append("(Expect r < 0: formal=1 wards should have LOWER EO4SD informal fraction)")

    # Simple agreement check: does a ward with EO4SD informal_fraction > threshold
    # match the training label?
    for thresh in [0.01, 0.05, 0.10]:
        pred_informal = (labeled["eo4sd_informal_fraction"] > thresh).astype(int)
        actual_informal = (labeled["label"] == 0).astype(int)
        agree = (pred_informal == actual_informal).mean()
        # also precision/recall for informal class
        tp = ((pred_informal == 1) & (actual_informal == 1)).sum()
        fp = ((pred_informal == 1) & (actual_informal == 0)).sum()
        fn = ((pred_informal == 0) & (actual_informal == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        line = (f"Threshold {thresh:.2f}: agreement={agree:.3f}  "
                f"informal-precision={precision:.3f}  informal-recall={recall:.3f}")
        print(line)
        report_lines.append(line)
else:
    report_lines.append("\nNo labeled wards found for comparison — check merge keys.")

# =============================================================================
# 5. VALIDATION AGAINST MODEL PREDICTIONS (if probability column found)
# =============================================================================

if prob_col:
    valid = df[df[prob_col].notnull() & df["eo4sd_informal_fraction"].notnull()]
    r2, p2 = stats.pearsonr(valid[prob_col], valid["eo4sd_informal_fraction"])
    rs, ps = stats.spearmanr(valid[prob_col], valid["eo4sd_informal_fraction"])
    print(f"\nModel probability vs EO4SD informal fraction:")
    print(f"  Pearson  r={r2:.4f}, p={p2:.4g}")
    print(f"  Spearman r={rs:.4f}, p={ps:.4g}")
    report_lines.append(f"\nModel predicted probability vs EO4SD informal fraction (n={len(valid)}):")
    report_lines.append(f"  Pearson  r={r2:.4f}, p={p2:.4g}")
    report_lines.append(f"  Spearman r={rs:.4f}, p={ps:.4g}")
else:
    report_lines.append("\nNo model probability column found in ward_predictions_extended.gpkg "
                         "— skipped model-vs-EO4SD correlation. Add this check manually once "
                         "the probability column name is confirmed.")

with open(OUT_REPORT, "w") as f:
    f.write("\n".join(report_lines))
print(f"\nReport written -> {OUT_REPORT}")

# =============================================================================
# 6. FIGURE
# =============================================================================

fig, axes = plt.subplots(1, 2 if prob_col else 1, figsize=(13 if prob_col else 7, 5.5))
if not prob_col:
    axes = [axes]

ax = axes[0]
if len(labeled) > 0:
    data_informal = labeled.loc[labeled["label"] == 0, "eo4sd_informal_fraction"]
    data_formal   = labeled.loc[labeled["label"] == 1, "eo4sd_informal_fraction"]
    ax.boxplot([data_informal, data_formal], labels=["Training label:\ninformal", "Training label:\nformal"])
    ax.set_ylabel("EO4SD informal-patch area fraction")
    ax.set_title("External check: training labels vs\nEO4SD 2017 informal-settlement footprint")

if prob_col:
    ax2 = axes[1]
    ax2.scatter(valid[prob_col], valid["eo4sd_informal_fraction"], alpha=0.5, s=18)
    ax2.set_xlabel("Model predicted P(informal)")
    ax2.set_ylabel("EO4SD informal-patch area fraction")
    ax2.set_title(f"Model probability vs EO4SD\n(Spearman r={rs:.2f})")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
print(f"Figure written -> {OUT_FIG}")

print("\n" + "="*70)
print("DONE")
print("="*70)
