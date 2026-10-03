# =============================================================================
# NOTEBOOK 13 — AGREEMENT PLOT (BLAND-ALTMAN STYLE) vs EO4SD
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
#
# PURPOSE
# -------
# NB08/10 report a Spearman correlation between the decoupled model's
# P(informal) and the EO4SD 2017 informal-area fraction. Correlation
# alone doesn't show WHERE the two measures disagree -- e.g. systematic
# over/under-prediction at high informality, or a handful of outlier
# wards driving the correlation. A Bland-Altman-style agreement plot
# (mean of the two measures on x, difference on y) makes any such
# pattern visible directly, and is a standard complement to a
# correlation coefficient when comparing two measures of a related but
# not identical quantity (here: a predicted probability vs. a mapped
# area fraction -- different units, same underlying construct).
#
# OUTPUT
# ------
#   outputs/figures/13_bland_altman.png
#   outputs/model/bland_altman_outliers.txt
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_IN = "data/processed/ward_decoupled_eo4sd_comparison.csv"
OUT_FIG = "outputs/figures/13_bland_altman.png"
OUT_TXT = "outputs/model/bland_altman_outliers.txt"

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/model", exist_ok=True)

print("="*70)
print("AGREEMENT PLOT (BLAND-ALTMAN STYLE) — DECOUPLED MODEL vs EO4SD")
print("="*70)

df = pd.read_csv(CSV_IN)
df = df.dropna(subset=["prob_informal_decoupled_all170", "eo4sd_informal_fraction"]).copy()

a = df["prob_informal_decoupled_all170"].values
b = df["eo4sd_informal_fraction"].values

mean_ab = (a + b) / 2
diff_ab = a - b  # model - EO4SD

bias = diff_ab.mean()
sd = diff_ab.std()
loa_upper = bias + 1.96 * sd
loa_lower = bias - 1.96 * sd

print(f"\nn = {len(df)}")
print(f"Mean difference (bias, model - EO4SD) : {bias:.4f}")
print(f"SD of difference                      : {sd:.4f}")
print(f"95% limits of agreement               : [{loa_lower:.4f}, {loa_upper:.4f}]")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(mean_ab, diff_ab, alpha=0.5, s=20, color="#2563eb")
ax.axhline(bias, color="#dc2626", linestyle="-", linewidth=1.5, label=f"Mean bias = {bias:.3f}")
ax.axhline(loa_upper, color="#dc2626", linestyle="--", linewidth=1, label=f"95% LoA = [{loa_lower:.3f}, {loa_upper:.3f}]")
ax.axhline(loa_lower, color="#dc2626", linestyle="--", linewidth=1)
ax.axhline(0, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Mean of model P(informal) and EO4SD informal-area fraction")
ax.set_ylabel("Difference (model P(informal) − EO4SD fraction)")
ax.set_title("Agreement plot: decoupled model vs. independent EO4SD 2017 inventory\n(n=203 wards)")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
print(f"\nFigure written -> {OUT_FIG}")

# flag wards outside limits of agreement as the clearest disagreement cases
outliers = df[(diff_ab > loa_upper) | (diff_ab < loa_lower)].copy()
outliers["diff"] = diff_ab[(diff_ab > loa_upper) | (diff_ab < loa_lower)]
outliers = outliers.sort_values("diff", key=abs, ascending=False)

with open(OUT_TXT, "w") as f:
    f.write("Wards outside the 95% limits of agreement (model vs EO4SD)\n")
    f.write("="*70 + "\n")
    f.write(f"n outliers: {len(outliers)} / {len(df)}\n\n")
    for _, row in outliers.iterrows():
        f.write(f"  {row['NAME_4']:20s} ({row['NAME_3']:15s})  "
                f"model={row['prob_informal_decoupled_all170']:.3f}  "
                f"EO4SD={row['eo4sd_informal_fraction']:.3f}  diff={row['diff']:+.3f}  "
                f"label_src={row['label_src']}\n")

print(f"Outlier list written -> {OUT_TXT} ({len(outliers)} wards outside 95% LoA)")

print("\n" + "="*70)
print("DONE")
print("="*70)
