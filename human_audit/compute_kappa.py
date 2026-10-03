"""
Run this AFTER filling in human_audit_BLIND_review_sheet.csv.

Usage:
    python3 compute_kappa.py

Reads:
    human_audit_BLIND_review_sheet.csv   (your filled-in judgments)
    human_audit_ANSWER_KEY_do_not_open_yet.csv   (the rule/anchor labels)

Prints:
    Confusion matrix, % agreement, Cohen's kappa, interpretation.
Writes:
    human_audit_kappa_report.txt
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REVIEW = "human_audit_BLIND_review_sheet.csv"
KEY = "human_audit_ANSWER_KEY_do_not_open_yet.csv"
OUT = "human_audit_kappa_report.txt"

review = pd.read_csv(REVIEW)
key = pd.read_csv(KEY)

judgment_col = "your_independent_judgment (type: informal / formal)"
review[judgment_col] = review[judgment_col].str.strip().str.lower()

missing = review[judgment_col].isna() | (review[judgment_col] == "")
if missing.any():
    print(f"WARNING: {missing.sum()} wards have no judgment filled in yet. Fill those in first.")
    print(review.loc[missing, "sample_id"].tolist())

merged = review.merge(key[["sample_id", "true_label"]], on="sample_id", how="left")
merged = merged.dropna(subset=[judgment_col, "true_label"])
merged = merged[merged[judgment_col].isin(["informal", "formal"])]

y_human = merged[judgment_col].values
y_rule = merged["true_label"].values

n = len(merged)
agreement = (y_human == y_rule).mean()
kappa = cohen_kappa_score(y_human, y_rule)
cm = confusion_matrix(y_human, y_rule, labels=["informal", "formal"])

if kappa < 0:
    interp = "poor (worse than chance)"
elif kappa < 0.21:
    interp = "slight"
elif kappa < 0.41:
    interp = "fair"
elif kappa < 0.61:
    interp = "moderate"
elif kappa < 0.81:
    interp = "substantial"
else:
    interp = "almost perfect"

report = f"""HUMAN LABEL AUDIT -- COHEN'S KAPPA REPORT
{"="*50}
n wards rated        : {n}
Raw agreement        : {agreement:.3f} ({agreement*100:.1f}%)
Cohen's kappa         : {kappa:.3f}
Interpretation        : {interp} agreement (Landis & Koch 1977 scale)

Confusion matrix (rows=human judgment, cols=rule/anchor label)
                  rule=informal   rule=formal
human=informal    {cm[0,0]:>13d}   {cm[0,1]:>11d}
human=formal      {cm[1,0]:>13d}   {cm[1,1]:>11d}
"""
print(report)
with open(OUT, "w") as f:
    f.write(report)
print(f"Report written -> {OUT}")
