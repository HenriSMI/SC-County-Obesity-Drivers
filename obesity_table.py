"""
Step 2: Obesity Table + Predictor Breakdown by Obesity Level
CDC PLACES 2025 — Santa Clara County Census Tracts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

# ── Load + Filter ─────────────────────────────────────────────────────────────
df = pd.read_csv("PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260308.csv")
df = df[df["CountyName"] == "Santa Clara"].copy()

# Convert all CrudePrev columns to numeric
prev_cols = [c for c in df.columns if "CrudePrev" in c]
for c in prev_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ── PART 1: Understand how PLACES quantifies obesity ─────────────────────────
print("=" * 60)
print("PART 1 — How PLACES Quantifies Obesity")
print("=" * 60)
print("\nOBESITY_CrudePrev = % of adults (18+) with BMI ≥ 30")
print("Source: BRFSS survey + small area estimation model\n")

obs = df["OBESITY_CrudePrev"]
print(f"  Tracts in SCC:     {len(obs)}")
print(f"  Min:               {obs.min():.1f}%")
print(f"  Max:               {obs.max():.1f}%")
print(f"  Mean:              {obs.mean():.1f}%")
print(f"  Median:            {obs.median():.1f}%")
print(f"  Std Dev:           {obs.std():.1f}%")
print(f"  Range:             {obs.max() - obs.min():.1f} percentage points")

# ── PART 2: Bin tracts into obesity levels ────────────────────────────────────
print("\n" + "=" * 60)
print("PART 2 — Obesity Level Bins")
print("=" * 60)

# Use tertiles (equal thirds) for natural breakpoints
df["obesity_level"] = pd.qcut(
    df["OBESITY_CrudePrev"],
    q=3,
    labels=["Low", "Medium", "High"]
)

bin_summary = df.groupby("obesity_level")["OBESITY_CrudePrev"].agg(
    Tracts="count", Min="min", Max="max", Mean="mean"
).round(1)
print(f"\n{bin_summary.to_string()}")

# ── PART 3: Predictor breakdown by obesity level ──────────────────────────────
print("\n" + "=" * 60)
print("PART 3 — Predictor Means by Obesity Level")
print("=" * 60)

PREDICTORS = [
    "DIABETES_CrudePrev",
    "STROKE_CrudePrev",
    "BPHIGH_CrudePrev",
    "DEPRESSION_CrudePrev",
    "CSMOKING_CrudePrev",
    "SLEEP_CrudePrev",
    "LPA_CrudePrev",
    "FOODSTAMP_CrudePrev",
    "CASTHMA_CrudePrev",
    "CHECKUP_CrudePrev",
    "CHOLSCREEN_CrudePrev",
    "MHLTH_CrudePrev",
    "PHLTH_CrudePrev",
]

table = df.groupby("obesity_level")[PREDICTORS].mean().round(2).T
table.columns = ["Low Obesity", "Medium Obesity", "High Obesity"]
table["Gap (High-Low)"] = (table["High Obesity"] - table["Low Obesity"]).round(2)
table = table.sort_values("Gap (High-Low)", ascending=False)
table.index = [i.replace("_CrudePrev", "") for i in table.index]

print(f"\n{table.to_string()}")
print("\n→ 'Gap' = difference in predictor % between high and low obesity tracts")
print("→ Larger gap = stronger driver of obesity at the neighborhood level")

# ── PART 4: Visualize the table as a grouped bar chart ───────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(table))
w = 0.25
colors = ["#2ecc71", "#f39c12", "#e74c3c"]

for i, (col, color) in enumerate(zip(["Low Obesity", "Medium Obesity", "High Obesity"], colors)):
    ax.bar(x + i * w, table[col], width=w, label=col, color=color, alpha=0.85)

ax.set_xticks(x + w)
ax.set_xticklabels(table.index, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Crude Prevalence (%)", fontsize=11)
ax.set_title(
    "Predictor Prevalence by Obesity Level\nSanta Clara County Census Tracts (CDC PLACES 2025)",
    fontsize=13, fontweight="bold"
)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/obesity_predictor_table.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved to outputs/obesity_predictor_table.png")

# ── PART 5: Correlation with obesity ranked ───────────────────────────────────
print("\n" + "=" * 60)
print("PART 5 — Correlation Ranking with OBESITY_CrudePrev")
print("=" * 60)
corrs = df[PREDICTORS + ["OBESITY_CrudePrev"]].corr()["OBESITY_CrudePrev"].drop("OBESITY_CrudePrev")
corrs = corrs.abs().sort_values(ascending=False)
corrs.index = [i.replace("_CrudePrev", "") for i in corrs.index]
print(f"\n{corrs.round(3).to_string()}")
print("\n→ These are your contour plot axis candidates (top 2-3)")