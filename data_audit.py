"""
Step 1: Data Audit + Correlation Heatmap
CDC PLACES 2025 — Santa Clara County Census Tracts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Load + Filter to Santa Clara County 
df = pd.read_csv("PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260308.csv")
df = df[df["CountyName"] == "Santa Clara"]
print(f"SCC tracts: {len(df)}")

# 
# PART 1 — AUDIT
# 
print("=" * 60)
print(f"Shape: {df.shape[0]} tracts × {df.shape[1]} columns")
print("\n── All columns ──")
for i, c in enumerate(df.columns):
    print(f"  {i:3d}  {c}")

print("\n── Missing values (columns with any NaN) ──")
missing = df.isnull().sum()
print(missing[missing > 0].to_string())

print("\n── Sample rows ──")
print(df.head(3).to_string())

# PART 2 — HEATMAP
# 

TARGET_MEASURES = [
    "DIABETES_CrudePrev",
    "OBESITY_CrudePrev",
    "STROKE_CrudePrev",
    "ARTHRITIS_CrudePrev",
    "BPHIGH_CrudePrev",
    "DEPRESSION_CrudePrev",
    "CSMOKING_CrudePrev",
    "SLEEP_CrudePrev",
    "CHECKUP_CrudePrev",
    "CHOLSCREEN_CrudePrev",
    "CASTHMA_CrudePrev",
    "CHD_CrudePrev",
    "HIGHCHOL_CrudePrev",
    "LPA_CrudePrev",
    "FOODSTAMP_CrudePrev",
]

available = [c for c in TARGET_MEASURES if c in df.columns]
missing_cols = [c for c in TARGET_MEASURES if c not in df.columns]
if missing_cols:
    print(f"\n⚠ Columns not found (check names): {missing_cols}")
print(f"\n✓ Building heatmap with {len(available)} measures")

heat_df = df[available].apply(pd.to_numeric, errors="coerce")
heat_df = heat_df.dropna(thresh=len(available) * 0.7)

corr = heat_df.corr()
labels = [c.replace("_CrudePrev", "") for c in available]
corr.index = labels
corr.columns = labels

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 8},
)

ax.set_title(
    "Correlation Heatmap — CDC PLACES 2025\nSanta Clara County Census Tracts",
    fontsize=14, fontweight="bold", pad=15
)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/heatmap_scc_places.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved to outputs/heatmap_scc_places.png")

# ── Top correlates with OBESITY ───────────────────────────────────────────────
print("\n── Top correlates with OBESITY ──")
if "OBESITY" in corr.columns:
    obs_corr = corr["OBESITY"].drop("OBESITY").sort_values(ascending=False)
    print(obs_corr.to_string())