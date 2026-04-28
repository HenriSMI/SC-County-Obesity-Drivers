"""
Step 4: Choropleth Map — Obesity by Census Tract
Santa Clara County, CDC PLACES 2025
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pygris
import os

os.makedirs("outputs", exist_ok=True)

# ── Load PLACES data ──────────────────────────────────────────────────────────
df = pd.read_csv("PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260308.csv")
df = df[df["CountyName"] == "Santa Clara"].copy()

cols = ["OBESITY_CrudePrev", "LPA_CrudePrev", "FOODSTAMP_CrudePrev",
        "DIABETES_CrudePrev", "DEPRESSION_CrudePrev", "TractFIPS"]
for c in cols[:-1]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# TractFIPS needs to be a zero-padded string for joining
df["TractFIPS"] = df["TractFIPS"].astype(str).str.zfill(11)

# ── Download SCC census tract shapefile via pygris ────────────────────────────
print("Downloading SCC census tract shapefile...")
tracts = pygris.tracts(state="CA", county="085", year=2020)  # 085 = Santa Clara
tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
print(f"Shapefile loaded: {len(tracts)} tracts")

# ── Merge ─────────────────────────────────────────────────────────────────────
merged = tracts.merge(df, left_on="GEOID", right_on="TractFIPS", how="left")
print(f"Merged: {merged['OBESITY_CrudePrev'].notna().sum()} tracts with data")

# ── Plot 4 maps in a grid ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(
    "Health Disparities by Census Tract — Santa Clara County\nCDC PLACES 2025",
    fontsize=16, fontweight="bold", y=0.98
)

maps = [
    ("OBESITY_CrudePrev",    "Obesity Prevalence (%)",         "RdYlGn_r", axes[0, 0]),
    ("LPA_CrudePrev",        "Physical Inactivity (%)",        "RdYlGn_r", axes[0, 1]),
    ("FOODSTAMP_CrudePrev",  "Food Stamp Usage (%)",           "RdYlGn_r", axes[1, 0]),
    ("DIABETES_CrudePrev",   "Diabetes Prevalence (%)",        "RdYlGn_r", axes[1, 1]),
]

for col, title, cmap, ax in maps:
    merged.plot(
        column=col,
        cmap=cmap,
        linewidth=0.3,
        edgecolor="white",
        legend=True,
        legend_kwds={},
        missing_kwds={"color": "lightgrey"},
        ax=ax,
        scheme="quantiles",   # equal # of tracts per color bucket
        k=6
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/map_scc_health_disparities.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to outputs/map_scc_health_disparities.png")

# ── Print top 10 highest obesity tracts with location info ───────────────────
print("\n── Top 10 Highest Obesity Tracts ──")
top10 = merged.nlargest(10, "OBESITY_CrudePrev")[
    ["GEOID", "NAMELSAD", "OBESITY_CrudePrev", "LPA_CrudePrev", "FOODSTAMP_CrudePrev", "DIABETES_CrudePrev"]
].reset_index(drop=True)
print(top10.to_string(index=False))