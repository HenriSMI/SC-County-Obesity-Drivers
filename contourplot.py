"""
Step 3: Contour Plot — Obesity as color gradient
X axis: LPA (Physical Inactivity %)
Y axis: FOODSTAMP (Food Stamp Usage %)
Color: OBESITY_CrudePrev
CDC PLACES 2025 — Santa Clara County Census Tracts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

os.makedirs("outputs", exist_ok=True)

# ── Load + Filter ─────────────────────────────────────────────────────────────
df = pd.read_csv("PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260308.csv")
df = df[df["CountyName"] == "Santa Clara"].copy()

cols = ["OBESITY_CrudePrev", "LPA_CrudePrev", "FOODSTAMP_CrudePrev", "TractFIPS"]
for c in cols[:-1]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=cols[:-1])

x = df["LPA_CrudePrev"].values
y = df["FOODSTAMP_CrudePrev"].values
z = df["OBESITY_CrudePrev"].values

# ── Interpolate onto a grid for the contour ───────────────────────────────────
xi = np.linspace(x.min(), x.max(), 300)
yi = np.linspace(y.min(), y.max(), 300)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))

# Filled contour (color gradient = obesity level)
contourf = ax.contourf(Xi, Yi, Zi, levels=20, cmap="RdYlGn_r", alpha=0.85)

# Contour lines
contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors="black", linewidths=0.5, alpha=0.4)
ax.clabel(contour_lines, inline=True, fontsize=7, fmt="%.1f%%")

# Scatter — actual census tracts
scatter = ax.scatter(
    x, y, c=z,
    cmap="RdYlGn_r",
    edgecolors="white",
    linewidths=0.5,
    s=50,
    zorder=5,
    vmin=z.min(), vmax=z.max()
)

# Colorbar
cbar = plt.colorbar(contourf, ax=ax)
cbar.set_label("Obesity Prevalence (%)", fontsize=11)

# Labels
ax.set_xlabel("Physical Inactivity — LPA_CrudePrev (%)", fontsize=12)
ax.set_ylabel("Food Stamp Usage — FOODSTAMP_CrudePrev (%)", fontsize=12)
ax.set_title(
    "Obesity Response Surface\nPhysical Inactivity × Food Insecurity — SCC Census Tracts",
    fontsize=14, fontweight="bold"
)

# Annotate top 5 highest obesity tracts
top5 = df.nlargest(5, "OBESITY_CrudePrev")
for _, row in top5.iterrows():
    ax.annotate(
        f"  {row['OBESITY_CrudePrev']:.1f}%",
        xy=(row["LPA_CrudePrev"], row["FOODSTAMP_CrudePrev"]),
        fontsize=7.5,
        color="darkred",
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("outputs/contour_obesity_lpa_foodstamp.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to outputs/contour_obesity_lpa_foodstamp.png")

# ── Print summary stats for top/bottom tracts ─────────────────────────────────
print("\n── Top 10 Highest Obesity Tracts ──")
top10 = df.nlargest(10, "OBESITY_CrudePrev")[["TractFIPS", "OBESITY_CrudePrev", "LPA_CrudePrev", "FOODSTAMP_CrudePrev"]]
print(top10.to_string(index=False))

print("\n── Top 10 Lowest Obesity Tracts ──")
bot10 = df.nsmallest(10, "OBESITY_CrudePrev")[["TractFIPS", "OBESITY_CrudePrev", "LPA_CrudePrev", "FOODSTAMP_CrudePrev"]]
print(bot10.to_string(index=False))