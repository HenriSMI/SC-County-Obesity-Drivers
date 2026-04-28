import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ── Load & filter to Santa Clara County ──────────────────────────
fp = "/Users/henri/SC_Project/PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260308.csv"
df = pd.read_csv(fp)
df = df[df["CountyName"] == "Santa Clara"].copy()

# ── Variables ────────────────────────────────────────────────────
x_col = "DEPRESSION_CrudePrev"
y_col = "CSMOKING_CrudePrev"
z_col = "OBESITY_CrudePrev"

x = df[x_col].values
y = df[y_col].values
z = df[z_col].values

# ── Build interpolation grid ─────────────────────────────────────
xi = np.linspace(x.min(), x.max(), 300)
yi = np.linspace(y.min(), y.max(), 300)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

# ── Plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

# Filled contour matching existing green-to-red palette
cmap = plt.cm.YlOrRd
cf = ax.contourf(Xi, Yi, Zi, levels=20, cmap=cmap, alpha=0.9)
cs = ax.contour(Xi, Yi, Zi, levels=10, colors="k", linewidths=0.4, alpha=0.5)
ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f%%")

# Scatter actual tracts
sc = ax.scatter(x, y, c=z, cmap=cmap, edgecolors="white", linewidths=0.5,
                s=40, alpha=0.8, vmin=z.min(), vmax=z.max())

# Annotate highest-obesity tracts
top = df.nlargest(5, z_col)
for _, row in top.iterrows():
    ax.annotate(f"{row[z_col]:.1f}%",
                xy=(row[x_col], row[y_col]),
                fontsize=8, fontweight="bold", color="darkred",
                textcoords="offset points", xytext=(6, 6))

cbar = fig.colorbar(cf, ax=ax, label="Obesity Prevalence (%)")

ax.set_xlabel("Depression Prevalence — DEPRESSION_CrudePrev (%)", fontsize=11)
ax.set_ylabel("Current Smoking — CSMOKING_CrudePrev (%)", fontsize=11)
ax.set_title("Obesity Response Surface\nDepression × Smoking — SCC Census Tracts",
             fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("/Users/henri/SC_Project/response_surface_depression_smoking.png", dpi=200)
plt.show()
print("Saved → response_surface_depression_smoking.png")