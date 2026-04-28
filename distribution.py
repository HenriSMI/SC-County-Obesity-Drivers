"""
Distribution Analysis: Obesity and Top Predictors in Santa Clara County
Tests for normality, skew, and bimodality at the census tract level.

Run locally:
    python3 distribution_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Paths
PROJECT_DIR = Path("/Users/henri/SC_Project")
DATA_FILE = PROJECT_DIR / "PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260326.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SCC_FIPS = "06085"

# Load and filter to Santa Clara County
print("Loading CDC PLACES data...")
df = pd.read_csv(DATA_FILE, dtype={"CountyFIPS": str, "TractFIPS": str})
df = df[df["CountyFIPS"] == SCC_FIPS].copy()
print(f"Santa Clara County tracts: {len(df)}")

# Variables to analyze
variables = {
    "OBESITY_CrudePrev": "Obesity Prevalence (%)",
    "DEPRESSION_CrudePrev": "Depression Prevalence (%)",
    "CSMOKING_CrudePrev": "Current Smoking (%)",
    "HOUSINSECU_CrudePrev": "Housing Insecurity (%)",
    "LPA_CrudePrev": "Physical Inactivity (%)",
    "FOODSTAMP_CrudePrev": "Food Stamp Usage (%)",
}

# Hartigan's dip test for unimodality (manual implementation, no extra deps)
# If diptest package isn't installed, we fall back to a bimodality coefficient (BC)
# BC > 5/9 (~0.555) suggests bimodality
def bimodality_coefficient(x):
    """Sarle's bimodality coefficient. BC > 0.555 suggests bimodal distribution."""
    n = len(x)
    g = stats.skew(x)
    k = stats.kurtosis(x)  # excess kurtosis
    bc = (g**2 + 1) / (k + (3 * (n - 1)**2) / ((n - 2) * (n - 3)))
    return bc

# Build the figure
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

stats_table = []

for i, (col, label) in enumerate(variables.items()):
    ax = axes[i]
    data = df[col].dropna().values

    # Histogram + KDE
    ax.hist(data, bins=30, density=True, alpha=0.55, color="#e07b3f", edgecolor="white")
    kde = stats.gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_grid, kde(x_grid), color="#2c3e50", linewidth=2, label="KDE")

    # Normal overlay for comparison
    mu, sigma = data.mean(), data.std()
    ax.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), "--",
            color="#3498db", linewidth=1.8, label="Normal fit")

    # Mean and median lines
    ax.axvline(mu, color="#c0392b", linestyle=":", linewidth=1.5, label=f"Mean={mu:.1f}")
    ax.axvline(np.median(data), color="#16a085", linestyle=":", linewidth=1.5,
               label=f"Median={np.median(data):.1f}")

    # Stats
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    sw_stat, sw_p = stats.shapiro(data)
    bc = bimodality_coefficient(data)

    shape_label = "Normal" if sw_p > 0.05 else ("Bimodal-leaning" if bc > 0.555 else "Skewed/non-normal")

    ax.set_title(f"{label}\nSkew={skew:.2f} | Kurt={kurt:.2f} | BC={bc:.3f} | {shape_label}",
                 fontsize=10)
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    stats_table.append({
        "Variable": col,
        "N": len(data),
        "Mean": round(mu, 2),
        "Median": round(np.median(data), 2),
        "SD": round(sigma, 2),
        "Skew": round(skew, 3),
        "Excess_Kurtosis": round(kurt, 3),
        "Shapiro_W": round(sw_stat, 4),
        "Shapiro_p": f"{sw_p:.2e}",
        "Bimodality_Coef": round(bc, 3),
        "Shape_Verdict": shape_label,
    })

plt.suptitle("Distribution Analysis: Obesity and Top Predictors\n"
             "Santa Clara County Census Tracts (CDC PLACES 2025)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()

out_png = OUTPUT_DIR / "distribution_analysis.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved: {out_png}")

# Save stats table
stats_df = pd.DataFrame(stats_table)
out_csv = OUTPUT_DIR / "distribution_stats.csv"
stats_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

print("\n" + "="*80)
print("DISTRIBUTION SUMMARY")
print("="*80)
print(stats_df.to_string(index=False))

print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("""
Shapiro-Wilk p < 0.05  -> reject normality
|Skew| > 1             -> strongly skewed
Bimodality Coef > 0.555 -> possible bimodal/multimodal structure
Bimodality Coef < 0.555 -> likely unimodal (one population)

For your project: if obesity is bimodal, that visually justifies the "two SCCs"
narrative (wealthy west vs working-class east). If unimodal but right-skewed,
the disparity is a continuous gradient with a long high-risk tail rather than
two distinct populations.
""")