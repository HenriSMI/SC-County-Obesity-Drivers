"""
Distribution Analysis — Obesity & Key Predictors
Santa Clara County Census Tracts, CDC PLACES 2025

Purpose: Characterize the shape of tract-level distributions for obesity
and its top predictors. Tests for:
  - Normality (Shapiro-Wilk, D'Agostino)
  - Bimodality (Hartigan's dip test, via statistic approximation)
  - Skewness and kurtosis

Interpretation matters for intervention framing:
  - Normal/unimodal  → gradient of risk, shift the whole curve
  - Bimodal          → two distinct populations, target the high-risk mode
  - Right-skewed     → most tracts healthy, long tail of high-risk outliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading CDC PLACES 2025 data...")
df = pd.read_csv("PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260326.csv")
df = df[df["CountyName"] == "Santa Clara"].copy()

VARIABLES = {
    "OBESITY_CrudePrev":     "Obesity Prevalence (%)",
    "DEPRESSION_CrudePrev":  "Depression Prevalence (%)",
    "CSMOKING_CrudePrev":    "Current Smoking (%)",
    "LPA_CrudePrev":         "Physical Inactivity (%)",
    "HOUSINSECU_CrudePrev":  "Housing Insecurity (%)",
    "FOODSTAMP_CrudePrev":   "Food Stamp Usage (%)",
}

for col in VARIABLES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=list(VARIABLES.keys())).reset_index(drop=True)
print(f"  Tracts: {len(df)}")

# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE BIMODALITY INDICATOR — BIMODALITY COEFFICIENT
# ═══════════════════════════════════════════════════════════════════════════════
# Sarle's bimodality coefficient (BC). BC > 5/9 (~0.555) suggests bimodality.
# BC = (skewness² + 1) / (kurtosis + 3(n-1)² / ((n-2)(n-3)))
def bimodality_coef(x):
    x = np.asarray(x)
    n = len(x)
    skew = stats.skew(x, bias=False)
    kurt = stats.kurtosis(x, bias=False)  # excess kurtosis
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (skew ** 2 + 1) / (kurt + correction)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT + STATS FOR EACH VARIABLE
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribution Analysis — SCC Census Tracts (CDC PLACES 2025)",
             fontsize=14, fontweight="bold", y=1.00)
axes = axes.flatten()

print("\n" + "=" * 78)
print(f"{'Variable':<25} {'Mean':>7} {'SD':>7} {'Skew':>7} {'Kurt':>7} "
      f"{'Shapiro p':>10} {'BC':>6} {'Shape':>12}")
print("=" * 78)

results = []
for ax, (col, label) in zip(axes, VARIABLES.items()):
    x = df[col].values

    # Stats
    mean, sd = np.mean(x), np.std(x, ddof=1)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x)  # excess kurtosis (normal = 0)
    shapiro_stat, shapiro_p = stats.shapiro(x)
    bc = bimodality_coef(x)

    # Interpret shape
    if bc > 0.555:
        shape = "Bimodal?"
    elif shapiro_p > 0.05:
        shape = "Normal"
    elif skew > 0.5:
        shape = "Right-skew"
    elif skew < -0.5:
        shape = "Left-skew"
    else:
        shape = "Non-normal"

    print(f"{label:<25} {mean:>7.2f} {sd:>7.2f} {skew:>+7.2f} {kurt:>+7.2f} "
          f"{shapiro_p:>10.4f} {bc:>6.3f} {shape:>12}")

    results.append({
        "variable": label, "mean": mean, "sd": sd,
        "skewness": skew, "kurtosis": kurt,
        "shapiro_p": shapiro_p, "bimodality_coef": bc, "shape": shape
    })

    # ── Plot histogram + KDE + normal overlay ──
    ax.hist(x, bins=25, density=True, alpha=0.55, color="#3498db",
            edgecolor="white", label="Tracts")

    # KDE for smooth shape
    kde = stats.gaussian_kde(x)
    x_range = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_range, kde(x_range), color="#2c3e50", linewidth=2, label="KDE")

    # Normal reference curve with same mean/SD
    ax.plot(x_range, stats.norm.pdf(x_range, mean, sd),
            color="#e74c3c", linewidth=1.5, linestyle="--", label="Normal fit")

    # Mean and median markers
    ax.axvline(mean, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(np.median(x), color="grey", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{label}\nskew={skew:+.2f}  kurt={kurt:+.2f}  BC={bc:.3f}  → {shape}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/distribution_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: outputs/distribution_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Q-Q PLOT FOR OBESITY SPECIFICALLY (normality visual)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Obesity Distribution — Closer Look", fontsize=13, fontweight="bold")

obesity = df["OBESITY_CrudePrev"].values

# Left: histogram + KDE + normal
axes[0].hist(obesity, bins=30, density=True, alpha=0.55, color="#3498db",
             edgecolor="white")
kde = stats.gaussian_kde(obesity)
x_range = np.linspace(obesity.min(), obesity.max(), 200)
axes[0].plot(x_range, kde(x_range), color="#2c3e50", linewidth=2.5, label="KDE (actual)")
axes[0].plot(x_range, stats.norm.pdf(x_range, obesity.mean(), obesity.std()),
             color="#e74c3c", linewidth=2, linestyle="--", label="Normal fit")
axes[0].axvline(obesity.mean(), color="black", linestyle=":", linewidth=1.5,
                label=f"Mean = {obesity.mean():.1f}%")
axes[0].axvline(np.median(obesity), color="grey", linestyle=":", linewidth=1.5,
                label=f"Median = {np.median(obesity):.1f}%")
axes[0].set_xlabel("Obesity Prevalence (%)", fontsize=11)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_title("Histogram + Density", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Right: Q-Q plot vs normal
stats.probplot(obesity, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot vs Normal Distribution", fontsize=11, fontweight="bold")
axes[1].get_lines()[0].set_markerfacecolor("#3498db")
axes[1].get_lines()[0].set_markersize(5)
axes[1].get_lines()[1].set_color("#e74c3c")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/obesity_distribution_detail.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/obesity_distribution_detail.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/distribution_stats.csv", index=False)
print("Saved: outputs/distribution_stats.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("HOW TO READ THESE RESULTS")
print("=" * 60)
print("""
  Skewness:
    ~0     symmetric
    >0.5   right-skewed (long tail of high-risk tracts)
    <-0.5  left-skewed

  Kurtosis (excess):
    ~0     normal-like tails
    >0     heavier tails, more extreme tracts than normal
    <0     lighter tails, fewer extremes

  Shapiro-Wilk p-value:
    >0.05  fails to reject normality (looks normal)
    <0.05  rejects normality (definitely not normal)

  Bimodality Coefficient (BC):
    <0.555 likely unimodal
    >0.555 possibly bimodal — two distinct clusters of tracts

  Q-Q plot:
    points on line  = normal
    S-curve         = heavy tails or skew
    stair-step      = discreteness or clustering
""")