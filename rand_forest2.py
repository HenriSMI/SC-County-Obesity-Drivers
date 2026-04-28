#!/usr/bin/env python3
"""
Random Forest + SHAP: Original 11 Health Features + Demographics
Uses the same curated feature set from the original analysis,
then adds demographic variables to test if depression holds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from census import Census
import shap

# ============================================================
# CONFIG
# ============================================================
CENSUS_API_KEY = "YOUR_CENSUS_API_KEY"  # Get from https://api.census.gov/data/key_signup.html
CDC_CSV_PATH = "PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260308.csv"
OUTPUT_DIR = "outputs/"
STATE_FIPS = "06"
COUNTY_FIPS = "085"

# ============================================================
# CURATED HEALTH FEATURES (same as original analysis)
# ============================================================
# These are the 11 features from your original RF/SHAP run.
# Adjust column names if your CSV uses slightly different naming.
HEALTH_FEATURE_KEYWORDS = [
    "DEPRESSION",
    "CSMOKING",
    "HOUSINSECU",
    "LPA",
    "LONELINESS",
    "BINGE",
    "FOODSTAMP",
    "LACKTRPT",
    "SLEEP",
    "FOODINSECU",
]

DEMO_COLS = [
    "pct_hispanic", "pct_white", "pct_black", "pct_asian",
    "median_income", "pct_poverty", "pct_bachelors_plus"
]

# ============================================================
# 1. LOAD CDC PLACES DATA
# ============================================================
print("Loading CDC PLACES data...")
cdc = pd.read_csv(CDC_CSV_PATH)

loc_col = next((c for c in cdc.columns if c.upper() in ["LOCATIONID", "TRACTFIPS", "GEOID", "FIPS"]), None)
county_col = next((c for c in cdc.columns if "county" in c.lower() and "fips" in c.lower()), None)
obesity_col = next((c for c in cdc.columns if "obesity" in c.lower() and "crude" in c.lower()), None)

cdc_scc = cdc[cdc[county_col] == 6085].copy()
cdc_scc["GEOID"] = cdc_scc[loc_col].astype(str).str.zfill(11)

# Match curated features to actual column names
crude_cols = [c for c in cdc_scc.columns if "CrudePrev" in c]
health_features = []
health_display = []

for keyword in HEALTH_FEATURE_KEYWORDS:
    match = [c for c in crude_cols if keyword.upper() in c.upper()]
    if match:
        health_features.append(match[0])
        health_display.append(keyword)
        print(f"  Matched: {keyword} -> {match[0]}")
    else:
        print(f"  WARNING: No match for {keyword}")

print(f"\n  {len(health_features)} of {len(HEALTH_FEATURE_KEYWORDS)} features matched")

cdc_slim = cdc_scc[["GEOID", obesity_col] + health_features].copy()
cdc_slim = cdc_slim.rename(columns={obesity_col: "OBESITY"})

# ============================================================
# 2. FETCH ACS DEMOGRAPHIC DATA
# ============================================================
print("\nFetching ACS data...")
c = Census(CENSUS_API_KEY)

acs_vars = [
    "NAME",
    "B01003_001E",
    "B03003_001E", "B03003_003E",
    "B02001_001E", "B02001_002E", "B02001_003E", "B02001_005E",
    "B19013_001E",
    "B17001_001E", "B17001_002E",
    "B15003_001E", "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
]

for yr in [2023, 2022]:
    try:
        acs_raw = c.acs5.state_county_tract(
            fields=acs_vars,
            state_fips=STATE_FIPS,
            county_fips=COUNTY_FIPS,
            tract="*",
            year=yr
        )
        print(f"  Using ACS 5-Year {yr}")
        break
    except Exception as e:
        print(f"  ACS {yr} not available ({e}), trying older year...")

acs = pd.DataFrame(acs_raw)
acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"]

race_total = acs["B02001_001E"].astype(float).replace(0, np.nan)
hisp_total = acs["B03003_001E"].astype(float).replace(0, np.nan)
pov_total = acs["B17001_001E"].astype(float).replace(0, np.nan)
ed_total = acs["B15003_001E"].astype(float).replace(0, np.nan)

acs["pct_hispanic"] = (acs["B03003_003E"].astype(float) / hisp_total * 100).round(1)
acs["pct_white"] = (acs["B02001_002E"].astype(float) / race_total * 100).round(1)
acs["pct_black"] = (acs["B02001_003E"].astype(float) / race_total * 100).round(1)
acs["pct_asian"] = (acs["B02001_005E"].astype(float) / race_total * 100).round(1)
acs["median_income"] = acs["B19013_001E"].astype(float)
acs["pct_poverty"] = (acs["B17001_002E"].astype(float) / pov_total * 100).round(1)
acs["pct_bachelors_plus"] = (
    (acs["B15003_022E"].astype(float) +
     acs["B15003_023E"].astype(float) +
     acs["B15003_024E"].astype(float) +
     acs["B15003_025E"].astype(float)) / ed_total * 100
).round(1)

acs_clean = acs[["GEOID"] + DEMO_COLS].copy()
acs_clean = acs_clean.replace(-666666666.0, np.nan)

# ============================================================
# 3. MERGE
# ============================================================
print("\nMerging datasets...")
merged = cdc_slim.merge(acs_clean, on="GEOID", how="inner")
merged = merged.dropna()
print(f"  {len(merged)} tracts with complete data")

# ============================================================
# 4. THREE MODELS
# ============================================================
feature_sets = {
    "Health Only (Original 11)": (health_features, health_display),
    "Demographics Only": (DEMO_COLS, DEMO_COLS),
    "Original 11 + Demographics": (health_features + DEMO_COLS, health_display + DEMO_COLS),
}

results = {}

for name, (features, display) in feature_sets.items():
    print(f"\n{'='*55}")
    print(f"Model: {name} ({len(features)} features)")
    print(f"{'='*55}")

    X = merged[features].values
    y = merged["OBESITY"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    train_r2 = rf.score(X_train, y_train)
    test_r2 = rf.score(X_test, y_test)
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_order = np.argsort(mean_shap)[::-1]

    print(f"\n  Full ranking (SHAP):")
    for rank, idx in enumerate(importance_order):
        marker = " <--" if display[idx].upper() == "DEPRESSION" else ""
        print(f"    {rank+1:>2}. {display[idx]:25s}  |SHAP| = {mean_shap[idx]:.3f}{marker}")

    results[name] = {
        "rf": rf,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "shap_values": shap_values,
        "X_test": X_test,
        "features": features,
        "display_names": display,
        "mean_shap": mean_shap,
        "importance_order": importance_order,
    }

# ============================================================
# 5. SIDE-BY-SIDE SHAP BAR CHARTS
# ============================================================
print("\nGenerating comparison figure...")

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for ax, (name, res) in zip(axes, results.items()):
    order = res["importance_order"]
    names = [res["display_names"][i] for i in order]
    vals = [res["mean_shap"][i] for i in order]

    colors = []
    for i in order:
        feat = res["features"][i]
        if feat in DEMO_COLS:
            colors.append("#3498db")
        elif "DEPRESSION" in res["display_names"][i].upper():
            colors.append("#e74c3c")  # red for depression
        else:
            colors.append("#e67e22")

    y_pos = range(len(names) - 1, -1, -1)
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (% obesity impact)", fontsize=11)
    ax.set_title(f"{name}\nTest R² = {res['test_r2']:.3f}", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="Depression"),
    Patch(facecolor="#e67e22", label="Other health measure"),
    Patch(facecolor="#3498db", label="Demographic variable"),
]
axes[2].legend(handles=legend_elements, loc="lower right", fontsize=10)

fig.suptitle(
    "Does Depression Hold Up? SHAP Rankings With and Without Demographics\n"
    "Original 11 Health Features, Santa Clara County Census Tracts",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}shap_curated_health_vs_demographics.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUTPUT_DIR}shap_curated_health_vs_demographics.png")
plt.close()

# ============================================================
# 6. SHAP DOT PLOTS (Health Only vs Combined)
# ============================================================
print("Generating SHAP dot plots...")

fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8))

# Health Only dot plot
plt.sca(ax_left)
shap.summary_plot(
    results["Health Only (Original 11)"]["shap_values"],
    features=results["Health Only (Original 11)"]["X_test"],
    feature_names=results["Health Only (Original 11)"]["display_names"],
    show=False,
    max_display=11
)
ax_left.set_title("Health Only (Original 11)", fontsize=12, fontweight="bold")

# Combined dot plot
plt.sca(ax_right)
shap.summary_plot(
    results["Original 11 + Demographics"]["shap_values"],
    features=results["Original 11 + Demographics"]["X_test"],
    feature_names=results["Original 11 + Demographics"]["display_names"],
    show=False,
    max_display=18
)
ax_right.set_title("Original 11 + Demographics", fontsize=12, fontweight="bold")

fig2.suptitle(
    "SHAP Dot Plots: Before and After Adding Demographics\nSanta Clara County Census Tracts",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}shap_dotplots_before_after.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUTPUT_DIR}shap_dotplots_before_after.png")
plt.close()

# ============================================================
# 7. SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY: Model Comparison")
print("="*60)
print(f"{'Model':<35s} {'Features':>8s} {'Train R²':>10s} {'Test R²':>10s}")
print("-"*65)
for name, res in results.items():
    n_feat = len(res["features"])
    print(f"{name:<35s} {n_feat:>8d} {res['train_r2']:>10.4f} {res['test_r2']:>10.4f}")

print(f"\n{'='*60}")
print("DEPRESSION TRACKING")
print("="*60)
for name, res in results.items():
    for rank, idx in enumerate(res["importance_order"]):
        if "DEPRESSION" in res["display_names"][idx].upper():
            print(f"  {name:<35s}  Rank: #{rank+1}  |SHAP| = {res['mean_shap'][idx]:.3f}")
            break
    else:
        print(f"  {name:<35s}  (not in feature set)")

print("\nDone!")