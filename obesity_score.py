"""
Obesity Risk Score — Neighborhood Targeting Tool
Santa Clara County Census Tracts
CDC PLACES 2025

Purpose: Generate a composite risk score per tract so Rohan's team
can target interventions in a proposal. The score combines:
  1. Predicted obesity from the tuned Random Forest
  2. Key driver levels (depression, smoking, housing insecurity, physical inactivity)
  3. Preventive care gap (low checkup/screening in high-risk areas)

Output: Ranked CSV + choropleth map of risk tiers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import pygris
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD + PREP
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading CDC PLACES 2025 data...")
df = pd.read_csv("PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260326.csv")
df = df[df["CountyName"] == "Santa Clara"].copy()

# Core behavioral/socioeconomic predictors (from your SHAP analysis)
PREDICTORS = [
    "CSMOKING_CrudePrev",
    "SLEEP_CrudePrev",
    "LPA_CrudePrev",
    "DEPRESSION_CrudePrev",
    "BINGE_CrudePrev",
    "FOODSTAMP_CrudePrev",
    "FOODINSECU_CrudePrev",
    "HOUSINSECU_CrudePrev",
    "LACKTRPT_CrudePrev",
    "LONELINESS_CrudePrev",
]
TARGET = "OBESITY_CrudePrev"

# Additional measures for the composite score
EXTRA_COLS = ["CHECKUP_CrudePrev", "CHOLSCREEN_CrudePrev", "DIABETES_CrudePrev",
              "BPHIGH_CrudePrev", "MHLTH_CrudePrev", "PHLTH_CrudePrev"]

ALL_COLS = PREDICTORS + [TARGET] + EXTRA_COLS

for c in ALL_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=PREDICTORS + [TARGET]).reset_index(drop=True)
df["TractFIPS"] = df["TractFIPS"].astype(str).str.zfill(11)

print(f"  Tracts loaded: {len(df)}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: TUNED RANDOM FOREST → PREDICTED OBESITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1 — Tuned Random Forest for Predicted Obesity")
print("=" * 60)

X = df[PREDICTORS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [8, 12, None],
    "min_samples_leaf": [3, 5],
    "max_features": ["sqrt", 0.5],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid, cv=5, scoring="r2", n_jobs=-1
)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
y_pred_test = best_rf.predict(X_test)
print(f"  Best params:  {grid.best_params_}")
print(f"  Test R²:      {r2_score(y_test, y_pred_test):.3f}")
print(f"  Test MAE:     {mean_absolute_error(y_test, y_pred_test):.2f}%")

# Refit on full data for risk scoring
best_rf.fit(X, y)
df["predicted_obesity"] = best_rf.predict(X)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: COMPOSITE RISK SCORE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Building Composite Risk Score")
print("=" * 60)

# Risk components (all normalized 0-1, higher = worse)
scaler = MinMaxScaler()

# Component 1: Predicted obesity (from RF model)
# Component 2: Top SHAP drivers — depression, smoking, housing insecurity
# Component 3: Physical inactivity + food stamp usage (socioeconomic stress)
# Component 4: Preventive care gap (inverted — low checkup = high risk)

risk_inputs = pd.DataFrame(index=df.index)

# Direct risk factors (higher = worse)
risk_inputs["predicted_obesity"]  = df["predicted_obesity"]
risk_inputs["depression"]         = df["DEPRESSION_CrudePrev"]
risk_inputs["smoking"]            = df["CSMOKING_CrudePrev"]
risk_inputs["housing_insecurity"] = df["HOUSINSECU_CrudePrev"]
risk_inputs["physical_inactivity"]= df["LPA_CrudePrev"]
risk_inputs["food_stamps"]        = df["FOODSTAMP_CrudePrev"]
risk_inputs["mental_health_days"] = df["MHLTH_CrudePrev"]

# Inverted: low preventive care = high risk
risk_inputs["low_checkup"]      = risk_inputs["predicted_obesity"].max() - df["CHECKUP_CrudePrev"].fillna(df["CHECKUP_CrudePrev"].median())
risk_inputs["low_screening"]    = risk_inputs["predicted_obesity"].max() - df["CHOLSCREEN_CrudePrev"].fillna(df["CHOLSCREEN_CrudePrev"].median())

# Normalize all components to 0-1
risk_normalized = pd.DataFrame(
    scaler.fit_transform(risk_inputs),
    columns=risk_inputs.columns,
    index=risk_inputs.index
)

# Weighted composite score (weights reflect SHAP importance ranking)
WEIGHTS = {
    "predicted_obesity":   0.30,   # model prediction is the anchor
    "depression":          0.15,   # #1 SHAP driver
    "smoking":             0.10,   # #2 SHAP driver
    "housing_insecurity":  0.10,   # #3 SHAP driver
    "physical_inactivity": 0.08,   # #4 SHAP driver
    "food_stamps":         0.07,   # socioeconomic proxy
    "mental_health_days":  0.07,   # mental health burden
    "low_checkup":         0.07,   # preventive care gap
    "low_screening":       0.06,   # preventive care gap
}

df["risk_score"] = sum(
    risk_normalized[col] * weight
    for col, weight in WEIGHTS.items()
)

# Scale to 0-100 for readability
df["risk_score"] = (df["risk_score"] * 100).round(1)

# Assign tiers
df["risk_tier"] = pd.qcut(
    df["risk_score"], q=5,
    labels=["Very Low", "Low", "Moderate", "High", "Critical"]
)

print("\n── Risk Tier Summary ──")
tier_summary = df.groupby("risk_tier", observed=True).agg(
    Tracts=("risk_score", "count"),
    Score_Min=("risk_score", "min"),
    Score_Max=("risk_score", "max"),
    Score_Mean=("risk_score", "mean"),
    Obesity_Mean=("predicted_obesity", "mean"),
    Depression_Mean=("DEPRESSION_CrudePrev", "mean"),
    Smoking_Mean=("CSMOKING_CrudePrev", "mean"),
).round(1)
print(tier_summary.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: MAP TRACT FIPS → NEIGHBORHOOD / CITY NAMES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Adding Neighborhood Names")
print("=" * 60)

print("  Downloading tract + place shapefiles for spatial join...")
tracts_gdf = pygris.tracts(state="CA", county="085", year=2020)
tracts_gdf["GEOID"] = tracts_gdf["GEOID"].astype(str).str.zfill(11)

# Get Census Designated Places (cities/CDPs) for Santa Clara County
places_gdf = pygris.places(state="CA", year=2020)

# Compute tract centroids and spatial join to nearest place
tracts_gdf = tracts_gdf.to_crs(epsg=4326)
places_gdf = places_gdf.to_crs(epsg=4326)

tract_centroids = tracts_gdf.copy()
tract_centroids["geometry"] = tract_centroids.geometry.centroid

# Spatial join: which place does each tract centroid fall in?
joined = gpd.sjoin(tract_centroids[["GEOID", "geometry"]],
                   places_gdf[["NAME", "geometry"]],
                   how="left", predicate="within")

# Some tracts may fall outside any place (unincorporated areas)
# For those, do a nearest join
unmatched = joined[joined["NAME"].isna()]["GEOID"].tolist()
if unmatched:
    unmatched_gdf = tract_centroids[tract_centroids["GEOID"].isin(unmatched)]
    nearest = gpd.sjoin_nearest(unmatched_gdf[["GEOID", "geometry"]],
                                places_gdf[["NAME", "geometry"]],
                                how="left")
    # Update the main join with nearest matches
    for _, row in nearest.iterrows():
        joined.loc[joined["GEOID"] == row["GEOID"], "NAME"] = row["NAME"] + " (unincorp.)"

# Create the lookup: GEOID → neighborhood/city name
tract_to_place = joined.drop_duplicates(subset="GEOID").set_index("GEOID")["NAME"].to_dict()

# Add to main dataframe
df["neighborhood"] = df["TractFIPS"].map(tract_to_place).fillna("Unknown")

print(f"  Mapped {(df['neighborhood'] != 'Unknown').sum()} / {len(df)} tracts to place names")

# Show a sample
print("\n── Sample Mappings ──")
sample = df[["TractFIPS", "neighborhood"]].drop_duplicates().head(10)
print(sample.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: TOP 20 HIGHEST-RISK TRACTS (WITH NAMES)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — Top 20 Highest-Risk Tracts")
print("=" * 60)

top20 = df.nlargest(20, "risk_score")[
    ["TractFIPS", "neighborhood", "risk_score", "risk_tier", "predicted_obesity",
     TARGET, "DEPRESSION_CrudePrev", "CSMOKING_CrudePrev",
     "HOUSINSECU_CrudePrev", "LPA_CrudePrev", "FOODSTAMP_CrudePrev"]
].reset_index(drop=True)

top20.columns = ["TractFIPS", "Neighborhood", "Risk Score", "Tier", "Predicted Obesity%",
                 "Actual Obesity%", "Depression%", "Smoking%",
                 "Housing Insec%", "Phys Inactive%", "Food Stamps%"]

print(top20.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: SAVE FULL OUTPUT CSV
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — Saving Full Dataset")
print("=" * 60)

output_cols = [
    "TractFIPS", "neighborhood", "risk_score", "risk_tier", "predicted_obesity", TARGET,
    "DEPRESSION_CrudePrev", "CSMOKING_CrudePrev", "HOUSINSECU_CrudePrev",
    "LPA_CrudePrev", "FOODSTAMP_CrudePrev", "FOODINSECU_CrudePrev",
    "LONELINESS_CrudePrev", "MHLTH_CrudePrev", "PHLTH_CrudePrev",
    "CHECKUP_CrudePrev", "CHOLSCREEN_CrudePrev",
    "DIABETES_CrudePrev", "BPHIGH_CrudePrev",
    "BINGE_CrudePrev", "SLEEP_CrudePrev", "LACKTRPT_CrudePrev",
]

# Only include columns that exist
output_cols = [c for c in output_cols if c in df.columns]

df_out = df[output_cols].sort_values("risk_score", ascending=False).reset_index(drop=True)
df_out.to_csv("outputs/obesity_risk_scores_by_tract.csv", index=False)
print(f"  Saved: outputs/obesity_risk_scores_by_tract.csv ({len(df_out)} tracts)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: RISK SCORE MAP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — Risk Score Choropleth Map")
print("=" * 60)

print("  Using already-downloaded tract shapefiles...")
merged = tracts_gdf.merge(
    df[["TractFIPS", "risk_score", "risk_tier", "predicted_obesity", TARGET]],
    left_on="GEOID", right_on="TractFIPS", how="left"
)

# ── Map 1: Continuous risk score ──
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Obesity Risk Score by Census Tract — Santa Clara County\n"
             "Composite Score: RF Prediction + Depression + Smoking + Housing Insecurity + Care Gaps",
             fontsize=13, fontweight="bold")

# Left: continuous risk score
merged.plot(
    column="risk_score", cmap="YlOrRd", linewidth=0.3,
    edgecolor="white", legend=True,
    legend_kwds={"label": "Risk Score (0-100)", "shrink": 0.6},
    missing_kwds={"color": "lightgrey"}, ax=axes[0]
)
axes[0].set_title("Composite Risk Score", fontsize=12, fontweight="bold")
axes[0].axis("off")

# Right: risk tiers (categorical)
tier_colors = {
    "Very Low": "#2b83ba",
    "Low":      "#abdda4",
    "Moderate": "#ffffbf",
    "High":     "#fdae61",
    "Critical": "#d7191c"
}

for tier, color in tier_colors.items():
    subset = merged[merged["risk_tier"] == tier]
    if len(subset) > 0:
        subset.plot(ax=axes[1], color=color, linewidth=0.3, edgecolor="white")

# Plot missing/NaN tracts in grey
missing = merged[merged["risk_tier"].isna()]
if len(missing) > 0:
    missing.plot(ax=axes[1], color="lightgrey", linewidth=0.3, edgecolor="white")

patches = [mpatches.Patch(color=c, label=t) for t, c in tier_colors.items()]
axes[1].legend(handles=patches, loc="lower left", fontsize=9, title="Risk Tier")
axes[1].set_title("Risk Tiers for Intervention Targeting", fontsize=12, fontweight="bold")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("outputs/obesity_risk_score_map.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: outputs/obesity_risk_score_map.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: RISK SCORE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Risk Score Distribution — Santa Clara County Census Tracts",
             fontsize=13, fontweight="bold")

# Histogram
axes[0].hist(df["risk_score"], bins=25, color="#e74c3c", edgecolor="white", alpha=0.85)
axes[0].axvline(df["risk_score"].median(), color="black", linestyle="--", linewidth=1.5,
                label=f'Median: {df["risk_score"].median():.1f}')
axes[0].set_xlabel("Risk Score (0-100)")
axes[0].set_ylabel("Number of Tracts")
axes[0].set_title("Distribution of Risk Scores")
axes[0].legend()

# Risk score vs actual obesity scatter
colors = df["risk_tier"].map(tier_colors)
axes[1].scatter(df["risk_score"], df[TARGET], c=colors, alpha=0.6, edgecolors="white", linewidth=0.5, s=40)
axes[1].set_xlabel("Composite Risk Score")
axes[1].set_ylabel("Actual Obesity Prevalence (%)")
axes[1].set_title("Risk Score vs Actual Obesity")

# Add tier patches to scatter legend
patches = [mpatches.Patch(color=c, label=t) for t, c in tier_colors.items()]
axes[1].legend(handles=patches, fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("outputs/obesity_risk_score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: outputs/obesity_risk_score_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: CITY-LEVEL SUMMARY (for proposals)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8 — Risk Summary by City/Neighborhood")
print("=" * 60)

# Clean up neighborhood names (remove unincorp. suffix for grouping)
df["city"] = df["neighborhood"].str.replace(r"\s*\(unincorp\.\)", "", regex=True)

city_summary = df.groupby("city", observed=True).agg(
    Tracts=("risk_score", "count"),
    Avg_Risk_Score=("risk_score", "mean"),
    Max_Risk_Score=("risk_score", "max"),
    Avg_Obesity=("predicted_obesity", "mean"),
    Critical_Tracts=("risk_tier", lambda x: (x == "Critical").sum()),
    High_Tracts=("risk_tier", lambda x: (x == "High").sum()),
).round(1)

city_summary = city_summary.sort_values("Avg_Risk_Score", ascending=False)

print("\n── Cities ranked by average risk score ──")
print(city_summary.to_string())

print("\n── Priority cities (≥1 Critical tract) ──")
priority = city_summary[city_summary["Critical_Tracts"] > 0]
print(priority.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
  Total tracts scored:    {len(df)}
  Risk score range:       {df['risk_score'].min():.1f} – {df['risk_score'].max():.1f}
  Critical tier tracts:   {(df['risk_tier'] == 'Critical').sum()}
  High tier tracts:       {(df['risk_tier'] == 'High').sum()}

  Composite score weights:
    Predicted obesity (RF):   30%
    Depression prevalence:    15%
    Smoking prevalence:       10%
    Housing insecurity:       10%
    Physical inactivity:       8%
    Food stamp usage:          7%
    Mental health days:        7%
    Low checkup rate:          7%
    Low screening rate:        6%

  Files saved:
    outputs/obesity_risk_scores_by_tract.csv
    outputs/obesity_risk_score_map.png
    outputs/obesity_risk_score_distribution.png

  → Use the CSV to identify which tracts to target in your proposal.
  → Critical + High tier tracts = priority neighborhoods for intervention.
""")