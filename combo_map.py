#!/usr/bin/env python3
"""
ACS Demographic & Socioeconomic Maps — Santa Clara County
Overlays Census demographic data with CDC PLACES health data.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from census import Census
from us import states
import numpy as np

# ============================================================
# CONFIG — UPDATE THESE
# ============================================================
CENSUS_API_KEY = "YOUR_CENSUS_API_KEY"  # Get from https://api.census.gov/data/key_signup.html
CDC_CSV_PATH = "PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260308.csv"
OUTPUT_DIR = "outputs/"

# Santa Clara County FIPS
STATE_FIPS = "06"
COUNTY_FIPS = "085"

# ============================================================
# 1. FETCH ACS 5-YEAR DATA (2019-2023)
# ============================================================
print("Fetching ACS 5-Year data from Census API...")
c = Census(CENSUS_API_KEY)

# ACS 5-Year 2023 variables at tract level
# B03003: Hispanic/Latino origin
# B02001: Race
# B19013: Median household income
# B17001: Poverty status
# B15003: Educational attainment
acs_vars = [
    "NAME",
    # Total population
    "B01003_001E",
    # Hispanic/Latino
    "B03003_001E",  # Total for Hispanic origin
    "B03003_003E",  # Hispanic or Latino
    # Race (alone)
    "B02001_001E",  # Total
    "B02001_002E",  # White alone
    "B02001_003E",  # Black alone
    "B02001_005E",  # Asian alone
    # Income & Poverty
    "B19013_001E",  # Median household income
    "B17001_001E",  # Total for poverty status
    "B17001_002E",  # Below poverty level
    # Education (pop 25+)
    "B15003_001E",  # Total 25+
    "B15003_022E",  # Bachelor's degree
    "B15003_023E",  # Master's degree
    "B15003_024E",  # Professional degree
    "B15003_025E",  # Doctorate
]

# Try 2023 first, fall back to 2022 if not yet available
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
print(f"  Retrieved {len(acs)} tracts from ACS")

# ============================================================
# 2. COMPUTE DEMOGRAPHIC PERCENTAGES
# ============================================================
# Build GEOID to match shapefile (state + county + tract = 11 digits)
acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"]

total_pop = acs["B01003_001E"].astype(float)
race_total = acs["B02001_001E"].astype(float)
hisp_total = acs["B03003_001E"].astype(float)
pov_total = acs["B17001_001E"].astype(float)
ed_total = acs["B15003_001E"].astype(float)

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

# Keep only what we need
acs_clean = acs[["GEOID", "pct_hispanic", "pct_white", "pct_black", "pct_asian",
                  "median_income", "pct_poverty", "pct_bachelors_plus"]].copy()

# Replace sentinel values (-666666666 = data not available)
acs_clean = acs_clean.replace(-666666666.0, np.nan)

print("  Demographic percentages computed")

# ============================================================
# 3. LOAD SHAPEFILE + CDC PLACES DATA
# ============================================================
print("Loading shapefile via pygris...")
from pygris import tracts
scc = tracts(state="06", county="085", year=2023)
scc = scc.to_crs(epsg=3857)  # Project for mapping
print(f"  {len(scc)} tracts loaded")

print("Loading CDC PLACES data...")
cdc = pd.read_csv(CDC_CSV_PATH)

# Debug: find the right column names
print(f"  CDC columns: {list(cdc.columns[:15])}...")  # Print first 15 cols

# Find the FIPS/location column (varies by release)
loc_col = next((c for c in cdc.columns if c.upper() in ["LOCATIONID", "TRACTFIPS", "GEOID", "FIPS"]), None)
county_col = next((c for c in cdc.columns if "county" in c.lower() and "fips" in c.lower()), None)

if loc_col is None or county_col is None:
    print(f"  WARNING: Could not auto-detect columns. All columns:\n  {list(cdc.columns)}")
    raise ValueError("Please check column names above and update script")

print(f"  Using location column: '{loc_col}', county column: '{county_col}'")
cdc_scc = cdc[cdc[county_col] == 6085].copy()
cdc_scc["GEOID"] = cdc_scc[loc_col].astype(str).str.zfill(11)

# Find the obesity column
obesity_col = next((c for c in cdc_scc.columns if "obesity" in c.lower() and "crude" in c.lower()), None)
if obesity_col is None:
    print(f"  WARNING: Could not find obesity column. Health columns:")
    print(f"  {[c for c in cdc_scc.columns if 'crude' in c.lower()][:20]}")
    raise ValueError("Please check column names above and update script")

print(f"  Using obesity column: '{obesity_col}'")
cdc_slim = cdc_scc[["GEOID", obesity_col]].copy()
cdc_slim = cdc_slim.rename(columns={obesity_col: "OBESITY_CrudePrev"})
print(f"  {len(cdc_slim)} CDC tracts for SCC")

# ============================================================
# 4. MERGE EVERYTHING
# ============================================================
merged = scc.merge(acs_clean, on="GEOID", how="left")
merged = merged.merge(cdc_slim, on="GEOID", how="left")
print(f"  Final merged dataset: {len(merged)} tracts")

# ============================================================
# 5. DEMOGRAPHIC MAPS (6-panel)
# ============================================================
print("Generating demographic maps...")

fig = plt.figure(figsize=(22, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.25, wspace=0.15)

map_configs = [
    ("pct_hispanic", "% Hispanic/Latino", "YlOrRd", None),
    ("pct_asian", "% Asian", "YlOrRd", None),
    ("pct_white", "% White (Non-Hispanic)", "YlOrRd", None),
    ("median_income", "Median Household Income ($)", "Greens", None),
    ("pct_poverty", "% Below Poverty Line", "Reds", None),
    ("pct_bachelors_plus", "% Bachelor's Degree+", "Blues", None),
]

for idx, (col, title, cmap, _) in enumerate(map_configs):
    ax = fig.add_subplot(gs[idx])
    if col == "median_income":
        # Use quantiles but format as dollars
        merged.plot(column=col, ax=ax, cmap=cmap, scheme="quantiles", k=6,
                    edgecolor="white", linewidth=0.3, legend=True,
                    missing_kwds={"color": "lightgrey", "label": "No Data"})
    else:
        merged.plot(column=col, ax=ax, cmap=cmap, scheme="quantiles", k=6,
                    edgecolor="white", linewidth=0.3, legend=True,
                    missing_kwds={"color": "lightgrey", "label": "No Data"})
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

fig.suptitle("Demographic & Socioeconomic Profile — Santa Clara County\nACS 5-Year Estimates (2019–2023)",
             fontsize=16, fontweight="bold", y=0.98)

plt.savefig(f"{OUTPUT_DIR}demographic_maps_scc.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUTPUT_DIR}demographic_maps_scc.png")
plt.close()

# ============================================================
# 6. SIDE-BY-SIDE: DEMOGRAPHICS vs OBESITY (4-panel)
# ============================================================
print("Generating demographic vs obesity comparison maps...")

fig2 = plt.figure(figsize=(22, 14))
gs2 = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.15)

comparison_configs = [
    ("OBESITY_CrudePrev", "Obesity Prevalence (%)", "RdYlGn_r"),
    ("pct_hispanic", "% Hispanic/Latino", "YlOrRd"),
    ("median_income", "Median Household Income ($)", "RdYlGn"),
    ("pct_poverty", "% Below Poverty Line", "RdYlGn_r"),
]

for idx, (col, title, cmap) in enumerate(comparison_configs):
    ax = fig2.add_subplot(gs2[idx])
    merged.plot(column=col, ax=ax, cmap=cmap, scheme="quantiles", k=6,
                edgecolor="white", linewidth=0.3, legend=True,
                missing_kwds={"color": "lightgrey", "label": "No Data"})
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

fig2.suptitle("Health Disparities & Demographics — Santa Clara County\nCDC PLACES 2025 + ACS 2019–2023",
              fontsize=16, fontweight="bold", y=0.98)

plt.savefig(f"{OUTPUT_DIR}demographic_vs_obesity_scc.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUTPUT_DIR}demographic_vs_obesity_scc.png")
plt.close()

# ============================================================
# 7. CORRELATION: DEMOGRAPHICS vs HEALTH OUTCOMES
# ============================================================
print("Computing demographic-health correlations...")

corr_cols = ["OBESITY_CrudePrev", "pct_hispanic", "pct_asian", "pct_white",
             "pct_black", "median_income", "pct_poverty", "pct_bachelors_plus"]

corr_matrix = merged[corr_cols].corr()

# Print key correlations with obesity
print("\n  Correlations with Obesity Prevalence:")
obesity_corrs = corr_matrix["OBESITY_CrudePrev"].drop("OBESITY_CrudePrev").sort_values(ascending=False)
for var, r in obesity_corrs.items():
    print(f"    {var:25s}  r = {r:+.3f}")

# Save correlation table
obesity_corrs.to_csv(f"{OUTPUT_DIR}demographic_obesity_correlations.csv")
print(f"\n  Saved: {OUTPUT_DIR}demographic_obesity_correlations.csv")

# ============================================================
# 8. SCATTER PLOTS: KEY RELATIONSHIPS
# ============================================================
print("Generating scatter plots...")

fig3, axes = plt.subplots(2, 2, figsize=(14, 12))

scatter_configs = [
    ("pct_hispanic", "% Hispanic/Latino", axes[0, 0]),
    ("median_income", "Median Household Income ($)", axes[0, 1]),
    ("pct_poverty", "% Below Poverty Line", axes[1, 0]),
    ("pct_bachelors_plus", "% Bachelor's Degree+", axes[1, 1]),
]

for col, label, ax in scatter_configs:
    subset = merged[[col, "OBESITY_CrudePrev"]].dropna()
    ax.scatter(subset[col], subset["OBESITY_CrudePrev"], alpha=0.5, s=30, color="#e67e22")
    # Fit line
    z = np.polyfit(subset[col], subset["OBESITY_CrudePrev"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(subset[col].min(), subset[col].max(), 100)
    ax.plot(x_range, p(x_range), "--", color="#c0392b", linewidth=2)
    r = subset[col].corr(subset["OBESITY_CrudePrev"])
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Obesity Prevalence (%)", fontsize=11)
    ax.set_title(f"{label} vs Obesity (r={r:.2f})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

fig3.suptitle("Demographic Predictors of Obesity — Santa Clara County\nCDC PLACES 2025 + ACS 2019–2023",
              fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}demographic_scatter_obesity.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUTPUT_DIR}demographic_scatter_obesity.png")
plt.close()

print("\nDone! All outputs saved to outputs/")