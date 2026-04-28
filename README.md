# Santa Clara County Obesity Drivers

Tract-level analysis of obesity prevalence across Santa Clara County's ~370 census tracts, plus a composite risk score (0–100) per tract for community health targeting.

**TL;DR.** Behavioral health beats demographics as a predictor of tract-level obesity. Depression, smoking, and housing insecurity rank above race, income, or education — and depression's signal holds up after controlling for demographics. The geography is sharply divided east/west, with priority communities in Alum Rock, Gilroy, Morgan Hill, East San Jose, and a Stanford student-housing tract.

## Three findings

1. **Behavioral health > demographics.** Health-only Random Forest (Test R² = 0.953) outperformed the combined model (R² = 0.930). Demographics added noise rather than new signal.
2. **East/west fault line.** Average risk scores differ ~5x between Cupertino/Saratoga and Alum Rock/Gilroy.
3. **Preventive care is inverted.** Tracts with the highest obesity have the lowest checkup and screening rates.

## Priority cities

| City        | Tracts | Avg Score | Critical | High |
|-------------|--------|-----------|----------|------|
| Alum Rock   | 3      | 63.2      | 3        | 0    |
| Gilroy      | 12     | 56.9      | 8        | 4    |
| Morgan Hill | 9      | 45.0      | 3        | 4    |
| San Jose    | 213    | 42.0      | 65       | 53   |
| Stanford    | 5      | 40.1      | 2        | 0    |

The composite score is a weighted blend of nine components from the SHAP rankings: predicted obesity (30%), depression (15%), smoking (10%), housing insecurity (10%), physical inactivity (8%), food stamp usage (7%), mental health days (7%), low checkup rate (7%), and low screening rate (6%). Tract-level CSV in `outputs/`.

## Why depression is the anchor

Depression ranked #1 across every interpretability method (mean |SHAP| ~2.0, RF importance, linear coefficient). To check whether it was just a demographic proxy, the RF was run three ways:

| Model                  | Features | Test R² |
|------------------------|----------|---------|
| Health only            | 11       | 0.953   |
| Demographics only      | 7        | 0.877   |
| Health + Demographics  | 18       | 0.930   |

When demographics were added, depression dropped from rank #1 to #2 but kept a strong SHAP value (1.40) and held its directionality. If it were purely a proxy, the signal would have collapsed. It didn't.

This is consistent with Hemmingsson's neighborhood-disadvantage cycle: structural disadvantage → chronic stress → depression → emotional eating and inactivity → obesity → stigma → more stress. Depression is a proximal mediator, not a proxy — which is why its signal survives demographic controls.

## What's in this repo

| Script | Purpose |
|---|---|
| `data_audit.py` | Load + merge CDC PLACES and ACS at tract level |
| `rand_forest2.py` | Random Forest + SHAP on health features |
| `rf_validation.py` | Learning curves, train/test diagnostics |
| `model_comparison.py` | Health-only vs. demographics-only vs. combined |
| `trilayer_analysis.py` | SES × race × region overlay |
| `obesity_score.py` | Composite 0–100 risk score and tier assignments |
| `obesity_table.py` | City-level rollup |
| `contourplot.py` | Response surface: inactivity × food stamps |
| `depression_contourplot.py` | Response surface: depression × smoking (threshold plot) |
| `distribution.py` | Skew, kurtosis, bimodality tests |
| `combo_map.py`, `obesit_map.py` | Choropleth maps |
| `outputs/` | Ranked tract CSV, city summary, distribution stats, figures |

## Data sources

Raw data is not committed. Sources:
- **CDC PLACES 2025** (Census Tract, GIS-Friendly Format) — [cdc.gov/places](https://www.cdc.gov/places/)
- **ACS 5-Year Estimates 2019–2023** — via Census API ([free key](https://api.census.gov/data/key_signup.html)), pulled with the `census` package
- **TIGER/Line shapefiles** — via `pygris` (state CA, county FIPS 085, year 2022)

```bash
pip install pandas scikit-learn shap geopandas pygris matplotlib census us
export CENSUS_API_KEY=your_key_here
```

## Limitations

- **Ecological.** Tract-level correlations don't establish individual-level causation. The depression–obesity link could reflect reverse causality or shared upstream causes.
- **Bounded by county geography.** ~370 tracts in SCC; doesn't generalize beyond the county.
- **Linear regression R² = 1.00.** Almost certainly multicollinearity (several features intercorrelated >0.8). The Random Forest (R² = 0.96) is the model the analysis leans on.
- **Composite score ≠ obesity prevalence.** It captures vulnerability beyond current rates. One Stanford tract scored Critical (67.7) at only 23.7% obesity, driven by 28.8% depression in likely student housing.

## What's next

- Layer insurance type, in-network access, and GLP-1 availability onto the tract map
- Overlay FQHC and CBO service locations to quantify the preventive-care gap directly
- Test the 20% depression threshold for whether it's a real phase transition or a small-area-estimation artifact

---

Built with CDC PLACES, US Census ACS, and TIGER/Line data. Feedback and collaboration welcome.
