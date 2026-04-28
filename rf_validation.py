"""
Step 6: Random Forest Deep Dive
- Overfitting check
- Hyperparameter tuning
- Tract-level risk scores
- SHAP values for interpretability
CDC PLACES 2025 — Santa Clara County Census Tracts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import geopandas as gpd
import pygris
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import os

os.makedirs("outputs", exist_ok=True)

# ── Load + Filter ─────────────────────────────────────────────────────────────
df = pd.read_csv("PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260308.csv")
df = df[df["CountyName"] == "Santa Clara"].copy()

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

for c in PREDICTORS + [TARGET]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=PREDICTORS + [TARGET]).reset_index(drop=True)
df["TractFIPS"] = df["TractFIPS"].astype(str).str.zfill(11)

X = df[PREDICTORS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Overfitting Check
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("PART 1 — Overfitting Check (Learning Curve)")
print("=" * 55)

base_rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
train_sizes = np.linspace(0.1, 1.0, 10)
train_r2, test_r2 = [], []

for size in train_sizes:
    n = max(10, int(size * len(X_train)))
    Xi, yi = X_train.iloc[:n], y_train.iloc[:n]
    base_rf.fit(Xi, yi)
    train_r2.append(r2_score(yi, base_rf.predict(Xi)))
    test_r2.append(r2_score(y_test, base_rf.predict(X_test)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes * len(X_train), train_r2, "o-", label="Train R²", color="steelblue")
ax.plot(train_sizes * len(X_train), test_r2,  "o-", label="Test R²",  color="tomato")
ax.set_xlabel("Training Samples")
ax.set_ylabel("R²")
ax.set_title("Learning Curve — Random Forest")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/rf_learning_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Train R²: {train_r2[-1]:.3f}  |  Test R²: {test_r2[-1]:.3f}")
gap = train_r2[-1] - test_r2[-1]
print(f"  Gap: {gap:.3f} {'⚠ Possible overfitting' if gap > 0.05 else '✓ Looks healthy'}")

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Hyperparameter Tuning
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PART 2 — Hyperparameter Tuning (Grid Search)")
print("=" * 55)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth":    [None, 5, 10],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n  Best params: {grid_search.best_params_}")
print(f"  Best CV R²:  {grid_search.best_score_:.3f}")

best_rf = grid_search.best_estimator_
y_pred  = best_rf.predict(X_test)
print(f"  Test R²:     {r2_score(y_test, y_pred):.3f}")
print(f"  Test MAE:    {mean_absolute_error(y_test, y_pred):.2f}%")

cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring="r2")
print(f"  5-fold CV:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — Tract-Level Risk Scores
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PART 3 — Tract-Level Risk Scores")
print("=" * 55)

best_rf.fit(X, y)  # refit on full data for risk scores
df["predicted_obesity"] = best_rf.predict(X)
df["residual"]           = df[TARGET] - df["predicted_obesity"]

# Risk tier based on predicted obesity
df["risk_tier"] = pd.qcut(
    df["predicted_obesity"], q=4,
    labels=["Low", "Moderate", "High", "Critical"]
)

print("\n── Risk Tier Summary ──")
print(df.groupby("risk_tier")["predicted_obesity"].agg(
    Tracts="count", Min="min", Max="max", Mean="mean"
).round(1).to_string())

print("\n── Top 15 Highest Risk Tracts ──")
top15 = df.nlargest(15, "predicted_obesity")[
    ["TractFIPS", "predicted_obesity", TARGET, "DEPRESSION_CrudePrev",
     "HOUSINSECU_CrudePrev", "CSMOKING_CrudePrev", "LPA_CrudePrev"]
].reset_index(drop=True)
top15.columns = ["TractFIPS", "Predicted%", "Actual%", "Depression%", "HousingInsec%", "Smoking%", "Inactivity%"]
print(top15.round(1).to_string(index=False))

df.to_csv("outputs/scc_tracts_with_risk_scores.csv", index=False)
print("\nSaved: outputs/scc_tracts_with_risk_scores.csv")

# ═════════════════════════════════════════════════════════════════════════════
# PART 4 — SHAP Values
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PART 4 — SHAP Values")
print("=" * 55)

explainer   = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X)
shap_df     = pd.DataFrame(shap_values, columns=PREDICTORS)
shap_df.columns = [c.replace("_CrudePrev", "") for c in shap_df.columns]
X_display   = X.copy()
X_display.columns = [c.replace("_CrudePrev", "") for c in X_display.columns]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("SHAP Analysis — Obesity Predictors\nSanta Clara County Census Tracts",
             fontsize=14, fontweight="bold")

# Global SHAP bar chart
mean_shap = shap_df.abs().mean().sort_values(ascending=True)
axes[0].barh(mean_shap.index, mean_shap.values, color="darkorange")
axes[0].set_title("Mean |SHAP Value|\n(Global Feature Importance)")
axes[0].set_xlabel("Mean |SHAP Value| (% obesity impact)")

# SHAP beeswarm-style dot plot
ax2 = axes[1]
feature_order = mean_shap.index.tolist()
for i, feat in enumerate(feature_order):
    vals   = shap_df[feat].values
    feats  = X_display[feat].values
    norm   = (feats - feats.min()) / (feats.max() - feats.min() + 1e-9)
    colors = cm.RdYlGn_r(norm)
    jitter = np.random.uniform(-0.3, 0.3, size=len(vals))
    ax2.scatter(vals, i + jitter, c=colors, s=12, alpha=0.6)

ax2.set_yticks(range(len(feature_order)))
ax2.set_yticklabels(feature_order)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("SHAP Value (impact on obesity %)")
ax2.set_title("SHAP Dot Plot\nRed = high feature value, Green = low")

plt.tight_layout()
plt.savefig("outputs/shap_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/shap_analysis.png")

# ═════════════════════════════════════════════════════════════════════════════
# PART 5 — Risk Score Map
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("PART 5 — Risk Score Map")
print("=" * 55)

tracts = pygris.tracts(state="CA", county="085", year=2020)
tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
merged = tracts.merge(df[["TractFIPS", "predicted_obesity", "risk_tier", TARGET]],
                      left_on="GEOID", right_on="TractFIPS", how="left")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Obesity Risk Scores by Census Tract — Santa Clara County",
             fontsize=14, fontweight="bold")

for ax, col, title in [
    (axes[0], "predicted_obesity", "Predicted Obesity Risk (%)"),
    (axes[1], TARGET,              "Actual Obesity Prevalence (%)")
]:
    merged.plot(column=col, cmap="RdYlGn_r", linewidth=0.3,
                edgecolor="white", legend=True,
                legend_kwds={}, missing_kwds={"color": "lightgrey"},
                scheme="quantiles", k=6, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("outputs/map_risk_scores.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/map_risk_scores.png")