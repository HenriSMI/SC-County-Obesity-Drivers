"""
Step 5: Regression Model — Predicting Obesity Prevalence
Linear Regression + Random Forest, compared
CDC PLACES 2025 — Santa Clara County Census Tracts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import os

os.makedirs("outputs", exist_ok=True)

# ── Load + Filter ─────────────────────────────────────────────────────────────
df = pd.read_csv("PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260308.csv")
df = df[df["CountyName"] == "Santa Clara"].copy()

# ── Define predictors and target ──────────────────────────────────────────────
PREDICTORS = [
    # Health behaviors
    "CSMOKING_CrudePrev",
    "SLEEP_CrudePrev",
    "LPA_CrudePrev",
    "DEPRESSION_CrudePrev",
    "BINGE_CrudePrev",
    # Socioeconomic
    "FOODSTAMP_CrudePrev",
    "FOODINSECU_CrudePrev",
    "HOUSINSECU_CrudePrev",
    "LACKTRPT_CrudePrev",
    "LONELINESS_CrudePrev",
]
TARGET = "OBESITY_CrudePrev"

all_cols = PREDICTORS + [TARGET]
for c in all_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=all_cols)
print(f"Tracts used for modeling: {len(df)}")

X = df[PREDICTORS]
y = df[TARGET]

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize for linear regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Linear Regression
# ═════════════════════════════════════════════════════════════════════════════
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

r2_lr  = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
cv_lr  = cross_val_score(lr, scaler.transform(X), y, cv=5, scoring="r2").mean()

print("\n" + "=" * 50)
print("LINEAR REGRESSION")
print("=" * 50)
print(f"  R²  (test):       {r2_lr:.3f}")
print(f"  R²  (5-fold CV):  {cv_lr:.3f}")
print(f"  MAE (test):       {mae_lr:.2f}%")

# Coefficients
coef_df = pd.DataFrame({
    "Predictor": PREDICTORS,
    "Coefficient": lr.coef_
}).sort_values("Coefficient", ascending=False)
coef_df["Predictor"] = coef_df["Predictor"].str.replace("_CrudePrev", "")
print(f"\n  Coefficients (standardized):")
print(coef_df.to_string(index=False))
print("  → Positive = increases obesity, Negative = decreases obesity")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Random Forest
# ═════════════════════════════════════════════════════════════════════════════
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf  = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
cv_rf  = cross_val_score(rf, X, y, cv=5, scoring="r2").mean()

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)
print(f"  R²  (test):       {r2_rf:.3f}")
print(f"  R²  (5-fold CV):  {cv_rf:.3f}")
print(f"  MAE (test):       {mae_rf:.2f}%")

# Feature importance
feat_df = pd.DataFrame({
    "Predictor": PREDICTORS,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)
feat_df["Predictor"] = feat_df["Predictor"].str.replace("_CrudePrev", "")
print(f"\n  Feature Importances:")
print(feat_df.to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Obesity Regression Models — Santa Clara County", fontsize=14, fontweight="bold")

# 1. Actual vs Predicted (both models)
ax = axes[0]
ax.scatter(y_test, y_pred_lr, alpha=0.6, label=f"Linear (R²={r2_lr:.2f})", color="steelblue", s=40)
ax.scatter(y_test, y_pred_rf, alpha=0.6, label=f"Random Forest (R²={r2_rf:.2f})", color="tomato", s=40)
lims = [y.min() - 1, y.max() + 1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
ax.set_xlabel("Actual Obesity %")
ax.set_ylabel("Predicted Obesity %")
ax.set_title("Actual vs Predicted")
ax.legend(fontsize=9)

# 2. Linear Regression Coefficients
ax = axes[1]
colors = ["tomato" if c > 0 else "steelblue" for c in coef_df["Coefficient"]]
ax.barh(coef_df["Predictor"], coef_df["Coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Linear Regression\nCoefficients (standardized)")
ax.set_xlabel("Coefficient")

# 3. Random Forest Feature Importance
ax = axes[2]
ax.barh(feat_df["Predictor"], feat_df["Importance"], color="darkorange")
ax.set_title("Random Forest\nFeature Importance")
ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("outputs/regression_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved to outputs/regression_results.png")