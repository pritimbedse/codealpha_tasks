"""
CodeAlpha Data Science Internship - Task 4: Sales Prediction using Python
-------------------------------------------------------------------------
Goal: predict Sales from advertising spend on TV, Radio and Newspaper,
measure how each channel moves sales, and turn that into marketing advice.

Run:  python sales_prediction.py
Out:  outputs/ (charts, tables, cleaned data, metrics)
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sales_outputs")
os.makedirs(OUT, exist_ok=True)
SEED = 42
sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ----------------------------------------------------------------------------
df = pd.read_csv(os.path.join(HERE, "Advertising.csv"))
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""])  # index column
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum().to_string())
print("Duplicate rows:", df.duplicated().sum())
print("Zero spend rows:", (df[["TV", "Radio", "Newspaper"]] == 0).sum().to_dict())

# Outliers (IQR rule) - report, keep (they are real spend levels, not errors)
for c in ["TV", "Radio", "Newspaper", "Sales"]:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    n = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    print(f"IQR outliers in {c}: {n}")
df.to_csv(os.path.join(OUT, "advertising_clean.csv"), index=False)

# ----------------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------------
print("\n", df.describe().T.round(2).to_string())
corr = df.corr()
print("\nCorrelation with Sales:\n", corr["Sales"].round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax[0])
ax[0].set_title("Correlation matrix")
df.drop(columns="Sales").plot.box(ax=ax[1])
ax[1].set_title("Spend distribution by channel")
plt.tight_layout(); plt.savefig(f"{OUT}/01_correlation_and_spread.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, c in zip(ax, ["TV", "Radio", "Newspaper"]):
    sns.regplot(x=c, y="Sales", data=df, ax=a, scatter_kws=dict(alpha=.6, s=25),
                line_kws=dict(color="crimson"))
    a.set_title(f"{c} vs Sales (r = {corr.loc[c, 'Sales']:.2f})")
plt.tight_layout(); plt.savefig(f"{OUT}/02_spend_vs_sales.png", dpi=150); plt.close()

# ----------------------------------------------------------------------------
# 3. FEATURE ENGINEERING / SELECTION
# ----------------------------------------------------------------------------
X_all = df[["TV", "Radio", "Newspaper"]]
vif = {}
for c in X_all.columns:
    others = X_all.drop(columns=c)
    r2 = LinearRegression().fit(others, X_all[c]).score(others, X_all[c])
    vif[c] = 1 / (1 - r2)
print("\nVIF:", {k: round(v, 2) for k, v in vif.items()})

df["TV_x_Radio"] = df["TV"] * df["Radio"]   # do the channels reinforce each other?
y = df["Sales"]
feature_sets = {
    "TV+Radio+Newspaper": ["TV", "Radio", "Newspaper"],
    "TV+Radio": ["TV", "Radio"],
    "TV+Radio+TVxRadio": ["TV", "Radio", "TV_x_Radio"],
}

# ----------------------------------------------------------------------------
# 4. MODELLING
# ----------------------------------------------------------------------------
Xtr_all, Xte_all, ytr, yte = train_test_split(df, y, test_size=0.2, random_state=SEED)
kf = KFold(5, shuffle=True, random_state=SEED)

def evaluate(name, model, cols):
    Xtr, Xte = Xtr_all[cols], Xte_all[cols]
    cv = cross_val_score(model, Xtr, ytr, cv=kf, scoring="r2")
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return dict(Model=name, Features="+".join(cols), CV_R2=cv.mean(), CV_R2_std=cv.std(),
                Test_R2=r2_score(yte, p), Test_MAE=mean_absolute_error(yte, p),
                Test_RMSE=np.sqrt(mean_squared_error(yte, p))), model, p

rows, fitted = [], {}
specs = [
    ("Linear (all 3)", LinearRegression(), feature_sets["TV+Radio+Newspaper"]),
    ("Linear (TV+Radio)", LinearRegression(), feature_sets["TV+Radio"]),
    ("Linear + interaction", LinearRegression(), feature_sets["TV+Radio+TVxRadio"]),
    ("Ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))), feature_sets["TV+Radio+Newspaper"]),
    ("Lasso", make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=SEED)), feature_sets["TV+Radio+Newspaper"]),
    ("Poly deg-2 + Ridge", make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                          RidgeCV(alphas=np.logspace(-3, 3, 30))), feature_sets["TV+Radio"]),
    ("Random Forest", RandomForestRegressor(300, min_samples_leaf=3, random_state=SEED), feature_sets["TV+Radio+Newspaper"]),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=.05,
                                                     subsample=.8, random_state=SEED), feature_sets["TV+Radio+Newspaper"]),
]
for name, m, cols in specs:
    r, fm, p = evaluate(name, m, cols)
    rows.append(r); fitted[name] = (fm, cols, p)
res = pd.DataFrame(rows).sort_values("CV_R2", ascending=False).reset_index(drop=True)
print("\nModel comparison:\n", res.round(3).to_string())
res.round(4).to_csv(f"{OUT}/model_comparison.csv", index=False)

best_name = res.loc[0, "Model"]
print("\nBest model by CV R2:", best_name)

fig, ax = plt.subplots(figsize=(8, 4.5))
r_plot = res.sort_values("CV_R2")
ax.barh(r_plot["Model"], r_plot["CV_R2"], xerr=r_plot["CV_R2_std"], color="#2a6f97")
ax.set_xlim(0.8, 1.0); ax.set_xlabel("5-fold CV R² (training set)")
ax.set_title("Model comparison")
plt.tight_layout(); plt.savefig(f"{OUT}/03_model_comparison.png", dpi=150); plt.close()

# ----------------------------------------------------------------------------
# 5. EXPLAINABLE MODEL: OLS with p-values and confidence intervals
# ----------------------------------------------------------------------------
def ols_summary(cols, data):
    X = np.column_stack([np.ones(len(data)), data[cols].values]); yy = data["Sales"].values
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    resid = yy - X @ beta
    dof = len(yy) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t = beta / se; p = 2 * stats.t.sf(np.abs(t), dof); tcrit = stats.t.ppf(.975, dof)
    out = pd.DataFrame({"coef": beta, "std_err": se, "t": t, "p_value": p,
                        "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se},
                       index=["Intercept"] + cols)
    r2 = 1 - resid @ resid / ((yy - yy.mean()) @ (yy - yy.mean()))
    return out, r2, resid

ols3, r2_3, _ = ols_summary(["TV", "Radio", "Newspaper"], df)
print(f"\nOLS (all 3 channels)  R2={r2_3:.4f}\n", ols3.round(4).to_string())
ols_int, r2_int, _ = ols_summary(["TV", "Radio", "TV_x_Radio"], df)
print(f"\nOLS (TV, Radio, TVxRadio)  R2={r2_int:.4f}\n", ols_int.round(5).to_string())
ols3.round(5).to_csv(f"{OUT}/ols_coefficients_all3.csv")
ols_int.round(6).to_csv(f"{OUT}/ols_coefficients_interaction.csv")

# ----------------------------------------------------------------------------
# 6. DIAGNOSTICS of the best model
# ----------------------------------------------------------------------------
bm, bcols, bpred = fitted[best_name]
resid = yte.values - bpred
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].scatter(yte, bpred, alpha=.7); lim = [yte.min() - 1, yte.max() + 1]
ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("Actual sales"); ax[0].set_ylabel("Predicted sales")
ax[0].set_title(f"{best_name}: actual vs predicted (test)")
ax[1].scatter(bpred, resid, alpha=.7); ax[1].axhline(0, color="r", ls="--")
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Residual"); ax[1].set_title("Residuals vs predicted")
stats.probplot(resid, plot=ax[2]); ax[2].set_title("Residual Q-Q plot")
plt.tight_layout(); plt.savefig(f"{OUT}/04_best_model_diagnostics.png", dpi=150); plt.close()

pi = permutation_importance(bm, Xte_all[bcols], yte, n_repeats=30, random_state=SEED, scoring="r2")
imp = pd.Series(pi.importances_mean, index=bcols).sort_values()
print("\nPermutation importance (drop in R2):\n", imp.round(3).to_string())
fig, ax = plt.subplots(figsize=(6, 3.5)); imp.plot.barh(ax=ax, color="#2a6f97")
ax.set_title("Which inputs matter? (permutation importance)"); ax.set_xlabel("Drop in test R² when shuffled")
plt.tight_layout(); plt.savefig(f"{OUT}/05_feature_importance.png", dpi=150); plt.close()

# ----------------------------------------------------------------------------
# 7. HOW DOES ADVERTISING CHANGE SALES?  (what-if)
# ----------------------------------------------------------------------------
final = LinearRegression().fit(df[["TV", "Radio", "TV_x_Radio"]], y)
def predict_sales(tv, radio):
    return final.predict(pd.DataFrame({"TV": [tv], "Radio": [radio], "TV_x_Radio": [tv * radio]}))[0]

base = dict(tv=df.TV.mean(), radio=df.Radio.mean())
base_sales = predict_sales(**base)
print(f"\nBaseline (avg spend TV={base['tv']:.1f}, Radio={base['radio']:.1f}) -> predicted sales {base_sales:.2f}")
scen = []
for ch in ["tv", "radio"]:
    for pct in [-20, -10, 10, 20]:
        b = dict(base); b[ch] = b[ch] * (1 + pct / 100)
        extra = base[ch] * pct / 100
        d = predict_sales(**b) - base_sales
        scen.append(dict(Channel=ch.upper(), Change_pct=pct, Spend_change=extra, Sales_change=d,
                         Sales_change_pct=100 * d / base_sales, Sales_per_1_spend=d / extra))
scen = pd.DataFrame(scen)
print(scen.round(3).to_string(index=False))
scen.round(4).to_csv(f"{OUT}/what_if_scenarios.csv", index=False)

tv_grid = np.linspace(df.TV.min(), df.TV.max(), 100)
rad_grid = np.linspace(df.Radio.min(), df.Radio.max(), 100)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for r, lab in [(base["radio"], f"Radio = {base['radio']:.0f} (avg)"), (40, "Radio = 40 (high)"), (5, "Radio = 5 (low)")]:
    ax[0].plot(tv_grid, [predict_sales(t, r) for t in tv_grid], label=lab)
ax[0].set_xlabel("TV spend"); ax[0].set_ylabel("Predicted sales"); ax[0].legend()
ax[0].set_title("TV effect gets stronger with more Radio")
for t, lab in [(base["tv"], f"TV = {base['tv']:.0f} (avg)"), (250, "TV = 250 (high)"), (30, "TV = 30 (low)")]:
    ax[1].plot(rad_grid, [predict_sales(t, r) for r in rad_grid], label=lab)
ax[1].set_xlabel("Radio spend"); ax[1].legend(); ax[1].set_title("Radio effect vs TV level")
plt.tight_layout(); plt.savefig(f"{OUT}/06_what_if_curves.png", dpi=150); plt.close()

# Budget allocation across TV/Radio within the observed range (Newspaper = 0)
mix_tv = df.TV.mean() / (df.TV.mean() + df.Radio.mean())
alloc = []
for B in [100, 150, 200, 250, 300]:
    best = (-1, 0, 0)
    for r in np.linspace(0, min(B, df.Radio.max()), 400):
        t = B - r
        if 0 <= t <= df.TV.max():
            s = predict_sales(t, r)
            if s > best[0]:
                best = (s, t, r)
    cur = predict_sales(B * mix_tv, B * (1 - mix_tv))
    alloc.append(dict(Budget=B, TV=best[1], Radio=best[2], Radio_share_pct=100 * best[2] / B,
                      Pred_sales_optimal=best[0], Pred_sales_current_mix=cur, Uplift=best[0] - cur))
alloc = pd.DataFrame(alloc)
print("\nBudget allocation (TV+Radio only):\n", alloc.round(2).to_string(index=False))
alloc.round(3).to_csv(f"{OUT}/budget_allocation.csv", index=False)

print("\nSales change per +1 unit spend (3-channel OLS):")
print(ols3.loc[["TV", "Radio", "Newspaper"], ["coef", "ci_low", "ci_high", "p_value"]].round(4).to_string())

# ----------------------------------------------------------------------------
# 8. SAVE
# ----------------------------------------------------------------------------
joblib.dump(final, f"{OUT}/final_model_interaction_ols.joblib")
json.dump(dict(best_model_cv=best_name, results=res.round(4).to_dict("records"),
               ols_all3_r2=round(r2_3, 4), ols_interaction_r2=round(r2_int, 4)),
          open(f"{OUT}/metrics.json", "w"), indent=2)
print("\nDone. Outputs in", OUT)