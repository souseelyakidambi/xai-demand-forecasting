import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv('data/train.csv', parse_dates=['date'])

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
df['year']        = df['date'].dt.year
df['month']       = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year']= df['date'].dt.isocalendar().week.astype(int)
df['quarter']     = df['date'].dt.quarter
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

# Lag features (previous sales context)
df = df.sort_values(['store','item','date'])
df['lag_7']  = df.groupby(['store','item'])['sales'].shift(7)
df['lag_30'] = df.groupby(['store','item'])['sales'].shift(30)
df['rolling_mean_7']  = df.groupby(['store','item'])['sales'].transform(lambda x: x.shift(1).rolling(7).mean())
df['rolling_mean_30'] = df.groupby(['store','item'])['sales'].transform(lambda x: x.shift(1).rolling(30).mean())
df = df.dropna()

# ── 3. PREPARE FEATURES ──────────────────────────────────────────────────────
features = ['store','item','year','month','day_of_week','week_of_year',
            'quarter','is_weekend','lag_7','lag_30','rolling_mean_7','rolling_mean_30']
X = df[features]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 4. TRAIN MODELS ──────────────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost':           xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                                          random_state=42, verbosity=0)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mape  = np.mean(np.abs((y_test - preds) / y_test)) * 100
    results[name] = {'MAE': round(mae,4), 'RMSE': round(rmse,4), 'MAPE': round(mape,4)}
    print(f"{name}: MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

# Save results table
results_df = pd.DataFrame(results).T
results_df.to_csv('model_results.csv')
print("\nModel results saved.")

# ── 5. SHAP EXPLANATIONS (XGBoost) ───────────────────────────────────────────
print("\nComputing SHAP values (this may take a minute)...")
best_model = models['XGBoost']

# Use a sample for speed
X_sample = X_test.sample(2000, random_state=42)
explainer   = shap.Explainer(best_model, X_train.sample(500, random_state=42))
shap_values = explainer(X_sample)

# Global feature importance (mean |SHAP|)
shap_importance = pd.DataFrame({
    'feature':    features,
    'mean_shap':  np.abs(shap_values.values).mean(axis=0)
}).sort_values('mean_shap', ascending=False)
shap_importance.to_csv('shap_importance.csv', index=False)
print("\nSHAP feature importance:")
print(shap_importance.to_string(index=False))

# ── 6. SAVE PLOTS ────────────────────────────────────────────────────────────
# Plot 1: SHAP bar summary
plt.figure(figsize=(9,6))
shap.plots.bar(shap_values, max_display=12, show=False)
plt.title("SHAP Feature Importance — XGBoost Demand Forecasting", fontsize=13, pad=14)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: SHAP beeswarm
plt.figure(figsize=(9,7))
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title("SHAP Value Distribution per Feature", fontsize=13, pad=14)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Model comparison bar chart
fig, ax = plt.subplots(figsize=(8,5))
metrics_plot = results_df[['MAE','RMSE']].copy()
metrics_plot.plot(kind='bar', ax=ax, color=['#2E75B6','#1F4E79'], edgecolor='white', width=0.6)
ax.set_title("Model Comparison: MAE and RMSE", fontsize=13, pad=12)
ax.set_xlabel("")
ax.set_ylabel("Error")
ax.set_xticklabels(metrics_plot.index, rotation=15, ha='right')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll plots saved. Done.")
print("\nFiles to share with me:")
print("  model_results.csv")
print("  shap_importance.csv")
print("  shap_bar.png")
print("  shap_beeswarm.png")
print("  model_comparison.png")