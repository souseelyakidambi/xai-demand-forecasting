import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

# ── LOAD & FEATURE ENGINEERING ───────────────────────────────────────────────
df = pd.read_csv('data/train.csv', parse_dates=['date'])
df['year']         = df['date'].dt.year
df['month']        = df['date'].dt.month
df['day_of_week']  = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['quarter']      = df['date'].dt.quarter
df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
df = df.sort_values(['store','item','date'])
df['lag_7']           = df.groupby(['store','item'])['sales'].shift(7)
df['lag_30']          = df.groupby(['store','item'])['sales'].shift(30)
df['rolling_mean_7']  = df.groupby(['store','item'])['sales'].transform(lambda x: x.shift(1).rolling(7).mean())
df['rolling_mean_30'] = df.groupby(['store','item'])['sales'].transform(lambda x: x.shift(1).rolling(30).mean())
df = df.dropna()

features = ['store','item','year','month','day_of_week','week_of_year',
            'quarter','is_weekend','lag_7','lag_30','rolling_mean_7','rolling_mean_30']

# ── TEMPORAL SPLIT (80/20 by date) ───────────────────────────────────────────
df_sorted  = df.sort_values('date')
split_idx  = int(len(df_sorted) * 0.8)
train_df   = df_sorted.iloc[:split_idx]
test_df    = df_sorted.iloc[split_idx:]
Xtr, ytr   = train_df[features], train_df['sales']
Xte, yte   = test_df[features],  test_df['sales']
print(f"Train size: {len(Xtr):,}  |  Test size: {len(Xte):,}")
print(f"Train dates: {train_df['date'].min().date()} → {train_df['date'].max().date()}")
print(f"Test  dates: {test_df['date'].min().date()}  → {test_df['date'].max().date()}")

# ── MODEL TRAINING ───────────────────────────────────────────────────────────
print("\nTraining models...")

lr = LinearRegression()
lr.fit(Xtr, ytr)
lr_preds = lr.predict(Xte)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)
rf_preds = rf.predict(Xte)

xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1,
                               max_depth=6, random_state=42, verbosity=0)
xgb_model.fit(Xtr, ytr)
xgb_preds = xgb_model.predict(Xte)

# ── TABLE 1: MODEL COMPARISON ─────────────────────────────────────────────────
def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return round(mae,2), round(rmse,2), round(mape,2)

results = {
    'Linear Regression': metrics(yte, lr_preds),
    'Random Forest':     metrics(yte, rf_preds),
    'XGBoost':           metrics(yte, xgb_preds),
}
print("\n=== TABLE 1: Model Comparison ===")
print(f"{'Model':<25} {'MAE':>6} {'RMSE':>6} {'MAPE':>7}")
print("-" * 48)
for model, (mae, rmse, mape) in results.items():
    print(f"{model:<25} {mae:>6.2f} {rmse:>6.2f} {mape:>6.2f}%")

results_df = pd.DataFrame(results, index=['MAE','RMSE','MAPE']).T
results_df.to_csv('output/global_model_comparison.csv')

# ── GLOBAL SHAP ANALYSIS ─────────────────────────────────────────────────────
print("\nRunning global SHAP analysis...")
sample_test = Xte.sample(2000, random_state=42)
explainer   = shap.Explainer(xgb_model, Xtr.sample(500, random_state=42))
shap_values = explainer(sample_test)

mean_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0),
    index=features
).sort_values(ascending=False)

print("\n=== TABLE 2: Global SHAP Feature Importance ===")
print(mean_shap.round(3).to_string())
mean_shap.to_csv('output/global_shap_importance.csv', header=['mean_abs_shap'])

# ── FIGURE 3: MODEL COMPARISON BAR ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor('white')
models    = list(results.keys())
mae_vals  = [results[m][0] for m in models]
rmse_vals = [results[m][1] for m in models]
x = np.arange(len(models))
w = 0.35
b1 = ax.bar(x - w/2, mae_vals,  w, label='MAE',  color='#1565C0', edgecolor='white')
b2 = ax.bar(x + w/2, rmse_vals, w, label='RMSE', color='#E53935', edgecolor='white')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, fontweight='bold')
ax.set_ylabel('Error (sales units)', fontsize=10)
ax.set_title('Model Comparison: MAE and RMSE', fontsize=12, fontweight='bold',
             pad=10, loc='left')
ax.legend(fontsize=10)
ax.set_ylim(0, max(rmse_vals) + 1.5)
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#FAFAFA')
plt.tight_layout()
plt.savefig('output/Figure3_model_comparison.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 3 saved.")

# ── FIGURE 4: GLOBAL SHAP BAR CHART ──────────────────────────────────────────
tier_colors = {
    'rolling_mean_30': '#B71C1C', 'rolling_mean_7': '#B71C1C',
    'day_of_week': '#E65100',     'month': '#E65100', 'week_of_year': '#E65100',
    'lag_7': '#F9A825',           'lag_30': '#F9A825',
    'year':  '#A5D6A7',           'store': '#A5D6A7', 'item': '#A5D6A7',
    'is_weekend': '#CFD8DC',      'quarter': '#CFD8DC',
}
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
colors = [tier_colors.get(f, '#90A4AE') for f in mean_shap.index]
bars   = ax.barh(range(len(mean_shap)), mean_shap.values,
                 color=colors, edgecolor='white', height=0.65)
for bar, val in zip(bars, mean_shap.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
ax.set_yticks(range(len(mean_shap)))
ax.set_yticklabels(mean_shap.index, fontsize=10)
ax.set_xlabel('Mean |SHAP value|', fontsize=10)
ax.set_title('Global SHAP Feature Importance (XGBoost)', fontsize=12,
             fontweight='bold', pad=10, loc='left')
ax.invert_yaxis()
ax.grid(axis='x', linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#FAFAFA')

# Tier legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#B71C1C', label='Tier 1 — Dominant'),
    Patch(facecolor='#E65100', label='Tier 2 — Meaningful'),
    Patch(facecolor='#F9A825', label='Tier 3 — Marginal'),
    Patch(facecolor='#CFD8DC', label='Zero contribution'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right',
          framealpha=0.9, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig('output/Figure4_global_shap_bar.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 4 saved.")

# ── FIGURE 5: BEESWARM PLOT ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('white')
shap.plots.beeswarm(shap_values, max_display=12, show=False)
plt.title('SHAP Value Distribution by Feature', fontsize=12,
          fontweight='bold', pad=10, loc='left')
plt.tight_layout()
plt.savefig('output/Figure5_beeswarm.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Figure 5 saved.")

print("\n=== ALL DONE ===")
print("Files saved to output/:")
print("  global_model_comparison.csv  → update Table 1")
print("  global_shap_importance.csv   → update Table 2")
print("  Figure3_model_comparison.png → replace Figure 3 in paper")
print("  Figure4_global_shap_bar.png  → replace Figure 4 in paper")
print("  Figure5_beeswarm.png         → replace Figure 5 in paper")
