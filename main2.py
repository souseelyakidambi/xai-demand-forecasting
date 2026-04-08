import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

# ── LOAD ────────────────────────────────────────────────────────────────────
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

# ── STUDY 1: SHAP BY STORE ───────────────────────────────────────────────────
print("Study 1: SHAP importance by store...")
store_shap = {}
for store_id in sorted(df['store'].unique()):
    sub = df[df['store'] == store_id]
    X, y = sub[features], sub['sales']
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1,
                             max_depth=6, random_state=42, verbosity=0)
    model.fit(X, y)
    sample = X.sample(min(500, len(X)), random_state=42)
    exp = shap.Explainer(model, X.sample(200, random_state=42))
    sv  = exp(sample)
    store_shap[store_id] = pd.Series(
        np.abs(sv.values).mean(axis=0), index=features)
    print(f"  Store {store_id} done")

store_shap_df = pd.DataFrame(store_shap).T
store_shap_df.to_csv('output/shap_by_store.csv')

# ── STUDY 2: SHAP BY ITEM CATEGORY ──────────────────────────────────────────
print("\nStudy 2: SHAP importance by item...")
item_shap = {}
for item_id in sorted(df['item'].unique()):
    sub = df[df['item'] == item_id]
    X, y = sub[features], sub['sales']
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1,
                             max_depth=6, random_state=42, verbosity=0)
    model.fit(X, y)
    sample = X.sample(min(500, len(X)), random_state=42)
    exp = shap.Explainer(model, X.sample(200, random_state=42))
    sv  = exp(sample)
    item_shap[item_id] = pd.Series(
        np.abs(sv.values).mean(axis=0), index=features)
    print(f"  Item {item_id} done")

item_shap_df = pd.DataFrame(item_shap).T
item_shap_df.to_csv('output/shap_by_item.csv')

# ── STUDY 3: ZERO-CONTRIBUTION FEATURES AUDIT ────────────────────────────────
print("\nStudy 3: Zero-contribution feature audit...")
zero_store = (store_shap_df < 0.01).sum()
zero_item  = (item_shap_df  < 0.01).sum()
audit = pd.DataFrame({
    'stores_near_zero': zero_store,
    'items_near_zero' : zero_item
})
audit.to_csv('output/zero_contribution_audit.csv')
print(audit)

# ── STUDY 4: ACCURACY BY STORE ───────────────────────────────────────────────
print("\nStudy 4: Accuracy by store...")
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

store_acc = {}
for store_id in sorted(df['store'].unique()):
    sub = df[df['store'] == store_id]
    X, y = sub[features], sub['sales']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1,
                             max_depth=6, random_state=42, verbosity=0)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    mape = np.mean(np.abs((yte - preds) / yte)) * 100
    mae  = mean_absolute_error(yte, preds)
    store_acc[store_id] = {'MAE': round(mae,4), 'MAPE': round(mape,4)}
    print(f"  Store {store_id}: MAE={mae:.4f}  MAPE={mape:.2f}%")

store_acc_df = pd.DataFrame(store_acc).T
store_acc_df.to_csv('output/accuracy_by_store.csv')

# ── PLOT 1: SHAP HEATMAP BY STORE ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(store_shap_df.T.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(store_shap_df.index)))
ax.set_xticklabels([f'Store {i}' for i in store_shap_df.index], fontsize=9)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=9)
ax.set_title('SHAP Feature Importance by Store\n(darker = more important)', fontsize=12, pad=12)
plt.colorbar(im, ax=ax, label='Mean |SHAP value|')
plt.tight_layout()
plt.savefig('output/shap_heatmap_store.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nStore heatmap saved.")

# ── PLOT 2: SHAP HEATMAP BY ITEM ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(item_shap_df.T.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(item_shap_df.index)))
ax.set_xticklabels([f'Item {i}' for i in item_shap_df.index], fontsize=8)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=9)
ax.set_title('SHAP Feature Importance by Item\n(darker = more important)', fontsize=12, pad=12)
plt.colorbar(im, ax=ax, label='Mean |SHAP value|')
plt.tight_layout()
plt.savefig('output/shap_heatmap_item.png', dpi=150, bbox_inches='tight')
plt.close()
print("Item heatmap saved.")

# ── PLOT 3: TOP FEATURE PER STORE (bar) ──────────────────────────────────────
top_feature_store = store_shap_df.idxmax(axis=1)
top_feature_item  = item_shap_df.idxmax(axis=1)

unique_feats = list(set(list(top_feature_store) + list(top_feature_item)))
color_map = {f: cm.tab10(i/10) for i, f in enumerate(features)}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, top_f, label in zip(axes,
                            [top_feature_store, top_feature_item],
                            ['By Store', 'By Item']):
    counts = top_f.value_counts()
    colors = [color_map[f] for f in counts.index]
    ax.bar(counts.index, counts.values, color=colors, edgecolor='white')
    ax.set_title(f'Most Important Feature — {label}', fontsize=11)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.savefig('output/top_feature_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Top feature distribution saved.")

# ── PLOT 4: MAPE VARIANCE ACROSS STORES ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(store_acc_df.index, store_acc_df['MAPE'],
       color='#2E75B6', edgecolor='white')
ax.axhline(store_acc_df['MAPE'].mean(), color='red',
           linestyle='--', linewidth=1.5, label=f"Mean MAPE = {store_acc_df['MAPE'].mean():.2f}%")
ax.set_title('Forecast Accuracy (MAPE) by Store', fontsize=12)
ax.set_xlabel('Store')
ax.set_ylabel('MAPE (%)')
ax.set_xticks(store_acc_df.index)
ax.set_xticklabels([f'Store {i}' for i in store_acc_df.index], rotation=15)
ax.legend()
plt.tight_layout()
plt.savefig('output/mape_by_store.png', dpi=150, bbox_inches='tight')
plt.close()
print("MAPE by store saved.")

print("\n=== ALL DONE ===")
print("Upload these files:")
print("  shap_by_store.csv, shap_by_item.csv")
print("  zero_contribution_audit.csv, accuracy_by_store.csv")
print("  shap_heatmap_store.png, shap_heatmap_item.png")
print("  top_feature_distribution.png, mape_by_store.png")