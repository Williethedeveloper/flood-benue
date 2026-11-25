from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import shap
from PyALE import ale

# -----------------------------------------------------------------------------
# 1. SETUP & PATH RESOLUTION
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

csv_path = DATA_DIR / "UpperBenue_Flood_8000_Samples.csv"
df = pd.read_csv(csv_path)

y = df['flood']

# Drop non-predictive metadata/spatial coordinates
drop_cols = ['system:index', '.geo', 'flood']
feature_cols = [col for col in df.columns if col not in drop_cols]
X = df[feature_cols]

# -----------------------------------------------------------------------------
# 2. VIF FILTERING (RETAIN 14 CLEAN PREDICTORS)
# -----------------------------------------------------------------------------
def calculate_vif(df_features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features)
    vif_df = pd.DataFrame()
    vif_df['Feature'] = df_features.columns
    vif_df['VIF'] = [
        variance_inflation_factor(scaled_data, i)
        for i in range(scaled_data.shape[1])
    ]
    return vif_df.sort_values(by='VIF', ascending=False)

vif_threshold = 10.0
X_vif = X.copy()

while True:
    vif_df = calculate_vif(X_vif)
    max_vif = vif_df['VIF'].iloc[0]
    if max_vif > vif_threshold:
        dropped_feature = vif_df['Feature'].iloc[0]
        X_vif = X_vif.drop(columns=[dropped_feature])
    else:
        break

X_selected = X_vif.copy()
print(f"Dataset ready: {len(X_selected)} samples with {X_selected.shape[1]} VIF-filtered features.")

# -----------------------------------------------------------------------------
# 3. TRAIN WINNING CATBOOST MODEL
# -----------------------------------------------------------------------------
print("\nFitting final CatBoost model...")
cat_model = CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.04,
    l2_leaf_reg=3,
    verbose=0,
    random_seed=42
)
cat_model.fit(X_selected, y)

# -----------------------------------------------------------------------------
# 4. SHAP GLOBAL & LOCAL INTERPRETATION
# -----------------------------------------------------------------------------
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(cat_model)
shap_values = explainer(X_selected)

# A. Export SHAP Summary Beeswarm Plot
plt.figure(figsize=(10, 8), dpi=300)
shap.summary_plot(shap_values, X_selected, show=False)
plt.title("SHAP Global Feature Importance (CatBoost Model)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved SHAP summary plot to 'results/fig_shap_summary.png'.")

# B. Export Feature Importance Bar Plot & Metrics Table
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
df_shap_imp = pd.DataFrame({
    'Feature': X_selected.columns,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

df_shap_imp.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)

plt.figure(figsize=(9, 6), dpi=300)
sns.barplot(data=df_shap_imp, x='Mean_Abs_SHAP', y='Feature', palette='crest')
plt.title("Mean Absolute SHAP Value (Global Impact)", fontsize=13)
plt.xlabel("Mean |SHAP value| (Average Impact on Model Output)", fontsize=11)
plt.ylabel("Conditioning Factor", fontsize=11)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_shap_bar.png", dpi=300)
plt.close()
print("Saved SHAP bar plot to 'results/fig_shap_bar.png'.")

# -----------------------------------------------------------------------------
# 5. ACCUMULATED LOCAL EFFECTS (ALE) PLOTS FOR TOP 6 PREDICTORS
# -----------------------------------------------------------------------------
print("\nGenerating Accumulated Local Effects (ALE) curves...")
top_6_features = df_shap_imp['Feature'].head(6).tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
axes = axes.flatten()

for idx, feat in enumerate(top_6_features):
    ax = axes[idx]
    # Calculate 1D ALE via PyALE
    ale_eff = ale(
        X=X_selected,
        model=cat_model,
        feature=[feat],
        grid_size=30,
        include_CI=True,
        plot=False
    )
    
    # Plot ALE line and confidence region
    x_vals = ale_eff.index
    y_vals = ale_eff['eff']
    ax.plot(x_vals, y_vals, color='#1f77b4', lw=2.5, label='ALE Main Effect')
    
    if 'lowerCI' in ale_eff.columns and 'upperCI' in ale_eff.columns:
        ax.fill_between(x_vals, ale_eff['lowerCI'], ale_eff['upperCI'], color='#1f77b4', alpha=0.2)
        
    ax.axhline(0, color='crimson', linestyle='--', alpha=0.7, lw=1)
    ax.set_title(f"ALE Curve: {feat}", fontsize=12, fontweight='bold')
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel("Accumulated Local Effect", fontsize=10)
    ax.grid(alpha=0.3)

plt.suptitle("Accumulated Local Effects (ALE) for Top 6 Flood Drivers", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_ale_top6.png", dpi=300, bbox_inches='tight')
plt.close()

print("Saved top-6 ALE curves grid to 'results/fig_ale_top6.png'.")
print("\nModel interpretability pipeline completed successfully!")