from pathlib import Path
import os
import numpy as np
import pandas as pd
import jenkspy
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -----------------------------------------------------------------------------
# 1. SETUP & PATH RESOLUTION
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

csv_path = DATA_DIR / "UpperBenue_Flood_8000_Samples.csv"
df = pd.read_csv(csv_path)

y = df['flood']
drop_cols = ['system:index', '.geo', 'flood']
feature_cols = [col for col in df.columns if col not in drop_cols]
X = df[feature_cols]

# -----------------------------------------------------------------------------
# 2. VIF FEATURE SELECTION
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

# -----------------------------------------------------------------------------
# 3. BASELINE MODEL TRAINING
# -----------------------------------------------------------------------------
cat_model = CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.04,
    l2_leaf_reg=3,
    verbose=0,
    random_seed=42
)
cat_model.fit(X_selected, y)

base_probs = cat_model.predict_proba(X_selected)[:, 1]
breaks = jenkspy.jenks_breaks(base_probs, n_classes=5)
high_threshold = breaks[3]

pixel_area_km2 = 0.0009
base_high_area = np.sum(base_probs >= high_threshold) * pixel_area_km2

# -----------------------------------------------------------------------------
# 4. HYDRO-CALIBRATED CLIMATE SCALING
# -----------------------------------------------------------------------------
scenarios = {
    'SSP2-4.5 (Mid 2041-2060)': {'rain_delta': 0.085, 'rx5_delta': 0.122},
    'SSP2-4.5 (Far 2081-2100)': {'rain_delta': 0.134, 'rx5_delta': 0.178},
    'SSP5-8.5 (Mid 2041-2060)': {'rain_delta': 0.148, 'rx5_delta': 0.205},
    'SSP5-8.5 (Far 2081-2100)': {'rain_delta': 0.246, 'rx5_delta': 0.321}
}

# Floodplain mask: low elevation and low distance to river
floodplain_mask = (X_selected['elevation'] <= 250) & (X_selected['distance_to_river'] <= 3000)

results = []

for name, deltas in scenarios.items():
    X_fut = X_selected.copy()
    
    # Apply rainfall deltas proportional to local topographic wetness (TWI)
    twi_weight = (X_selected['twi'] - X_selected['twi'].min()) / (X_selected['twi'].max() - X_selected['twi'].min())
    
    if 'mean_annual_rainfall' in X_fut.columns:
        X_fut.loc[floodplain_mask, 'mean_annual_rainfall'] *= (1.0 + deltas['rain_delta'] * twi_weight[floodplain_mask])
    if 'max_5day_rainfall' in X_fut.columns:
        X_fut.loc[floodplain_mask, 'max_5day_rainfall'] *= (1.0 + deltas['rx5_delta'] * twi_weight[floodplain_mask])
        
    fut_probs = cat_model.predict_proba(X_fut)[:, 1]
    fut_high_area = np.sum(fut_probs >= high_threshold) * pixel_area_km2
    
    expansion_km2 = fut_high_area - base_high_area
    expansion_pct = (expansion_km2 / base_high_area) * 100
    
    results.append({
        'Scenario': name,
        'Baseline_High_Risk_km2': round(base_high_area, 2),
        'Future_High_Risk_km2': round(fut_high_area, 2),
        'Area_Expansion_km2': round(expansion_km2, 2),
        'Expansion_%': round(expansion_pct, 2)
    })

df_res = pd.DataFrame(results)
df_res.to_csv(RESULTS_DIR / "future_exposure_deltas.csv", index=False)

print("--- HYDRO-CALIBRATED CMIP6 PROJECTIONS ---")
print(df_res.to_string(index=False))