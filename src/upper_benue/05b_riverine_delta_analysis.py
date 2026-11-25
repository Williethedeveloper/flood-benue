from pathlib import Path
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
drop_cols = ['system:index', '.geo', 'flood']
feature_cols = [col for col in df.columns if col not in drop_cols]
X = df[feature_cols]

# -----------------------------------------------------------------------------
# 2. VIF FEATURE SELECTION (14 CLEAN PREDICTORS)
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
# 3. TRAIN CATBOOST MODEL
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

# -----------------------------------------------------------------------------
# 4. STRATIFY BY RIVERINE FLOODPLAIN VS. UPLAND ZONES
# -----------------------------------------------------------------------------
# Riverine zone: elevation <= 200m AND distance_to_river <= 2000m
riverine_mask = (X_selected['elevation'] <= 200) & (X_selected['distance_to_river'] <= 2000)

print("--- DATASET STRATIFICATION ---")
print(f"Total samples: {len(df)}")
print(f"Riverine Floodplain points (elevation <= 200m, dist_to_river <= 2km): {riverine_mask.sum()}")
print(f"Upland / Non-riverine points: {(~riverine_mask).sum()}\n")

# -----------------------------------------------------------------------------
# 5. DELTA PROBABILITY ANALYSIS (delta P = P_future - P_baseline)
# -----------------------------------------------------------------------------
scenarios = {
    'SSP2-4.5 (Mid 2041-2060)': {'mean_rain_mult': 1.085, 'max5day_mult': 1.122},
    'SSP2-4.5 (Far 2081-2100)': {'mean_rain_mult': 1.134, 'max5day_mult': 1.178},
    'SSP5-8.5 (Mid 2041-2060)': {'mean_rain_mult': 1.148, 'max5day_mult': 1.205},
    'SSP5-8.5 (Far 2081-2100)': {'mean_rain_mult': 1.246, 'max5day_mult': 1.321}
}

for scen_name, deltas in scenarios.items():
    X_future = X_selected.copy()
    if 'mean_annual_rainfall' in X_future.columns:
        X_future['mean_annual_rainfall'] *= deltas['mean_rain_mult']
    if 'max_5day_rainfall' in X_future.columns:
        X_future['max_5day_rainfall'] *= deltas['max5day_mult']
        
    fut_probs = cat_model.predict_proba(X_future)[:, 1]
    delta_p = fut_probs - base_probs
    
    delta_p_riverine = delta_p[riverine_mask]
    delta_p_upland = delta_p[~riverine_mask]
    
    print(f"=== {scen_name} ===")
    print(f" Riverine Floodplain Mean Probability Shift (delta P): {delta_p_riverine.mean():+.4f}")
    print(f"   - % Points with Increase (delta P > 0): {(delta_p_riverine > 0).mean() * 100:.1f}%")
    print(f" Upland / Non-riverine Mean Probability Shift (delta P): {delta_p_upland.mean():+.4f}")
    print(f"   - % Points with Increase (delta P > 0): {(delta_p_upland > 0).mean() * 100:.1f}%\n")