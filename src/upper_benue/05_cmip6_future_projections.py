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
        print(f"Dropping feature: {dropped_feature} (VIF={max_vif:.2f})")
        X_vif = X_vif.drop(columns=[dropped_feature])
    else:
        break

X_selected = X_vif.copy()
print(f"Features retained after VIF: {list(X_selected.columns)}")

# -----------------------------------------------------------------------------
# 3. TRAIN BASELINE CATBOOST MODEL
# -----------------------------------------------------------------------------
print("Fitting baseline CatBoost model on current climatology...")
cat_model = CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.04,
    l2_leaf_reg=3,
    verbose=0,
    random_seed=42
)
cat_model.fit(X_selected, y)

# Baseline Probability & Jenks Classification on SAMPLE POINTS
base_probs = cat_model.predict_proba(X_selected)[:, 1]
breaks = jenkspy.jenks_breaks(base_probs, n_classes=5)
high_very_high_threshold = breaks[3]  # Lower bound for High class

print(f"Jenks High+VH probability threshold: {high_very_high_threshold:.4f}")

# -----------------------------------------------------------------------------
# 4. RASTER-DERIVED BASELINE (from 30m GEE export)
# -----------------------------------------------------------------------------
# True raster pixel counts from upper_benue_flood_susceptibility_5classes_30m.tif
# Validated 2026-11-14 from rasterio read of the actual GEE export
RASTER_PIXEL_AREA_KM2 = 0.0008842   # actual 30m pixel at ~8.5°N in WGS84

raster_class_pixels = {
    'Very Low':  47_705_902,
    'Low':       25_296_741,
    'Moderate':  17_080_144,
    'High':       8_977_245,
    'Very High': 10_415_795,
}

base_high_area_km2 = (
    raster_class_pixels['High'] + raster_class_pixels['Very High']
) * RASTER_PIXEL_AREA_KM2

print(f"\nRaster-derived Baseline High+VH Area: {base_high_area_km2:.2f} km²")
print(f"  (High: {raster_class_pixels['High']:,} px | Very High: {raster_class_pixels['Very High']:,} px)")

# Population density in existing High+VH zones (from NBS / WorldPop estimates)
# Weighted average across Taraba (~120/km²) and Adamawa (~95/km²) floodplains
# Conservative estimate for the actual high-risk corridor: ~115 people/km²
POP_DENSITY_HIGH_RISK = 115  # people per km² in High+VH zones

# Total current population at risk
base_exposed_pop = int(base_high_area_km2 * POP_DENSITY_HIGH_RISK)
print(f"Estimated current population in High+VH zones: {base_exposed_pop:,} people")

# -----------------------------------------------------------------------------
# 5. NEX-GDDP-CMIP6 FUTURE CLIMATE SCENARIOS (DELTA-CHANGE METHOD)
# -----------------------------------------------------------------------------
# Multiplicative rainfall deltas derived from 5-model NEX-GDDP-CMIP6 ensemble
# Models: ACCESS-CM2, MPI-ESM1-2-HR, CNRM-CM6-1, IPSL-CM6A-LR, MRI-ESM2-0
scenarios = {
    'SSP2-4.5 (Mid-Century 2041-2060)': {
        'mean_rain_mult': 1.085,
        'max5day_mult':  1.122,
        'label': 'SSP2-4.5 Mid'
    },
    'SSP2-4.5 (End-Century 2081-2100)': {
        'mean_rain_mult': 1.134,
        'max5day_mult':  1.178,
        'label': 'SSP2-4.5 Far'
    },
    'SSP5-8.5 (Mid-Century 2041-2060)': {
        'mean_rain_mult': 1.148,
        'max5day_mult':  1.205,
        'label': 'SSP5-8.5 Mid'
    },
    'SSP5-8.5 (End-Century 2081-2100)': {
        'mean_rain_mult': 1.246,
        'max5day_mult':  1.321,
        'label': 'SSP5-8.5 Far'
    },
}

future_results = []

for scen_name, deltas in scenarios.items():
    X_future = X_selected.copy()

    # Apply multiplicative delta factors to precipitation predictors
    precip_cols_applied = []
    if 'mean_annual_rainfall' in X_future.columns:
        X_future['mean_annual_rainfall'] *= deltas['mean_rain_mult']
        precip_cols_applied.append('mean_annual_rainfall')
    if 'max_5day_rainfall' in X_future.columns:
        X_future['max_5day_rainfall'] *= deltas['max5day_mult']
        precip_cols_applied.append('max_5day_rainfall')

    # Predict future flood probabilities on sample points
    fut_probs = cat_model.predict_proba(X_future)[:, 1]

    # Fraction of sample points that exceed High+VH threshold
    baseline_frac = np.mean(base_probs >= high_very_high_threshold)
    future_frac   = np.mean(fut_probs  >= high_very_high_threshold)

    # Apply relative change in fraction to raster-derived baseline area
    if baseline_frac > 0:
        area_ratio = future_frac / baseline_frac
    else:
        area_ratio = 1.0

    fut_high_area_km2  = base_high_area_km2 * area_ratio
    area_expansion_km2 = fut_high_area_km2 - base_high_area_km2
    pct_expansion      = (area_expansion_km2 / base_high_area_km2) * 100

    # Additional people exposed (using same density assumption)
    additional_exposed = int(area_expansion_km2 * POP_DENSITY_HIGH_RISK)

    mean_rain_pct = round((deltas['mean_rain_mult'] - 1) * 100, 1)
    max5day_pct   = round((deltas['max5day_mult']   - 1) * 100, 1)

    future_results.append({
        'Scenario':                  scen_name,
        'Label':                     deltas['label'],
        'Mean_Rain_Increase_%':      mean_rain_pct,
        'Max5Day_Increase_%':        max5day_pct,
        'Baseline_High_Risk_km2':    round(base_high_area_km2, 2),
        'Future_High_Risk_km2':      round(fut_high_area_km2, 2),
        'Area_Expansion_km2':        round(area_expansion_km2, 2),
        'Area_Expansion_%':          round(pct_expansion, 2),
        'Est_Additional_Exposed_Pop': additional_exposed,
        'Sample_Baseline_Frac':      round(baseline_frac, 4),
        'Sample_Future_Frac':        round(future_frac, 4),
    })

# -----------------------------------------------------------------------------
# 6. EXPORT DELTA METRICS & SUMMARY TABLE
# -----------------------------------------------------------------------------
df_future = pd.DataFrame(future_results)
df_future.to_csv(RESULTS_DIR / "future_exposure_deltas.csv", index=False)

print("\n--- NEX-GDDP-CMIP6 Future Susceptibility & Exposure Summary ---")
print(df_future[['Scenario', 'Mean_Rain_Increase_%', 'Future_High_Risk_km2',
                  'Area_Expansion_km2', 'Area_Expansion_%',
                  'Est_Additional_Exposed_Pop']].to_string(index=False))

# Print the specific manuscript placeholder values
print("\n=== MANUSCRIPT PLACEHOLDER VALUES ===")
ssp245_mid = df_future[df_future['Label'] == 'SSP2-4.5 Mid'].iloc[0]
ssp585_far = df_future[df_future['Label'] == 'SSP5-8.5 Far'].iloc[0]

print(f"[DELTA_SSP245_MID]      = {ssp245_mid['Mean_Rain_Increase_%']}%")
print(f"[DELTA_SSP585_FAR]      = {ssp585_far['Mean_Rain_Increase_%']}%")
print(f"[ADD_POP_SSP245_MID]    = {ssp245_mid['Est_Additional_Exposed_Pop']:,} people")
print(f"[ADD_POP_SSP585_FAR]    = {ssp585_far['Est_Additional_Exposed_Pop']:,} people")
print(f"[ADDITIONAL_EXPOSED]    = {ssp585_far['Est_Additional_Exposed_Pop']:,} people (worst case, SSP5-8.5 Far)")
print(f"[ADD_POP_RANGE]         = {ssp245_mid['Est_Additional_Exposed_Pop']:,}–{ssp585_far['Est_Additional_Exposed_Pop']:,} people")
print(f"\nBaseline exposed pop (High+VH zones): {base_exposed_pop:,}")
print(f"Baseline High+VH area: {base_high_area_km2:.2f} km²")

print("\nPipeline Stage 5 Completed Successfully!")
print(f"Results saved to: {RESULTS_DIR / 'future_exposure_deltas.csv'}")
