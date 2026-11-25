from pathlib import Path
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
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

# Extract coordinates from .geo JSON string for spatial raster construction
import json
coords = df['.geo'].apply(lambda g: json.loads(g)['coordinates'])
lons = np.array([c[0] for c in coords])
lats = np.array([c[1] for c in coords])

# -----------------------------------------------------------------------------
# 2. VIF FEATURE SELECTION & MODEL TRAINING
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

X_vif = X.copy()
while True:
    vif_df = calculate_vif(X_vif)
    max_vif = vif_df['VIF'].iloc[0]
    if max_vif > 10.0:
        dropped_feature = vif_df['Feature'].iloc[0]
        X_vif = X_vif.drop(columns=[dropped_feature])
    else:
        break

X_selected = X_vif.copy()

print("Training CatBoost model...")
cat_model = CatBoostClassifier(
    iterations=800, depth=8, learning_rate=0.04, l2_leaf_reg=3, verbose=0, random_seed=42
)
cat_model.fit(X_selected, y)

# Predict continuous flood probability (0.0 to 1.0)
y_probs = cat_model.predict_proba(X_selected)[:, 1]

# Reclassify into 5 Jenks Natural Breaks classes (1: Very Low, 2: Low, 3: Moderate, 4: High, 5: Very High)
breaks = jenkspy.jenks_breaks(y_probs, n_classes=5)
classified = np.digitize(y_probs, bins=breaks[1:-1]) + 1

import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import rasterio.features

# -----------------------------------------------------------------------------
# 3. VECTOR LAYER EXPORT FOR QGIS (POINT CLOUD WITH SUSCEPTIBILITY CLASSES)
# -----------------------------------------------------------------------------
print("Exporting QGIS-ready vector point layer...")

class_labels = {1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very High'}
class_names = [class_labels[c] for c in classified]

geometry = [Point(xy) for xy in zip(lons, lats)]
gdf_points = gpd.GeoDataFrame({
    'flood_observed': y.values,
    'flood_probability': np.round(y_probs, 4),
    'susceptibility_class': classified,
    'class_name': class_names
}, geometry=geometry, crs='EPSG:4326')

gpkg_path = RESULTS_DIR / "upper_benue_flood_susceptibility_points.gpkg"
gdf_points.to_file(gpkg_path, driver='GPKG', layer='flood_susceptibility')
print(f"Exported Vector Points: {gpkg_path}")

# -----------------------------------------------------------------------------
# 4. CONTINUOUS SPATIAL RASTER INTERPOLATION & GEOTIFF EXPORT
# -----------------------------------------------------------------------------
print("\nGenerating continuous susceptibility raster surface for QGIS...")

# Load AOI boundary for clipping and grid definition
aoi_path = DATA_DIR / "Study_Area.geojson"
gdf_aoi = gpd.read_file(aoi_path)
if gdf_aoi.crs != "EPSG:4326":
    gdf_aoi = gdf_aoi.to_crs("EPSG:4326")

bounds = gdf_aoi.total_bounds  # [minx, miny, maxx, maxy]
min_x, min_y, max_x, max_y = bounds

# Grid resolution (~500m / 0.005 degrees in WGS84)
res = 0.005
ncols = int(np.ceil((max_x - min_x) / res))
nrows = int(np.ceil((max_y - min_y) / res))

# Grid cell coordinates (pixel centers)
x_coords = np.linspace(min_x + res / 2, max_x - res / 2, ncols)
y_coords = np.linspace(max_y - res / 2, min_y + res / 2, nrows)
grid_x, grid_y = np.meshgrid(x_coords, y_coords)
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

# Fast Inverse Distance Weighting (IDW) interpolation using KD-Tree
print("Interpolating continuous probability surface via spatial KD-Tree...")
tree = cKDTree(np.column_stack([lons, lats]))
k_neighbors = 12
distances, indices = tree.query(grid_points, k=k_neighbors)

# Prevent division by zero for exact match points
distances = np.maximum(distances, 1e-6)
weights = 1.0 / (distances ** 2.0)
weights_sum = weights.sum(axis=1, keepdims=True)
normalized_weights = weights / weights_sum

interpolated_probs = np.sum(normalized_weights * y_probs[indices], axis=1)
prob_grid = interpolated_probs.reshape((nrows, ncols)).astype(np.float32)

# Classify interpolated surface into 5 Jenks classes
class_grid = (np.digitize(prob_grid, bins=breaks[1:-1]) + 1).astype(np.int16)

# Mask pixels outside the study boundary polygon
transform = from_origin(min_x, max_y, res, res)
mask = rasterio.features.geometry_mask(
    gdf_aoi.geometry,
    out_shape=(nrows, ncols),
    transform=transform,
    invert=True  # True inside polygon, False outside
)

prob_grid[~mask] = -9999.0
class_grid[~mask] = -9999

# A. Export Continuous Probability Raster (.tif)
prob_tif_path = RESULTS_DIR / "upper_benue_flood_probability.tif"
with rasterio.open(
    prob_tif_path, 'w',
    driver='GTiff',
    height=nrows,
    width=ncols,
    count=1,
    dtype='float32',
    crs='EPSG:4326',
    transform=transform,
    nodata=-9999.0,
    compress='deflate'
) as dst:
    dst.write(prob_grid, 1)

print(f"Exported Continuous Probability Raster: {prob_tif_path}")

# B. Export Classified 5-Class Susceptibility Raster (.tif)
class_tif_path = RESULTS_DIR / "upper_benue_flood_susceptibility_5classes.tif"
with rasterio.open(
    class_tif_path, 'w',
    driver='GTiff',
    height=nrows,
    width=ncols,
    count=1,
    dtype='int16',
    crs='EPSG:4326',
    transform=transform,
    nodata=-9999,
    compress='deflate'
) as dst:
    dst.write(class_grid, 1)

print(f"Exported Classified 5-Class Raster: {class_tif_path}")
print("\n" + "=" * 70)
print("SUCCESS: Layers ready for QGIS!")
print(f"1. Continuous Heatmap: {prob_tif_path.name}")
print(f"2. 5 Susceptibility Zones: {class_tif_path.name}")
print(f"3. Sample Points (CatBoost predictions): {gpkg_path.name}")
print("=" * 70)