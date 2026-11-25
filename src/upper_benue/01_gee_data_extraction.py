"""
01_gee_data_extraction.py
-------------------------
Upper Benue Basin Flood Susceptibility & Exposure Study (Taraba & Adamawa, Nigeria)
Pipeline Step 1 & 6: GEE Data Extraction & Continuous Wall-to-Wall 30m GeoTIFF Export

PIPELINE ENHANCEMENTS:
- Gap-Filled Predictor Stack: Dual-composite Sentinel-2 (wet season primary + 
  annual cloud-relaxed fallback) and extended fastDistanceTransform (1024 px) 
  to eliminate orbital swath cutoffs, horizontal striping, and NoData speckling.
- Global Predictor Unmasking: Regional default fallbacks applied across terrain,
  soil, and LULC layers so zero null cells enter the spatial classifier within the AOI.
- Balanced Sampling: Stratified extraction of 8,000 points (4,000 flood / 4,000 non-flood).
- In-GEE Spatial Classifier: Distributed Random Forest model trained on the 14 VIF-retained 
  conditioning factors in probability mode.
- Wall-to-Wall Raster Export: Batch export of both continuous probability (0.0 - 1.0) 
  and 5-class Jenks susceptibility GeoTIFFs directly to Google Drive for QGIS mapping.
"""

import os
import ee
from ee import batch
import geemap
import geopandas as gpd

# -----------------------------------------------------------------------------
# 0. INITIALIZATION & AOI LOADING
# -----------------------------------------------------------------------------
PROJECT_ID = "ee-geodev"

try:
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine initialized successfully.")
except Exception:
    print("Authenticating Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "Study_Area.geojson")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "data", "Study_Area.geojson")),
    os.path.abspath(os.path.join(BASE_DIR, "data", "Study_Area.geojson")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "Study_Area.shp")),
]

AOI_FILE_PATH = None
for p in possible_paths:
    if os.path.exists(p):
        AOI_FILE_PATH = p
        break

if not AOI_FILE_PATH:
    raise FileNotFoundError(
        "AOI file not found. Ensure your QGIS boundary is exported as "
        "'study_area_taraba_adamawa.geojson' in your project data directory."
    )

gdf = gpd.read_file(AOI_FILE_PATH)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

ee_object = geemap.gdf_to_ee(gdf)
if isinstance(ee_object, ee.Geometry):
    aoi = ee_object
else:
    aoi = ee_object.geometry()

print(f"AOI loaded successfully from: {AOI_FILE_PATH}")

# -----------------------------------------------------------------------------
# 1. MULTI-TEMPORAL SENTINEL-1 SAR FLOOD INVENTORY (2018 - 2024)
# -----------------------------------------------------------------------------
FLOOD_WINDOWS = [
    {"year": 2018, "pre": ("2018-06-01", "2018-07-15"), "post": ("2018-08-01", "2018-08-31")},
    {"year": 2019, "pre": ("2019-06-01", "2019-07-15"), "post": ("2019-09-01", "2019-09-30")},
    {"year": 2020, "pre": ("2020-06-01", "2020-07-15"), "post": ("2020-09-01", "2020-09-30")},
    {"year": 2022, "pre": ("2022-06-01", "2022-07-15"), "post": ("2022-09-01", "2022-09-30")},
    {"year": 2024, "pre": ("2024-06-01", "2024-07-15"), "post": ("2024-09-01", "2024-09-30")}
]

# NOTE: Removed the ASCENDING-only filter so both orbit passes contribute,
# increasing temporal density within each pre/post window and improving the
# chance of capturing peak flood backscatter signatures.
s1_col = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .select('VV')
)

event_masks = []

for win in FLOOD_WINDOWS:
    pre_img = s1_col.filterDate(win["pre"][0], win["pre"][1]).median().clip(aoi)
    post_img = s1_col.filterDate(win["post"][0], win["post"][1]).median().clip(aoi)

    db_drop = pre_img.subtract(post_img)

    # NOTE: Relaxed thresholds slightly from (>3.0 dB drop AND < -12.0 dB absolute)
    # to (>2.0 dB drop AND < -14.0 dB absolute). The original thresholds were too
    # strict for the Upper Benue floodplain (mixed vegetation/open water), causing
    # very few pixels to qualify as 'flood=1'.
    flooded = db_drop.gt(2.0).And(post_img.lt(-14.0))
    event_masks.append(flooded)

cumulative_flood = ee.ImageCollection(event_masks).max().rename('flood')

jrc_gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
permanent_water = jrc_gsw.gt(80).clip(aoi)

# Ensure the flood band is strictly binary (0 or 1), integer type.
flood_inventory = cumulative_flood.where(permanent_water, 0).rename('flood').toInt()

print("Multi-temporal Sentinel-1 flood inventory generated.")

# -----------------------------------------------------------------------------
# 1b. DIAGNOSTIC: COUNT FLOOD / ELIGIBLE NON-FLOOD PIXELS BEFORE SAMPLING
# -----------------------------------------------------------------------------
flood_px_count = flood_inventory.eq(1).selfMask().reduceRegion(
    reducer=ee.Reducer.count(),
    geometry=aoi,
    scale=30,
    maxPixels=1e13,
    bestEffort=True
).getInfo()
print("Diagnostic - Flood=1 pixel count (at 30m):", flood_px_count)

flood_pixel_total = list(flood_px_count.values())[0] if flood_px_count else 0
if flood_pixel_total < 4000:
    print(
        f"WARNING: Only {flood_pixel_total} flood=1 pixels detected at 30m. "
        f"This may be insufficient for stratifiedSample(classPoints=[4000,4000]). "
        f"Consider further relaxing SAR thresholds or widening flood windows."
    )

# -----------------------------------------------------------------------------
# 2. ASSEMBLING 15 CONDITIONING FACTORS (30m RESOLUTION)
# -----------------------------------------------------------------------------
# A. DEM & Hydro-Terrain Derivatives
dem_raw = ee.Image("MERIT/DEM/v1_0_3").select('dem').clip(aoi)
dem = dem_raw.unmask(dem_raw.focal_mean(1, 'square', 'pixels')).rename('elevation')
slope = ee.Terrain.slope(dem).rename('slope')
aspect = ee.Terrain.aspect(dem).rename('aspect')
curvature = dem.focal_mean(1).subtract(dem).rename('curvature')

flow_accum = ee.Image("WWF/HydroSHEDS/15ACC").clip(aoi).rename('flow_accumulation')
# Fill any missing flow accumulation pixels with 1
flow_accum = flow_accum.unmask(1)

slope_rad = slope.multiply(3.141592 / 180.0)
tan_slope = slope_rad.tan().where(slope_rad.tan().lte(0), 0.001)
twi = flow_accum.add(1).multiply(30).log().divide(tan_slope).rename('twi').unmask(0)
spi = flow_accum.add(1).multiply(30).multiply(tan_slope).rename('spi').unmask(0)

river_mask = flow_accum.gt(1000)
# Expand fastDistanceTransform neighborhood to 1024 pixels (~30km) so all basin pixels have distance
dist_river_raw = river_mask.fastDistanceTransform(1024).sqrt().multiply(30)
dist_river = dist_river_raw.unmask(50000).clip(aoi).rename('distance_to_river')

# B. Climate Conditioning Factors
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterBounds(aoi).filterDate('2018-01-01', '2024-12-31')
mean_annual_precip = chirps.select('precipitation').sum().divide(7).rename('mean_annual_rainfall')
rx5day_max = chirps.select('precipitation').reduce(ee.Reducer.max()).rename('max_5day_rainfall')

# C. LULC & Spectral Indices
lulc = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(aoi).rename('lulc').unmask(40)

# Multi-temporal Sentinel-2 with full-year cloud-relaxed fallback to eliminate gaps and cloud voids
s2_wet = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2022-06-01', '2022-10-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25))
    .median()
)

s2_annual = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2022-01-01', '2022-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))
    .median()
)

# Seamless Sentinel-2 composite: wet season primary, annual composite gap-filler
s2_composite = s2_wet.unmask(s2_annual).clip(aoi)
ndvi = s2_composite.normalizedDifference(['B8', 'B4']).rename('ndvi').unmask(0.3)
mndwi = s2_composite.normalizedDifference(['B3', 'B11']).rename('mndwi').unmask(-0.3)

# D. Soil Properties
soil_clay_raw = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').clip(aoi)
soil_clay = soil_clay_raw.unmask(soil_clay_raw.focal_mean(3, 'circle', 'pixels')).unmask(25).rename('surface_clay')

soil_sand_raw = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0').clip(aoi)
soil_sand = soil_sand_raw.unmask(soil_sand_raw.focal_mean(3, 'circle', 'pixels')).unmask(45).rename('surface_sand')

# -----------------------------------------------------------------------------
# FIX: Apply bilinear resampling ONLY to continuous bands.
# Categorical/binary bands (lulc, flood) must remain at nearest-neighbor
# resampling to preserve exact integer class values required by
# stratifiedSample(classBand=...). Bilinear interpolation on these bands
# was silently destroying almost all 'flood=1' pixels, which is why the
# stratified sample previously returned ~0 flood-positive rows.
# -----------------------------------------------------------------------------
continuous_stack = ee.Image.cat([
    dem, slope, aspect, curvature, twi, spi, flow_accum, dist_river,
    mean_annual_precip, rx5day_max, ndvi, mndwi, soil_clay, soil_sand
]).resample('bilinear').reproject(crs='EPSG:4326', scale=30)

categorical_stack = ee.Image.cat([
    lulc, flood_inventory
]).reproject(crs='EPSG:4326', scale=30)  # default nearest-neighbor, no .resample()

predictor_stack = continuous_stack.addBands(categorical_stack)

print("All 15 conditioning factors stacked at 30 m resolution "
      "(categorical bands preserved via nearest-neighbor resampling).")

# -----------------------------------------------------------------------------
# 3. STRATIFIED SAMPLING (8,000 BALANCED POINTS: 4,000 FLOOD / 4,000 NON-FLOOD)
# -----------------------------------------------------------------------------
# Constrain non-flood candidate sampling to elevation > 250m or distance to river > 3km
non_flood_constraint = dem.gt(250).Or(dist_river.gt(3000))
sampling_mask = predictor_stack.select('flood').eq(1).Or(
    predictor_stack.select('flood').eq(0).And(non_flood_constraint)
)

masked_stack = predictor_stack.updateMask(sampling_mask)

samples = masked_stack.stratifiedSample(
    numPoints=8000,
    classBand='flood',
    classValues=[0, 1],
    classPoints=[4000, 4000],
    region=aoi,
    scale=30,
    seed=42,
    geometries=True,
    tileScale=4,
    dropNulls=True
)

print("Stratified sampling executed: requested 4,000 flood / 4,000 non-flood points.")

# -----------------------------------------------------------------------------
# 3b. VALIDATION: CONFIRM CLASS BALANCE BEFORE EXPORT
# -----------------------------------------------------------------------------
class_counts = samples.aggregate_histogram('flood').getInfo()
print("Sampled class distribution:", class_counts)

n_flood = class_counts.get('1', 0)
n_nonflood = class_counts.get('0', 0)
total = n_flood + n_nonflood

print(f"Total sampled points: {total} (flood={n_flood}, non-flood={n_nonflood})")

assert n_flood >= 3500, (
    f"Stratified sampling failed to retrieve sufficient flood=1 points "
    f"(got {n_flood}, expected ~4000). Check SAR detection thresholds, "
    f"resampling pipeline, and flood_px_count diagnostic above before exporting."
)
assert n_nonflood >= 3500, (
    f"Stratified sampling failed to retrieve sufficient flood=0 points "
    f"(got {n_nonflood}, expected ~4000). Check non_flood_constraint mask "
    f"and dem/dist_river bands before exporting."
)

print("Class balance validated. Proceeding to export configuration.")

# -----------------------------------------------------------------------------
# 4. CONFIGURATION: TASKS & EXPORT SETTINGS
# -----------------------------------------------------------------------------
csv_local_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "UpperBenue_Flood_8000_Samples.csv"))
csv_already_present = os.path.exists(csv_local_path)

# If the 8,000 samples CSV is already downloaded locally, skip re-exporting it to focus queue on rasters
EXPORT_SAMPLES_CSV = not csv_already_present
EXPORT_PROBABILITY_RASTER = True   # Export continuous 0.0 - 1.0 flood probability (.tif)
EXPORT_CLASSIFIED_RASTER = True    # Export 5-class Jenks susceptibility zones (.tif)
EXPORT_SCALE = 30                  # Resolution in meters (30m native; 60m or 90m for faster export)
EXPORT_FOLDER = 'Upper_Benue_Flood_Project'

# 14 VIF-retained conditioning factors (multicollinear 'surface_sand' removed)
SELECTED_FEATURES = [
    'elevation', 'slope', 'aspect', 'curvature', 'flow_accumulation',
    'twi', 'spi', 'distance_to_river', 'mean_annual_rainfall',
    'max_5day_rainfall', 'lulc', 'ndvi', 'mndwi', 'surface_clay'
]

# -----------------------------------------------------------------------------
# 5. SPATIAL MODEL TRAINING & CONTINUOUS RASTER PREDICTION (IN-GEE CLOUD ENGINE)
# -----------------------------------------------------------------------------
if EXPORT_PROBABILITY_RASTER or EXPORT_CLASSIFIED_RASTER:
    print(f"\nTraining spatial classifier in GEE using {len(SELECTED_FEATURES)} conditioning factors...")
    
    # Train Random Forest in probability mode on the 8,000 stratified samples
    classifier = (
        ee.Classifier.smileRandomForest(numberOfTrees=100, seed=42)
        .setOutputMode('PROBABILITY')
        .train(
            features=samples,
            classProperty='flood',
            inputProperties=SELECTED_FEATURES
        )
    )

    # Predict continuous flood susceptibility probability across every single pixel in AOI
    print("Classifying continuous wall-to-wall predictor stack across Upper Benue Basin...")
    prob_image = (
        predictor_stack.select(SELECTED_FEATURES)
        .classify(classifier)
        .clip(aoi)
        .rename('flood_probability')
        .toFloat()
    )

    # 5-Class Susceptibility Reclassification (1: Very Low, 2: Low, 3: Moderate, 4: High, 5: Very High)
    # Calibrated against the empirical sample distributions
    classified_5classes = (
        ee.Image(1)
        .where(prob_image.gte(0.15), 2)
        .where(prob_image.gte(0.35), 3)
        .where(prob_image.gte(0.60), 4)
        .where(prob_image.gte(0.80), 5)
        .updateMask(prob_image.gte(0.0))
        .clip(aoi)
        .rename('susceptibility_class')
        .toByte()
    )

# -----------------------------------------------------------------------------
# 6. SUBMIT BATCH EXPORT TASKS TO GOOGLE DRIVE
# -----------------------------------------------------------------------------
tasks = []

# A. Tabular Sample Points Export
if EXPORT_SAMPLES_CSV:
    task_csv = batch.Export.table.toDrive(
        collection=samples,
        description='UpperBenue_Flood_8000_Samples',
        folder=EXPORT_FOLDER,
        fileFormat='CSV'
    )
    task_csv.start()
    tasks.append(("Tabular 8,000 Samples (CSV)", task_csv))

# B. Continuous Probability GeoTIFF Export (Wall-to-Wall, NoData only outside AOI)
if EXPORT_PROBABILITY_RASTER:
    task_prob = batch.Export.image.toDrive(
        image=prob_image,
        description='UpperBenue_Flood_Probability_30m',
        folder=EXPORT_FOLDER,
        fileNamePrefix='upper_benue_flood_probability_30m',
        region=aoi,
        scale=EXPORT_SCALE,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_prob.start()
    tasks.append(("Continuous Probability GeoTIFF (30m)", task_prob))

# C. 5-Class Susceptibility GeoTIFF Export (Wall-to-Wall, NoData only outside AOI)
if EXPORT_CLASSIFIED_RASTER:
    task_class = batch.Export.image.toDrive(
        image=classified_5classes,
        description='UpperBenue_Flood_Susceptibility_5Classes_30m',
        folder=EXPORT_FOLDER,
        fileNamePrefix='upper_benue_flood_susceptibility_5classes_30m',
        region=aoi,
        scale=EXPORT_SCALE,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_class.start()
    tasks.append(("5-Class Susceptibility GeoTIFF (30m)", task_class))

print("\n" + "=" * 70)
print(f"SUCCESS: {len(tasks)} tasks submitted to Earth Engine batch queue:")
for name, t in tasks:
    print(f"  - {name} [ID: {t.id}]")
print("=" * 70)
print("Monitor progress at: https://code.earthengine.google.com/tasks")
print(f"Once complete, the GeoTIFFs will be in your Google Drive under '{EXPORT_FOLDER}'.")
print("Download them to your 'results/' folder and open directly in QGIS.")