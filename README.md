# Explainable GeoAI Framework for Transboundary Dam-Driven Flood Susceptibility and Future Exposure Mapping in the Upper Benue Basin, Nigeria

[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-managed-blueviolet.svg)](https://docs.astral.sh/uv/)
[![Target Journal](https://img.shields.io/badge/Journal-Natural%20Hazards%20(Springer)-green.svg)]()

Reproducible geospatial machine learning and explainable AI (XAI) pipeline for flood susceptibility mapping, population exposure assessment, and CMIP6 climate-scenario projections across **Taraba and Adamawa States (Upper Benue Basin, Nigeria)**.

**Authors:**
- **Temitope Benedict Waheeb**¹\* (*Bherney Spatial Consult, Ibadan; Department of Geographic Information Systems, University of Ibadan, Nigeria*) — `waheebtemitope@gmail.com`
- **Williams Omotayo Adaji-Agbane**² (*Anadata Technologies, Lagos; TETFund Center of Excellence in Computational Intelligence Research, University of Uyo, Nigeria*) — `williamsadaji2017@gmail.com`

---

## Overview

Flooding in the Upper Benue Basin is driven by the compounding effects of intense seasonal Sudan-Guinea savanna rainfall and transboundary water releases from the **Lagdo Dam** in northern Cameroon. This repository implements a five-phase GeoAI workflow integrating **Google Earth Engine (GEE)**, **Python** (`CatBoost`, `XGBoost`, `TabNet`, `SHAP`, `PyALE`), and **QGIS**:

1. **Multi-temporal SAR Flood Inventory (2018–2024):** Built from Sentinel-1 GRD ascending VV backscatter across five major flood events (August 2018, September 2019, September 2020, September 2022, September 2024).
2. **Conditioning Factor Assembly & VIF Filtering:** 15 candidate predictors at 30 m resolution (MERIT Hydro DEM, HydroSHEDS, CHIRPS v2.0, ESA WorldCover, Sentinel-2 SR, ISRIC SoilGrids). Multicollinearity screening (`VIF > 10`) drops `surface_sand`, retaining **14 predictors**.
3. **Model Benchmarking (5-Fold Stratified CV):**
   - **CatBoost:** AUC = `0.9520`, Accuracy = `0.8750`, F1 = `0.8714`
   - **XGBoost:** AUC = `0.9517`, Accuracy = `0.8751`, F1 = `0.8718`
   - **TabNet:** AUC = `0.9357`, Accuracy = `0.8481`, F1 = `0.8463`
4. **Explainable AI (SHAP + ALE):** Identifies `distance_to_river`, `elevation`, `twi`, `lulc`, `slope`, and `surface_clay` as dominant drivers, revealing sharp susceptibility thresholds below ~250 m elevation and within ~3.0 km of mainstem channels.
5. **Susceptibility Mapping, Exposure & CMIP6 Projections:**
   - **Baseline High + Very High susceptibility:** `17,139.68 km²` (`17.71%` of the `96,771.54 km²` basin), exposing `404,700` residents across focal hotspot LGAs (Yola North, Ibi, Numan, Ardo Kola) and `~1.97M` basin-wide.
   - **NEX-GDDP-CMIP6 (5-model ensemble):** Projects High + Very High susceptibility expansion of `+8.50%` (`+1,456.87 km²`; `+167,000` exposed) under **SSP2-4.5 Mid (2041–2060)** up to `+24.60%` (`+4,216.36 km²`; `+484,000` exposed) under **SSP5-8.5 Far (2081–2100)**.

---

## Repository Structure

```text
Upper_Benue/
├── data/
│   ├── UpperBenue_Flood_8000_Samples.csv   # 8,000 stratified flood/non-flood samples (15 predictors)
│   ├── Study_Area.geojson                  # Upper Benue Basin boundary (EPSG:32632)
│   └── NGA_rvrsl_1m_esri.*                 # Nigeria river network shapefile
├── src/upper_benue/
│   ├── 01_gee_data_extraction.py           # GEE SAR flood inventory & conditioning factor extraction
│   ├── 02_model_training_and_vif.py        # VIF filtering, correlation matrix, 5-fold CV & ROC curves
│   ├── 03_shap_and_ale_analysis.py         # Global/local SHAP & Accumulated Local Effects (ALE)
│   ├── 04_spatial_prediction_and_exposure.py # State/LGA susceptibility & GHS-POP/NBS exposure tables
│   ├── 05_cmip6_future_projections.py      # NEX-GDDP-CMIP6 scenario delta-change analysis
│   ├── 05b_riverine_delta_analysis.py      # Riverine sensitivity diagnostics
│   ├── 05c_hydro_calibrated_cmip6.py       # Hydrologically calibrated CMIP6 sensitivity run
│   ├── 06_export_geotiff_for_qgis.py       # Export 30 m probability & 5-class GeoTIFFs for QGIS
│   └── 08_fig8_cmip6_future_susceptibility.py # Generate multi-panel CMIP6 projection figure (Fig. 8)
├── results/
│   ├── model_metrics.csv                   # Cross-validated benchmark metrics (Table 2)
│   ├── vif.csv                             # Retained predictors & VIF scores
│   ├── shap_importance.csv                 # Mean |SHAP| feature rankings
│   ├── exposure_by_state.csv               # State-level susceptibility & area totals (Table 3)
│   ├── exposure_by_lga.csv                 # Focal LGA exposure breakdown (Table 3)
│   ├── exposure_by_class.csv               # Five-class basin area breakdown
│   ├── future_exposure_deltas.csv          # CMIP6 scenario deltas
│   └── fig_*.png / fig8_*.png              # Publication figures
├── pyproject.toml                          # Project metadata & Python dependencies
├── uv.lock                                 # Locked dependency versions
└── .gitignore                              # Excludes .venv/, catboost_info/, *.qgz, and large *.tif rasters
```

> **Note on Large Rasters:** Full 30 m GeoTIFF outputs in `results/` (`upper_benue_flood_probability*.tif`, `upper_benue_flood_susceptibility_5classes*.tif`, ~1.9 GB total) are excluded via `.gitignore` and can be regenerated via GEE / `06_export_geotiff_for_qgis.py`.

---

## Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) with Python `>=3.11`.

```bash
# Clone the repository
git clone https://github.com/Williethedeveloper/flood-benue.git
cd flood-benue

# Create virtual environment and install locked dependencies
uv sync
```

*(If running GEE extraction scripts `01_gee_data_extraction.py` or `06_export_geotiff_for_qgis.py`, authenticate Earth Engine first with `earthengine authenticate`.)*

---

## Running the Pipeline

Execute scripts in order using the project virtual environment:

```bash
# 1. (Optional) Extract training samples from Google Earth Engine
uv run python src/upper_benue/01_gee_data_extraction.py

# 2. VIF filtering, correlation heatmap, 5-fold CV benchmark & ROC curves (Fig. 4)
uv run python src/upper_benue/02_model_training_and_vif.py

# 3. SHAP & ALE explainability plots (Figs. 5 & 6)
uv run python src/upper_benue/03_shap_and_ale_analysis.py

# 4. Spatial susceptibility statistics & population exposure (Table 3)
uv run python src/upper_benue/04_spatial_prediction_and_exposure.py

# 5. CMIP6 future projections & multi-panel Figure 8
uv run python src/upper_benue/05_cmip6_future_projections.py
uv run python src/upper_benue/08_fig8_cmip6_future_susceptibility.py
```

---

## Open Data Sources

| Dataset | Use | Resolution |
|---|---|---|
| **Copernicus Sentinel-1 GRD** | Multi-temporal SAR flood inventory (2018–2024) | 10 m → 30 m |
| **MERIT Hydro DEM / HydroSHEDS** | Elevation, slope, aspect, curvature, TWI, SPI, flow accumulation, river distance | 30 m |
| **CHIRPS v2.0** | Mean annual rainfall & Rx5day (1991–2020 climatology) | ~5.5 km → 30 m |
| **ESA WorldCover v200 (2021)** | Land Use / Land Cover (LULC) | 10 m → 30 m |
| **Sentinel-2 SR Level-2A** | NDVI & MNDWI wet-season composites | 10 m → 30 m |
| **ISRIC SoilGrids v2.0** | Surface clay & sand content | 250 m → 30 m |
| **NASA NEX-GDDP-CMIP6** | 5-GCM ensemble (SSP2-4.5 & SSP5-8.5; 2041–2060 & 2081–2100) | 0.25° |
| **GHS-POP 2020 & NBS** | Gridded & administrative population exposure | 100 m |
