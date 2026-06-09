# Informal Settlement Classifier — Dhaka, Bangladesh

A supervised machine learning pipeline for classifying informal settlements across 203 wards in Dhaka Metropolitan Region using multi-source remote sensing and geospatial data.

---

## Project Overview

This project classifies Dhaka's 203 administrative wards as **formal** or **informal settlements** using an XGBoost classifier trained on 12 geospatial features derived from satellite imagery, population data, and OpenStreetMap building footprints. The final output is a ward-level informality probability map with risk tiers.

**Author:** Fahim Ahmed  
**Institution:** Department of Urban and Regional Planning, BUET  
**Study Area:** Dhaka Metropolitan Region (~960 km², ~12.5M population)  
**CRS:** EPSG:32646 (UTM Zone 46N)

---

## Key Results

| Metric | Baseline Model (8 features) | Extended Model (12 features) |
|--------|----------------------------|------------------------------|
| Test Accuracy | 84.85% | **90.91%** |
| CV F1 Score | 0.9444 ± 0.0237 | 0.9321 ± 0.0409 |
| ROC-AUC | 0.9654 | **0.9692** |
| Algorithm | XGBoost | XGBoost |

**Prediction Summary (Extended Model):**
- High informal risk: 95 wards (46.8%)
- Moderate informal risk: 2 wards (1.0%)
- Low informal risk: 12 wards (5.9%)
- Formal / planned: 94 wards (46.3%)
- Both models agree on 198/203 wards (97.5%)

---

## Data Sources

| Dataset | Source | Resolution | Purpose |
|---------|--------|-----------|---------|
| Landsat 9 OLI/TIRS (Path 137, Row 44, Feb 2022) | USGS | 30m | NDVI, SAVI, NDBI proxy, LST |
| ESA WorldCover 2021 | ESA | 10m | Built-up fraction |
| SRTM DEM | NASA | 30m | Slope |
| WorldPop BGD 2020 | WorldPop | 92m | Population density |
| GADM Level 4 Bangladesh | GADM | — | Ward boundaries |
| Sentinel-2 L1C (T46QBM, Dec 2021) | Copernicus | 10/20m | True NDBI, MNDWI |
| OSM Bangladesh Buildings | GeoFabrik | — | Building density, mean area |
| GHSL GHS-BUILT-S R2023A | JRC | 90m | Built-up surface (reference) |

---

## Features

### Baseline Features (8)
| Feature | Description | Source |
|---------|-------------|--------|
| ndvi_mean | Mean NDVI per ward | Landsat B4/B5 |
| savi_mean | Mean SAVI per ward | Landsat B4/B5 |
| ndbi_mean | NDBI proxy (−NDVI) | Landsat B4/B5 |
| lst_mean | Mean Land Surface Temperature (°C) | Landsat B10 |
| slope_mean | Mean slope (degrees) | SRTM DEM |
| pop_mean | Mean population density | WorldPop |
| pop_std | Population density std dev | WorldPop |
| built_fraction | Fraction of ward classified as built-up | ESA WorldCover |

### Extended Features (4 additional)
| Feature | Description | Source |
|---------|-------------|--------|
| s2_ndbi_mean | True NDBI = (B11−B08)/(B11+B08) | Sentinel-2 |
| s2_mndwi_mean | MNDWI = (B03−B11)/(B03+B11) | Sentinel-2 |
| osm_building_density | Buildings per km² | OSM |
| osm_mean_building_area | Mean building footprint area (m²) | OSM |

---

## Project Structure

```
informal-settlement-classifier/
│
├── notebooks/
│   ├── 01_data_prep.py              # Reproject and clip all rasters
│   ├── 02_feature_extraction.py     # Extract 8 features per ward
│   ├── 02b_extended_features.py     # Extract 4 additional features
│   ├── 03_labeling.py               # Rule-based + anchor labeling
│   ├── 04_model_training.py         # Baseline model (8 features)
│   ├── 04b_retrain.py               # Extended model (12 features)
│   ├── 05_prediction_mapping.py     # Baseline prediction maps
│   └── 05b_prediction_mapping.py    # Extended prediction maps
│
├── data/
│   ├── raw/
│   │   ├── landsat/scene_01_dhaka/  # Landsat 9 bands + MTL
│   │   ├── sentinel2/scene_01_dhaka/ # Sentinel-2 bands
│   │   ├── osm/                     # OSM building footprints
│   │   ├── ghsl/                    # GHSL built-up surface
│   │   ├── dem/                     # SRTM DEM
│   │   ├── lulc/                    # ESA WorldCover
│   │   ├── population/              # WorldPop
│   │   └── admin/                   # GADM boundaries
│   └── processed/
│       ├── dem_utm46n.tif
│       ├── lulc_dhaka_clipped.tif
│       ├── pop_dhaka_clipped.tif
│       ├── gadm_dhaka_l4.shp
│       ├── ward_features.csv
│       ├── ward_features_extended.csv
│       ├── ward_labels.csv
│       ├── ward_predictions.gpkg
│       └── ward_predictions_extended.gpkg
│
├── outputs/
│   ├── figures/                     # All maps and plots
│   └── model/                       # Saved models and reports
│
├── dhaka_boundary_utm.shp           # Study area boundary
├── informal_settlement_classifier.qgz # QGIS project
├── requirements.txt
└── README.md
```

---

## How to Run

### Prerequisites

```
pip install -r requirements.txt
```

### Run the full pipeline in order

```bash
cd informal-settlement-classifier

python notebooks/01_data_prep.py
python notebooks/02_feature_extraction.py
python notebooks/02b_extended_features.py
python notebooks/03_labeling.py
python notebooks/04_model_training.py
python notebooks/04b_retrain.py
python notebooks/05_prediction_mapping.py
python notebooks/05b_prediction_mapping.py
```

Each notebook prints a detailed log and saves its outputs before the next one begins.

---

## Labeling Methodology

163 out of 203 wards were labeled for training using a two-stage approach:

**Stage 1 — Ground truth anchors (49 wards)**
Hard-coded labels based on local knowledge and published slum studies:
- Informal: Lalbagh, Hazaribagh, Demra, Mohammadpur, Jatrabari, Kadamtali, Shyampur, Kotwali, Sutrapur, Lalbagh
- Formal: Gulshan, Uttara, Cantonment, Banani

**Stage 2 — Rule-based labeling (114 wards)**
- Informal if 2+ of: built_fraction > 0.85, ndvi < 0.15, pop > 800, lst > 31°C
- Formal if: built_fraction < 0.60, ndvi > 0.25, pop < 600

40 wards were left unlabeled and are predicted-only.

---

## Most Important Features (XGBoost)

1. NDVI (Landsat) — 34.7%
2. NDBI proxy — 15.1%
3. LST — 12.1%
4. SAVI — 10.0%
5. Built-up fraction — 8.2%
6. Population density — 5.1%
7. **OSM Building density** ★ — 4.1%
8. **OSM Mean building area** ★ — 2.7%
9. **S2 True NDBI** ★ — 2.7%

★ = new features added in extended model

---

## Known Limitations

- Sentinel-2 scene is L1C (top-of-atmosphere) rather than L2A (surface reflectance) — atmospheric correction would improve S2 index accuracy
- GHSL tile did not overlap with the Dhaka ward boundaries and was excluded from the final model
- Rule-based labeling has some misclassifications (e.g. dense formal areas like parts of Dhanmondi labeled informal by rules)
- OSM building coverage is incomplete in some rural fringe wards

---

## Output Maps

All maps are saved in `outputs/figures/`:

| File | Description |
|------|-------------|
| 05b_dashboard.png | **Main deliverable** — 4-panel map |
| 05b_probability_map.png | Informality probability choropleth |
| 05b_baseline_vs_extended_map.png | Side-by-side model comparison |
| 04b_feature_importance.png | Feature importance comparison |
| 04b_baseline_vs_extended.png | Accuracy comparison bar chart |

---

## Viewing in QGIS

1. Open QGIS → New Empty Project
2. Layer → Add Layer → Add Vector Layer
3. Select `data/processed/ward_predictions_extended.gpkg`
4. Right-click layer → Properties → Symbology
5. Set: Graduated, Value = `prob_informal`, Color ramp = RdYlGn (inverted), 5 classes, Natural Breaks
6. Click Classify → Apply → OK

---

## Dependencies

```
rasterio
geopandas
numpy
pandas
matplotlib
scikit-learn
scikit-image
xgboost
scipy
shapely
pyproj
folium
joblib
requests
```

---

## Citation

If you use this project, please cite the data sources:

- Landsat 9: U.S. Geological Survey. https://doi.org/10.5066/P9OGBGM6
- ESA WorldCover: Zanaga et al. (2021). ESA WorldCover 10m 2021 v200
- WorldPop: WorldPop (2020). Bangladesh 100m Population
- GADM: Database of Global Administrative Areas. https://gadm.org
- Sentinel-2: Copernicus Open Access Hub. European Space Agency
- OSM Buildings: OpenStreetMap contributors. GeoFabrik Bangladesh extract
- GHSL: Pesaresi et al. (2023). GHS-BUILT-S R2023A. European Commission JRC
