# =============================================================================
# NOTEBOOK 02 — FEATURE EXTRACTION
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
# Author  : (your name)
# Date    : 2024
#
# PURPOSE
# -------
# Extract one row of features per ward from all processed rasters.
# Each ward ends up as a single row in a feature table (CSV + GeoPackage).
#
# FEATURES EXTRACTED
# ------------------
#   Spectral  : NDVI, NDBI, MNDWI, SAVI  (from Landsat B4/B5)
#   Thermal   : mean LST per ward         (from Landsat B10 + MTL)
#   Topo      : mean slope per ward       (derived from DEM)
#   Population: mean pop density per ward (WorldPop)
#   LULC      : fraction built-up per ward (ESA WorldCover class 50)
#
# INPUT FILES
# -----------
#   data/processed/gadm_dhaka_l4.shp
#   data/processed/dem_utm46n.tif
#   data/processed/lulc_dhaka_clipped.tif
#   data/processed/pop_dhaka_clipped.tif
#   data/raw/landsat/scene_01_dhaka/*SR_B4.TIF
#   data/raw/landsat/scene_01_dhaka/*SR_B5.TIF
#   data/raw/landsat/scene_01_dhaka/*ST_B10.TIF
#   data/raw/landsat/scene_01_dhaka/*MTL.txt
#
# OUTPUT FILES
# ------------
#   data/processed/ward_features.csv        ← feature table (no geometry)
#   data/processed/ward_features.gpkg       ← feature table WITH geometry
#   outputs/figures/02_feature_maps.png     ← one map per feature
#   outputs/figures/02_correlation_matrix.png
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import os
import glob
import re
import warnings
warnings.filterwarnings("ignore")

# ── Numerical ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Spatial ───────────────────────────────────────────────────────────────────
import rasterio
from rasterio.mask import mask as rasterio_mask
import geopandas as gpd
from scipy.ndimage import generic_filter

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

TARGET_CRS   = "EPSG:32646"
LANDSAT_DIR  = "data/raw/landsat/scene_01_dhaka/"

# ── Processed inputs ──────────────────────────────────────────────────────────
WARDS_PATH   = "data/processed/gadm_dhaka_l4.shp"
DEM_PATH     = "data/processed/dem_utm46n.tif"
LULC_PATH    = "data/processed/lulc_dhaka_clipped.tif"
POP_PATH     = "data/processed/pop_dhaka_clipped.tif"

# ── Outputs ───────────────────────────────────────────────────────────────────
OUT_CSV      = "data/processed/ward_features.csv"
OUT_GPKG     = "data/processed/ward_features.gpkg"
OUT_MAPS     = "outputs/figures/02_feature_maps.png"
OUT_CORR     = "outputs/figures/02_correlation_matrix.png"

os.makedirs("data/processed",   exist_ok=True)
os.makedirs("outputs/figures",  exist_ok=True)


# =============================================================================
# 1.  LOAD WARD BOUNDARIES
# =============================================================================

print("="*60)
print("STEP 1 — Load ward boundaries")
print("="*60)

wards = gpd.read_file(WARDS_PATH)
print(f"  CRS    : {wards.crs}")
print(f"  Wards  : {len(wards)}")
print(f"  Columns: {list(wards.columns)}")

# We will build features into this GeoDataFrame
# Keep only the columns we need for identification
wards = wards[["GID_4", "NAME_4", "NAME_3", "NAME_2", "geometry"]].copy()
wards = wards.reset_index(drop=True)


# =============================================================================
# 2.  HELPER — ZONAL STATISTICS
#     For every ward polygon, read the pixels inside it from a raster
#     and return a summary statistic (mean, std, or custom function).
# =============================================================================

def zonal_stats(raster_path, gdf, stat="mean", nodata=None, band=1):
    """
    Compute a per-polygon statistic from a raster.

    Parameters
    ----------
    raster_path : str   — path to the raster file
    gdf         : GeoDataFrame — polygons in the SAME CRS as the raster
    stat        : str   — "mean", "std", "median", "sum", or "count_nonzero"
    nodata      : float — override nodata value (None = use raster nodata)
    band        : int   — band index (1-based)

    Returns
    -------
    np.ndarray of shape (len(gdf),) — one value per polygon
    """
    results = np.full(len(gdf), np.nan)

    with rasterio.open(raster_path) as src:
        raster_nodata = nodata if nodata is not None else src.nodata

        for i, row in gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            try:
                # Crop raster to the bounding box of this ward
                out_image, _ = rasterio_mask(src, geom, crop=True,
                                             nodata=raster_nodata)
                arr = out_image[band - 1].astype(float)

                # Mask out nodata pixels
                if raster_nodata is not None:
                    arr = arr[arr != raster_nodata]

                # Remove remaining fill values (0 can be nodata for Landsat)
                arr = arr[np.isfinite(arr)]

                if len(arr) == 0:
                    continue   # no valid pixels — leave as NaN

                if stat == "mean":
                    results[i] = np.nanmean(arr)
                elif stat == "std":
                    results[i] = np.nanstd(arr)
                elif stat == "median":
                    results[i] = np.nanmedian(arr)
                elif stat == "sum":
                    results[i] = np.nansum(arr)
                elif stat == "count_nonzero":
                    results[i] = np.count_nonzero(arr)

            except Exception as e:
                # Ward might be too small or outside raster extent
                pass

    return results


# =============================================================================
# 3.  SPECTRAL INDICES — NDVI, NDBI, MNDWI, SAVI
#     Formula reference:
#       NDVI  = (NIR - Red)  / (NIR + Red)          range -1 to +1
#       NDBI  = (SWIR - NIR) / (SWIR + NIR)         range -1 to +1  *
#       MNDWI = (Green - SWIR) / (Green + SWIR)     range -1 to +1  *
#       SAVI  = ((NIR - Red) / (NIR + Red + L)) * (1 + L),  L = 0.5
#
#     * We only have B4 (Red) and B5 (NIR) from Landsat in the raw folder.
#       NDBI and MNDWI need SWIR (B6) and Green (B3) bands.
#       We derive NDVI and SAVI from B4/B5.
#       NDBI proxy = 1 - NDVI (inverse relationship, well established).
#       MNDWI will be added in NB 02b once Sentinel-2 is downloaded.
#
#     Landsat L2 DN → reflectance:
#       reflectance = DN * MULT + ADD   (from MTL file)
#       MULT = 2.75e-05,  ADD = -0.2    (confirmed from your MTL)
# =============================================================================

print("\n" + "="*60)
print("STEP 2 — Compute spectral indices (NDVI, SAVI, NDBI proxy)")
print("="*60)

# ── Parse scale factors from MTL ──────────────────────────────────────────────
mtl_path = glob.glob(os.path.join(LANDSAT_DIR, "*MTL.txt"))[0]
with open(mtl_path, "r") as f:
    mtl_text = f.read()

SR_MULT = float(re.search(r"REFLECTANCE_MULT_BAND_4\s*=\s*([0-9Ee.+-]+)",
                           mtl_text, re.IGNORECASE).group(1))
SR_ADD  = float(re.search(r"REFLECTANCE_ADD_BAND_4\s*=\s*([0-9Ee.+-]+)",
                           mtl_text, re.IGNORECASE).group(1))
print(f"  Landsat SR scale — MULT: {SR_MULT},  ADD: {SR_ADD}")
print(f"  Expected: MULT ≈ 2.75e-05,  ADD ≈ -0.2")
print(f"  Landsat SR scale — MULT: {SR_MULT},  ADD: {SR_ADD}")

# Landsat band paths
B4_PATH = glob.glob(os.path.join(LANDSAT_DIR, "*SR_B4.TIF"))[0]  # Red
B5_PATH = glob.glob(os.path.join(LANDSAT_DIR, "*SR_B5.TIF"))[0]  # NIR

# ── Read B4 and B5 as reflectance arrays (full scene) ────────────────────────
# We compute indices at pixel level, THEN do zonal stats per ward.
# This is more accurate than computing per-ward means of bands separately.

print("  Reading B4 (Red) and B5 (NIR) ...")
with rasterio.open(B4_PATH) as src:
    b4_dn    = src.read(1).astype(float)
    b4_nd    = src.nodata        # 0 for Landsat L2
    profile  = src.profile.copy()
    transform = src.transform
    crs       = src.crs

with rasterio.open(B5_PATH) as src:
    b5_dn = src.read(1).astype(float)

# Mask nodata (DN == 0 means fill in Landsat L2)
b4_dn[b4_dn == 0] = np.nan
b5_dn[b5_dn == 0] = np.nan

# Convert DN to surface reflectance
b4 = b4_dn * SR_MULT + SR_ADD   # Red
b5 = b5_dn * SR_MULT + SR_ADD   # NIR

# Clip reflectance to valid range [0, 1]
b4 = np.clip(b4, 0, 1)
b5 = np.clip(b5, 0, 1)

print(f"  B4 reflectance — min: {np.nanmin(b4):.4f}  max: {np.nanmax(b4):.4f}")
print(f"  B5 reflectance — min: {np.nanmin(b5):.4f}  max: {np.nanmax(b5):.4f}")

# ── Compute indices ───────────────────────────────────────────────────────────
# Small epsilon avoids division by zero
eps = 1e-10

# NDVI — Normalized Difference Vegetation Index
ndvi = (b5 - b4) / (b5 + b4 + eps)

# SAVI — Soil Adjusted Vegetation Index (L = 0.5)
L    = 0.5
savi = ((b5 - b4) / (b5 + b4 + L + eps)) * (1 + L)

# NDBI proxy — higher values = more built-up
# True NDBI needs SWIR; proxy = -NDVI (inverse, well-correlated)
ndbi_proxy = -ndvi

print(f"  NDVI  — min: {np.nanmin(ndvi):.3f}  max: {np.nanmax(ndvi):.3f}")
print(f"  SAVI  — min: {np.nanmin(savi):.3f}  max: {np.nanmax(savi):.3f}")
print(f"  NDBI* — min: {np.nanmin(ndbi_proxy):.3f}  max: {np.nanmax(ndbi_proxy):.3f}")

# ── Save index rasters to temp files for zonal stats ─────────────────────────
# We write single-band float32 GeoTIFFs so zonal_stats() can read them

def save_index_raster(array, path, ref_profile, ref_transform, ref_crs):
    """Save a 2D numpy array as a single-band float32 GeoTIFF."""
    p = ref_profile.copy()
    p.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype(np.float32), 1)

TMP_NDVI = "data/processed/_tmp_ndvi.tif"
TMP_SAVI = "data/processed/_tmp_savi.tif"
TMP_NDBI = "data/processed/_tmp_ndbi.tif"

save_index_raster(ndvi,       TMP_NDVI, profile, transform, crs)
save_index_raster(savi,       TMP_SAVI, profile, transform, crs)
save_index_raster(ndbi_proxy, TMP_NDBI, profile, transform, crs)
print("  Index rasters saved to temp files.")

# ── Zonal stats — mean index per ward ────────────────────────────────────────
# Wards are in EPSG:32646, Landsat is also EPSG:32646 — no reprojection needed
print("  Computing zonal statistics (this may take 1–2 minutes) ...")

wards["ndvi_mean"] = zonal_stats(TMP_NDVI, wards, stat="mean", nodata=np.nan)
wards["savi_mean"] = zonal_stats(TMP_SAVI, wards, stat="mean", nodata=np.nan)
wards["ndbi_mean"] = zonal_stats(TMP_NDBI, wards, stat="mean", nodata=np.nan)

print(f"  NDVI  per ward — mean: {wards['ndvi_mean'].mean():.3f}")
print(f"  SAVI  per ward — mean: {wards['savi_mean'].mean():.3f}")
print(f"  NDBI* per ward — mean: {wards['ndbi_mean'].mean():.3f}")


# =============================================================================
# 4.  LAND SURFACE TEMPERATURE (LST)
#     Landsat L2 ST band (ST_B10) stores temperature as scaled integer.
#     Formula:  LST_kelvin = DN * MULT + ADD
#               LST_celsius = LST_kelvin - 273.15
#     Scale factors from MTL (confirmed): MULT=0.00341802, ADD=149.0
# =============================================================================

print("\n" + "="*60)
print("STEP 3 — Compute Land Surface Temperature (LST)")
print("="*60)

# Parse LST scale factors
LST_MULT = float(re.search(r"TEMPERATURE_MULT_BAND_ST_B10\s*=\s*([\d.E+-]+)",
                            mtl_text).group(1))
LST_ADD  = float(re.search(r"TEMPERATURE_ADD_BAND_ST_B10\s*=\s*([\d.E+-]+)",
                            mtl_text).group(1))
print(f"  LST scale — MULT: {LST_MULT},  ADD: {LST_ADD}")

B10_PATH = glob.glob(os.path.join(LANDSAT_DIR, "*ST_B10.TIF"))[0]

with rasterio.open(B10_PATH) as src:
    b10_dn    = src.read(1).astype(float)
    b10_nd    = src.nodata    # 0 for fill pixels
    b10_prof  = src.profile.copy()

b10_dn[b10_dn == 0] = np.nan   # mask fill
lst_kelvin  = b10_dn * LST_MULT + LST_ADD
lst_celsius = lst_kelvin - 273.15

print(f"  LST range — min: {np.nanmin(lst_celsius):.1f}°C  "
      f"max: {np.nanmax(lst_celsius):.1f}°C")

# Save LST raster
TMP_LST = "data/processed/_tmp_lst.tif"
save_index_raster(lst_celsius, TMP_LST, b10_prof, b10_prof["transform"], crs)

# Zonal stats
wards["lst_mean"] = zonal_stats(TMP_LST, wards, stat="mean", nodata=np.nan)
print(f"  LST per ward — mean: {wards['lst_mean'].mean():.1f}°C  "
      f"std: {wards['lst_mean'].std():.1f}°C")


# =============================================================================
# 5.  TOPOGRAPHIC SLOPE
#     Slope is derived from the DEM using a simple 3×3 Sobel-based gradient.
#     Formula: slope_degrees = arctan(sqrt(dz/dx² + dz/dy²))
#     where dz/dx and dz/dy are computed from neighbouring pixels.
# =============================================================================

print("\n" + "="*60)
print("STEP 4 — Compute slope from DEM")
print("="*60)

with rasterio.open(DEM_PATH) as src:
    dem_arr  = src.read(1).astype(float)
    dem_nd   = src.nodata        # -32767
    dem_res  = src.res[0]        # pixel size in metres (~29.6 m)
    dem_prof = src.profile.copy()

# Replace nodata with NaN
dem_arr[dem_arr == dem_nd] = np.nan

print(f"  DEM resolution: {dem_res:.1f} m")
print(f"  DEM range     : {np.nanmin(dem_arr):.1f} – {np.nanmax(dem_arr):.1f} m")

# ── Compute slope using numpy gradient ───────────────────────────────────────
# np.gradient returns [dz/dy, dz/dx] — note row order (y first)
dz_dy, dz_dx = np.gradient(dem_arr, dem_res, dem_res)
slope_rad     = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
slope_deg     = np.degrees(slope_rad)

print(f"  Slope range   : {np.nanmin(slope_deg):.2f}° – {np.nanmax(slope_deg):.2f}°")

# Save slope raster
TMP_SLOPE = "data/processed/_tmp_slope.tif"
save_index_raster(slope_deg, TMP_SLOPE, dem_prof,
                  dem_prof["transform"], dem_prof["crs"])

# Zonal stats
wards["slope_mean"] = zonal_stats(TMP_SLOPE, wards, stat="mean", nodata=np.nan)
print(f"  Slope per ward — mean: {wards['slope_mean'].mean():.2f}°  "
      f"max: {wards['slope_mean'].max():.2f}°")


# =============================================================================
# 6.  POPULATION DENSITY
#     WorldPop gives estimated population count per pixel (~90m resolution).
#     We compute mean population per pixel per ward.
#     (Total population would require summing and knowing pixel area.)
# =============================================================================

print("\n" + "="*60)
print("STEP 5 — Extract population density per ward")
print("="*60)

with rasterio.open(POP_PATH) as src:
    print(f"  Population raster — CRS: {src.crs}, res: {src.res}")

wards["pop_mean"] = zonal_stats(POP_PATH, wards, stat="mean", nodata=-99.0)
wards["pop_std"]  = zonal_stats(POP_PATH, wards, stat="std",  nodata=-99.0)

# Extra safety — replace any remaining negative values with NaN
wards.loc[wards["pop_mean"] < 0, "pop_mean"] = np.nan
wards.loc[wards["pop_std"]  < 0, "pop_std"]  = np.nan

print(f"  Pop density per ward — mean: {wards['pop_mean'].mean():.1f}  "
      f"max: {wards['pop_mean'].max():.1f}")


# =============================================================================
# 7.  BUILT-UP FRACTION FROM LULC
#     ESA WorldCover class 50 = Built-up.
#     We compute: fraction_built = (pixels with value 50) / (total pixels)
#     This gives a 0–1 score per ward.
# =============================================================================

print("\n" + "="*60)
print("STEP 6 — Compute built-up fraction from LULC (ESA WorldCover)")
print("="*60)

LULC_BUILT_CLASS = 50   # ESA WorldCover built-up class value

built_fractions = np.full(len(wards), np.nan)

with rasterio.open(LULC_PATH) as src:
    print(f"  LULC raster — CRS: {src.crs}, res: {src.res}")

    for i, row in wards.iterrows():
        geom = [row.geometry.__geo_interface__]
        try:
            out_image, _ = rasterio_mask(src, geom, crop=True, nodata=0)
            arr = out_image[0]

            # Total valid pixels (exclude nodata=0 which is also ocean)
            # ESA WorldCover nodata is 0; land classes start at 10
            valid_pixels = np.sum(arr > 0)
            built_pixels = np.sum(arr == LULC_BUILT_CLASS)

            if valid_pixels > 0:
                built_fractions[i] = built_pixels / valid_pixels

        except Exception:
            pass

wards["built_fraction"] = built_fractions

print(f"  Built fraction per ward — "
      f"mean: {wards['built_fraction'].mean():.3f}  "
      f"max: {wards['built_fraction'].max():.3f}")


# =============================================================================
# 8.  CLEAN UP TEMPORARY FILES
# =============================================================================

for tmp in [TMP_NDVI, TMP_SAVI, TMP_NDBI, TMP_LST, TMP_SLOPE]:
    if os.path.exists(tmp):
        os.remove(tmp)
print("\n  Temporary rasters removed.")


# =============================================================================
# 9.  HANDLE MISSING VALUES
#     Some small wards at the boundary may have NaN for some features.
#     We fill with the column median — a safe choice for skewed distributions.
# =============================================================================

print("\n" + "="*60)
print("STEP 7 — Handle missing values")
print("="*60)

feature_cols = ["ndvi_mean", "savi_mean", "ndbi_mean",
                "lst_mean", "slope_mean",
                "pop_mean", "pop_std",
                "built_fraction"]

print("  NaN counts before fill:")
for col in feature_cols:
    n = wards[col].isna().sum()
    print(f"    {col:20s}: {n} NaN")

# Fill with column median
for col in feature_cols:
    median_val = wards[col].median()
    wards[col] = wards[col].fillna(median_val)

print("\n  NaN counts after fill:")
for col in feature_cols:
    n = wards[col].isna().sum()
    print(f"    {col:20s}: {n} NaN")


# =============================================================================
# 10.  SAVE FEATURE TABLE
# =============================================================================

print("\n" + "="*60)
print("STEP 8 — Save feature table")
print("="*60)

# CSV (no geometry — for model training)
csv_cols = ["GID_4", "NAME_4", "NAME_3", "NAME_2"] + feature_cols
wards[csv_cols].to_csv(OUT_CSV, index=False)
print(f"  [SAVED] {OUT_CSV}  ({len(wards)} rows × {len(feature_cols)} features)")

# GeoPackage (with geometry — for mapping)
wards.to_file(OUT_GPKG, driver="GPKG")
print(f"  [SAVED] {OUT_GPKG}")

# Preview the feature table
print("\n  Feature table preview (first 5 rows):")
print(wards[csv_cols].head().to_string(index=False))


# =============================================================================
# 11.  FEATURE MAPS — one choropleth per feature
# =============================================================================

print("\n" + "="*60)
print("STEP 9 — Generate feature maps")
print("="*60)

# Feature display settings: (column, title, colormap)
map_features = [
    ("ndvi_mean",     "Mean NDVI\n(vegetation)",          "RdYlGn"),
    ("ndbi_mean",     "Mean NDBI proxy\n(built-up)",      "RdYlBu_r"),
    ("lst_mean",      "Mean LST (°C)\n(thermal)",         "hot_r"),
    ("slope_mean",    "Mean Slope (°)\n(topography)",     "YlOrBr"),
    ("pop_mean",      "Mean Pop Density\n(WorldPop)",     "YlOrRd"),
    ("built_fraction","Built-up Fraction\n(ESA LULC)",   "Reds"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    "Notebook 02 — Extracted Features per Ward\nDhaka Metropolitan Region",
    fontsize=14, fontweight="bold", y=1.01
)

for ax, (col, title, cmap) in zip(axes.flatten(), map_features):
    wards.plot(
        column     = col,
        ax         = ax,
        cmap       = cmap,
        legend     = True,
        edgecolor  = "grey",
        linewidth  = 0.3,
        legend_kwds= {"shrink": 0.7, "label": col}
    )
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Easting (m)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(OUT_MAPS, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_MAPS}")


# =============================================================================
# 12.  CORRELATION MATRIX
#      Helps spot multicollinearity before model training.
# =============================================================================

print("\n  Generating correlation matrix ...")

corr = wards[feature_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label="Pearson r")

ax.set_xticks(range(len(feature_cols)))
ax.set_yticks(range(len(feature_cols)))
ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(feature_cols, fontsize=9)

# Annotate each cell with the correlation value
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")

ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_CORR, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_CORR}")


# =============================================================================
# 13.  FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("NOTEBOOK 02 COMPLETE — Feature extraction summary")
print("="*60)

print(f"\n  Wards processed : {len(wards)}")
print(f"  Features extracted: {len(feature_cols)}")
print(f"\n  Feature statistics:")
print(wards[feature_cols].describe().round(3).to_string())

print("\n  Output files:")
for label, path in [("Feature CSV",  OUT_CSV),
                     ("Feature GPKG", OUT_GPKG),
                     ("Feature maps", OUT_MAPS),
                     ("Corr matrix",  OUT_CORR)]:
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"    {status}  {label:15s}  →  {path}")

print("\nReady for Notebook 03 (Labeling).")
print("="*60)
