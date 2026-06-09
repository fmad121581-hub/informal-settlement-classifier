# =============================================================================
# NOTEBOOK 01 — DATA PREPARATION
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
# Author  : (your name)
# Date    : 2024
#
# PURPOSE
# -------
# This notebook does three things:
#   1. Verifies that every required input file exists on disk
#   2. Loads, reprojects, and clips all rasters to EPSG:32646 + Dhaka boundary
#   3. Loads, reprojects, and saves the ward shapefile to EPSG:32646
#
# After running this notebook every processed file will be in EPSG:32646
# and clipped to the Dhaka boundary — ready for feature extraction in NB 02.
#
# INPUT FILES (all relative to project root)
# -------------------------------------------
#   dhaka_boundary_utm.shp                       ← boundary, EPSG:32646
#   data/raw/dem/srtm_dhaka_30m.tif              ← raw DEM, any CRS
#   data/raw/lulc/ESA_WorldCover_10m_2021_*.tif  ← LULC, EPSG:4326
#   data/raw/population/BGD_ppp_2020_adj_v2.tif  ← population, any CRS
#   data/raw/admin/gadm41_BGD_4.shp              ← admin L4, EPSG:4326
#   data/raw/landsat/scene_01_dhaka/             ← B4, B5, ST_B10, MTL
#
# OUTPUT FILES
# ------------
#   data/processed/dem_utm46n.tif
#   data/processed/lulc_dhaka_clipped.tif
#   data/processed/pop_dhaka_clipped.tif
#   data/processed/gadm_dhaka_l4.shp
#   outputs/figures/01_data_overview.png
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import os
import glob

# ── Numerical / spatial ───────────────────────────────────────────────────────
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rasterio_mask
import geopandas as gpd
from shapely.geometry import box

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ── Suppress minor warnings that clutter the output ───────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*winding order.*")


# =============================================================================
# 0.  CONFIGURATION — edit paths here if your folder layout differs
# =============================================================================

# Target CRS for the entire project
TARGET_CRS = "EPSG:32646"

# ── Input paths ───────────────────────────────────────────────────────────────
BOUNDARY_UTM   = "dhaka_boundary_utm.shp"
RAW_DEM        = "data/raw/dem/srtm_dhaka_30m.tif"
RAW_LULC_GLOB  = "data/raw/lulc/ESA_WorldCover_10m_2021_*.tif"   # glob pattern
RAW_POP        = "data/raw/population/BGD_ppp_2020_adj_v2.tif"
RAW_ADMIN      = "data/raw/admin/gadm41_BGD_4.shp"
LANDSAT_DIR    = "data/raw/landsat/scene_01_dhaka/"

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DEM        = "data/processed/dem_utm46n.tif"
OUT_LULC       = "data/processed/lulc_dhaka_clipped.tif"
OUT_POP        = "data/processed/pop_dhaka_clipped.tif"
OUT_WARDS      = "data/processed/gadm_dhaka_l4.shp"
OUT_FIGURE     = "outputs/figures/01_data_overview.png"

# ── Create output directories if they don't exist yet ─────────────────────────
for folder in ["data/processed", "outputs/figures", "outputs/model"]:
    os.makedirs(folder, exist_ok=True)
    print(f"[OK] Directory ready: {folder}")


# =============================================================================
# 1.  FILE-EXISTENCE CHECKLIST
#     Stops early with a clear message if anything is missing.
# =============================================================================

print("\n" + "="*60)
print("STEP 1 — File existence check")
print("="*60)

def check_file(label, path):
    """Print a pass/fail line for a single file path."""
    exists = os.path.exists(path)
    status = "✓  FOUND" if exists else "✗  MISSING"
    print(f"  {status} | {label:25s} | {path}")
    return exists

# Resolve the LULC glob to an actual file path
lulc_matches = glob.glob(RAW_LULC_GLOB)
RAW_LULC = lulc_matches[0] if lulc_matches else RAW_LULC_GLOB  # keep pattern if missing

checks = {
    "Boundary (UTM)":   BOUNDARY_UTM,
    "DEM (raw)":        RAW_DEM,
    "LULC (raw)":       RAW_LULC,
    "Population (raw)": RAW_POP,
    "Admin L4 (raw)":   RAW_ADMIN,
    "Landsat dir":      LANDSAT_DIR,
}

all_ok = all(check_file(label, path) for label, path in checks.items())

# Check individual Landsat files
landsat_required = ["SR_B4.TIF", "SR_B5.TIF", "ST_B10.TIF", "MTL.txt"]
for suffix in landsat_required:
    matches = glob.glob(os.path.join(LANDSAT_DIR, f"*{suffix}"))
    found   = len(matches) > 0
    label   = f"Landsat {suffix}"
    status  = "✓  FOUND" if found else "✗  MISSING"
    print(f"  {status} | {label:25s} | {'(found)' if found else '(not found)'}")
    all_ok = all_ok and found

if not all_ok:
    raise FileNotFoundError(
        "\nOne or more required files are missing (see ✗ above). "
        "Copy them into the correct folders before continuing."
    )

print("\n[PASS] All required files found.\n")


# =============================================================================
# 2.  LOAD THE STUDY-AREA BOUNDARY
#     This polygon is used to clip every raster and the ward shapefile.
# =============================================================================

print("="*60)
print("STEP 2 — Load study-area boundary")
print("="*60)

boundary = gpd.read_file(BOUNDARY_UTM)

# Confirm CRS — must be EPSG:32646
if boundary.crs.to_epsg() != 32646:
    print(f"  WARNING: boundary CRS is {boundary.crs}. Reprojecting to {TARGET_CRS}...")
    boundary = boundary.to_crs(TARGET_CRS)
else:
    print(f"  CRS OK: {boundary.crs}")

print(f"  Shape:  {boundary.shape}")
print(f"  Bounds: {boundary.total_bounds.round(1)}")

# Extract geometry list — rasterio.mask expects a list of geometries
boundary_geom = [geom.__geo_interface__ for geom in boundary.geometry]


# =============================================================================
# 3.  HELPER FUNCTIONS
#     Two reusable functions used for every raster below.
# =============================================================================

def reproject_raster(src_path, dst_path, target_crs=TARGET_CRS,
                     resampling=Resampling.bilinear):
    """
    Reproject a raster to target_crs and save to dst_path.
    Returns the output path.

    Parameters
    ----------
    src_path   : str  — input raster file
    dst_path   : str  — output raster file
    target_crs : str  — e.g. "EPSG:32646"
    resampling : rasterio.warp.Resampling method
    """
    with rasterio.open(src_path) as src:
        # Calculate the new transform, width, height in the target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        # Copy metadata and update CRS / dimensions
        kwargs = src.meta.copy()
        kwargs.update({
            "crs":       target_crs,
            "transform": transform,
            "width":     width,
            "height":    height,
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source      = rasterio.band(src, band_idx),
                    destination = rasterio.band(dst, band_idx),
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = transform,
                    dst_crs       = target_crs,
                    resampling    = resampling,
                )
    return dst_path


def clip_raster_to_boundary(src_path, dst_path, geom_list, nodata=None):
    """
    Clip a raster to a polygon boundary and save to dst_path.
    The raster CRS and boundary CRS must already match.

    Parameters
    ----------
    src_path  : str  — input raster (must be in same CRS as boundary)
    dst_path  : str  — output clipped raster
    geom_list : list — list of geometry dicts (from __geo_interface__)
    nodata    : optional override for nodata value
    """
    with rasterio.open(src_path) as src:
        out_image, out_transform = rasterio_mask(
            src, geom_list, crop=True
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "height":    out_image.shape[1],
            "width":     out_image.shape[2],
            "transform": out_transform,
        })
        if nodata is not None:
            out_meta["nodata"] = nodata
        with rasterio.open(dst_path, "w", **out_meta) as dst:
            dst.write(out_image)
    return dst_path


def print_raster_info(label, path):
    """Print a short summary of a raster file."""
    with rasterio.open(path) as src:
        print(f"\n  [{label}]")
        print(f"    CRS:      {src.crs}")
        print(f"    Shape:    {src.height} x {src.width}")
        print(f"    Bands:    {src.count}")
        print(f"    Dtype:    {src.dtypes}")
        print(f"    Res (m):  {src.res[0]:.2f} x {src.res[1]:.2f}")
        print(f"    Bounds:   {[round(x,1) for x in src.bounds]}")
        print(f"    NoData:   {src.nodata}")


# =============================================================================
# 4.  PROCESS DEM
#     Raw DEM → reproject to EPSG:32646 → clip to boundary
# =============================================================================

print("\n" + "="*60)
print("STEP 3 — Process DEM")
print("="*60)

# Temporary reprojected DEM (will be overwritten by clipped version)
TMP_DEM = "data/processed/_tmp_dem_reproj.tif"

print("  Reprojecting DEM to EPSG:32646 ...")
reproject_raster(RAW_DEM, TMP_DEM, resampling=Resampling.bilinear)
print("  Clipping DEM to Dhaka boundary ...")
clip_raster_to_boundary(TMP_DEM, OUT_DEM, boundary_geom, nodata=-32767)
os.remove(TMP_DEM)   # clean up temp file

print_raster_info("DEM output", OUT_DEM)
print(f"\n  [SAVED] {OUT_DEM}")


# =============================================================================
# 5.  PROCESS LULC
#     Raw ESA WorldCover (EPSG:4326) → reproject → clip
# =============================================================================

print("\n" + "="*60)
print("STEP 4 — Process LULC (ESA WorldCover)")
print("="*60)

TMP_LULC = "data/processed/_tmp_lulc_reproj.tif"

print(f"  Source file: {RAW_LULC}")
print("  Reprojecting LULC to EPSG:32646 ...")
# Use nearest-neighbour for categorical land-cover data
reproject_raster(RAW_LULC, TMP_LULC, resampling=Resampling.nearest)
print("  Clipping LULC to Dhaka boundary ...")
clip_raster_to_boundary(TMP_LULC, OUT_LULC, boundary_geom)
os.remove(TMP_LULC)

print_raster_info("LULC output", OUT_LULC)
print(f"\n  [SAVED] {OUT_LULC}")


# =============================================================================
# 6.  PROCESS POPULATION
#     WorldPop raster → reproject → clip
# =============================================================================

print("\n" + "="*60)
print("STEP 5 — Process Population (WorldPop)")
print("="*60)

TMP_POP = "data/processed/_tmp_pop_reproj.tif"

print("  Reprojecting population raster to EPSG:32646 ...")
reproject_raster(RAW_POP, TMP_POP, resampling=Resampling.bilinear)
print("  Clipping population raster to Dhaka boundary ...")
clip_raster_to_boundary(TMP_POP, OUT_POP, boundary_geom, nodata=-99)
os.remove(TMP_POP)

print_raster_info("Population output", OUT_POP)
print(f"\n  [SAVED] {OUT_POP}")


# =============================================================================
# 7.  PROCESS WARD SHAPEFILE
#     gadm41_BGD_4 covers all of Bangladesh → filter to Dhaka district
#     → reproject to EPSG:32646 → save
# =============================================================================

print("\n" + "="*60)
print("STEP 6 — Process ward boundaries (GADM L4)")
print("="*60)

print("  Loading GADM L4 shapefile ...")
gdf_all = gpd.read_file(RAW_ADMIN)
print(f"  Full dataset: {len(gdf_all)} features, CRS: {gdf_all.crs}")
print(f"  Columns: {list(gdf_all.columns)}")

# ── Filter to Dhaka district only ─────────────────────────────────────────────
# NAME_2 contains the district name; 'Dhaka' covers the metropolitan region
dhaka_mask = gdf_all["NAME_2"].str.contains("Dhaka", case=False, na=False)
gdf_dhaka  = gdf_all[dhaka_mask].copy()
print(f"\n  After filtering to Dhaka district: {len(gdf_dhaka)} wards")

# If filter returns too few rows, print unique NAME_2 values to help diagnose
if len(gdf_dhaka) < 10:
    print("  WARNING: very few wards found. Unique NAME_2 values in dataset:")
    print(" ", gdf_all["NAME_2"].unique()[:20])

# ── Reproject to EPSG:32646 ───────────────────────────────────────────────────
print(f"  Reprojecting from {gdf_dhaka.crs} → {TARGET_CRS} ...")
gdf_dhaka = gdf_dhaka.to_crs(TARGET_CRS)

# ── Clip to boundary (removes any edge wards that spill outside) ───────────────
print("  Clipping wards to Dhaka boundary ...")
gdf_dhaka = gpd.clip(gdf_dhaka, boundary)

print(f"\n  Final ward count : {len(gdf_dhaka)}")
print(f"  CRS              : {gdf_dhaka.crs}")
print(f"  Bounds           : {gdf_dhaka.total_bounds.round(1)}")
print(f"  Columns          : {list(gdf_dhaka.columns)}")
print(f"\n  Sample ward names:")
print(gdf_dhaka[["NAME_4", "NAME_3", "NAME_2"]].head(10).to_string(index=False))

# Fix invalid winding order (pyogrio warning) before saving
gdf_dhaka["geometry"] = gdf_dhaka["geometry"].buffer(0)

gdf_dhaka.to_file(OUT_WARDS)
print(f"\n  [SAVED] {OUT_WARDS}")


# =============================================================================
# 8.  VERIFY LANDSAT FILES
#     We don't reproject Landsat here — that's done in NB 02 during
#     feature extraction. We just confirm files exist and print metadata.
# =============================================================================

print("\n" + "="*60)
print("STEP 7 — Verify Landsat files")
print("="*60)

landsat_bands = {
    "B4 (Red)":      glob.glob(os.path.join(LANDSAT_DIR, "*SR_B4.TIF"))[0],
    "B5 (NIR)":      glob.glob(os.path.join(LANDSAT_DIR, "*SR_B5.TIF"))[0],
    "B10 (Thermal)": glob.glob(os.path.join(LANDSAT_DIR, "*ST_B10.TIF"))[0],
}

for label, path in landsat_bands.items():
    print_raster_info(label, path)

# Print MTL scale factors we will use later for LST
mtl_path = glob.glob(os.path.join(LANDSAT_DIR, "*MTL.txt"))[0]
print(f"\n  MTL file: {mtl_path}")

# Parse the two key scale factors from the MTL text file
with open(mtl_path, "r") as f:
    mtl_text = f.read()

import re
mult = re.search(r"TEMPERATURE_MULT_BAND_ST_B10\s*=\s*([\d.E+-]+)", mtl_text)
add  = re.search(r"TEMPERATURE_ADD_BAND_ST_B10\s*=\s*([\d.E+-]+)",  mtl_text)

if mult and add:
    print(f"  LST scale factor (mult): {mult.group(1)}")
    print(f"  LST scale factor (add):  {add.group(1)}")
else:
    print("  WARNING: Could not parse LST scale factors from MTL file.")


# =============================================================================
# 9.  OVERVIEW MAP
#     Plot all four processed layers together to confirm spatial alignment.
# =============================================================================

print("\n" + "="*60)
print("STEP 8 — Generate overview map")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle(
    "Notebook 01 — Processed Data Overview\nDhaka Metropolitan Region · EPSG:32646",
    fontsize=14, fontweight="bold", y=0.98
)

# ── Panel 1: DEM ─────────────────────────────────────────────────────────────
ax = axes[0, 0]
with rasterio.open(OUT_DEM) as src:
    dem_data = src.read(1).astype(float)
    dem_data[dem_data == src.nodata] = np.nan
    extent = [src.bounds.left, src.bounds.right,
              src.bounds.bottom, src.bounds.top]

im1 = ax.imshow(dem_data, cmap="terrain", extent=extent, origin="upper")
boundary.boundary.plot(ax=ax, color="red", linewidth=1.5)
plt.colorbar(im1, ax=ax, label="Elevation (m)", shrink=0.8)
ax.set_title("DEM — SRTM 30m", fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

# ── Panel 2: LULC ────────────────────────────────────────────────────────────
ax = axes[0, 1]
with rasterio.open(OUT_LULC) as src:
    lulc_data = src.read(1).astype(float)
    extent    = [src.bounds.left, src.bounds.right,
                 src.bounds.bottom, src.bounds.top]

# ESA WorldCover class colours (simplified)
lulc_cmap = ListedColormap([
    "#006400",  # 10 Tree cover
    "#ffbb22",  # 20 Shrubland
    "#ffff4c",  # 30 Grassland
    "#f096ff",  # 40 Cropland
    "#fa0000",  # 50 Built-up
    "#b4b4b4",  # 60 Bare/sparse
    "#f0f0f0",  # 70 Snow/ice
    "#0064c8",  # 80 Water
    "#0096a0",  # 90 Wetland
    "#00cf75",  # 95 Mangroves
    "#fae6a0",  # 100 Moss/lichen
])

ax.imshow(lulc_data, cmap="tab20", extent=extent, origin="upper",
          vmin=10, vmax=100)
boundary.boundary.plot(ax=ax, color="red", linewidth=1.5)
ax.set_title("LULC — ESA WorldCover 10m", fontweight="bold")
ax.set_xlabel("Easting (m)")

# ── Panel 3: Population ───────────────────────────────────────────────────────
ax = axes[1, 0]
with rasterio.open(OUT_POP) as src:
    pop_data = src.read(1).astype(float)
    pop_data[pop_data == src.nodata] = np.nan
    extent   = [src.bounds.left, src.bounds.right,
                src.bounds.bottom, src.bounds.top]

im3 = ax.imshow(pop_data, cmap="hot_r", extent=extent, origin="upper")
boundary.boundary.plot(ax=ax, color="blue", linewidth=1.5)
plt.colorbar(im3, ax=ax, label="Population / pixel", shrink=0.8)
ax.set_title("Population Density — WorldPop 92m", fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

# ── Panel 4: Ward boundaries ─────────────────────────────────────────────────
ax = axes[1, 1]
gdf_wards = gpd.read_file(OUT_WARDS)
gdf_wards.plot(ax=ax, edgecolor="black", facecolor="lightyellow",
               linewidth=0.4)
boundary.boundary.plot(ax=ax, color="red", linewidth=2, label="Dhaka boundary")
ax.set_title(f"Ward Boundaries — GADM L4 ({len(gdf_wards)} wards)",
             fontweight="bold")
ax.set_xlabel("Easting (m)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUT_FIGURE, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  [SAVED] {OUT_FIGURE}")


# =============================================================================
# 10.  FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("NOTEBOOK 01 COMPLETE — Summary of processed outputs")
print("="*60)

outputs = {
    "DEM (UTM)":       OUT_DEM,
    "LULC (clipped)":  OUT_LULC,
    "Population":      OUT_POP,
    "Wards shapefile": OUT_WARDS,
    "Overview figure": OUT_FIGURE,
}

for label, path in outputs.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status}  {label:22s}  →  {path}")

print("\nAll outputs saved. You are ready to run Notebook 02 (Feature Extraction).")
print("="*60)
