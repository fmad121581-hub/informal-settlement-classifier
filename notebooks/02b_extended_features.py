# =============================================================================
# NOTEBOOK 02b — EXTENDED FEATURE EXTRACTION (fixed version)
# Project : Informal Settlement Classifier — Dhaka, Bangladesh
# =============================================================================

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
import matplotlib.pyplot as plt

# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

TARGET_CRS   = "EPSG:32646"

FEATURES_CSV  = "data/processed/ward_features.csv"
WARDS_GPKG    = "data/processed/ward_features.gpkg"
GHSL_DIR      = "data/raw/ghsl/"
S2_DIR        = "data/raw/sentinel2/scene_01_dhaka/"
OSM_BUILDINGS = "data/raw/osm/gis_osm_buildings_a_free_1.shp"

OUT_CSV       = "data/processed/ward_features_extended.csv"
OUT_GPKG      = "data/processed/ward_features_extended.gpkg"
OUT_MAPS      = "outputs/figures/02b_new_features.png"
OUT_CORR      = "outputs/figures/02b_feature_correlation.png"

os.makedirs("data/processed",      exist_ok=True)
os.makedirs("data/processed/_tmp", exist_ok=True)
os.makedirs("outputs/figures",     exist_ok=True)

ORIGINAL_FEATURES = [
    "ndvi_mean", "savi_mean", "ndbi_mean", "lst_mean",
    "slope_mean", "pop_mean", "pop_std", "built_fraction"
]

NEW_FEATURES = [
    "ghsl_built_density",
    "s2_ndbi_mean",
    "s2_mndwi_mean",
    "osm_building_density",
    "osm_mean_building_area",
]


# =============================================================================
# 1.  LOAD EXISTING FEATURES AND WARD GEOMETRIES
# =============================================================================

print("="*60)
print("STEP 1 — Load existing features and ward geometries")
print("="*60)

df_orig = pd.read_csv(FEATURES_CSV)
print(f"  Existing features : {df_orig.shape}")

gdf = gpd.read_file(WARDS_GPKG)
print(f"  Ward GeoPackage   : {len(gdf)} features, CRS: {gdf.crs}")

for col in NEW_FEATURES:
    gdf[col] = np.nan


# =============================================================================
# 2.  HELPER FUNCTIONS
# =============================================================================

def zonal_stats_raster(raster_path, gdf, stat="mean", nodata=-9999):
    """Compute per-polygon statistic from a raster."""
    results = np.full(len(gdf), np.nan)
    with rasterio.open(raster_path) as src:
        for i, row in gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            try:
                out_image, _ = rasterio_mask(src, geom, crop=True,
                                             nodata=nodata)
                arr = out_image[0].astype(float)
                arr[arr == nodata] = np.nan
                arr = arr[np.isfinite(arr)]
                if len(arr) == 0:
                    continue
                if stat == "mean":
                    results[i] = np.nanmean(arr)
                elif stat == "sum":
                    results[i] = np.nansum(arr)
            except Exception:
                pass
    return results


def save_tif(array, path, ref_path, nodata=-9999):
    """Save 2D numpy array as float32 GeoTIFF."""
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
    # Force GeoTIFF — important when ref is JP2
    profile.update(
        driver  = "GTiff",
        dtype   = rasterio.float32,
        count   = 1,
        nodata  = nodata,
        compress= "lzw",
    )
    array = array.astype(np.float32)
    array[~np.isfinite(array)] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def reproject_to_utm(src_path, dst_path, target_crs=TARGET_CRS,
                     resampling=Resampling.bilinear):
    """Reproject a raster to target CRS and save as GeoTIFF."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs":      target_crs,
            "transform": transform,
            "width":    width,
            "height":   height,
            "driver":   "GTiff",   # always write as GeoTIFF
        })
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source        = rasterio.band(src, i),
                    destination   = rasterio.band(dst, i),
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = transform,
                    dst_crs       = target_crs,
                    resampling    = resampling,
                )
    return dst_path


# =============================================================================
# 3.  GHSL BUILT-UP DENSITY
# =============================================================================

print("\n" + "="*60)
print("STEP 2 — Extract GHSL built-up density per ward")
print("="*60)

ghsl_files = glob.glob(os.path.join(GHSL_DIR, "*.tif"))
if not ghsl_files:
    print("  WARNING: No GHSL .tif found — skipping.")
else:
    ghsl_raw = ghsl_files[0]
    print(f"  GHSL file : {os.path.basename(ghsl_raw)}")

    with rasterio.open(ghsl_raw) as src:
        print(f"  CRS: {src.crs}  res: {src.res}  nodata: {src.nodata}")
        ghsl_nodata = src.nodata if src.nodata is not None else -1

    # Reproject to UTM
    ghsl_utm = "data/processed/_tmp/ghsl_utm.tif"
    print("  Reprojecting GHSL to EPSG:32646 ...")
    reproject_to_utm(ghsl_raw, ghsl_utm, resampling=Resampling.bilinear)

    with rasterio.open(ghsl_utm) as src:
        print(f"  Reprojected — CRS: {src.crs}  res: {src.res[0]:.1f}m")
        # Replace nodata with -9999 for consistent handling
        data = src.read(1).astype(float)
        if ghsl_nodata is not None:
            data[data == ghsl_nodata] = np.nan
        data[data < 0] = np.nan

    # Save clean version
    ghsl_clean = "data/processed/_tmp/ghsl_utm_clean.tif"
    save_tif(data, ghsl_clean, ghsl_utm)

    print("  Computing GHSL zonal statistics ...")
    gdf["ghsl_built_density"] = zonal_stats_raster(
        ghsl_clean, gdf, stat="mean", nodata=-9999
    )

    valid = gdf["ghsl_built_density"].notna().sum()
    print(f"  Valid wards: {valid}/203")
    print(f"  GHSL density — mean: {gdf['ghsl_built_density'].mean():.2f}  "
          f"max: {gdf['ghsl_built_density'].max():.2f}")


# =============================================================================
# 4.  SENTINEL-2 NDBI AND MNDWI
# =============================================================================

print("\n" + "="*60)
print("STEP 3 — Compute Sentinel-2 NDBI and MNDWI")
print("="*60)

s2_band_paths = {
    "B03": os.path.join(S2_DIR, "S2_correct_B03.jp2"),
    "B08": os.path.join(S2_DIR, "S2_correct_B08.jp2"),
    "B11": os.path.join(S2_DIR, "S2_correct_B11.jp2"),
}

missing = [k for k, v in s2_band_paths.items() if not os.path.exists(v)]
if missing:
    print(f"  WARNING: Missing bands {missing} — skipping S2 features.")
else:
    print("  Reprojecting S2 bands to UTM ...")
    s2_utm = {}
    for band, path in s2_band_paths.items():
        out = f"data/processed/_tmp/s2_{band}_utm.tif"
        reproject_to_utm(path, out)
        s2_utm[band] = out
        print(f"    {band} done.")

    # Read B08 reference array
    with rasterio.open(s2_utm["B08"]) as src:
        b08         = src.read(1).astype(float)
        ref_profile = src.profile.copy()
        ref_transform = src.transform
        h, w        = src.height, src.width
    b08[b08 <= 0] = np.nan

    # Read B03
    with rasterio.open(s2_utm["B03"]) as src:
        b03 = src.read(1).astype(float)
    b03[b03 <= 0] = np.nan

    # Read B11 and resample to B08 shape
    b11 = np.zeros((h, w), dtype=float)
    with rasterio.open(s2_utm["B11"]) as src:
        reproject(
            source        = rasterio.band(src, 1),
            destination   = b11,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = ref_transform,
            dst_crs       = TARGET_CRS,
            resampling    = Resampling.bilinear,
        )
    b11[b11 <= 0] = np.nan

    print(f"  Arrays — B03:{b03.shape} B08:{b08.shape} B11:{b11.shape}")

    eps      = 1e-10
    s2_ndbi  = (b11 - b08) / (b11 + b08 + eps)
    s2_mndwi = (b03 - b11) / (b03 + b11 + eps)

    print(f"  NDBI  range: {np.nanmin(s2_ndbi):.3f} – {np.nanmax(s2_ndbi):.3f}")
    print(f"  MNDWI range: {np.nanmin(s2_mndwi):.3f} – {np.nanmax(s2_mndwi):.3f}")

    # Save as GeoTIFF (force GTiff driver)
    TMP_NDBI  = "data/processed/_tmp/s2_ndbi.tif"
    TMP_MNDWI = "data/processed/_tmp/s2_mndwi.tif"

    ref_profile.update(driver="GTiff", dtype=rasterio.float32,
                       count=1, nodata=-9999, compress="lzw")

    for arr, path in [(s2_ndbi, TMP_NDBI), (s2_mndwi, TMP_MNDWI)]:
        out = arr.astype(np.float32)
        out[~np.isfinite(out)] = -9999
        with rasterio.open(path, "w", **ref_profile) as dst:
            dst.write(out, 1)

    print("  Computing S2 zonal statistics (2-3 minutes) ...")
    gdf["s2_ndbi_mean"]  = zonal_stats_raster(TMP_NDBI,  gdf,
                                               stat="mean", nodata=-9999)
    gdf["s2_mndwi_mean"] = zonal_stats_raster(TMP_MNDWI, gdf,
                                               stat="mean", nodata=-9999)

    valid_ndbi  = gdf["s2_ndbi_mean"].notna().sum()
    valid_mndwi = gdf["s2_mndwi_mean"].notna().sum()
    print(f"  S2 NDBI  — valid: {valid_ndbi}/203  "
          f"mean: {gdf['s2_ndbi_mean'].mean():.3f}")
    print(f"  S2 MNDWI — valid: {valid_mndwi}/203  "
          f"mean: {gdf['s2_mndwi_mean'].mean():.3f}")


# =============================================================================
# 5.  OSM BUILDING FEATURES
# =============================================================================

print("\n" + "="*60)
print("STEP 4 — Extract OSM building features per ward")
print("="*60)

if not os.path.exists(OSM_BUILDINGS):
    print(f"  WARNING: OSM file not found — skipping.")
else:
    print("  Loading OSM buildings ...")

    # Read only buildings within Dhaka bounding box
    # This avoids loading all 11M Bangladesh buildings into memory
    # Dhaka bbox in WGS84: lon 90.2-90.6, lat 23.5-24.1
    dhaka_bbox = box(90.20, 23.50, 90.60, 24.10)

    print("  Reading only Dhaka-area buildings (bbox filter) ...")
    osm = gpd.read_file(
        OSM_BUILDINGS,
        bbox=(90.20, 23.50, 90.60, 24.10)   # pass bbox to read_file directly
    )
    print(f"  Buildings loaded: {len(osm)}")
    print(f"  CRS: {osm.crs}")

    # Reproject to UTM — now only Dhaka buildings, much smaller
    print("  Reprojecting to EPSG:32646 ...")
    osm = osm.to_crs(TARGET_CRS)

    # Compute area
    osm["area_m2"] = osm.geometry.area
    print(f"  Building area — mean: {osm['area_m2'].mean():.1f} m²  "
          f"max: {osm['area_m2'].max():.1f} m²")

    # Spatial join
    print("  Spatial join buildings → wards ...")
    joined = gpd.sjoin(
        osm[["geometry", "area_m2"]],
        gdf[["GID_4", "geometry"]],
        how="inner",
        predicate="within"
    )
    print(f"  Buildings matched: {len(joined)}")

    # Per-ward stats
    stats = joined.groupby("GID_4").agg(
        building_count=("area_m2", "count"),
        mean_area     =("area_m2", "mean"),
    ).reset_index()

    gdf["ward_area_km2"] = gdf.geometry.area / 1e6
    gdf = gdf.merge(stats, on="GID_4", how="left")

    gdf["osm_building_density"]   = (
        gdf["building_count"] / gdf["ward_area_km2"]
    ).fillna(0)
    gdf["osm_mean_building_area"] = gdf["mean_area"].fillna(0)

    print(f"  Building density — mean: {gdf['osm_building_density'].mean():.1f}  "
          f"max: {gdf['osm_building_density'].max():.1f} bldgs/km²")
    print(f"  Mean area        — mean: {gdf['osm_mean_building_area'].mean():.1f} m²")


# =============================================================================
# 6.  MERGE ORIGINAL FEATURES + FILL NaN
# =============================================================================

print("\n" + "="*60)
print("STEP 5 — Merge original features and fill missing values")
print("="*60)

df_slim = df_orig[["GID_4"] + ORIGINAL_FEATURES]
gdf = gdf.merge(df_slim, on="GID_4", how="left", suffixes=("", "_orig"))
for col in ORIGINAL_FEATURES:
    if f"{col}_orig" in gdf.columns:
        gdf[col] = gdf[f"{col}_orig"]
        gdf = gdf.drop(columns=[f"{col}_orig"])

all_features = ORIGINAL_FEATURES + NEW_FEATURES
existing     = [c for c in all_features if c in gdf.columns]

print("  NaN counts before fill:")
for col in existing:
    n = gdf[col].isna().sum()
    if n > 0:
        print(f"    {col:30s}: {n}")

for col in existing:
    gdf[col] = gdf[col].fillna(gdf[col].median())

print("  All NaN filled with column median.")


# =============================================================================
# 7.  SAVE
# =============================================================================

print("\n" + "="*60)
print("STEP 6 — Save extended feature table")
print("="*60)

id_cols   = ["GID_4", "NAME_4", "NAME_3", "NAME_2"]
save_cols = [c for c in id_cols + existing if c in gdf.columns]

gdf[save_cols].to_csv(OUT_CSV, index=False)
print(f"  [SAVED] {OUT_CSV}  ({len(gdf)} rows × {len(existing)} features)")

gdf.to_file(OUT_GPKG, driver="GPKG")
print(f"  [SAVED] {OUT_GPKG}")

print("\n  Preview:")
print(gdf[save_cols].head(5).to_string(index=False))


# =============================================================================
# 8.  PLOTS
# =============================================================================

print("\n" + "="*60)
print("STEP 7 — Generate plots")
print("="*60)

plot_cfg = [
    ("ghsl_built_density",    "GHSL Built-up Density",    "YlOrRd"),
    ("s2_ndbi_mean",          "S2 True NDBI",             "RdYlGn_r"),
    ("s2_mndwi_mean",         "S2 MNDWI (water)",        "Blues"),
    ("osm_building_density",  "OSM Building Density",     "Reds"),
    ("osm_mean_building_area","OSM Mean Building Area m²","YlGn"),
]

plot_cfg = [(c, t, m) for c, t, m in plot_cfg
            if c in gdf.columns and gdf[c].sum() > 0]

if plot_cfg:
    n   = len(plot_cfg)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 8))
    if n == 1:
        axes = [axes]
    fig.suptitle("Extended Features per Ward — Dhaka",
                 fontsize=13, fontweight="bold")
    for ax, (col, title, cmap) in zip(axes, plot_cfg):
        gdf.plot(column=col, ax=ax, cmap=cmap,
                 edgecolor="white", linewidth=0.3, legend=True,
                 legend_kwds={"shrink": 0.6})
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(OUT_MAPS, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [SAVED] {OUT_MAPS}")

# Correlation matrix
corr = gdf[existing].corr()
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(len(existing)))
ax.set_yticks(range(len(existing)))
ax.set_xticklabels(existing, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(existing, fontsize=8)
for i in range(len(existing)):
    for j in range(len(existing)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(corr.iloc[i,j]) > 0.6 else "black")
ax.set_title("Extended Feature Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_CORR, dpi=150, bbox_inches="tight")
plt.show()
print(f"  [SAVED] {OUT_CORR}")


# =============================================================================
# 9.  CLEANUP
# =============================================================================

import shutil
if os.path.exists("data/processed/_tmp"):
    shutil.rmtree("data/processed/_tmp")
    print("\n  Temp files cleaned up.")


# =============================================================================
# 10.  SUMMARY
# =============================================================================

print("\n" + "="*60)
print("NOTEBOOK 02b COMPLETE")
print("="*60)
print(f"\n  Total features : {len(existing)}")
print(f"  Original       : {len(ORIGINAL_FEATURES)}")
print(f"  New            : {len([c for c in NEW_FEATURES if c in gdf.columns])}")
print(f"\n  Feature stats:")
print(gdf[existing].describe().round(3).to_string())
print("\n  Output files:")
for label, path in [("CSV", OUT_CSV), ("GPKG", OUT_GPKG),
                    ("Maps", OUT_MAPS), ("Corr", OUT_CORR)]:
    status = "✓" if os.path.exists(path) else "✗"
    print(f"    {status}  {label:5s}  →  {path}")
print("\nReady to retrain — run Notebook 04b next.")
print("="*60)
