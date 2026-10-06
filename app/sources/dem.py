"""Terrain from USGS 3DEP LiDAR (US, 1-3 m source, read at 5 m) with the Copernicus DEM
GLO-30 (public COGs on AWS) as the worldwide fallback.

What we compute on the grid for a field (buffered by 300 m so depressions that
drain across the boundary are seen):
  - elevation range and mean slope
  - depression depth: fill the DEM (Planchon-Darboux) and subtract the original
  - relative elevation: cell height minus the mean of a ~270 m neighbourhood
  - share of the field area sitting in depressions (> 0.15 m on LiDAR, > 0.3 m on the
    30 m DSM) and in the lowest decile

The 30 m DSM with 1.5-4 m vertical error is only good for "which way does it drain" and
"where are the broad low spots". LiDAR at 5 m resolves potholes and their depth; a design
survey still wants the 1 m product or RTK.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping

from ..geo import Field

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

THREEDEP_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
THREEDEP_CELL_M = 5.0      # LiDAR resampled to 5 m: sharp enough for potholes, small enough to fill quickly
THREEDEP_MAX_PX = 700


def _read_3dep(bounds: tuple[float, float, float, float], lat: float) -> tuple[np.ndarray, Any, str] | None:
    """USGS 3DEP seamless DEM (best available source, mostly 1 m LiDAR in the Corn Belt) as a
    GeoTIFF window in WGS84 at ~5 m. Returns None outside CONUS or on any service problem."""
    import httpx
    from rasterio.io import MemoryFile

    if not (-125.5 <= bounds[0] and bounds[2] <= -66.5 and 24.0 <= bounds[1] and bounds[3] <= 49.5):
        return None
    minx, miny, maxx, maxy = bounds
    w_m = (maxx - minx) * 111_320.0 * math.cos(math.radians(lat))
    h_m = (maxy - miny) * 111_320.0
    w = int(min(THREEDEP_MAX_PX, max(40, round(w_m / THREEDEP_CELL_M))))
    h = int(min(THREEDEP_MAX_PX, max(40, round(h_m / THREEDEP_CELL_M))))
    params = {"bbox": f"{minx},{miny},{maxx},{maxy}", "bboxSR": "4326", "imageSR": "4326", "size": f"{w},{h}",
              "format": "tiff", "pixelType": "F32", "noDataInterpretation": "esriNoDataMatchAny",
              "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    r = httpx.get(THREEDEP_URL, params=params, timeout=90)
    if r.status_code != 200 or not r.content.startswith((b"II*\x00", b"MM\x00*")):
        return None
    with MemoryFile(r.content) as mem, mem.open() as ds:
        z = ds.read(1).astype("float64")
        tr = ds.transform
        nodata = ds.nodata
    if nodata is not None:
        z[z == nodata] = np.nan
    z[(np.abs(z) > 1e5) | (z < -500)] = np.nan
    if z.size == 0 or np.isnan(z).mean() > 0.3:
        return None
    return z, tr, "USGS 3DEP (LiDAR where available) via the National Map, resampled to 5 m"


def _tile_name(lat: float, lon: float) -> str:
    la = math.floor(lat)
    lo = math.floor(lon)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    base = f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"
    return f"/vsis3/copernicus-dem-30m/{base}/{base}.tif"


def _fill_depressions(z: np.ndarray, eps: float = 0.001) -> np.ndarray:
    """Planchon & Darboux (2001) depression filling; fine for a few hundred thousand cells."""
    w = np.full_like(z, np.inf)
    w[0, :] = z[0, :]
    w[-1, :] = z[-1, :]
    w[:, 0] = z[:, 0]
    w[:, -1] = z[:, -1]
    changed = True
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while changed:
        changed = False
        for di, dj in nbrs:
            shifted = np.full_like(w, np.inf)
            si = slice(max(di, 0), w.shape[0] + min(di, 0))
            sj = slice(max(dj, 0), w.shape[1] + min(dj, 0))
            ti = slice(max(-di, 0), w.shape[0] + min(-di, 0))
            tj = slice(max(-dj, 0), w.shape[1] + min(-dj, 0))
            shifted[ti, tj] = w[si, sj] + eps
            cand = np.maximum(z, shifted)
            lower = cand < w
            if lower.any():
                w[lower] = cand[lower]
                changed = True
    return w


def analyse_terrain(field: Field, buffer_m: float = 300.0) -> dict[str, Any]:
    lon, lat = field.centroid
    minx, miny, maxx, maxy = field.bbox
    dlat = buffer_m / 111_320.0
    dlon = buffer_m / (111_320.0 * math.cos(math.radians(lat)))
    bounds = (minx - dlon, miny - dlat, maxx + dlon, maxy + dlat)

    source = None
    lidar = False
    try:
        got = _read_3dep(bounds, lat)
    except Exception as e:  # noqa: BLE001
        print(f"3DEP unavailable, using Copernicus: {type(e).__name__}: {str(e)[:100]}", flush=True)
        got = None
    if got is not None:
        z, tr, source = got
        lidar = True
    else:
        # Assume the buffered bbox stays inside one 1x1 degree tile; stitch later if needed.
        path = _tile_name(lat, lon)
        with rasterio.open(path) as ds:
            win = from_bounds(*bounds, transform=ds.transform)
            z = ds.read(1, window=win).astype("float64")
            tr = ds.window_transform(win)
            nodata = ds.nodata
        if nodata is not None:
            z[z == nodata] = np.nan
        source = "Copernicus DEM GLO-30 (AWS public COG)"
    path = source
    if z.size == 0 or np.isnan(z).all():
        return {"error": "no DEM data in window", "source": path}
    z = np.where(np.isnan(z), np.nanmean(z), z)

    mask_in = ~geometry_mask([mapping(field.geom_wgs84)], out_shape=z.shape, transform=tr, invert=False)
    if not mask_in.any():
        return {"error": "field smaller than one DEM cell", "source": path}

    filled = _fill_depressions(z)
    dep = filled - z  # depression depth, metres

    # Relative elevation vs a ~270 m neighbourhood mean.
    cell_m_est = abs(tr.e) * 111_320.0
    k = max(3, int(round(270.0 / cell_m_est)) | 1)
    pad = k // 2
    zp = np.pad(z, pad, mode="edge")
    cs = np.cumsum(np.cumsum(zp, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    tot = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
    local_mean = tot / (k * k)
    rel = z - local_mean

    # Slope in percent from central differences (cell size in metres).
    cell_y = abs(tr.e) * 111_320.0
    cell_x = abs(tr.a) * 111_320.0 * math.cos(math.radians(lat))
    gy, gx = np.gradient(z, cell_y, cell_x)
    slope_pct = np.hypot(gx, gy) * 100.0

    zin = z[mask_in]
    low_decile = np.quantile(zin, 0.1)
    dep_thresh = 0.15 if lidar else 0.3
    return {
        "source": source,
        "resolution_m": round(cell_m_est, 1),
        "vertical_accuracy_note": ("LiDAR-derived, ~0.1-0.3 m vertical error at 5 m: good for pothole depth and drainage direction; "
                                   "a design survey still needs the 1 m product or RTK.") if lidar else
                                  "30 m DSM, 1.5-4 m vertical error. Suitable for drainage direction and broad low spots only, never for grades or depths.",
        "cells_in_field": int(mask_in.sum()),
        "elevation_min_m": round(float(zin.min()), 1),
        "elevation_max_m": round(float(zin.max()), 1),
        "relief_m": round(float(zin.max() - zin.min()), 1),
        "mean_slope_pct": round(float(slope_pct[mask_in].mean()), 2),
        "depression_share": round(float((dep[mask_in] > dep_thresh).mean()), 3),
        "depression_threshold_m": dep_thresh,
        "max_depression_depth_m": round(float(dep[mask_in].max()), 2),
        "low_relative_share": round(float((rel[mask_in] < -0.5).mean()), 3),
        "lowest_decile_elev_m": round(float(low_decile), 1),
        "aspect_outlet_hint": _outlet_hint(z, mask_in),
    }


def _outlet_hint(z: np.ndarray, mask_in: np.ndarray) -> str:
    """Which edge of the field is lowest: a first guess at where an outlet would sit."""
    rows, cols = np.where(mask_in)
    edges = {
        "north": z[rows.min(), :][mask_in[rows.min(), :]],
        "south": z[rows.max(), :][mask_in[rows.max(), :]],
        "west": z[:, cols.min()][mask_in[:, cols.min()]],
        "east": z[:, cols.max()][mask_in[:, cols.max()]],
    }
    means = {k: float(v.mean()) for k, v in edges.items() if v.size}
    if not means:
        return "unknown"
    return min(means, key=means.get)
