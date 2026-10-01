"""Terrain from the Copernicus DEM GLO-30 (public COGs on AWS, no credentials).

What we compute on the 30 m grid for a field (buffered by 300 m so depressions that
drain across the boundary are seen):
  - elevation range and mean slope
  - depression depth: fill the DEM (Planchon-Darboux) and subtract the original
  - relative elevation: cell height minus the mean of a ~270 m neighbourhood
  - share of the field area sitting in depressions > 0.3 m and in the lowest decile

30 m with 1.5-4 m vertical error is only good for "which way does it drain" and
"where are the broad low spots". It is never good enough for grades. The output
carries a confidence note saying so; swap in 1 m LiDAR where it exists.
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


def _tile_name(lat: float, lon: float) -> str:
    la = math.floor(lat)
    lo = math.floor(lon)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    base = f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"
    return f"/vsis3/copernicus-dem-30m/{base}/{base}.tif"


def _fill_depressions(z: np.ndarray, eps: float = 0.001) -> np.ndarray:
    """Planchon & Darboux (2001) depression filling; fine for a few thousand cells."""
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

    # Assume the buffered bbox stays inside one 1x1 degree tile; stitch later if needed.
    path = _tile_name(lat, lon)
    with rasterio.open(path) as ds:
        win = from_bounds(*bounds, transform=ds.transform)
        z = ds.read(1, window=win).astype("float64")
        tr = ds.window_transform(win)
        nodata = ds.nodata
    if nodata is not None:
        z[z == nodata] = np.nan
    if z.size == 0 or np.isnan(z).all():
        return {"error": "no DEM data in window", "source": path}
    z = np.where(np.isnan(z), np.nanmean(z), z)

    mask_in = ~geometry_mask([mapping(field.geom_wgs84)], out_shape=z.shape, transform=tr, invert=False)
    if not mask_in.any():
        return {"error": "field smaller than one DEM cell", "source": path}

    filled = _fill_depressions(z)
    dep = filled - z  # depression depth, metres

    # Relative elevation vs 9x9 neighbourhood mean (~270 m).
    k = 9
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
    return {
        "source": "Copernicus DEM GLO-30 (AWS public COG)",
        "vertical_accuracy_note": "30 m DSM, 1.5-4 m vertical error. Suitable for drainage direction and broad low spots only, never for grades or depths.",
        "cells_in_field": int(mask_in.sum()),
        "elevation_min_m": round(float(zin.min()), 1),
        "elevation_max_m": round(float(zin.max()), 1),
        "relief_m": round(float(zin.max() - zin.min()), 1),
        "mean_slope_pct": round(float(slope_pct[mask_in].mean()), 2),
        "depression_share": round(float((dep[mask_in] > 0.3).mean()), 3),
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
