"""Existing-tile evidence for a field, from two public sources.

1. AgTile-US (Valayamkunnath et al. 2020, Scientific Data; CC BY 4.0). A 30 m map of
   cropland likely to be tile drained, circa 2017, built from the USDA Census of Agriculture
   county tile acres, SSURGO drainage class and CDL cropland. Reported accuracy ~86% at the
   county scale. Hosted in our bucket as agtile/2020.tif (scripts/load_agtile.py); we read a
   window per field and report the share of the field flagged as tiled.
   Caveat for the report: it is a probability map built from county statistics, not a survey
   of real tile lines. "Appears tiled" means "fields like this one in this county are usually
   tiled", nothing more.

2. USDA Census of Agriculture, county tile-drained acres (2022; updated every five years,
   next 2027). Static file data/census_tile_2022.csv with columns fips, state, county,
   tile_acres, cropland_acres. Used for the sentence "x% of cropland in this county is tiled".
"""
from __future__ import annotations

import csv
import functools
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape

from ..config import settings
from ..geo import Field
from .boundary import _presigned, _url_cache
from .fields import county_fips

AGTILE_KEY = "agtile/2020.tif"
CENSUS_PATH = Path(__file__).resolve().parents[2] / "data" / "census_tile_2022.csv"


def agtile_share(field: Field) -> dict[str, Any] | None:
    """Share of the field's pixels flagged as tiled in AgTile-US. Blocking; run in a thread.

    Returns None when the raster is not hosted (bucket not configured or file not loaded).
    GDAL caches a 404 per URL for the life of the process, so on failure the presigned URL is
    dropped and a fresh one (new signature, new URL) is tried once.
    """
    try:
        return _agtile_read(field)
    except Exception:  # noqa: BLE001
        _url_cache.pop(AGTILE_KEY, None)
        return _agtile_read(field)


def _agtile_read(field: Field) -> dict[str, Any] | None:
    url = _presigned(AGTILE_KEY)
    if not url:
        return None
    t0 = time.time()
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES"):
        with rasterio.open(f"/vsicurl/{url}") as ds:
            geom = transform_geom("EPSG:4326", ds.crs, mapping(field.geom_wgs84))
            minx, miny, maxx, maxy = shape(geom).bounds
            if maxx < ds.bounds.left or minx > ds.bounds.right or maxy < ds.bounds.bottom or miny > ds.bounds.top:
                return {"source": "AgTile-US (2020)", "covered": False, "note": "field is outside the AgTile-US extent"}
            win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform).round_offsets().round_lengths()
            if win.width < 1 or win.height < 1:
                return {"source": "AgTile-US (2020)", "covered": False, "note": "field smaller than one 30 m pixel"}
            arr = ds.read(1, window=win)
            tr = ds.window_transform(win)
            nodata = ds.nodata
    mask = geometry_mask([geom], out_shape=arr.shape, transform=tr, invert=True)
    vals = arr[mask]
    if nodata is not None:
        vals = vals[vals != nodata]
    if vals.size == 0:
        return {"source": "AgTile-US (2020)", "covered": False, "note": "no AgTile-US data under this field"}
    tiled = float(np.mean(vals == 1))
    label = "appears tiled" if tiled >= 0.6 else ("partly tiled" if tiled >= 0.25 else "appears untiled")
    return {
        "source": "AgTile-US (Valayamkunnath et al. 2020), 30 m, CC BY 4.0",
        "covered": True,
        "tiled_share": round(tiled, 3),
        "pixels": int(vals.size),
        "label": label,
        "caveat": "a county-statistics model of where tile is likely (about 86% county-level accuracy), "
                  "not a survey of real tile lines; a farmer's own knowledge beats it",
        "elapsed_s": round(time.time() - t0, 1),
    }


@functools.lru_cache(maxsize=1)
def _census() -> dict[str, dict[str, Any]]:
    if not CENSUS_PATH.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(CENSUS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            fips = str(row.get("fips", "")).zfill(5)
            try:
                tile = float(row.get("tile_acres") or 0)
                crop = float(row.get("cropland_acres") or 0)
            except ValueError:
                continue
            out[fips] = {"fips": fips, "state": row.get("state"), "county": row.get("county"),
                         "tile_acres": tile, "cropland_acres": crop,
                         "tiled_share_of_cropland": round(tile / crop, 3) if crop else None}
    return out


def census_county(fips: str | None) -> dict[str, Any] | None:
    """County tile acres from the 2022 Census of Agriculture, by 5-digit FIPS."""
    if not fips:
        return None
    rec = _census().get(str(fips).zfill(5))
    if not rec:
        return None
    return {"source": "USDA Census of Agriculture 2022 (county tile-drained acres); next release 2027", **rec}


async def get_tile(field: Field) -> dict[str, Any]:
    import asyncio
    out: dict[str, Any] = {}
    fips = None
    try:
        lon, lat = field.centroid
        cf = await county_fips(lon, lat)
        if cf:
            out["county"] = cf
            fips = cf["fips"]
    except Exception as e:  # noqa: BLE001
        out["county_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        a = await asyncio.to_thread(agtile_share, field)
        if a is not None:
            out["agtile"] = a
        else:
            out["agtile_skipped"] = "AgTile-US raster not hosted yet (run scripts/load_agtile.py)"
    except Exception as e:  # noqa: BLE001
        out["agtile_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    c = census_county(fips)
    if c:
        out["census_county"] = c
    return out
