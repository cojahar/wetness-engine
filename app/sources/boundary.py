"""Snap a pin to a field boundary.

First choice: a Fields of The World polygon (Sentinel-2 segmentation at 10 m, see
fields.py). Fallback: the USDA Cropland Data Layer (CDL) crop-sequence flood fill, the
idea behind USDA's own Crop Sequence Boundaries: a field is a block of land that carried
the same crop sequence for the last few years, so we flood-fill outward from the pin
across cells whose three-year sequence matches. The CDL window is also how the crop
rotation for the economics is read, whichever boundary wins.

CDL is USA only. Service: https://nassgeodata.gmu.edu/CropScape/devhelp/help.html
"""
from __future__ import annotations

import asyncio
import datetime as dt
import math
import re
import time
from collections import deque
from typing import Any

import httpx
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
from pyproj import Transformer
from shapely.geometry import shape, Point, mapping
from shapely.ops import transform as shp_transform

from ..config import settings
from ..geo import Field
from . import fields as ftw

CDL_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
ALBERS = "EPSG:5070"
_to_albers = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
_to_wgs = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)

CROP_NAMES = {
    1: "corn", 2: "cotton", 3: "rice", 4: "sorghum", 5: "soybeans", 6: "sunflower", 10: "peanuts",
    12: "sweet corn", 21: "barley", 22: "durum wheat", 23: "spring wheat", 24: "winter wheat", 26: "winter wheat/soybeans",
    27: "rye", 28: "oats", 29: "millet", 31: "canola", 32: "flax", 33: "safflower", 36: "alfalfa", 37: "other hay",
    41: "sugarbeets", 42: "dry beans", 43: "potatoes", 53: "peas", 61: "fallow/idle", 176: "grass/pasture",
}
NON_CROP = {0, 63, 64, 65, 81, 82, 83, 87, 88, 92, 111, 112, 121, 122, 123, 124, 131, 141, 142, 143, 152, 190, 195}


def _cdl_years(today: dt.date) -> list[int]:
    # CDL for year Y is published around February of Y+1.
    latest = today.year - 1 if today.month >= 3 else today.year - 2
    return [latest, latest - 1, latest - 2]


async def _fetch_cdl(client: httpx.AsyncClient, year: int, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    b = ",".join(str(int(round(v))) for v in bbox)
    last_exc: Exception | None = None
    for attempt in range(2):  # the CropScape service is intermittently slow
        try:
            r = await client.get(CDL_URL, params={"year": year, "bbox": b})
            r.raise_for_status()
            break
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_exc = e
            await asyncio.sleep(2)
    else:
        raise last_exc  # type: ignore[misc]
    m = re.search(r"https?://[^<\s]+\.tif", r.text)
    if not m:
        return None
    tif = await client.get(m.group(0))
    tif.raise_for_status()
    with MemoryFile(tif.content) as mem, mem.open() as ds:
        arr = ds.read(1)
        _fetch_cdl.last_transform = ds.transform  # type: ignore[attr-defined]
        _fetch_cdl.last_crs = ds.crs  # type: ignore[attr-defined]
    return arr


_url_cache: dict[str, tuple[float, str]] = {}


def _presigned(key: str) -> str | None:
    """Short-lived URL for an object in our bucket; None when the bucket is not configured."""
    if not (settings.s3_endpoint and settings.s3_bucket and settings.s3_access_key_id):
        return None
    now = time.time()
    hit = _url_cache.get(key)
    if hit and hit[0] > now + 300:
        return hit[1]
    import boto3  # local import: optional dependency path

    client = boto3.client("s3", endpoint_url=settings.s3_endpoint, aws_access_key_id=settings.s3_access_key_id,
                          aws_secret_access_key=settings.s3_secret_access_key, region_name=settings.s3_region)
    url = client.generate_presigned_url("get_object", Params={"Bucket": settings.s3_bucket, "Key": key}, ExpiresIn=3600)
    _url_cache[key] = (now + 3600, url)
    return url


def _read_hosted(year: int, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, Any] | None:
    """Read a window of our hosted COG for one year via HTTP range requests. Blocking; run in a thread."""
    url = _presigned(f"cdl/{year}.tif")
    if not url:
        return None
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES"):
        with rasterio.open(f"/vsicurl/{url}") as ds:
            win = from_bounds(*bbox, transform=ds.transform)
            win = win.round_offsets().round_lengths()
            arr = ds.read(1, window=win)
            return arr, ds.window_transform(win)


async def _load_cube(bbox: tuple[float, float, float, float], years: list[int]) -> tuple[list[np.ndarray], Any, str] | None:
    """Three years of CDL for the bbox: from our bucket when hosted, else from USDA's service."""
    if _presigned(f"cdl/{years[0]}.tif"):
        try:
            outs = await asyncio.gather(*(asyncio.to_thread(_read_hosted, y, bbox) for y in years))
            if all(o is not None for o in outs):
                return [o[0] for o in outs], outs[0][1], "hosted CDL COG"  # type: ignore[index]
        except Exception as e:  # noqa: BLE001
            print(f"hosted CDL read failed, falling back to USDA: {type(e).__name__}: {str(e)[:120]}", flush=True)
    stack: list[np.ndarray] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20), follow_redirects=True) as client:
        for yr in years:
            arr = await _fetch_cdl(client, yr, bbox)
            if arr is None:
                return None
            stack.append(arr)
    return stack, _fetch_cdl.last_transform, "USDA CropScape service"  # type: ignore[attr-defined]


def _sequence_in(poly_albers, cube: np.ndarray, tr: Any) -> tuple[int, ...] | None:
    """Most common 3-year CDL crop sequence among crop pixels inside a polygon."""
    from rasterio.features import geometry_mask
    h, w = cube.shape[1:]
    m = geometry_mask([mapping(poly_albers)], out_shape=(h, w), transform=tr, invert=True)
    if not m.any():
        return None
    block = cube[:, m].T
    seqs, counts = np.unique(block, axis=0, return_counts=True)
    crop_idx = [i for i, s in enumerate(seqs) if int(s[0]) not in NON_CROP]
    if not crop_idx:
        return None
    best = max(crop_idx, key=lambda i: counts[i])
    return tuple(int(v) for v in seqs[best])


async def snap_to_field(lon: float, lat: float, window_m: float = 1000.0, max_ha: float = 300.0,
                        name: str | None = None, prefer: str = "ftw") -> dict[str, Any] | None:
    """Field polygon under a pin.

    1. Fields of The World (Sentinel-2 segmentation, 10 m, 2024/2025) when a polygon of
       acceptable confidence contains the pin; the CDL cube is still read for the crop
       sequence (US).
    2. Otherwise the CDL crop-sequence flood fill (30 m, US only).
    """
    x, y = _to_albers.transform(lon, lat)
    bbox = (x - window_m, y - window_m, x + window_m, y + window_m)
    years = _cdl_years(dt.date.today())
    county: dict[str, Any] | None = None
    try:
        county = await ftw.county_fips(lon, lat)
    except Exception:  # noqa: BLE001
        county = None
    ftw_hit: dict[str, Any] | None = None
    ftw_err: str | None = None
    if prefer == "ftw":
        try:
            ftw_hit = await ftw.field_at(lon, lat, (county or {}).get("state_fips"))
        except Exception as e:  # noqa: BLE001
            ftw_err = f"{type(e).__name__}: {str(e)[:160]}"
    try:
        loaded = await _load_cube(bbox, years)
    except Exception as e:  # noqa: BLE001
        if not ftw_hit:
            raise
        print(f"CDL cube unavailable (crop sequence skipped): {type(e).__name__}: {str(e)[:100]}", flush=True)
        loaded = None
    if ftw_hit:
        geom_wgs = shape(ftw_hit["geometry"])
        seq = None
        source = None
        if loaded:
            stack, tr, source = loaded
            h = min(a.shape[0] for a in stack)
            w = min(a.shape[1] for a in stack)
            cube = np.stack([a[:h, :w] for a in stack])
            seq = _sequence_in(shp_transform(_to_albers.transform, geom_wgs), cube, tr)
        field = Field(geom_wgs84=geom_wgs, name=name)
        if field.area_ha <= max_ha:
            return {
                "found": True,
                "method": f"Fields of The World field unit, {ftw_hit['year']} (Sentinel-2 segmentation at 10 m)",
                "source": ftw_hit["source"],
                "years": years if seq else [],
                "crop_sequence": [CROP_NAMES.get(c, f"class {c}") for c in seq] if seq else None,
                "crop_sequence_source": f"USDA CDL via {source}" if seq else None,
                "area_ha": round(field.area_ha, 1),
                "confidence": ftw_hit.get("confidence"),
                "alternatives": ftw_hit.get("alternatives"),
                "truncated_at_max_ha": False,
                "geometry": mapping(geom_wgs),
                "county": county,
                "note": ftw_hit["note"],
            }
    if not loaded:
        return None
    stack, tr, source = loaded
    h = min(a.shape[0] for a in stack)
    w = min(a.shape[1] for a in stack)
    cube = np.stack([a[:h, :w] for a in stack])  # (3, h, w)

    # Pin cell
    col = int((x - tr.c) / tr.a)
    row = int((y - tr.f) / tr.e)
    if not (0 <= row < h and 0 <= col < w):
        return None
    # Seed sequence = the most common 3-year sequence in a 5x5 block around the pin, so one
    # stray CDL pixel under the pin does not define the field.
    r0, r1 = max(0, row - 2), min(h, row + 3)
    c0, c1 = max(0, col - 2), min(w, col + 3)
    block = cube[:, r0:r1, c0:c1].reshape(3, -1).T
    seqs, counts = np.unique(block, axis=0, return_counts=True)
    crop_idx = [i for i, s in enumerate(seqs) if int(s[0]) not in NON_CROP]
    if not crop_idx:
        return {"found": False, "reason": f"pin is on non-crop land (CDL class {int(cube[0, row, col])})", "years": years}
    best = max(crop_idx, key=lambda i: counts[i])
    seq = tuple(int(v) for v in seqs[best])

    # Flood fill on exact 3-year sequence match, 4-connected, seeded from every matching
    # cell in the 5x5 block.
    match = np.all(cube == np.array(seq)[:, None, None], axis=0)
    mask = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            if match[rr, cc]:
                mask[rr, cc] = True
                q.append((rr, cc))
    cell_ha = abs(tr.a * tr.e) / 10_000.0
    max_cells = int(max_ha / cell_ha)
    n = 1
    while q and n < max_cells:
        r0, c0 = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r1, c1 = r0 + dr, c0 + dc
            if 0 <= r1 < h and 0 <= c1 < w and match[r1, c1] and not mask[r1, c1]:
                mask[r1, c1] = True
                q.append((r1, c1))
                n += 1
    truncated = n >= max_cells

    # Polygonize, keep the piece under (or nearest) the pin, and absorb interior holes of a
    # few stray pixels by closing with a 1-cell buffer.
    polys = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=tr) if v == 1]
    if not polys:
        return None
    pin = Point(x, y)
    poly = next((p for p in polys if p.contains(pin)), min(polys, key=lambda p: p.distance(pin)))
    poly = poly.buffer(31).buffer(-31).simplify(12)  # close one-cell holes, smooth stair-steps
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    if poly.is_empty:
        return None
    geom_wgs = shp_transform(_to_wgs.transform, poly)
    field = Field(geom_wgs84=geom_wgs, name=name)
    return {
        "found": True,
        "method": f"CDL crop-sequence flood fill (USDA Cropland Data Layer via {source})",
        "years": years,
        "crop_sequence": [CROP_NAMES.get(c, f"class {c}") for c in seq],
        "area_ha": round(field.area_ha, 1),
        "truncated_at_max_ha": truncated,
        "geometry": mapping(geom_wgs),
        "county": county,
        "ftw": "no field unit under the pin" if ftw_err is None else f"lookup failed: {ftw_err}",
        "note": "30 m raster edges; expect boundaries to be within one cell (~30 m) of the true fence line. "
                "Adjacent fields with the identical three-year crop sequence will merge.",
    }


def field_from_snap(snap: dict[str, Any], name: str | None = None) -> Field:
    return Field(geom_wgs84=shape(snap["geometry"]), name=name)
