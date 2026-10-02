"""Within-field wet-zone map from Sentinel-2 via the Sentinel Hub Process API (CDSE).

For every season since 2019 we ask Sentinel Hub for one small raster of the field at
10 m with, per pixel:
  band 1  peak-season NDVI maximum (the best the crop ever looked that year)
  band 2  share of spring scenes in which the pixel read as standing water (NDWI > 0 or SCL water)
  band 3  number of clear peak-season scenes behind band 1

A pixel that is well below the field's median peak NDVI in most years, or that ponds in
spring more often than not, is a persistent problem spot: exactly the places a tile line
pays for itself. We polygonize those spots, measure them, and render a small preview PNG.

Docs: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process.html
"""
from __future__ import annotations

import asyncio
import base64
import datetime as dt
import math
import warnings
from typing import Any

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Dataset has no geotransform")

import httpx
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform, unary_union

from ..geo import Field
from .sentinel import _get_token, _sem

PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
RES_M = 10.0
LOW_NDVI_DELTA = 0.08   # below the field median by this much = "low" that year
MIN_ZONE_HA = 0.15      # drop specks smaller than this
MIN_PIXEL_COVER = 0.5   # a year counts only if half the field had a clear peak scene

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B03", "B04", "B08", "SCL", "dataMask"]}],
    output: {bands: 3, sampleType: "FLOAT32"},
    mosaicking: "ORBIT"
  };
}
var PEAK = __PEAK__;
var SPRING = __SPRING__;
function evaluatePixel(samples, scenes) {
  var peakMax = -2, wet = 0, nSpring = 0, nPeak = 0;
  for (var i = 0; i < samples.length; i++) {
    var s = samples[i];
    if (!s.dataMask) continue;
    if (!(s.SCL == 4 || s.SCL == 5 || s.SCL == 6 || s.SCL == 7)) continue;
    var m = new Date(scenes.orbits[i].dateFrom).getUTCMonth() + 1;
    if (PEAK.indexOf(m) >= 0) {
      var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
      if (ndvi > peakMax) peakMax = ndvi;
      nPeak++;
    }
    if (SPRING.indexOf(m) >= 0) {
      nSpring++;
      var ndwi = (s.B03 - s.B08) / (s.B03 + s.B08 + 1e-6);
      if (ndwi > 0.0 || s.SCL == 6) wet++;
    }
  }
  return [nPeak > 0 ? peakMax : -9999, nSpring > 0 ? wet / nSpring : -9999, nPeak];
}
"""


TRUECOLOR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {input: [{bands: ["B04", "B03", "B02", "dataMask"]}], output: {bands: 4, sampleType: "UINT8"}};
}
function evaluatePixel(s) {
  var g = 2.8;  // simple gain; reflectances ~0-0.35 on cropland
  return [Math.min(255, s.B04 * g * 255), Math.min(255, s.B03 * g * 255), Math.min(255, s.B02 * g * 255), s.dataMask * 255];
}
"""

UPSCALE = 3  # report images at 10 m / 3 so a 40 ha field is ~200 px wide


async def _fetch_truecolor(client: httpx.AsyncClient, headers: dict[str, str], geom_utm: dict[str, Any], crs_uri: str,
                           w: int, h: int, northern: bool) -> tuple[bytes | None, str | None]:
    """Least-cloudy recent peak-season true-colour picture of the field, PNG, UPSCALE x grid."""
    today = dt.date.today()
    year = today.year
    _, _, peak, _ = _season(year, northern)
    if today.month < peak[0] + 1:   # this season's peak not over yet: use last year
        year -= 1
    start, end = dt.date(year, peak[0], 1), dt.date(year, peak[-1] + 1, 1) - dt.timedelta(days=1)
    body = {
        "input": {
            "bounds": {"geometry": geom_utm, "properties": {"crs": crs_uri}},
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                                     "maxCloudCoverage": 30, "mosaickingOrder": "leastCC"}}],
        },
        "output": {"width": w * UPSCALE, "height": h * UPSCALE,
                   "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": TRUECOLOR_EVALSCRIPT,
    }
    try:
        async with _sem:
            r = await client.post(PROCESS_URL, headers=headers, json=body)
        if r.status_code != 200:
            return None, f"{r.status_code} {r.text[:120]}"
        return r.content, f"{start:%b}-{end:%b %Y}"
    except httpx.HTTPError as e:
        return None, f"{type(e).__name__}"


def _season(year: int, northern: bool) -> tuple[dt.date, dt.date, list[int], list[int]]:
    if northern:
        return dt.date(year, 4, 1), dt.date(year, 9, 30), [6, 7, 8], [4, 5, 6]
    # Western Australia: cereals sown Apr-Jun, canopy peak Aug-Oct, wettest months May-Jul
    return dt.date(year, 4, 1), dt.date(year, 11, 30), [8, 9, 10], [5, 6, 7]


def _grid(field: Field) -> tuple[dict[str, Any], str, Any, int, int]:
    """UTM geometry, CRS URI, affine transform and raster size for a 10 m grid over the field."""
    utm = field.utm_crs
    g = field.to_utm()
    minx, miny, maxx, maxy = g.bounds
    minx, miny = math.floor(minx / RES_M) * RES_M, math.floor(miny / RES_M) * RES_M
    maxx, maxy = math.ceil(maxx / RES_M) * RES_M, math.ceil(maxy / RES_M) * RES_M
    w, h = int((maxx - minx) / RES_M), int((maxy - miny) / RES_M)
    transform = from_origin(minx, maxy, RES_M, RES_M)
    return mapping(g), f"http://www.opengis.net/def/crs/EPSG/0/{utm.to_epsg()}", transform, w, h


async def _fetch_year(client: httpx.AsyncClient, headers: dict[str, str], geom_utm: dict[str, Any], crs_uri: str,
                      w: int, h: int, year: int, northern: bool) -> tuple[int, np.ndarray | None, str | None]:
    start, end, peak, spring = _season(year, northern)
    end = min(end, dt.date.today())
    if start >= end:
        return year, None, "season not started"
    body = {
        "input": {
            "bounds": {"geometry": geom_utm, "properties": {"crs": crs_uri}},
            "data": [{"type": "sentinel-2-l2a",
                      "dataFilter": {"timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                                     "maxCloudCoverage": 60, "mosaickingOrder": "mostRecent"}}],
        },
        "output": {"width": w, "height": h, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT.replace("__PEAK__", str(peak)).replace("__SPRING__", str(spring)),
    }
    last: httpx.Response | None = None
    for attempt in range(3):
        async with _sem:
            r = await client.post(PROCESS_URL, headers=headers, json=body)
        if r.status_code < 500 and r.status_code != 429:
            break
        last = r
        await asyncio.sleep(3 * (attempt + 1))
    else:
        assert last is not None
        return year, None, f"{last.status_code} {last.text[:160]}"
    if r.status_code != 200:
        return year, None, f"{r.status_code} {r.text[:160]}"
    with MemoryFile(r.content) as mem, mem.open() as ds:
        return year, ds.read().astype("float32"), None  # (3, h, w)


def _box3(a: np.ndarray) -> np.ndarray:
    """3x3 mean with edge padding; NaN-aware."""
    p = np.pad(a, 1, mode="edge")
    stack = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in (0, 1, 2) for j in (0, 1, 2)])
    return np.nanmean(stack, axis=0)


def _polygons(mask: np.ndarray, transform: Any, utm_crs: Any) -> list[dict[str, Any]]:
    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform
    out = []
    polys = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask, transform=transform) if v == 1]
    if not polys:
        return out
    merged = unary_union([p.buffer(RES_M).buffer(-RES_M) for p in polys])
    geoms = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
    for p in geoms:
        ha = p.area / 10_000
        if ha < MIN_ZONE_HA:
            continue
        c = p.centroid
        out.append({"area_ha": round(ha, 2), "geometry": mapping(shp_transform(to_wgs, p.simplify(5))),
                    "centroid_utm": (c.x, c.y)})
    out.sort(key=lambda z: -z["area_ha"])
    return out


def _compass(dx: float, dy: float) -> str:
    if math.hypot(dx, dy) < 40:
        return "centre"
    ang = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int((ang + 22.5) // 45) % 8]


def _png(freq: np.ndarray, field_mask: np.ndarray, transform: Any, utm_crs: Any) -> tuple[str, list[float]]:
    """Reproject the low-frequency raster to WGS84 and colour it: transparent outside the
    field, green (never low) through yellow to red (low every year)."""
    src = np.where(field_mask, np.nan_to_num(freq, nan=0.0), np.nan).astype("float32")
    h, w = src.shape
    bounds = rasterio.transform.array_bounds(h, w, transform)
    dst_transform, dw, dh = calculate_default_transform(utm_crs, "EPSG:4326", w, h, *bounds)
    dst = np.full((dh, dw), np.nan, dtype="float32")
    reproject(src, dst, src_transform=transform, src_crs=utm_crs, src_nodata=np.nan,
              dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=Resampling.nearest)
    wb = rasterio.transform.array_bounds(dh, dw, dst_transform)  # (west, south, east, north)
    return base64.b64encode(_colour_png(dst)).decode(), [round(x, 6) for x in wb]


def _colour_png(freq: np.ndarray, alpha: int = 190) -> bytes:
    """Colour a 0-1 raster: green (0) through yellow (0.5) to red (1); NaN transparent."""
    dh, dw = freq.shape
    t = np.nan_to_num(np.clip(freq, 0, 1), nan=0.0)
    rgba = np.zeros((4, dh, dw), dtype="uint8")
    r = np.where(t < 0.5, 46 + (255 - 46) * t * 2, 255 - 55 * (t - 0.5) * 2)
    g = np.where(t < 0.5, 160 + (214 - 160) * t * 2, 214 - (214 - 30) * (t - 0.5) * 2)
    b = np.where(t < 0.5, 67 - 67 * t * 2, 0 + 30 * (t - 0.5) * 2)
    rgba[0], rgba[1], rgba[2] = r.astype("uint8"), g.astype("uint8"), b.astype("uint8")
    rgba[3] = np.where(np.isnan(freq), 0, alpha).astype("uint8")
    with MemoryFile() as mem:
        with mem.open(driver="PNG", width=dw, height=dh, count=4, dtype="uint8") as ds:
            ds.write(rgba)
        return mem.read()


async def get_zones(field: Field, start_year: int = 2019) -> dict[str, Any]:
    """Persistent low-vigour / ponding zones inside the field. Returns summary, GeoJSON zones
    and a PNG overlay (base64) with its WGS84 bounds for a web map."""
    lon, lat = field.centroid
    northern = lat >= 0
    geom_utm, crs_uri, transform, w, h = _grid(field)
    if w * h > 4_000_000:
        return {"skipped": "field too large for a 10 m zone map (over 400 km2)"}
    years = list(range(start_year, dt.date.today().year + 1))

    async with httpx.AsyncClient(timeout=180) as client:
        token = await _get_token(client)
        if not token:
            return {"skipped": "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set"}
        headers = {"Authorization": f"Bearer {token}"}
        *results, tc = await asyncio.gather(
            *(_fetch_year(client, headers, geom_utm, crs_uri, w, h, y, northern) for y in years),
            _fetch_truecolor(client, headers, geom_utm, crs_uri, w, h, northern))
        truecolor_png, truecolor_note = tc

    utm_crs = field.utm_crs
    field_mask = ~geometry_mask([geom_utm], out_shape=(h, w), transform=transform, invert=False)
    n_field = int(field_mask.sum())
    low_layers, pond_layers, used, errors = [], [], [], {}
    for year, arr, err in results:
        if err:
            if err != "season not started":
                errors[str(year)] = err
            continue
        ndvi, wet, n = arr[0], arr[1], arr[2]
        valid = field_mask & (n > 0) & (ndvi > -1)
        if valid.sum() < MIN_PIXEL_COVER * n_field:
            errors[str(year)] = f"only {valid.sum() / max(n_field, 1):.0%} of the field had a clear peak-season scene"
            continue
        med = float(np.median(ndvi[valid]))
        low = np.full((h, w), np.nan, dtype="float32")
        low[valid] = (ndvi[valid] < med - LOW_NDVI_DELTA).astype("float32")
        low_layers.append(low)
        pond = np.full((h, w), np.nan, dtype="float32")
        pv = field_mask & (wet > -1)
        pond[pv] = wet[pv]
        pond_layers.append(pond)
        used.append({"year": year, "median_peak_ndvi": round(med, 3),
                     "low_share": round(float(np.nanmean(low[field_mask])), 3),
                     "spring_pond_share": round(float(np.nanmean(pond[field_mask])) if pv.any() else 0.0, 3)})
    if len(low_layers) < 2:
        return {"skipped": "fewer than two usable seasons", "errors": errors, "years_tried": years}

    with np.errstate(all="ignore"):
        low_freq = np.nanmean(np.stack(low_layers), axis=0)
        pond_freq = np.nanmean(np.stack(pond_layers), axis=0)
    low_s, pond_s = _box3(low_freq), _box3(pond_freq)
    problem = field_mask & ((low_s >= 0.5) | (pond_s >= 0.2))
    watch = field_mask & ~problem & ((low_s >= 0.3) | (pond_s >= 0.1))

    pz = _polygons(problem, transform, utm_crs)
    wz = _polygons(watch, transform, utm_crs)
    fc = field.to_utm().centroid
    for z in pz + wz:
        cx, cy = z.pop("centroid_utm")
        z["position"] = _compass(cx - fc.x, cy - fc.y)
    cell_ha = RES_M * RES_M / 10_000
    png, png_bounds = _png(low_freq, field_mask, transform, utm_crs)
    # Report-grid images (UTM, UPSCALE x): overlay of the same low-frequency raster, plus the picture
    report_freq = np.where(field_mask, np.nan_to_num(low_freq, nan=0.0), np.nan).astype("float32")
    report_overlay = _colour_png(np.repeat(np.repeat(report_freq, UPSCALE, axis=0), UPSCALE, axis=1), alpha=150)
    return {
        "report_overlay_png_base64": base64.b64encode(report_overlay).decode(),
        "report_truecolor_png_base64": base64.b64encode(truecolor_png).decode() if truecolor_png else None,
        "report_truecolor_note": truecolor_note,
        "source": "Sentinel-2 L2A via Sentinel Hub Process API on CDSE, 10 m",
        "seasons_used": used,
        "errors": errors,
        "problem_share": round(float(problem.sum()) / n_field, 3),
        "watch_share": round(float(watch.sum()) / n_field, 3),
        "problem_ha": round(float(problem.sum()) * cell_ha, 2),
        "watch_ha": round(float(watch.sum()) * cell_ha, 2),
        "problem_zones": pz[:12],
        "watch_zones": wz[:12],
        "overlay_png_base64": png,
        "overlay_bounds_wgs84": png_bounds,
        "legend": "green: peak-season vigour never below the field median; red: below it every year "
                  f"(by at least {LOW_NDVI_DELTA} NDVI). Problem = low in half the seasons or ponded in a fifth of spring scenes.",
    }
