"""Within-field wet-zone map from every clear Sentinel-2 pass and every Sentinel-1 pass
since 2019, via the Sentinel Hub Process API (CDSE).

Per season we make two requests for the field at 10 m in UTM:
  optical  one band per Sentinel-2 orbit (every pass, 5-day revisit): NDVI where the pixel
           was clear, -1.5 where it read as standing water (NDWI > 0 or SCL water),
           -9999 where cloud, shadow, snow or no data. Scene dates ride along as userdata.
  radar    one band per Sentinel-1 orbit (6-12 day revisit, sees through cloud): VV gamma0
           in dB, -9999 where no data.

Then, per pass, each pixel is compared with the median of its own crop (CDL class, when
hosted; else the whole field) on that same date, which cancels planting date, crop stage,
haze, and a merged neighbour growing a different crop. A pixel is "stressed" on a date
when its crop is at full canopy (reference NDVI >= 0.55) and the pixel is 0.10 NDVI under
the reference. Stress days per season = the sum of the intervals around stressed passes
(capped at 15 d per pass). Ponding = share of spring passes in which the pixel read as
water on optical, or as a dark return on radar (VV < -17 dB and 5 dB under the field
median that day). Pixels CDL calls non-crop in most years, and pixels that never green up
in any season (lanes, yards, waterways), are dropped from the field mask.

Problem zone = at least 21 stress days per season on average, or ponded in a fifth of
spring passes. Watch = 10 stress days or a tenth of passes. The old "seasonal maximum"
statistics are still computed from the same data and reported for comparison.

Docs: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process.html
"""
from __future__ import annotations

import asyncio
import base64
import datetime as dt
import io
import json
import math
import tarfile
import warnings
from typing import Any

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice")
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
from . import boundary as cdl
from .sentinel import _get_token, _sem

PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
RES_M = 10.0
NODATA, WATER = -9999.0, -1.5
STRESS_DELTA = 0.10        # NDVI under the same-date reference = stressed
CROP_MIN_NDVI = 0.55       # full canopy: below this, pixel differences are emergence/maturity, not water
MAX_GAP_DAYS = 15.0        # a stressed pass counts for at most this many days
PROBLEM_DAYS, WATCH_DAYS = 21.0, 10.0
PROBLEM_POND, WATCH_POND = 0.20, 0.10
LEGACY_LOW_DELTA = 0.08    # old method: seasonal max under the field's median max
MIN_ZONE_HA = 0.15
MIN_PIXEL_COVER = 0.5
SAR_WATER_DB, SAR_REL_DB = -17.0, -5.0
UPSCALE = 3
NONCROP_MAX_NDVI = 0.45    # never above this in any season = not a crop pixel (lane, yard, water)

S2_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B03", "B04", "B08", "SCL", "dataMask"]}],
    output: {id: "default", bands: 1, sampleType: "FLOAT32"},
    mosaicking: "ORBIT"
  };
}
function updateOutput(outputs, collection) {
  outputs.default.bands = Math.max(1, collection.scenes.orbits.length);
}
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {dates: scenes.orbits.map(function (o) { return o.dateFrom; })};
}
function evaluatePixel(samples, scenes) {
  var n = Math.max(1, scenes.orbits.length);
  var out = [];
  for (var i = 0; i < n; i++) {
    var s = samples[i];
    if (!s || !s.dataMask || !(s.SCL == 4 || s.SCL == 5 || s.SCL == 6 || s.SCL == 7)) { out.push(-9999); continue; }
    var ndwi = (s.B03 - s.B08) / (s.B03 + s.B08 + 1e-6);
    if (ndwi > 0.0 || s.SCL == 6) { out.push(-1.5); continue; }
    out.push((s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6));
  }
  return out;
}
"""

S1_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "dataMask"]}],
    output: {id: "default", bands: 1, sampleType: "FLOAT32"},
    mosaicking: "ORBIT"
  };
}
function updateOutput(outputs, collection) {
  outputs.default.bands = Math.max(1, collection.scenes.orbits.length);
}
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {dates: scenes.orbits.map(function (o) { return o.dateFrom; })};
}
function evaluatePixel(samples, scenes) {
  var n = Math.max(1, scenes.orbits.length);
  var out = [];
  for (var i = 0; i < n; i++) {
    var s = samples[i];
    if (!s || !s.dataMask || s.VV <= 0) { out.push(-9999); continue; }
    out.push(10 * Math.log(s.VV) / Math.LN10);
  }
  return out;
}
"""

TRUECOLOR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {input: [{bands: ["B04", "B03", "B02", "dataMask"]}], output: {bands: 4, sampleType: "UINT8"}};
}
function evaluatePixel(s) {
  var g = 2.8;
  return [Math.min(255, s.B04 * g * 255), Math.min(255, s.B03 * g * 255), Math.min(255, s.B02 * g * 255), s.dataMask * 255];
}
"""


def _season(year: int, northern: bool) -> tuple[dt.date, dt.date, list[int], list[int]]:
    """(start, end, peak months, spring months). Peak months only steer the true-colour
    snapshot and the legacy statistic; the stress tally uses every pass in the season."""
    if northern:
        return dt.date(year, 4, 1), dt.date(year, 9, 30), [6, 7, 8], [4, 5, 6]
    return dt.date(year, 4, 1), dt.date(year, 11, 30), [8, 9, 10], [5, 6, 7]


def _grid(field: Field) -> tuple[dict[str, Any], str, Any, int, int]:
    utm = field.utm_crs
    g = field.to_utm()
    minx, miny, maxx, maxy = g.bounds
    minx, miny = math.floor(minx / RES_M) * RES_M, math.floor(miny / RES_M) * RES_M
    maxx, maxy = math.ceil(maxx / RES_M) * RES_M, math.ceil(maxy / RES_M) * RES_M
    w, h = int((maxx - minx) / RES_M), int((maxy - miny) / RES_M)
    return mapping(g), f"http://www.opengis.net/def/crs/EPSG/0/{utm.to_epsg()}", from_origin(minx, maxy, RES_M, RES_M), w, h


async def _post(client: httpx.AsyncClient, headers: dict[str, str], body: dict[str, Any],
                accept: str) -> tuple[bytes | None, str | None]:
    last: httpx.Response | None = None
    for attempt in range(3):
        try:
            async with _sem:
                r = await client.post(PROCESS_URL, headers={**headers, "Accept": accept}, json=body)
        except httpx.HTTPError as e:
            return None, type(e).__name__
        if r.status_code == 200:
            return r.content, None
        if r.status_code < 500 and r.status_code != 429:
            return None, f"{r.status_code} {r.text[:160]}"
        last = r
        await asyncio.sleep(3 * (attempt + 1))
    assert last is not None
    return None, f"{last.status_code} {last.text[:160]}"


def _untar(blob: bytes) -> tuple[np.ndarray, list[dt.date]]:
    """Process API multipart response: default.tif (one band per orbit) + userdata.json (dates)."""
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}  # type: ignore[union-attr]
    tif = next(v for k, v in members.items() if k.endswith(".tif") or k.endswith(".tiff"))
    meta = json.loads(next(v for k, v in members.items() if k.endswith(".json")))
    dates = [dt.date.fromisoformat(d[:10]) for d in meta.get("dates", [])]
    with MemoryFile(tif) as mem, mem.open() as ds:
        arr = ds.read().astype("float32")
    return arr[: len(dates)] if dates else arr[:0], dates


async def _fetch_stack(client: httpx.AsyncClient, headers: dict[str, str], geom_utm: dict[str, Any], crs_uri: str,
                       w: int, h: int, start: dt.date, end: dt.date, sensor: str) -> tuple[np.ndarray | None, list[dt.date], str | None]:
    if sensor == "s2":
        data = {"type": "sentinel-2-l2a",
                "dataFilter": {"timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"}, "maxCloudCoverage": 70}}
        script = S2_EVALSCRIPT
    else:
        data = {"type": "sentinel-1-grd",
                "dataFilter": {"timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                               "acquisitionMode": "IW", "polarization": "DV"},
                "processing": {"orthorectify": True, "backCoeff": "GAMMA0_TERRAIN", "demInstance": "COPERNICUS"}}
        script = S1_EVALSCRIPT
    body = {
        "input": {"bounds": {"geometry": geom_utm, "properties": {"crs": crs_uri}}, "data": [data]},
        "output": {"width": w, "height": h,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}},
                                 {"identifier": "userdata", "format": {"type": "application/json"}}]},
        "evalscript": script,
    }
    blob, err = await _post(client, headers, body, "application/tar")
    if err or blob is None:
        return None, [], err
    try:
        arr, dates = _untar(blob)
    except Exception as e:  # noqa: BLE001
        return None, [], f"bad response: {type(e).__name__}: {str(e)[:80]}"
    return arr, dates, None


async def _fetch_truecolor(client: httpx.AsyncClient, headers: dict[str, str], geom_utm: dict[str, Any], crs_uri: str,
                           w: int, h: int, northern: bool) -> tuple[bytes | None, str | None]:
    today = dt.date.today()
    year = today.year
    _, _, peak, _ = _season(year, northern)
    if today.month < peak[0] + 1:
        year -= 1
    start, end = dt.date(year, peak[0], 1), dt.date(year, peak[-1] + 1, 1) - dt.timedelta(days=1)
    body = {
        "input": {"bounds": {"geometry": geom_utm, "properties": {"crs": crs_uri}},
                  "data": [{"type": "sentinel-2-l2a",
                            "dataFilter": {"timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                                           "maxCloudCoverage": 30, "mosaickingOrder": "leastCC"}}]},
        "output": {"width": w * UPSCALE, "height": h * UPSCALE,
                   "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": TRUECOLOR_EVALSCRIPT,
    }
    blob, err = await _post(client, headers, body, "image/png")
    return blob, (err or f"{start:%b}-{end:%b %Y}")


def _box3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    stack = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in (0, 1, 2) for j in (0, 1, 2)])
    return np.nanmean(stack, axis=0)


def _polygons(mask: np.ndarray, transform: Any, utm_crs: Any) -> list[dict[str, Any]]:
    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform
    out: list[dict[str, Any]] = []
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
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((ang + 22.5) // 45) % 8]


def _colour_png(v01: np.ndarray, alpha: int = 190) -> bytes:
    """Colour a 0-1 raster green -> yellow -> red; NaN transparent."""
    dh, dw = v01.shape
    t = np.nan_to_num(np.clip(v01, 0, 1), nan=0.0)
    rgba = np.zeros((4, dh, dw), dtype="uint8")
    r = np.where(t < 0.5, 46 + (255 - 46) * t * 2, 255 - 55 * (t - 0.5) * 2)
    g = np.where(t < 0.5, 160 + (214 - 160) * t * 2, 214 - (214 - 30) * (t - 0.5) * 2)
    b = np.where(t < 0.5, 67 - 67 * t * 2, 0 + 30 * (t - 0.5) * 2)
    rgba[0], rgba[1], rgba[2] = r.astype("uint8"), g.astype("uint8"), b.astype("uint8")
    rgba[3] = np.where(np.isnan(v01), 0, alpha).astype("uint8")
    with MemoryFile() as mem:
        with mem.open(driver="PNG", width=dw, height=dh, count=4, dtype="uint8") as ds:
            ds.write(rgba)
        return mem.read()


def _png_wgs84(v01: np.ndarray, field_mask: np.ndarray, transform: Any, utm_crs: Any) -> tuple[str, list[float]]:
    src = np.where(field_mask, np.nan_to_num(v01, nan=0.0), np.nan).astype("float32")
    h, w = src.shape
    bounds = rasterio.transform.array_bounds(h, w, transform)
    dst_transform, dw, dh = calculate_default_transform(utm_crs, "EPSG:4326", w, h, *bounds)
    dst = np.full((dh, dw), np.nan, dtype="float32")
    reproject(src, dst, src_transform=transform, src_crs=utm_crs, src_nodata=np.nan,
              dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=Resampling.nearest)
    wb = rasterio.transform.array_bounds(dh, dw, dst_transform)
    return base64.b64encode(_colour_png(dst)).decode(), [round(x, 6) for x in wb]


def _class_medians(band: np.ndarray, veg: np.ndarray, classes: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Per-pixel reference NDVI: the median of the pixel's own CDL crop class on this date when
    that class covers at least a tenth of the field, else the whole-field median. This keeps a
    merged neighbour with a different crop, or a split-planted field, from reading as stress."""
    field_med = float(np.median(band[veg]))
    ref = np.full(band.shape, field_med, dtype="float32")
    if classes is None:
        return ref, field_med
    total = int(veg.sum())
    vals, counts = np.unique(classes[veg], return_counts=True)
    for v, c in zip(vals, counts):
        if int(v) in cdl.NON_CROP or c < max(50, 0.10 * total):
            continue
        sel = classes == v
        ref[sel] = float(np.median(band[veg & sel]))
    return ref, field_med


def _season_stress(arr: np.ndarray, dates: list[dt.date], field_mask: np.ndarray, spring: list[int],
                   classes: np.ndarray | None = None) -> dict[str, Any] | None:
    """Per-pixel stress days, optical ponding share and legacy max statistic for one season."""
    n_field = int(field_mask.sum())
    h, w = field_mask.shape
    stress_days = np.zeros((h, w), dtype="float32")
    obs_days = np.zeros((h, w), dtype="float32")        # days of crop-established coverage per pixel
    pond_hits = np.zeros((h, w), dtype="float32")
    pond_obs = np.zeros((h, w), dtype="float32")
    season_max = np.full((h, w), np.nan, dtype="float32")
    used_dates: list[tuple[str, float, float]] = []
    order = np.argsort([d.toordinal() for d in dates])
    dates_sorted = [dates[i] for i in order]
    for k, i in enumerate(order):
        band = arr[i]
        d = dates_sorted[k]
        clear = field_mask & (band > NODATA + 1)
        if clear.sum() < MIN_PIXEL_COVER * n_field:
            continue
        water = clear & (np.abs(band - WATER) < 1e-3)
        veg = clear & ~water
        # interval this pass stands for: half way to the neighbours, capped
        prev_d = dates_sorted[k - 1] if k > 0 else d - dt.timedelta(days=MAX_GAP_DAYS)
        next_d = dates_sorted[k + 1] if k + 1 < len(dates_sorted) else d + dt.timedelta(days=MAX_GAP_DAYS)
        span = min(MAX_GAP_DAYS, ((next_d - prev_d).days) / 2.0)
        if d.month in spring:
            pond_obs[clear] += 1
            pond_hits[water] += 1
        if veg.sum() < 0.3 * n_field:
            continue
        ref, med = _class_medians(band, veg, classes)
        season_max = np.where(veg, np.fmax(np.nan_to_num(season_max, nan=-2.0), band), season_max)
        established = clear & (ref >= CROP_MIN_NDVI)   # crop at full canopy for this pixel's own class
        if established.sum() < 0.3 * n_field:
            continue
        stressed = veg & established & (band < ref - STRESS_DELTA)
        stressed |= water & established  # standing water on an established crop is stress too
        obs_days[established] += span
        stress_days[stressed] += span
        used_dates.append((d.isoformat(), round(med, 3), round(float(stressed.sum()) / n_field, 3)))
    if not used_dates:
        return None
    # Scale stress days to the whole season's observed window so cloudy seasons are not penalised
    with np.errstate(all="ignore"):
        frac = np.where(obs_days > 0, stress_days / obs_days, np.nan)
    window = max(1.0, float(np.nanmax(obs_days[field_mask])))
    stress_days_scaled = frac * window
    pond_share = np.where(pond_obs > 0, pond_hits / np.maximum(pond_obs, 1), np.nan)
    med_max = float(np.nanmedian(season_max[field_mask]))
    legacy_low = np.where(np.isnan(season_max), np.nan, (season_max < med_max - LEGACY_LOW_DELTA).astype("float32"))
    return {
        "stress_days": stress_days_scaled, "pond": pond_share, "legacy_low": legacy_low, "season_max": season_max,
        "passes_used": len(used_dates), "window_days": round(window),
        "mean_stress_days": round(float(np.nanmean(stress_days_scaled[field_mask])), 1),
        "worst_pass": max(used_dates, key=lambda x: x[2]),
        "legacy_low_share": round(float(np.nanmean(legacy_low[field_mask])), 3),
        "optical_pond_share": round(float(np.nanmean(pond_share[field_mask])) if np.isfinite(pond_share[field_mask]).any() else 0.0, 3),
    }


def _season_sar(arr: np.ndarray, dates: list[dt.date], field_mask: np.ndarray, spring: list[int]) -> np.ndarray | None:
    """Share of spring radar passes in which the pixel was a dark (water-like) return."""
    hits = np.zeros(field_mask.shape, dtype="float32")
    obs = np.zeros(field_mask.shape, dtype="float32")
    for i, d in enumerate(dates):
        if d.month not in spring:
            continue
        band = arr[i]
        ok = field_mask & (band > NODATA + 1)
        if ok.sum() < MIN_PIXEL_COVER * field_mask.sum():
            continue
        med = float(np.median(band[ok]))
        dark = ok & (band < SAR_WATER_DB) & (band < med + SAR_REL_DB)
        obs[ok] += 1
        hits[dark] += 1
    if obs.max() == 0:
        return None
    return np.where(obs > 0, hits / np.maximum(obs, 1), np.nan)


async def get_zones(field: Field, start_year: int = 2019) -> dict[str, Any]:
    lon, lat = field.centroid
    northern = lat >= 0
    geom_utm, crs_uri, transform, w, h = _grid(field)
    if w * h > 4_000_000:
        return {"skipped": "field too large for a 10 m zone map (over 400 km2)"}
    years = [y for y in range(start_year, dt.date.today().year + 1)]

    async with httpx.AsyncClient(timeout=240) as client:
        token = await _get_token(client)
        if not token:
            return {"skipped": "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set"}
        headers = {"Authorization": f"Bearer {token}"}
        tasks = []
        for y in years:
            s, e, _, _ = _season(y, northern)
            e = min(e, dt.date.today())
            if s >= e:
                continue
            tasks.append(("s2", y, _fetch_stack(client, headers, geom_utm, crs_uri, w, h, s, e, "s2")))
            tasks.append(("s1", y, _fetch_stack(client, headers, geom_utm, crs_uri, w, h, s, e, "s1")))
        results = await asyncio.gather(*(t[2] for t in tasks), _fetch_truecolor(client, headers, geom_utm, crs_uri, w, h, northern))
    truecolor_png, truecolor_note = results[-1]
    fetched = {(t[0], t[1]): r for t, r in zip(tasks, results[:-1])}

    utm_crs = field.utm_crs
    field_mask = ~geometry_mask([geom_utm], out_shape=(h, w), transform=transform, invert=False)
    # CDL crop classes per season (US, hosted years only): per-crop reference NDVI and a
    # non-crop mask for lanes, yards, waterways and water inside the boundary.
    try:
        classes_by_year = await cdl.cdl_classes_on_grid(field, years, transform, utm_crs, (h, w))
    except Exception as e:  # noqa: BLE001
        print(f"CDL classes unavailable for zones: {type(e).__name__}: {str(e)[:80]}", flush=True)
        classes_by_year = {}
    cdl_noncrop_share = 0.0
    if classes_by_year:
        nc = np.stack([np.isin(c, list(cdl.NON_CROP)) for c in classes_by_year.values()])
        cdl_noncrop = field_mask & (nc.mean(axis=0) >= 0.5)
        cdl_noncrop_share = round(float(cdl_noncrop.sum()) / max(1, int(field_mask.sum())), 3)
        field_mask = field_mask & ~cdl_noncrop
    n_field = int(field_mask.sum())
    stress_layers, pond_layers, sar_layers, legacy_layers, max_layers = [], [], [], [], []
    seasons: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for y in years:
        _, _, _, spring = _season(y, northern)
        s2 = fetched.get(("s2", y))
        if s2 is None:
            continue
        arr, dates, err = s2
        if err or arr is None or not dates:
            errors[f"s2 {y}"] = err or "no passes"
            continue
        st = _season_stress(arr, dates, field_mask, spring, classes_by_year.get(y))
        if st is None:
            errors[f"s2 {y}"] = "no pass with a clear view of an established crop"
            continue
        stress_layers.append(st["stress_days"])
        pond_layers.append(st["pond"])
        legacy_layers.append(st["legacy_low"])
        max_layers.append(st["season_max"])
        row = {"year": y, "optical_passes_used": st["passes_used"], "window_days": st["window_days"],
               "crop_classes_used": y in classes_by_year,
               "mean_stress_days": st["mean_stress_days"], "optical_pond_share": st["optical_pond_share"],
               "worst_pass": {"date": st["worst_pass"][0], "field_median_ndvi": st["worst_pass"][1],
                              "stressed_share": st["worst_pass"][2]},
               "legacy_low_share_max_method": st["legacy_low_share"]}
        s1 = fetched.get(("s1", y))
        if s1 is not None:
            arr1, dates1, err1 = s1
            if err1 or arr1 is None or not dates1:
                errors[f"s1 {y}"] = err1 or "no passes"
            else:
                sar = _season_sar(arr1, dates1, field_mask, spring)
                if sar is not None:
                    sar_layers.append(sar)
                    row["radar_passes"] = len(dates1)
                    row["radar_pond_share"] = round(float(np.nanmean(sar[field_mask])), 3)
        seasons.append(row)
    if len(stress_layers) < 2:
        return {"skipped": "fewer than two usable seasons", "errors": errors, "years_tried": years}

    with np.errstate(all="ignore"):
        stress_mean = np.nanmean(np.stack(stress_layers), axis=0)
        stress_worst = np.nanmax(np.stack(stress_layers), axis=0)
        pond_opt = np.nanmean(np.stack(pond_layers), axis=0)
        pond_sar = np.nanmean(np.stack(sar_layers), axis=0) if sar_layers else np.full((h, w), np.nan, dtype="float32")
        legacy_freq = np.nanmean(np.stack(legacy_layers), axis=0)
    pond = np.fmax(np.nan_to_num(pond_opt, nan=0.0), np.nan_to_num(pond_sar, nan=0.0))
    # Pixels that never greened up in any season (lanes, yards, waterways, tree lines, permanent
    # water) are not crop and must not be scored as wet crop. A spot that ponds a lot is kept.
    with np.errstate(all="ignore"):
        alltime_max = np.nanmax(np.stack(max_layers), axis=0)
    noncrop = field_mask & (np.nan_to_num(alltime_max, nan=0.0) < NONCROP_MAX_NDVI) & (pond < 0.3)
    noncrop_share = round(float(noncrop.sum()) / max(1, n_field), 3)
    field_mask = field_mask & ~noncrop
    n_field = max(1, int(field_mask.sum()))
    stress_mean = np.where(field_mask, stress_mean, np.nan)
    stress_s, pond_s = _box3(stress_mean), _box3(pond)
    problem = field_mask & ((stress_s >= PROBLEM_DAYS) | (pond_s >= PROBLEM_POND))
    watch = field_mask & ~problem & ((stress_s >= WATCH_DAYS) | (pond_s >= WATCH_POND))
    legacy_s = _box3(legacy_freq)
    legacy_problem = field_mask & (legacy_s >= 0.5)

    pz = _polygons(problem, transform, utm_crs)
    wz = _polygons(watch, transform, utm_crs)
    fc = field.to_utm().centroid
    for z in pz + wz:
        cx, cy = z.pop("centroid_utm")
        z["position"] = _compass(cx - fc.x, cy - fc.y)
        zm = geometry_mask([shp_transform(Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform,
                                          shape(z["geometry"]))], out_shape=(h, w), transform=transform, invert=True)
        if zm.any():
            z["mean_stress_days_per_season"] = round(float(np.nanmean(stress_mean[zm])), 1)
            z["worst_season_stress_days"] = round(float(np.nanmax(stress_worst[zm])), 1)
            z["spring_ponding_share"] = round(float(np.nanmean(pond[zm])), 3)
    cell_ha = RES_M * RES_M / 10_000
    v01 = np.clip(stress_mean / 30.0, 0, 1)   # 30 stress days a season = full red
    png, png_bounds = _png_wgs84(v01, field_mask, transform, utm_crs)
    report_v = np.where(field_mask, np.nan_to_num(v01, nan=0.0), np.nan).astype("float32")
    report_overlay = _colour_png(np.repeat(np.repeat(report_v, UPSCALE, axis=0), UPSCALE, axis=1), alpha=150)
    # Compact per-pixel grid so a yield map can be compared against the zones later (app/validate.py)
    import zlib
    def _pack(a: np.ndarray) -> str:
        return base64.b64encode(zlib.compress(a.astype("uint8").tobytes(), 6)).decode()
    grid = {
        "epsg": utm_crs.to_epsg(), "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        "width": w, "height": h, "cell_m": RES_M,
        "stress_days_u8_zb64": _pack(np.clip(np.nan_to_num(stress_mean, nan=0.0), 0, 255)),
        "zone_u8_zb64": _pack(problem.astype("uint8") * 2 + watch.astype("uint8")),
        "field_mask_zb64": _pack(field_mask),
    }
    return {
        "grid": grid,
        "method": "every clear Sentinel-2 pass and every Sentinel-1 pass per season since 2019, 10 m; "
                  "stress = pixel 0.10 NDVI under its own crop's median on the same date while the crop is established; "
                  "ponding = water on optical, or dark radar return, in spring passes",
        "seasons_used": seasons,
        "errors": errors,
        "problem_share": round(float(problem.sum()) / n_field, 3),
        "watch_share": round(float(watch.sum()) / n_field, 3),
        "problem_ha": round(float(problem.sum()) * cell_ha, 2),
        "watch_ha": round(float(watch.sum()) * cell_ha, 2),
        "field_mean_stress_days_per_season": round(float(np.nanmean(stress_mean[field_mask])), 1),
        "non_crop_share_excluded": noncrop_share,
        "cdl_non_crop_share_excluded": cdl_noncrop_share,
        "field_spring_ponding_share": round(float(np.nanmean(pond[field_mask])), 3),
        "radar_used": bool(sar_layers),
        "legacy_max_method": {"problem_share": round(float(legacy_problem.sum()) / n_field, 3),
                              "problem_ha": round(float(legacy_problem.sum()) * cell_ha, 2)},
        "problem_zones": pz[:12],
        "watch_zones": wz[:12],
        "overlay_png_base64": png,
        "overlay_bounds_wgs84": png_bounds,
        "report_overlay_png_base64": base64.b64encode(report_overlay).decode(),
        "report_truecolor_png_base64": base64.b64encode(truecolor_png).decode() if truecolor_png else None,
        "report_truecolor_note": truecolor_note if truecolor_png else None,
        "legend": "green: the crop here keeps up with the field; red: on average 30 or more days a season "
                  f"spent well below the field (0.10 NDVI under the same-day median). Problem = {PROBLEM_DAYS:.0f}+ "
                  f"stress days a season or ponded in {PROBLEM_POND:.0%} of spring passes.",
    }
