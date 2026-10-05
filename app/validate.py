"""Compare a farmer's yield monitor export with the engine's stress map for the same field.

Accepts CSV (lat, lon, yield columns in any common naming), GeoJSON points, or a zipped
shapefile (Deere Operations Center, Climate FieldView, Ag Leader SMS all export one of
these). Points are binned onto the engine's 10 m grid; each cell gets the median yield of
the points inside it. Then:
  - correlation between stress days and yield across cells (negative = our map is right)
  - mean yield inside problem zones vs the rest of the field
  - mean yield by stress band

The result is a number a dealer can quote: "your own combine says the red areas yielded
24% less". It is also the accuracy figure the engine is tuned against over time.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import re
import zipfile
import zlib
from typing import Any

import numpy as np
from pyproj import Transformer

LON_NAMES = ("longitude", "lon", "long", "lng", "x", "easting")
LAT_NAMES = ("latitude", "lat", "y", "northing")
YIELD_PREFERRED = ("yld_vol_dr", "yld_mass_dr", "dry_yield", "yield_dry", "yield", "yld", "vryieldvol", "yld_vol_we",
                   "wet_yield", "yieldbuac", "yield_bu_ac", "crop_flw_m")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _pick(cols: list[str], candidates: tuple[str, ...], contains: str | None = None) -> str | None:
    normed = {_norm(c): c for c in cols}
    for cand in candidates:
        if _norm(cand) in normed:
            return normed[_norm(cand)]
    if contains:
        for n, c in normed.items():
            if contains in n:
                return c
    return None


def _to_float(v: Any) -> float | None:
    try:
        f = float(str(v).strip().replace(",", ""))
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def parse_points(filename: str, data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an (n, 3) array of lon, lat, yield and a note about the columns used."""
    name = filename.lower()
    if name.endswith(".zip"):
        return _parse_shapefile_zip(data)
    if name.endswith(".geojson") or name.endswith(".json"):
        return _parse_geojson(data)
    return _parse_csv(data)


def _parse_csv(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    text = data.decode("utf-8-sig", errors="replace")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    cols = reader.fieldnames or []
    lon_c, lat_c = _pick(cols, LON_NAMES), _pick(cols, LAT_NAMES)
    y_c = _pick(cols, YIELD_PREFERRED, contains="yield") or _pick(cols, (), contains="yld")
    if not (lon_c and lat_c and y_c):
        raise ValueError(f"could not find longitude, latitude and yield columns in: {', '.join(cols[:20])}")
    pts = []
    for row in reader:
        lon, lat, y = _to_float(row.get(lon_c)), _to_float(row.get(lat_c)), _to_float(row.get(y_c))
        if lon is None or lat is None or y is None:
            continue
        pts.append((lon, lat, y))
    return np.array(pts, dtype="float64").reshape(-1, 3), {"format": "csv", "columns": {"lon": lon_c, "lat": lat_c, "yield": y_c}}


def _parse_geojson(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    js = json.loads(data)
    feats = js.get("features") or []
    if not feats:
        raise ValueError("GeoJSON has no features")
    props = list((feats[0].get("properties") or {}).keys())
    y_c = _pick(props, YIELD_PREFERRED, contains="yield") or _pick(props, (), contains="yld")
    if not y_c:
        raise ValueError(f"no yield property found among: {', '.join(props[:20])}")
    pts = []
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        y = _to_float((f.get("properties") or {}).get(y_c))
        if y is None:
            continue
        lon, lat = g["coordinates"][:2]
        pts.append((lon, lat, y))
    return np.array(pts, dtype="float64").reshape(-1, 3), {"format": "geojson", "columns": {"yield": y_c}}


def _parse_shapefile_zip(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    import shapefile  # pyshp

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        shp = next((n for n in names if n.lower().endswith(".shp")), None)
        if not shp:
            raise ValueError("zip contains no .shp file")
        base = shp[:-4]
        parts = {ext: z.read(base + ext) for ext in (".shp", ".dbf", ".shx", ".prj") if base + ext in names}
    if ".dbf" not in parts:
        raise ValueError("shapefile is missing its .dbf attribute table")
    prj = parts.get(".prj", b"").decode("latin-1")
    transformer = None
    if prj and "GEOGCS" not in prj[:16].upper() and "PROJCS" in prj.upper():
        from pyproj import CRS
        try:
            transformer = Transformer.from_crs(CRS.from_wkt(prj), "EPSG:4326", always_xy=True)
        except Exception:  # noqa: BLE001
            transformer = None
    r = shapefile.Reader(shp=io.BytesIO(parts[".shp"]), dbf=io.BytesIO(parts[".dbf"]),
                         shx=io.BytesIO(parts[".shx"]) if ".shx" in parts else None)
    fields = [f[0] for f in r.fields if f[0] != "DeletionFlag"]
    y_c = _pick(fields, YIELD_PREFERRED, contains="yield") or _pick(fields, (), contains="yld")
    if not y_c:
        raise ValueError(f"no yield field found among: {', '.join(fields[:20])}")
    yi = fields.index(y_c)
    pts = []
    for sr in r.iterShapeRecords():
        if not sr.shape.points:
            continue
        x, y = sr.shape.points[0][:2]
        if transformer:
            x, y = transformer.transform(x, y)
        v = _to_float(sr.record[yi])
        if v is None:
            continue
        pts.append((x, y, v))
    return np.array(pts, dtype="float64").reshape(-1, 3), {"format": "shapefile", "columns": {"yield": y_c},
                                                            "reprojected": bool(transformer)}


def _unpack(s: str, h: int, w: int) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype="uint8").reshape(h, w)


def compare(grid: dict[str, Any], pts: np.ndarray, units: str = "bu/ac") -> dict[str, Any]:
    """Bin yield points to the engine grid and compare with stress days and zones."""
    if pts.shape[0] < 50:
        raise ValueError(f"only {pts.shape[0]} usable yield points; a yield map has thousands")
    if not (np.all(np.abs(pts[:, 0]) <= 180) and np.all(np.abs(pts[:, 1]) <= 90)):
        raise ValueError("coordinates are not longitude/latitude; export the yield map in WGS84")
    h, w = grid["height"], grid["width"]
    a, b, c, d, e, f = grid["transform"]
    stress = _unpack(grid["stress_days_u8_zb64"], h, w).astype("float32")
    zone = _unpack(grid["zone_u8_zb64"], h, w)
    mask = _unpack(grid["field_mask_zb64"], h, w).astype(bool)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{grid['epsg']}", always_xy=True)
    x, y = to_utm.transform(pts[:, 0], pts[:, 1])
    col = np.floor((x - c) / a).astype(int)
    row = np.floor((y - f) / e).astype(int)
    yld = pts[:, 2]
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h) & (yld > 0)
    if ok.sum() < 50:
        raise ValueError("fewer than 50 yield points fall inside this field boundary; is this the right field?")
    # Trim the usual combine artefacts (zeros at headlands, spikes at overlaps)
    lo, hi = np.percentile(yld[ok], [1, 99])
    ok &= (yld >= lo) & (yld <= hi)
    col, row, yld = col[ok], row[ok], yld[ok]
    idx = row * w + col
    order = np.argsort(idx)
    idx, yld = idx[order], yld[order]
    uniq, start, counts = np.unique(idx, return_index=True, return_counts=True)
    cell_yield = np.array([np.median(yld[s:s + n]) for s, n in zip(start, counts)])
    cell_mask = mask.ravel()[uniq] & (counts >= 2)
    uniq, cell_yield = uniq[cell_mask], cell_yield[cell_mask]
    if uniq.size < 30:
        raise ValueError("fewer than 30 grid cells with yield inside the field")
    cs, cz = stress.ravel()[uniq], zone.ravel()[uniq]
    field_mean = float(cell_yield.mean())
    r = float(np.corrcoef(cs, cell_yield)[0, 1]) if cs.std() > 0 else 0.0
    bands = [(0, 1, "no stress"), (1, 10, "1-9 days"), (10, 21, "10-20 days"), (21, 999, "21+ days")]
    by_band = []
    for lo_b, hi_b, label in bands:
        sel = (cs >= lo_b) & (cs < hi_b)
        if sel.sum() >= 5:
            by_band.append({"band": label, "cells": int(sel.sum()), "mean_yield": round(float(cell_yield[sel].mean()), 1),
                            "vs_field_pct": round((float(cell_yield[sel].mean()) / field_mean - 1) * 100, 1)})
    out: dict[str, Any] = {
        "points_used": int(ok.sum()), "cells_compared": int(uniq.size), "units": units,
        "field_mean_yield": round(field_mean, 1),
        "correlation_stress_vs_yield": round(r, 3),
        "variance_explained_pct": round(r * r * 100, 1),
        "by_stress_band": by_band,
    }
    for code, key in ((2, "problem"), (1, "watch")):
        sel = cz == code
        if sel.sum() >= 5:
            m = float(cell_yield[sel].mean())
            rest = float(cell_yield[cz != code].mean()) if (cz != code).sum() else field_mean
            out[f"{key}_zone"] = {"cells": int(sel.sum()), "mean_yield": round(m, 1),
                                  "vs_rest_pct": round((m / rest - 1) * 100, 1)}
    pz = out.get("problem_zone")
    if pz and pz["vs_rest_pct"] <= -10:
        verdict = f"Confirms the map: the problem zones yielded {abs(pz['vs_rest_pct']):.0f}% below the rest of the field."
    elif pz and pz["vs_rest_pct"] >= 5:
        verdict = (f"Does not confirm the map for this year: problem zones out-yielded the rest by {pz['vs_rest_pct']:.0f}%. "
                   "In a dry year wet spots often do; check which year this map is from.")
    elif r <= -0.3:
        verdict = f"Yield falls as stress days rise (r = {r:.2f}); the map explains {r * r * 100:.0f}% of yield variation."
    else:
        verdict = "Weak relationship between stress days and yield for this map; wetness may not be the main yield limiter here."
    out["verdict"] = verdict
    return out
