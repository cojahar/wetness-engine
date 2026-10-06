"""Wetlands and outlets around a field, from two free USGS/USFWS map services.

1. National Wetlands Inventory (USFWS). Mapped wetland polygons that overlap the field,
   with the Cowardin code (PEM1A = freshwater emergent, temporarily flooded, the classic
   prairie pothole; PUBH = open water pond; PFO = forested). Two uses:
     - compliance: draining a mapped wetland can trip USDA Swampbuster rules; the farmer
       needs an NRCS wetland determination before tiling there, so the report says so;
     - evidence: a pothole NWI mapped in the 1980s that still shows crop stress today is
       about as strong a "this spot is wet" signal as public data gives.
   Service: https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0

2. NHDPlus High Resolution (USGS). Streams, ditches and canals within 2 km: the nearest one
   is the first guess at an outlet, and no outlet nearby is the first reason a tile job is
   dearer than the per-acre number says. Also waterbodies inside the field.
   Service: https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer (layer 3
   NetworkNHDFlowline, 4 NonNetworkNHDFlowline, 9 NHDWaterbody)

Both are US only; outside the US the functions return None.
"""
from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import httpx
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

from ..geo import Field

NWI_URL = "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query"
NHD_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer/{layer}/query"
FTYPE = {336: "canal/ditch", 460: "stream/river", 428: "pipeline", 558: "artificial path", 334: "connector",
         390: "lake/pond", 436: "reservoir", 466: "swamp/marsh", 493: "estuary"}
# FCode refinements that matter to a tile outlet: an intermittent stream may be dry when the tile runs
FCODE = {46006: "perennial stream", 46003: "intermittent stream", 46007: "ephemeral stream", 33600: "canal/ditch",
         33601: "aqueduct", 33603: "stormwater ditch", 55800: "river centreline (artificial path)", 42801: "pipeline",
         42803: "pipeline (siphon)", 42807: "underground conduit", 39004: "lake/pond (perennial)",
         39009: "lake/pond (intermittent)", 43600: "reservoir"}


def _kind(p: dict[str, Any]) -> str:
    try:
        fc = int(p.get("fcode") or 0)
    except (TypeError, ValueError):
        fc = 0
    if fc in FCODE:
        return FCODE[fc]
    try:
        return FTYPE.get(int(p.get("ftype") or 0), str(p.get("ftype")))
    except (TypeError, ValueError):
        return str(p.get("ftype"))
HA_TO_AC = 2.47105


def _in_conus(lon: float, lat: float) -> bool:
    return -125.5 <= lon <= -66.5 and 24.0 <= lat <= 49.5


def _esri_polygon(field: Field) -> str:
    g = field.geom_wgs84
    rings = []
    polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
    for p in polys:
        rings.append([list(c) for c in p.exterior.coords])
        for r in p.interiors:
            rings.append([list(c) for c in r.coords])
    return json.dumps({"rings": rings, "spatialReference": {"wkid": 4326}})


async def wetlands(field: Field, client: httpx.AsyncClient) -> dict[str, Any] | None:
    lon, lat = field.centroid
    if not _in_conus(lon, lat):
        return None
    params = {"geometry": _esri_polygon(field), "geometryType": "esriGeometryPolygon", "inSR": "4326",
              "spatialRel": "esriSpatialRelIntersects", "outFields": "WETLAND_TYPE,ATTRIBUTE,ACRES",
              "returnGeometry": "true", "outSR": "4326", "f": "geojson"}
    r = await client.post(NWI_URL, data=params)
    if r.status_code != 200:
        return {"error": f"NWI {r.status_code}"}
    js = r.json()
    feats = js.get("features") or []
    to_utm = Transformer.from_crs("EPSG:4326", field.utm_crs, always_xy=True).transform
    fld = field.to_utm()
    rows = []
    total = 0.0
    by_type: dict[str, float] = {}
    for f in feats:
        try:
            g = shp_transform(to_utm, shape(f["geometry"]))
        except Exception:  # noqa: BLE001
            continue
        inter = g.intersection(fld)
        if inter.is_empty:
            continue
        ha = inter.area / 10_000
        total += ha
        p = f.get("properties") or {}
        t = p.get("WETLAND_TYPE") or "wetland"
        by_type[t] = by_type.get(t, 0.0) + ha
        c = shp_transform(Transformer.from_crs(field.utm_crs, "EPSG:4326", always_xy=True).transform, inter.centroid)
        rows.append({"type": t, "code": p.get("ATTRIBUTE"), "ha_in_field": round(ha, 2),
                     "ac_in_field": round(ha * HA_TO_AC, 1), "centroid": [round(c.x, 6), round(c.y, 6)]})
    rows.sort(key=lambda x: -x["ha_in_field"])
    share = total / field.area_ha if field.area_ha else 0.0
    return {
        "source": "USFWS National Wetlands Inventory (map service)",
        "mapped_wetland_share": round(share, 3),
        "mapped_wetland_ha": round(total, 2),
        "mapped_wetland_ac": round(total * HA_TO_AC, 1),
        "by_type_ha": {k: round(v, 2) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])},
        "polygons": rows[:12],
        "note": ("NWI is a map of wetland signatures from aerial photos, not a legal determination. Any tile work "
                 "on or draining a mapped wetland needs an NRCS certified wetland determination first (Swampbuster)."
                 if total > 0 else "no NWI wetland polygons overlap this field"),
    }


async def outlets(field: Field, client: httpx.AsyncClient, search_m: float = 2000.0) -> dict[str, Any] | None:
    lon, lat = field.centroid
    if not _in_conus(lon, lat):
        return None
    minx, miny, maxx, maxy = field.bbox
    dlat = search_m / 111_320.0
    dlon = search_m / (111_320.0 * math.cos(math.radians(lat)))
    env = json.dumps({"xmin": minx - dlon, "ymin": miny - dlat, "xmax": maxx + dlon, "ymax": maxy + dlat,
                      "spatialReference": {"wkid": 4326}})
    to_utm = Transformer.from_crs("EPSG:4326", field.utm_crs, always_xy=True).transform
    fld = field.to_utm()
    fc = fld.centroid

    async def q(layer: int, fields: str) -> list[dict[str, Any]]:
        params = {"geometry": env, "geometryType": "esriGeometryEnvelope", "inSR": "4326",
                  "spatialRel": "esriSpatialRelIntersects", "outFields": fields, "returnGeometry": "true",
                  "outSR": "4326", "f": "geojson", "resultRecordCount": "500"}
        r = await client.post(NHD_URL.format(layer=layer), data=params)
        if r.status_code != 200:
            return []
        return (r.json().get("features") or [])

    lines, lines2, bodies = await asyncio.gather(q(3, "gnis_name,ftype,fcode,lengthkm"),
                                                 q(4, "gnis_name,ftype,fcode,lengthkm"),
                                                 q(9, "gnis_name,ftype,fcode,areasqkm"))
    cands = []
    for f in lines + lines2:
        try:
            g = shp_transform(to_utm, shape(f["geometry"]))
        except Exception:  # noqa: BLE001
            continue
        p = {k.lower(): v for k, v in (f.get("properties") or {}).items()}
        d = g.distance(fld)
        near = g.interpolate(g.project(fc)) if g.geom_type == "LineString" else g.representative_point()
        ang = (math.degrees(math.atan2(near.x - fc.x, near.y - fc.y)) + 360) % 360
        direction = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((ang + 22.5) // 45) % 8]
        cands.append({"name": p.get("gnis_name") or None, "kind": _kind(p),
                      "fcode": p.get("fcode"), "distance_m": round(float(d)), "direction": direction,
                      "touches_field": bool(d < 1.0)})
    cands.sort(key=lambda c: c["distance_m"])
    inside = []
    for f in bodies:
        try:
            g = shp_transform(to_utm, shape(f["geometry"]))
        except Exception:  # noqa: BLE001
            continue
        inter = g.intersection(fld)
        if inter.is_empty:
            continue
        p = {k.lower(): v for k, v in (f.get("properties") or {}).items()}
        inside.append({"name": p.get("gnis_name") or None, "kind": _kind(p),
                       "ha_in_field": round(inter.area / 10_000, 2)})
    nearest = cands[0] if cands else None
    if nearest is None:
        verdict = f"no mapped stream, ditch or canal within {search_m / 1000:.0f} km: outlet is the first question to settle"
    elif nearest["touches_field"]:
        verdict = f"a mapped {nearest['kind']}{' (' + nearest['name'] + ')' if nearest['name'] else ''} touches the field: outlet likely at hand"
    elif nearest["distance_m"] <= 400:
        verdict = (f"nearest mapped {nearest['kind']}{' (' + nearest['name'] + ')' if nearest['name'] else ''} is "
                   f"{nearest['distance_m']} m to the {nearest['direction']}: a main of that length is a normal part of the job")
    else:
        verdict = (f"nearest mapped {nearest['kind']}{' (' + nearest['name'] + ')' if nearest['name'] else ''} is "
                   f"{nearest['distance_m']} m to the {nearest['direction']}: budget for a long main or a lift, or check for an existing county tile main")
    return {
        "source": "USGS NHDPlus High Resolution (map service)",
        "nearest_outlet": nearest,
        "candidates": cands[:6],
        "waterbodies_in_field": inside[:6],
        "verdict": verdict,
        "note": "NHD maps named streams and larger ditches; private tile mains and small farm ditches are not in it, so a dealer check beats this hint",
    }


async def get_hydro(field: Field) -> dict[str, Any]:
    out: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            w = await wetlands(field, client)
            if w is not None:
                out["wetlands"] = w
        except Exception as e:  # noqa: BLE001
            out["wetlands_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        try:
            o = await outlets(field, client)
            if o is not None:
                out["outlets"] = o
        except Exception as e:  # noqa: BLE001
            out["outlets_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return out
