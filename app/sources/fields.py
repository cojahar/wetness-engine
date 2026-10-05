"""Field boundaries from Fields of The World (FTW) global predictions.

FTW (fieldsofthe.world, CC BY 4.0) runs a segmentation model over Sentinel-2 and publishes
one GeoParquet of field polygons per country / state for 2024 and 2025 at 10 m:
  https://data.source.coop/ftw/global-data/predictions/vectors/alpha/results-by-admin-conf/
    admin:country_code={CC}/{CC}_{SUB}.parquet
Polygons carry a bbox struct, so a pin lookup reads the file footer plus the one or two row
groups whose bbox covers the pin, via HTTP range requests (a few MB, 1-3 s). No key, no
download. These are imagery-derived "field units" at 10 m, much closer to the fence line
than the 30 m CDL flood fill, and they exist outside the US (Western Australia included).

Lookup: all polygons containing the pin, newest year first, confidence >= MIN_CONF. The
caller still shows the outline for editing.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pyarrow.parquet as pq
from shapely import from_wkb
from shapely.geometry import Point, mapping
from shapely.validation import make_valid

FTW_BASE = ("https://data.source.coop/ftw/global-data/predictions/vectors/alpha/results-by-admin-conf/"
            "admin:country_code={cc}/{part}.parquet")
MIN_CONF = 50          # FTW suggests >= 69 for clean fields; we accept lower and say so
PAD_DEG = 0.0005       # ~50 m slack around the pin when pruning row groups

# State FIPS -> USPS, for the Census geocoder result
FIPS_TO_USPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
    "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
    "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
}
# Australian states by bounding box (lon_min, lat_min, lon_max, lat_max); good enough for a pin
AU_STATES = {
    "WA": (112.9, -35.2, 129.0, -13.6), "NT": (129.0, -26.0, 138.0, -10.9), "SA": (129.0, -38.1, 141.0, -26.0),
    "QLD": (138.0, -29.2, 153.6, -10.4), "NSW": (141.0, -37.6, 153.7, -28.1), "VIC": (140.9, -39.2, 150.0, -33.9),
    "TAS": (143.8, -43.7, 148.5, -39.5), "ACT": (148.7, -35.95, 149.4, -35.1),
}


async def county_fips(lon: float, lat: float) -> dict[str, Any] | None:
    """County FIPS and name from the free US Census geocoder (no key). None outside the US."""
    import httpx
    params = {"x": lon, "y": lat, "benchmark": "Public_AR_Current", "vintage": "Current_Current", "format": "json",
              "layers": "Counties"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get("https://geocoding.geo.census.gov/geocoder/geographies/coordinates", params=params)
        if r.status_code != 200:
            return None
        js = r.json()
    counties = ((js.get("result") or {}).get("geographies") or {}).get("Counties") or []
    if not counties:
        return None
    c = counties[0]
    return {"fips": f"{c.get('STATE', '')}{c.get('COUNTY', '')}", "county": c.get("NAME"), "state_fips": c.get("STATE")}


def partition_for(lon: float, lat: float, state_fips: str | None = None) -> tuple[str, str] | None:
    """(country_code, partition name) for a pin, or None where FTW has no partition we know."""
    if state_fips and state_fips in FIPS_TO_USPS:
        return "US", f"US_{FIPS_TO_USPS[state_fips]}"
    for st, (x0, y0, x1, y1) in AU_STATES.items():
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            return "AU", f"AU_{st}"
    return None


def _open(url: str):
    import fsspec
    fs = fsspec.filesystem("https", block_size=4 * 1024 * 1024)
    return fs.open(url, "rb")


_meta_cache: dict[str, Any] = {}   # parquet footer per partition, so repeat pins skip the footer read


def _lookup_sync(url: str, lon: float, lat: float) -> list[dict[str, Any]]:
    t0 = time.time()
    with _open(url) as f:
        pf = pq.ParquetFile(f, metadata=_meta_cache.get(url))
        md = pf.metadata
        _meta_cache[url] = md
        schema = pf.schema_arrow
        names = [schema.field(i).name for i in range(len(schema))]
        # Column indices of the bbox struct children in the parquet (leaf) schema
        leaf = [md.row_group(0).column(j).path_in_schema for j in range(md.num_columns)] if md.num_row_groups else []
        idx = {p: j for j, p in enumerate(leaf)}
        keys = {k: idx.get(k) for k in ("bbox.xmin", "bbox.ymin", "bbox.xmax", "bbox.ymax")}
        groups: list[int] = []
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            ok = True
            if all(v is not None for v in keys.values()):
                st = {k: rg.column(j).statistics for k, j in keys.items()}  # type: ignore[arg-type]
                if all(s is not None and s.has_min_max for s in st.values()):
                    # A row group can hold the pin only if some polygon's bbox could span it
                    ok = (st["bbox.xmin"].min <= lon + PAD_DEG and st["bbox.xmax"].max >= lon - PAD_DEG
                          and st["bbox.ymin"].min <= lat + PAD_DEG and st["bbox.ymax"].max >= lat - PAD_DEG)
            if ok:
                groups.append(i)
        if not groups or len(groups) > 40:
            return []  # no pruning possible: do not pull hundreds of MB
        cols = [c for c in ("geometry", "bbox", "confidence", "determination:datetime", "metrics:area", "id") if c in names]
        tbl = pf.read_row_groups(groups, columns=cols)
    pin = Point(lon, lat)
    out: list[dict[str, Any]] = []
    bb = tbl.column("bbox").to_pylist() if "bbox" in cols else [None] * tbl.num_rows
    geoms = tbl.column("geometry").to_pylist()
    confs = tbl.column("confidence").to_pylist() if "confidence" in cols else [None] * tbl.num_rows
    dates = tbl.column("determination:datetime").to_pylist() if "determination:datetime" in cols else [None] * tbl.num_rows
    areas = tbl.column("metrics:area").to_pylist() if "metrics:area" in cols else [None] * tbl.num_rows
    ids = tbl.column("id").to_pylist() if "id" in cols else [None] * tbl.num_rows
    for g, b, c, d, a, i in zip(geoms, bb, confs, dates, areas, ids):
        if b and not (b["xmin"] <= lon <= b["xmax"] and b["ymin"] <= lat <= b["ymax"]):
            continue
        geom = make_valid(from_wkb(g))
        if not geom.contains(pin):
            continue
        year = str(d)[:4] if d is not None else None
        out.append({"id": i, "year": year, "confidence": c, "area_m2": a, "geometry": geom})
    out.sort(key=lambda r: (r["year"] or "", r["confidence"] or 0), reverse=True)
    for r in out:
        r["elapsed_s"] = round(time.time() - t0, 1)
    return out


async def field_at(lon: float, lat: float, state_fips: str | None = None) -> dict[str, Any] | None:
    """Best FTW polygon under the pin, or None. Runs the blocking parquet read in a thread."""
    part = partition_for(lon, lat, state_fips)
    if not part:
        return None
    cc, name = part
    url = FTW_BASE.format(cc=cc, part=name)
    hits = await asyncio.to_thread(_lookup_sync, url, lon, lat)
    good = [h for h in hits if (h["confidence"] or 0) >= MIN_CONF] or hits
    if not good:
        return None
    best = good[0]
    alt = [{"year": h["year"], "confidence": h["confidence"], "area_ha": round((h["area_m2"] or 0) / 10_000, 1)}
           for h in good[1:3]]
    return {
        "found": True,
        "source": "Fields of The World (Sentinel-2 segmentation, 10 m, CC BY 4.0)",
        "partition": name,
        "year": best["year"],
        "confidence": best["confidence"],
        "area_ha": round((best["area_m2"] or best["geometry"].area) / 10_000, 1) if best["area_m2"] else None,
        "geometry": mapping(best["geometry"]),
        "alternatives": alt,
        "elapsed_s": best.get("elapsed_s"),
        "note": "imagery-derived field unit at 10 m; a field split by crop or planting date may come back as two "
                "pieces, and two fields farmed as one may come back merged. Check the outline.",
    }
