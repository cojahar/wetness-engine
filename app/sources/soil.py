"""Soil properties for a field.

Global: SoilGrids 2.0 (ISRIC) point query at the centroid, 250 m.
USA:    SSURGO via Soil Data Access, which carries a true drainage class and Ksat.
"""
from __future__ import annotations

from typing import Any

import httpx

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SDA_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"


async def soilgrids_point(lon: float, lat: float) -> dict[str, Any]:
    """Centroid query; if the 250 m pixel is a SoilGrids gap, try one pixel to the north-east.

    SoilGrids allows only a handful of requests per minute, so never fan out further.
    """
    out = await _soilgrids_one(lon, lat)
    if not out["values"]:
        alt = await _soilgrids_one(lon + 0.0025, lat + 0.0022)
        if alt["values"]:
            alt["note"] = "centroid pixel had no data; value from the adjacent pixel"
            return alt
        out["note"] = "no SoilGrids data at this location"
    return out


async def _soilgrids_one(lon: float, lat: float) -> dict[str, Any]:
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["clay", "sand", "silt", "bdod", "soc"],
        # SoilGrids depth labels are fixed: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm.
        "depth": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
        "value": "mean",
    }
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.get(SOILGRIDS_URL, params=params)
        r.raise_for_status()
        js = r.json()
    out: dict[str, Any] = {}
    topsoil_weights = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}
    for layer in js.get("properties", {}).get("layers", []):
        name = layer["name"]
        factor = layer.get("unit_measure", {}).get("d_factor", 1) or 1
        top_num = top_den = 0.0
        for d in layer.get("depths", []):
            v = d.get("values", {}).get("mean")
            if v is None:
                continue
            out[f"{name}_{d['label']}"] = round(v / factor, 2)
            w = topsoil_weights.get(d["label"])
            if w:
                top_num += (v / factor) * w
                top_den += w
        if top_den:
            # thickness-weighted 0-30 cm value, the one the scoring uses
            out[f"{name}_0-30cm"] = round(top_num / top_den, 2)
    return {"source": "SoilGrids 2.0 (ISRIC), 250 m", "values": out}


async def ssurgo_polygon(wkt: str) -> dict[str, Any] | None:
    """Soils inside the field, weighted by how much of the field each covers.

    Step 1 intersects the SSURGO map-unit polygons with the field and returns the share of
    the field under each map unit. Step 2 lists each unit's major components with their
    drainage class, Ksat and clay. A soil's share of the field = unit share x component
    percent. Returns None outside SSURGO coverage (i.e. outside the USA).
    """
    area_q = f"""
    SELECT mukey, SUM(mupolygongeo.STIntersection(geometry::STGeomFromText('{wkt}', 4326)).STArea()) AS a
    FROM mupolygon
    WHERE mupolygongeo.STIntersects(geometry::STGeomFromText('{wkt}', 4326)) = 1
    GROUP BY mukey
    """
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(SDA_URL, json={"query": area_q, "format": "JSON+COLUMNNAME"})
        if r.status_code != 200:
            return None
        rows = (r.json().get("Table") or [])
        if len(rows) < 2:
            return None
        unit_area = {str(row[0]): float(row[1] or 0) for row in rows[1:]}
        total = sum(unit_area.values()) or 1.0
        unit_share = {k: v / total for k, v in unit_area.items()}
        keys = ",".join(f"'{k}'" for k in unit_share)
        comp_q = f"""
        SELECT mu.mukey, mu.muname, c.compname, c.comppct_r, c.drainagecl, c.hydgrp,
               (SELECT MIN(ch.ksat_r) FROM chorizon ch WHERE ch.cokey = c.cokey AND ch.hzdept_r < 150) AS ksat_min_um_s,
               (SELECT AVG(ch.claytotal_r) FROM chorizon ch WHERE ch.cokey = c.cokey AND ch.hzdept_r < 100) AS clay_pct
        FROM mapunit mu JOIN component c ON c.mukey = mu.mukey
        WHERE mu.mukey IN ({keys}) AND c.majcompflag = 'Yes'
        ORDER BY c.comppct_r DESC
        """
        r = await client.post(SDA_URL, json={"query": comp_q, "format": "JSON+COLUMNNAME"})
        if r.status_code != 200:
            return None
        js = r.json()
    rows = js.get("Table") or []
    if len(rows) < 2:
        return None
    cols = rows[0]
    recs = [dict(zip(cols, row)) for row in rows[1:]]
    poor = {"Somewhat poorly drained", "Poorly drained", "Very poorly drained"}
    # Normalise component percents within each unit, then weight by the unit's share of the field.
    pct_sum: dict[str, float] = {}
    for x in recs:
        pct_sum[str(x["mukey"])] = pct_sum.get(str(x["mukey"]), 0.0) + float(x.get("comppct_r") or 0)
    by_soil: dict[str, dict[str, Any]] = {}
    poor_share = 0.0
    for x in recs:
        mk = str(x["mukey"])
        w = unit_share.get(mk, 0.0) * (float(x.get("comppct_r") or 0) / (pct_sum.get(mk) or 1.0))
        x["field_share"] = round(w, 4)
        cls = x.get("drainagecl") or ""
        if cls in poor:
            poor_share += w
        name = x.get("compname") or "?"
        d = by_soil.setdefault(name, {"compname": name, "field_share": 0.0, "drainagecl": cls, "hydgrp": x.get("hydgrp"),
                                      "clay_pct": x.get("clay_pct"), "ksat_min_um_s": x.get("ksat_min_um_s")})
        d["field_share"] += w
    soils = sorted(by_soil.values(), key=lambda d: -d["field_share"])
    for d in soils:
        d["field_share"] = round(d["field_share"], 3)
    return {
        "source": "USDA SSURGO via Soil Data Access; map units intersected with the field boundary",
        "soils": soils[:10],
        "components": recs[:12],
        "poorly_drained_component_share": round(min(1.0, poor_share), 2),
        "poorly_drained_share_basis": "share of the field's area under soils rated somewhat poorly drained or worse",
    }


async def get_soil(lon: float, lat: float, wkt: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["soilgrids"] = await soilgrids_point(lon, lat)
    except Exception as e:  # noqa: BLE001
        out["soilgrids_error"] = str(e)[:200]
    try:
        ss = await ssurgo_polygon(wkt)
        if ss:
            out["ssurgo"] = ss
    except Exception as e:  # noqa: BLE001
        out["ssurgo_error"] = str(e)[:200]
    return out
