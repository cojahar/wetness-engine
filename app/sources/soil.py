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
    """Dominant components intersecting the field, with drainage class and Ksat.

    Returns None outside SSURGO coverage (i.e. outside the USA).
    """
    query = f"""
    SELECT mu.mukey, mu.muname, c.compname, c.comppct_r, c.drainagecl, c.hydgrp,
           (SELECT MIN(ch.ksat_r) FROM chorizon ch WHERE ch.cokey = c.cokey AND ch.hzdept_r < 150) AS ksat_min_um_s,
           (SELECT AVG(ch.claytotal_r) FROM chorizon ch WHERE ch.cokey = c.cokey AND ch.hzdept_r < 100) AS clay_pct
    FROM mapunit mu
    JOIN component c ON c.mukey = mu.mukey
    WHERE mu.mukey IN (SELECT DISTINCT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}'))
      AND c.majcompflag = 'Yes'
    ORDER BY c.comppct_r DESC
    """
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(SDA_URL, json={"query": query, "format": "JSON+COLUMNNAME"})
        if r.status_code != 200:
            return None
        js = r.json()
    rows = js.get("Table") or []
    if len(rows) < 2:
        return None
    cols = rows[0]
    recs = [dict(zip(cols, row)) for row in rows[1:]]
    # Summarise: share of components that are somewhat poorly drained or worse.
    poor = {"Somewhat poorly drained", "Poorly drained", "Very poorly drained"}
    n = len(recs)
    poor_share = sum(1 for x in recs if (x.get("drainagecl") or "") in poor) / n if n else 0.0
    return {
        "source": "USDA SSURGO via Soil Data Access",
        "components": recs[:12],
        "poorly_drained_component_share": round(poor_share, 2),
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
