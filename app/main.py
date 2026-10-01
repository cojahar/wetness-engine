"""Farm X wetness engine: HTTP service.

POST /analyze   {"boundary": <GeoJSON>, "name": "optional"}  -> full analysis JSON
GET  /selftest  -> hits every data source for a known Iowa field and reports status
GET  /health
"""
from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field as PField

from .config import settings
from .geo import parse_field
from .scoring import score
from .sources import dem, sentinel, soil, weather

app = FastAPI(title="Farm X wetness engine", version="0.1.0")


class AnalyzeRequest(BaseModel):
    boundary: dict[str, Any] = PField(..., description="GeoJSON Feature, FeatureCollection or geometry in WGS84")
    name: str | None = None


def _auth(x_api_key: str | None) -> None:
    if settings.engine_api_key and x_api_key != settings.engine_api_key:
        raise HTTPException(status_code=401, detail="bad api key")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "time": dt.datetime.utcnow().isoformat() + "Z",
        "open_meteo_commercial": bool(settings.open_meteo_api_key),
        "cdse_configured": bool(settings.cdse_client_id and settings.cdse_client_secret),
    }


async def _run(field) -> dict[str, Any]:
    lon, lat = field.centroid
    t0 = time.time()

    async def guarded(name: str, coro):
        try:
            return name, await coro, None
        except Exception as e:  # noqa: BLE001
            return name, {}, f"{type(e).__name__}: {str(e)[:300]}"

    results = await asyncio.gather(
        guarded("weather", weather.get_weather(lon, lat)),
        guarded("soil", soil.get_soil(lon, lat, field.wkt)),
        guarded("terrain", asyncio.to_thread(dem.analyse_terrain, field)),
        guarded("sentinel", sentinel.get_sentinel(field.geojson(), lat)),
    )
    data = {name: val for name, val, _ in results}
    errors = {name: err for name, _, err in results if err}

    out = {
        "field": {"name": field.name, "centroid": [lon, lat], "area_ha": round(field.area_ha, 2), "bbox": field.bbox},
        "wetness": score(data["soil"], data["terrain"], data["weather"], data["sentinel"]),
        "weather": data["weather"],
        "soil": data["soil"],
        "terrain": data["terrain"],
        "sentinel": data["sentinel"],
        "errors": errors,
        "elapsed_s": round(time.time() - t0, 1),
        "engine_version": app.version,
    }
    return out


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _auth(x_api_key)
    try:
        field = parse_field(req.boundary, req.name)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    if field.area_ha > 2000:
        raise HTTPException(status_code=400, detail="boundary over 2000 ha; split it")
    return await _run(field)


# ~64 ha rectangle on the flat Des Moines Lobe prairie north of Ames, Iowa (pothole country,
# the classic tile-drainage landscape). Used only to check the pipeline end to end.
SELFTEST_FIELD = {
    "type": "Feature",
    "properties": {"name": "selftest-story-county-iowa"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[-93.600, 42.180], [-93.590, 42.180], [-93.590, 42.187], [-93.600, 42.187], [-93.600, 42.180]]],
    },
}


@app.get("/selftest")
async def selftest(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _auth(x_api_key)
    field = parse_field(SELFTEST_FIELD)
    res = await _run(field)
    status = {
        "weather": "ok" if res["weather"].get("seasons") else "fail",
        "soilgrids": "ok" if (res["soil"].get("soilgrids") or {}).get("values")
        else ("no_data" if res["soil"].get("soilgrids") else "fail"),
        "ssurgo": "ok" if res["soil"].get("ssurgo") else "fail",
        "terrain": "ok" if "depression_share" in res["terrain"] else "fail",
        "sentinel": "ok" if res["sentinel"].get("s2_monthly") else ("skipped" if res["sentinel"].get("skipped") else "fail"),
    }
    return {"status": status, "errors": res["errors"], "wetness": res["wetness"], "elapsed_s": res["elapsed_s"],
            "detail": {k: res[k] for k in ("weather", "soil", "terrain")}, "sentinel_summary": {
                k: res["sentinel"].get(k) for k in ("skipped", "s2_error", "s1_error", "ndvi_seasons", "surface_water_months", "s1_low_backscatter_periods")}}
