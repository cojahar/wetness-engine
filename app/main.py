"""Farm X wetness engine: HTTP service.

POST /analyze   {"boundary": <GeoJSON>, "name": "optional"}  -> full analysis JSON
GET  /selftest  -> hits every data source for a known Iowa field and reports status
GET  /health
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field as PField

from . import jobs
from .config import settings
from .geo import parse_field, square_around
from .scoring import score
from .sources import boundary, dem, sentinel, soil, weather

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


@app.get("/boundary")
async def get_boundary(lat: float, lon: float, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Snap a pin to a field boundary (USA only, from the USDA Cropland Data Layer)."""
    _auth(x_api_key)
    try:
        snap = await boundary.snap_to_field(lon, lat)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"boundary service failed: {type(e).__name__}: {str(e)[:200]}")
    return snap or {"found": False, "reason": "no CDL coverage here (outside the contiguous US?)"}


@app.get("/analyze/point")
async def analyze_point(lat: float, lon: float, side_m: float = 400, name: str | None = None,
                        snap: bool = True, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Start an analysis from a pin. With snap=true (default) the pin is first snapped to a
    field boundary from the USDA Cropland Data Layer; if that fails (outside the US, or not
    cropland) a square of side_m metres centred on the pin is used instead.
    Returns a job id at once; fetch the result from /jobs/{id}."""
    _auth(x_api_key)
    if not (50 <= side_m <= 3000):
        raise HTTPException(status_code=400, detail="side_m must be between 50 and 3000")
    label = name or f"{lat:.4f},{lon:.4f}"

    async def job() -> dict[str, Any]:
        field = None
        snap_info: dict[str, Any] = {"used": False}
        if snap:
            try:
                s = await boundary.snap_to_field(lon, lat, name=label)
                if s and s.get("found"):
                    field = boundary.field_from_snap(s, label)
                    snap_info = {"used": True, **{k: v for k, v in s.items() if k != "geometry"}}
                else:
                    snap_info = {"used": False, "reason": (s or {}).get("reason", "no CDL coverage")}
            except Exception as e:  # noqa: BLE001
                snap_info = {"used": False, "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        if field is None:
            field = square_around(lon, lat, side_m, label)
            snap_info["fallback"] = f"{side_m:.0f} m square"
        res = await _run(field)
        res["boundary"] = snap_info
        res["field"]["geometry"] = field.geojson()
        return res

    job_id = jobs.start(label, job)
    return {"job_id": job_id, "check": f"/jobs/{job_id}"}


@app.get("/jobs")
async def list_jobs(x_api_key: str | None = Header(default=None)) -> list[dict[str, Any]]:
    _auth(x_api_key)
    return jobs.summaries()


@app.get("/jobs/{job_id}")
@app.get("/jobs/{job_id}/v/{nonce}")  # nonce defeats intermediate caches; ignored
async def get_job(job_id: str, nonce: str | None = None, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _auth(x_api_key)
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="no such job")
    return j


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


_last_selftest: dict[str, Any] = {}


async def _selftest_job() -> dict[str, Any]:
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
    out = {"status": status, "errors": res["errors"], "wetness": res["wetness"], "elapsed_s": res["elapsed_s"],
           "finished_at": dt.datetime.utcnow().isoformat() + "Z",
           "detail": {k: res[k] for k in ("weather", "soil", "terrain")}, "sentinel_summary": {
               k: res["sentinel"].get(k) for k in ("skipped", "s2_error", "s1_error", "ndvi_seasons", "surface_water_months", "s1_low_backscatter_periods")}}
    _last_selftest.clear()
    _last_selftest.update(out)
    # One-line summary in the service log so it can be read without a long HTTP call.
    print("SELFTEST", json.dumps({"status": status, "errors": res["errors"], "score": res["wetness"].get("score"),
                                  "confidence": res["wetness"].get("confidence"), "elapsed_s": res["elapsed_s"]}), flush=True)
    return out


@app.get("/selftest")
async def selftest(x_api_key: str | None = Header(default=None), background: bool = False) -> dict[str, Any]:
    """Run the pipeline on the known field. With ?background=true it returns at once and the
    result lands in /selftest/last and in the service log."""
    _auth(x_api_key)
    if background:
        asyncio.create_task(_selftest_job())
        return {"started": True, "check": "/selftest/last"}
    return await _selftest_job()


@app.get("/selftest/last")
async def selftest_last(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _auth(x_api_key)
    return _last_selftest or {"status": "no run yet"}
