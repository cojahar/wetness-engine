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

from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PField

from . import economics, jobs, report, store, validate
from .config import settings
from .geo import parse_field, square_around
from .scoring import score
from .sources import boundary, dem, sentinel, soil, tile, weather, zones

app = FastAPI(title="Farm X wetness engine", version="0.6.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
        "durable_jobs": store.enabled(),
        "version": app.version,
    }


async def _none() -> dict[str, Any]:
    return {"skipped": "not requested"}


async def _run(field, with_zones: bool = True, crop_sequence: list[str] | None = None,
               econ_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
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
        guarded("zones", zones.get_zones(field) if with_zones else _none()),
        guarded("tile", tile.get_tile(field)),
    )
    data = {name: val for name, val, _ in results}
    errors = {name: err for name, _, err in results if err}

    wet = score(data["soil"], data["terrain"], data["weather"], data["sentinel"])
    comps = wet.get("components") or {}
    symptoms = {"satellite": (comps.get("satellite") or {}).get("value"),
                "stress_days": (data["zones"] or {}).get("field_mean_stress_days_per_season"),
                "tile_label": ((data["tile"] or {}).get("agtile") or {}).get("label")}
    try:
        econ = economics.estimate(field.area_ha, wet.get("score"), data["zones"], crop_sequence, econ_overrides, symptoms)
    except Exception as e:  # noqa: BLE001
        econ = {}
        errors["economics"] = f"{type(e).__name__}: {str(e)[:300]}"
    out = {
        "field": {"name": field.name, "centroid": [lon, lat], "area_ha": round(field.area_ha, 2), "bbox": field.bbox},
        "wetness": wet,
        "economics": econ,
        "weather": data["weather"],
        "soil": data["soil"],
        "terrain": data["terrain"],
        "sentinel": data["sentinel"],
        "zones": data["zones"],
        "existing_tile": data["tile"],
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
    """Snap a pin to a field boundary (Fields of The World at 10 m, else the USDA CDL flood fill)."""
    _auth(x_api_key)
    try:
        snap = await boundary.snap_to_field(lon, lat)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"boundary service failed: {type(e).__name__}: {str(e)[:200]}")
    return snap or {"found": False, "reason": "no field boundary or CDL coverage here"}


@app.get("/analyze/point")
async def analyze_point(lat: float, lon: float, side_m: float = 400, name: str | None = None,
                        snap: bool = True, install_cost_per_ac: float | None = None,
                        own_plow_cost_per_ac: float | None = None, price_per_unit: float | None = None,
                        yield_per_ac: float | None = None,
                        x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Start an analysis from a pin. With snap=true (default) the pin is first snapped to a
    field boundary (Fields of The World, else the USDA Cropland Data Layer); if that fails
    a square of side_m metres centred on the pin is used instead.
    Optional economics overrides: install_cost_per_ac, own_plow_cost_per_ac, price_per_unit, yield_per_ac.
    Returns a job id at once; fetch the result from /jobs/{id}."""
    _auth(x_api_key)
    if not (50 <= side_m <= 3000):
        raise HTTPException(status_code=400, detail="side_m must be between 50 and 3000")
    label = name or f"{lat:.4f},{lon:.4f}"
    overrides = {k: v for k, v in {"install_cost_per_ac": install_cost_per_ac, "own_plow_cost_per_ac": own_plow_cost_per_ac,
                                   "price_per_unit": price_per_unit, "yield_per_ac": yield_per_ac}.items() if v is not None}

    async def job() -> dict[str, Any]:
        field = None
        crops: list[str] | None = None
        snap_info: dict[str, Any] = {"used": False}
        if snap:
            try:
                s = await boundary.snap_to_field(lon, lat, name=label)
                if s and s.get("found"):
                    field = boundary.field_from_snap(s, label)
                    crops = s.get("crop_sequence")
                    snap_info = {"used": True, **{k: v for k, v in s.items() if k != "geometry"}}
                else:
                    snap_info = {"used": False, "reason": (s or {}).get("reason", "no CDL coverage")}
            except Exception as e:  # noqa: BLE001
                snap_info = {"used": False, "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        if field is None:
            field = square_around(lon, lat, side_m, label)
            snap_info["fallback"] = f"{side_m:.0f} m square"
        res = await _run(field, crop_sequence=crops, econ_overrides=overrides)
        res["boundary"] = snap_info
        res["field"]["geometry"] = field.geojson()
        return res

    job_id = jobs.start(label, job)
    return {"job_id": job_id, "check": f"/jobs/{job_id}"}


class AnalyzePolygonRequest(AnalyzeRequest):
    install_cost_per_ac: float | None = None
    own_plow_cost_per_ac: float | None = None
    price_per_unit: float | None = None
    yield_per_ac: float | None = None


@app.post("/analyze/polygon")
async def analyze_polygon(req: AnalyzePolygonRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Start an analysis from a drawn boundary (GeoJSON). Returns a job id like /analyze/point."""
    _auth(x_api_key)
    try:
        field = parse_field(req.boundary, req.name)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    if field.area_ha > 2000:
        raise HTTPException(status_code=400, detail="boundary over 2000 ha; split it")
    if field.area_ha < 0.2:
        raise HTTPException(status_code=400, detail="boundary under half an acre; draw the whole field")
    overrides = {k: v for k, v in {"install_cost_per_ac": req.install_cost_per_ac, "own_plow_cost_per_ac": req.own_plow_cost_per_ac,
                                   "price_per_unit": req.price_per_unit, "yield_per_ac": req.yield_per_ac}.items() if v is not None}
    label = field.name or f"drawn field {field.area_ha:.1f} ha"

    async def job() -> dict[str, Any]:
        crops: list[str] | None = None
        try:  # crop rotation from the hosted crop map, for the economics basis (US only)
            lon, lat = field.centroid
            s = await boundary.snap_to_field(lon, lat, name=label, prefer="cdl")
            if s and s.get("found"):
                crops = s.get("crop_sequence")
        except Exception:  # noqa: BLE001
            pass
        res = await _run(field, crop_sequence=crops, econ_overrides=overrides)
        res["boundary"] = {"used": False, "method": "drawn by the user", "crop_sequence": crops}
        res["field"]["geometry"] = field.geojson()
        return res

    job_id = jobs.start(label, job)
    return {"job_id": job_id, "check": f"/jobs/{job_id}"}


@app.post("/zones")
async def zones_polygon(req: AnalyzeRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Within-field wet-zone map only (10 m, Sentinel-2 since 2019) for a drawn boundary."""
    _auth(x_api_key)
    try:
        field = parse_field(req.boundary, req.name)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    return await zones.get_zones(field)


@app.get("/zones/point")
async def zones_point(lat: float, lon: float, side_m: float = 400, name: str | None = None,
                      x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Zone map from a pin: snaps to a field boundary, else a square. Runs as a job."""
    _auth(x_api_key)
    label = name or f"zones {lat:.4f},{lon:.4f}"

    async def job() -> dict[str, Any]:
        field = None
        try:
            s = await boundary.snap_to_field(lon, lat, name=label)
            if s and s.get("found"):
                field = boundary.field_from_snap(s, label)
        except Exception:  # noqa: BLE001
            pass
        if field is None:
            field = square_around(lon, lat, side_m, label)
        t0 = time.time()
        z = await zones.get_zones(field)
        return {"field": {"name": label, "area_ha": round(field.area_ha, 2), "geometry": field.geojson()},
                "zones": z, "elapsed_s": round(time.time() - t0, 1)}

    job_id = jobs.start(label, job)
    return {"job_id": job_id, "check": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}/report.pdf")
async def job_report(job_id: str, prepared_by: str = "Farm X", x_api_key: str | None = Header(default=None)) -> Response:
    """Customer PDF for a finished job."""
    _auth(x_api_key)
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="no such job")
    if j["status"] != "done":
        raise HTTPException(status_code=409, detail=f"job is {j['status']}")
    pdf = await asyncio.to_thread(report.build_pdf, j["result"], prepared_by)
    name = (j.get("name") or job_id).replace(" ", "_").replace(",", "_")
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="drainage-assessment-{name}.pdf"'})


@app.post("/jobs/{job_id}/yield")
async def upload_yield(job_id: str, file: UploadFile = File(...), crop_year: int | None = Form(None),
                       units: str = Form("bu/ac"), x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Compare a farmer's yield-monitor export (CSV, GeoJSON or zipped shapefile) with the zone map.

    Stores the comparison on the job as result["yield_check"] so the front end and the PDF can show
    it. This is the accuracy check: the farmer's own combine against our problem zones.
    """
    _auth(x_api_key)
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="no such job")
    if j["status"] != "done":
        raise HTTPException(status_code=409, detail=f"job is {j['status']}")
    grid = ((j.get("result") or {}).get("zones") or {}).get("grid")
    if not grid:
        raise HTTPException(status_code=409, detail="this job has no zone grid (older job or zone map skipped); re-run the field")
    data = await file.read()
    if len(data) > 60_000_000:
        raise HTTPException(status_code=413, detail="file over 60 MB; export a single field, not the whole farm")
    try:
        pts, note = await asyncio.to_thread(validate.parse_points, file.filename or "upload.csv", data)
        out = await asyncio.to_thread(validate.compare, grid, pts, units)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    out["crop_year"] = crop_year
    out["source"] = note | {"filename": file.filename, "points_in_file": int(pts.shape[0])}
    out["checked_at"] = dt.datetime.utcnow().isoformat() + "Z"
    j["result"]["yield_check"] = out
    await asyncio.to_thread(jobs.update, job_id)
    print("YIELD", json.dumps({"id": job_id, "verdict": out["verdict"], "r": out["correlation_stress_vs_yield"],
                               "problem_zone": out.get("problem_zone")}), flush=True)
    return out


class ReportRequest(BaseModel):
    result: dict[str, Any] = PField(..., description="A full analysis result as returned by /analyze or /jobs/{id}.result")
    prepared_by: str = "Farm X"


@app.post("/report")
async def report_from_result(req: ReportRequest, x_api_key: str | None = Header(default=None)) -> Response:
    """Customer PDF from a stored result (the front end keeps results in its own database)."""
    _auth(x_api_key)
    pdf = await asyncio.to_thread(report.build_pdf, req.result, req.prepared_by)
    return Response(pdf, media_type="application/pdf")


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
    res = await _run(field, with_zones=False)
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
