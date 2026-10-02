"""Tiny in-process job store so long analyses can be started with one request and
collected with another. Railway runs one replica, so a dict is enough for now; the
Supabase job table replaces this when the front end arrives.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import uuid
from typing import Any, Awaitable, Callable

from . import store

_jobs: dict[str, dict[str, Any]] = {}
_MAX = 200


def start(name: str | None, coro_factory: Callable[[], Awaitable[dict[str, Any]]]) -> str:
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {"id": job_id, "name": name, "status": "running",
                     "started_at": dt.datetime.utcnow().isoformat() + "Z", "result": None, "error": None}
    if len(_jobs) > _MAX:
        for k in list(_jobs)[: len(_jobs) - _MAX]:
            _jobs.pop(k, None)

    async def runner() -> None:
        try:
            _jobs[job_id]["result"] = await coro_factory()
            _jobs[job_id]["status"] = "done"
        except Exception as e:  # noqa: BLE001
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        _jobs[job_id]["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
        j = _jobs[job_id]
        r = j.get("result") or {}
        try:
            await asyncio.to_thread(store.save, j, _summary(j))
        except Exception as e:  # noqa: BLE001
            print(f"job store save failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        print("JOB", json.dumps({"id": job_id, "name": name, "status": j["status"], "error": j.get("error"),
                                 "score": (r.get("wetness") or {}).get("score"),
                                 "confidence": (r.get("wetness") or {}).get("confidence"),
                                 "area_ha": (r.get("field") or {}).get("area_ha"),
                                 "boundary": {k: v for k, v in (r.get("boundary") or {}).items() if k != "note"},
                                 "zones": {k: v for k, v in (r.get("zones") or {}).items()
                                           if k in ("skipped", "errors", "problem_share", "watch_share", "problem_ha",
                                                    "watch_ha", "seasons_used", "field_mean_stress_days_per_season",
                                                    "field_spring_ponding_share", "radar_used", "legacy_max_method")},
                                 "zone_positions": [(z.get("position"), z.get("area_ha"))
                                                    for z in (r.get("zones") or {}).get("problem_zones", [])][:6],
                                 "economics": {k: (r.get("economics") or {}).get(k) for k in
                                               ("crop_basis", "expected_yield_gain_pct")} | {
                                     "own_plow_mid": ((r.get("economics") or {}).get("own_plow") or {}).get("mid"),
                                     "contractor_mid": ((r.get("economics") or {}).get("contractor") or {}).get("mid")},
                                 "errors": r.get("errors"),
                                 "elapsed_s": r.get("elapsed_s")}), flush=True)

    asyncio.create_task(runner())
    return job_id


def _summary(j: dict[str, Any]) -> dict[str, Any]:
    r = j.get("result") or {}
    w = r.get("wetness") or {}
    f = r.get("field") or {}
    z = r.get("zones") or {}
    e = r.get("economics") or {}
    c = f.get("centroid") or [None, None]
    return {"id": j["id"], "name": j["name"], "status": j["status"], "started_at": j["started_at"],
            "finished_at": j.get("finished_at"), "score": w.get("score"), "confidence": w.get("confidence"),
            "label": w.get("label"), "area_ha": f.get("area_ha"), "lon": c[0], "lat": c[1],
            "problem_share": z.get("problem_share"), "expected_yield_gain_pct": e.get("expected_yield_gain_pct"),
            "own_plow_payback_years": ((e.get("own_plow") or {}).get("mid") or {}).get("simple_payback_years"),
            "error": j.get("error")}


def get(job_id: str) -> dict[str, Any] | None:
    j = _jobs.get(job_id)
    if j is not None:
        return j
    j = store.load(job_id)
    if j is not None:
        _jobs[job_id] = j  # warm the cache
    return j


def summaries() -> list[dict[str, Any]]:
    """Running and recent jobs from memory, merged with the durable index (newest first)."""
    mem = {j["id"]: _summary(j) for j in _jobs.values()}
    out = dict(mem)
    for s in store.index():
        out.setdefault(s["id"], s)
    return sorted(out.values(), key=lambda s: s.get("started_at") or "", reverse=True)
