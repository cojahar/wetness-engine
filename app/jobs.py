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
        print("JOB", json.dumps({"id": job_id, "name": name, "status": j["status"], "error": j.get("error"),
                                 "score": (r.get("wetness") or {}).get("score"),
                                 "confidence": (r.get("wetness") or {}).get("confidence"),
                                 "area_ha": (r.get("field") or {}).get("area_ha"),
                                 "boundary": {k: v for k, v in (r.get("boundary") or {}).items() if k != "note"},
                                 "zones": {k: v for k, v in (r.get("zones") or {}).items()
                                           if k in ("skipped", "errors", "problem_share", "watch_share", "problem_ha",
                                                    "watch_ha", "seasons_used")},
                                 "zone_positions": [(z.get("position"), z.get("area_ha"))
                                                    for z in (r.get("zones") or {}).get("problem_zones", [])][:6],
                                 "errors": r.get("errors"),
                                 "elapsed_s": r.get("elapsed_s")}), flush=True)

    asyncio.create_task(runner())
    return job_id


def get(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def summaries() -> list[dict[str, Any]]:
    out = []
    for j in _jobs.values():
        r = j.get("result") or {}
        w = r.get("wetness") or {}
        out.append({"id": j["id"], "name": j["name"], "status": j["status"], "started_at": j["started_at"],
                    "score": w.get("score"), "confidence": w.get("confidence"), "label": w.get("label"),
                    "error": j.get("error")})
    return out
