"""Durable job store on the Railway bucket (S3 API). One replica, so a single index object
is safe. Objects: jobs/{id}.json (full job incl. result) and jobs/_index.json (summaries,
newest first, capped). When the bucket is not configured everything is a no-op and the
in-process dict in jobs.py is all there is.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from .config import settings

_INDEX_KEY = "jobs/_index.json"
_INDEX_MAX = 500
_lock = threading.Lock()
_client = None


def enabled() -> bool:
    return bool(settings.s3_endpoint and settings.s3_bucket and settings.s3_access_key_id)


def _s3():
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("s3", endpoint_url=settings.s3_endpoint, aws_access_key_id=settings.s3_access_key_id,
                               aws_secret_access_key=settings.s3_secret_access_key, region_name=settings.s3_region)
    return _client


def _get_json(key: str) -> Any | None:
    try:
        obj = _s3().get_object(Bucket=settings.s3_bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:  # noqa: BLE001  (NoSuchKey and friends)
        return None


def _put_json(key: str, data: Any) -> None:
    _s3().put_object(Bucket=settings.s3_bucket, Key=key, Body=json.dumps(data).encode(), ContentType="application/json")


def save(job: dict[str, Any], summary: dict[str, Any]) -> None:
    """Blocking; call from a thread."""
    if not enabled():
        return
    _put_json(f"jobs/{job['id']}.json", job)
    with _lock:
        index = _get_json(_INDEX_KEY) or []
        index = [s for s in index if s.get("id") != job["id"]]
        index.insert(0, summary)
        _put_json(_INDEX_KEY, index[:_INDEX_MAX])


def load(job_id: str) -> dict[str, Any] | None:
    if not enabled():
        return None
    return _get_json(f"jobs/{job_id}.json")


def index() -> list[dict[str, Any]]:
    if not enabled():
        return []
    return _get_json(_INDEX_KEY) or []


_SIGNUPS_KEY = "pilot/signups.json"


def add_signup(entry: dict[str, Any]) -> int:
    """Append a pilot sign-up (name, contact, county, notes). Blocking; call from a thread. Returns the count."""
    if not enabled():
        return 0
    with _lock:
        rows = _get_json(_SIGNUPS_KEY) or []
        rows.append(entry)
        _put_json(_SIGNUPS_KEY, rows[-2000:])
        return len(rows)


def signups() -> list[dict[str, Any]]:
    if not enabled():
        return []
    return _get_json(_SIGNUPS_KEY) or []
