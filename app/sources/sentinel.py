"""Sentinel-1 and Sentinel-2 field statistics via the Sentinel Hub Statistical API
on the Copernicus Data Space Ecosystem.

Needs CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (free CDSE account: 10,000 requests and
10,000 processing units per month; the CREODIAS Sentinel Hub plan raises that).
Docs: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Statistical.html
"""
from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Any

import httpx

from ..config import settings

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

_token: dict[str, Any] = {"value": None, "exp": 0.0}
# Sentinel Hub rejects bursts with "Too many execution errors"; keep requests small and few.
_sem = asyncio.Semaphore(2)


async def _post_stats(client: httpx.AsyncClient, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    """POST with up to 3 attempts on 5xx / 429, spaced 3, 6 s."""
    last: httpx.Response | None = None
    for attempt in range(3):
        async with _sem:
            r = await client.post(STATS_URL, headers=headers, json=body)
        if r.status_code < 500 and r.status_code != 429:
            r.raise_for_status()
            return r.json()
        last = r
        await asyncio.sleep(3 * (attempt + 1))
    assert last is not None
    last.raise_for_status()
    return {}

S2_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B03", "B04", "B08", "B11", "SCL", "dataMask"]}],
    output: [
      {id: "ndvi", bands: 1, sampleType: "FLOAT32"},
      {id: "ndmi", bands: 1, sampleType: "FLOAT32"},
      {id: "water", bands: 1, sampleType: "FLOAT32"},
      {id: "dataMask", bands: 1}
    ]
  };
}
function evaluatePixel(s) {
  // keep vegetation, bare soil, water, unclassified; drop cloud, shadow, snow
  var ok = (s.SCL == 4 || s.SCL == 5 || s.SCL == 6 || s.SCL == 7) ? 1 : 0;
  var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  var ndwi = (s.B03 - s.B08) / (s.B03 + s.B08 + 1e-6);
  var ndmi = (s.B08 - s.B11) / (s.B08 + s.B11 + 1e-6);
  return {
    ndvi: [ndvi],
    ndmi: [ndmi],
    water: [(ndwi > 0.0 || s.SCL == 6) ? 1 : 0],
    dataMask: [s.dataMask * ok]
  };
}
"""

S1_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH", "dataMask"]}],
    output: [
      {id: "vv_db", bands: 1, sampleType: "FLOAT32"},
      {id: "vh_db", bands: 1, sampleType: "FLOAT32"},
      {id: "lowvv", bands: 1, sampleType: "FLOAT32"},
      {id: "dataMask", bands: 1}
    ]
  };
}
function evaluatePixel(s) {
  var vv = 10 * Math.log(s.VV + 1e-6) / Math.LN10;
  var vh = 10 * Math.log(s.VH + 1e-6) / Math.LN10;
  return {vv_db: [vv], vh_db: [vh], lowvv: [vv < -18 ? 1 : 0], dataMask: [s.dataMask]};
}
"""


async def _get_token(client: httpx.AsyncClient) -> str | None:
    if not (settings.cdse_client_id and settings.cdse_client_secret):
        return None
    if _token["value"] and time.time() < _token["exp"] - 60:
        return _token["value"]
    r = await client.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": settings.cdse_client_id,
            "client_secret": settings.cdse_client_secret,
        },
    )
    r.raise_for_status()
    js = r.json()
    _token["value"] = js["access_token"]
    _token["exp"] = time.time() + float(js.get("expires_in", 600))
    return _token["value"]


def _body(geometry: dict[str, Any], data_type: str, evalscript: str, start: dt.date, end: dt.date,
          interval: str, extra_data: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"type": data_type}
    if extra_data:
        data.update(extra_data)
    return {
        "input": {
            "bounds": {"geometry": geometry, "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [data],
        },
        "aggregation": {
            "timeRange": {"from": f"{start.isoformat()}T00:00:00Z", "to": f"{end.isoformat()}T23:59:59Z"},
            "aggregationInterval": {"of": interval},
            "evalscript": evalscript,
            "resx": 0.0001,  # ~10 m in degrees
            "resy": 0.0001,
        },
        "calculations": {"default": {"statistics": {"default": {"percentiles": {"k": [10, 50, 90]}}}}},
    }


def _flatten(js: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in js.get("data", []):
        row: dict[str, Any] = {"from": item["interval"]["from"][:10], "to": item["interval"]["to"][:10]}
        for out_name, out in item.get("outputs", {}).items():
            b = out.get("bands", {}).get("B0", {}).get("stats", {})
            if b.get("sampleCount", 0) == 0 or b.get("noDataCount", 0) == b.get("sampleCount", 0):
                continue
            row[f"{out_name}_mean"] = b.get("mean")
            row[f"{out_name}_p10"] = b.get("percentiles", {}).get("10.0")
            row[f"{out_name}_p90"] = b.get("percentiles", {}).get("90.0")
            row["valid_px"] = b.get("sampleCount", 0) - b.get("noDataCount", 0)
        if len(row) > 2:
            rows.append(row)
    return rows


async def get_sentinel(geometry: dict[str, Any], lat: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=180) as client:
        token = await _get_token(client)
        if not token:
            return {"skipped": "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set"}
        headers = {"Authorization": f"Bearer {token}"}
        start = dt.date(settings.history_start_year, 1, 1)
        end = dt.date.today()

        out: dict[str, Any] = {"source": "Sentinel Hub Statistical API on CDSE"}

        # Sentinel-2: one request per calendar year (smaller requests fail far less often).
        s2_rows: list[dict[str, Any]] = []
        s2_errors: list[str] = []
        for year in range(start.year, end.year + 1):
            y0 = dt.date(year, 1, 1)
            y1 = min(dt.date(year, 12, 31), end)
            try:
                js = await _post_stats(client, headers,
                                       _body(geometry, "sentinel-2-l2a", S2_EVALSCRIPT, y0, y1, "P1M",
                                             {"dataFilter": {"maxCloudCoverage": 70}}))
                s2_rows.extend(_flatten(js))
            except httpx.HTTPStatusError as e:
                s2_errors.append(f"{year}: {e.response.status_code} {e.response.text[:160]}")
        if s2_rows:
            out["s2_monthly"] = s2_rows
        if s2_errors:
            out["s2_error"] = "; ".join(s2_errors)

        # Sentinel-1: last two years in two one-year requests.
        s1_rows: list[dict[str, Any]] = []
        s1_errors: list[str] = []
        s1_start = max(start, end - dt.timedelta(days=730))
        for y0, y1 in ((s1_start, s1_start + dt.timedelta(days=365)), (s1_start + dt.timedelta(days=366), end)):
            if y0 >= y1:
                continue
            try:
                js = await _post_stats(client, headers,
                                       _body(geometry, "sentinel-1-grd", S1_EVALSCRIPT, y0, y1, "P12D",
                                             {"processing": {"orthorectify": True, "backCoeff": "GAMMA0_TERRAIN",
                                                             "demInstance": "COPERNICUS"},
                                              "dataFilter": {"acquisitionMode": "IW", "polarization": "DV"}}))
                s1_rows.extend(_flatten(js))
            except httpx.HTTPStatusError as e:
                s1_errors.append(f"{y0}: {e.response.status_code} {e.response.text[:160]}")
        if s1_rows:
            out["s1_12day"] = s1_rows
        if s1_errors:
            out["s1_error"] = "; ".join(s1_errors)

    out.update(_derive(out, lat))
    return out


def _derive(out: dict[str, Any], lat: float) -> dict[str, Any]:
    """Season-level NDVI and surface-water signals from the monthly rows."""
    rows = out.get("s2_monthly") or []
    northern = lat >= 0
    peak_months = {6, 7, 8} if northern else {8, 9, 10}  # crop peak; WA cereals peak Aug-Oct
    by_season: dict[str, list[dict[str, Any]]] = {}
    water_months = []
    for r in rows:
        d = dt.date.fromisoformat(r["from"])
        if northern:
            key = str(d.year)
        else:
            key = f"{d.year}-{d.year + 1}" if d.month >= 10 else f"{d.year - 1}-{d.year}"
        if d.month in peak_months and r.get("ndvi_mean") is not None:
            by_season.setdefault(key, []).append(r)
        # Snow and ice score as "water" on NDWI, so in the northern hemisphere only count
        # April to October. Southern-hemisphere cropland we serve (WA) has no snow season.
        frost_free = (4 <= d.month <= 10) if northern else True
        if r.get("water_mean") is not None and frost_free:
            water_months.append({"month": r["from"][:7], "water_share": round(r["water_mean"], 3)})
    seasons = {}
    for k, rs in by_season.items():
        seasons[k] = {
            "peak_ndvi_mean": round(max(x["ndvi_mean"] for x in rs), 3),
            "peak_ndvi_p10": round(max(x.get("ndvi_p10") or 0 for x in rs), 3),
            # spread between the best and worst tenth of the field at peak: within-field unevenness
            "peak_ndvi_spread": round(max((x.get("ndvi_p90") or 0) - (x.get("ndvi_p10") or 0) for x in rs), 3),
        }
    s1 = out.get("s1_12day") or []
    low_vv = [{"period": r["from"], "low_vv_share": round(r["lowvv_mean"], 3)} for r in s1 if r.get("lowvv_mean") is not None]
    return {
        "ndvi_seasons": dict(sorted(seasons.items())),
        "surface_water_months": sorted(water_months, key=lambda m: -m["water_share"])[:6],
        "s1_low_backscatter_periods": sorted(low_vv, key=lambda m: -m["low_vv_share"])[:6],
    }
