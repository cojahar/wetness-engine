"""Open-Meteo historical weather (ERA5-Land based) for a field centroid.

Docs: https://open-meteo.com/en/docs/historical-weather-api
Commercial plans use the customer-* hostnames with an apikey parameter.
"""
from __future__ import annotations

import asyncio
import datetime as dt
from collections import defaultdict
from typing import Any

import httpx

from ..config import settings

DAILY_VARS = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "temperature_2m_mean",
]
HOURLY_SOIL_VARS = [
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
]


def _archive_url() -> str:
    if settings.open_meteo_api_key:
        return "https://customer-archive-api.open-meteo.com/v1/archive"
    return "https://archive-api.open-meteo.com/v1/archive"


class WeatherDataError(RuntimeError):
    pass


def _valid_fraction(raw: dict[str, Any]) -> float:
    p = (raw.get("daily") or {}).get("precipitation_sum") or []
    if not p:
        return 0.0
    return sum(1 for v in p if v is not None) / len(p)


async def fetch_history(lon: float, lat: float, start: dt.date, end: dt.date) -> dict[str, Any]:
    """Daily history. Open-Meteo occasionally answers 200 with every value null; we never
    let that through as 'zero rain', so a null-heavy answer is retried once, then raised."""
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
    }
    if settings.open_meteo_api_key:
        params["apikey"] = settings.open_meteo_api_key
    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(2):
            r = await client.get(_archive_url(), params=params)
            r.raise_for_status()
            raw = r.json()
            frac = _valid_fraction(raw)
            if frac >= 0.9:
                return raw
            if attempt == 0:
                await asyncio.sleep(3)
    raise WeatherDataError(f"Open-Meteo returned {frac:.0%} valid daily values; refusing to score on it")


def summarise(raw: dict[str, Any], lat: float) -> dict[str, Any]:
    """Per growing-season water balance and a wet/dry classification of each year.

    Growing season: Apr-Sep in the northern hemisphere, Oct-Mar (spanning years) in the south.
    Returns totals per season, a z-score of season rainfall against the record, and the list of
    wet and dry seasons. Also returns the top rain events (for Sentinel-1 after-rain sampling).
    """
    daily = raw.get("daily", {})
    dates = [dt.date.fromisoformat(d) for d in daily.get("time", [])]
    p = daily.get("precipitation_sum") or []
    et0 = daily.get("et0_fao_evapotranspiration") or []

    northern = lat >= 0
    seasons: dict[str, dict[str, float]] = defaultdict(lambda: {"rain_mm": 0.0, "et0_mm": 0.0, "days": 0})
    for d, pi, ei in zip(dates, p, et0):
        if northern:
            if 4 <= d.month <= 9:
                key = str(d.year)
            else:
                continue
        else:
            if d.month >= 10:
                key = f"{d.year}-{d.year + 1}"
            elif d.month <= 3:
                key = f"{d.year - 1}-{d.year}"
            else:
                continue
        s = seasons[key]
        s["rain_mm"] += pi or 0.0
        s["et0_mm"] += ei or 0.0
        s["days"] += 1

    # Drop partial seasons (fewer than 150 of ~183 days).
    full = {k: v for k, v in seasons.items() if v["days"] >= 150}
    if not full:
        return {"seasons": {}, "wet_seasons": [], "dry_seasons": [], "rain_events": []}
    rains = [v["rain_mm"] for v in full.values()]
    mean = sum(rains) / len(rains)
    sd = (sum((x - mean) ** 2 for x in rains) / max(len(rains) - 1, 1)) ** 0.5 or 1.0
    for v in full.values():
        v["rain_z"] = round((v["rain_mm"] - mean) / sd, 2)
        v["balance_mm"] = round(v["rain_mm"] - v["et0_mm"], 1)
        v["rain_mm"] = round(v["rain_mm"], 1)
        v["et0_mm"] = round(v["et0_mm"], 1)
    wet = sorted([k for k, v in full.items() if v["rain_z"] >= 0.5])
    dry = sorted([k for k, v in full.items() if v["rain_z"] <= -0.5])

    # Largest 3-day rain totals in the last two years: candidate dates for SAR wetness checks.
    events = []
    for i in range(2, len(dates)):
        tot = sum((p[j] or 0.0) for j in range(i - 2, i + 1))
        if tot >= 40 and dates[i] >= dates[-1] - dt.timedelta(days=730):
            events.append({"end_date": dates[i].isoformat(), "rain_3day_mm": round(tot, 1)})
    events = sorted(events, key=lambda e: -e["rain_3day_mm"])[:8]

    return {
        "seasons": dict(sorted(full.items())),
        "season_rain_mean_mm": round(mean, 1),
        "wet_seasons": wet,
        "dry_seasons": dry,
        "rain_events": events,
    }


async def get_weather(lon: float, lat: float) -> dict[str, Any]:
    start = dt.date(settings.history_start_year, 1, 1)
    end = dt.date.today() - dt.timedelta(days=6)  # ERA5-Land has a ~5 day lag
    raw = await fetch_history(lon, lat, start, end)
    out = summarise(raw, lat)
    out["source"] = "Open-Meteo historical (ERA5-Land)"
    out["commercial_endpoint"] = bool(settings.open_meteo_api_key)
    return out
