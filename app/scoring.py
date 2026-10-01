"""Combine soil, terrain, weather and satellite signals into a chronic-wetness score.

This is a screening score (0-100) with a confidence label, not a measurement.
Each component is scaled 0-1 and weighted; missing components reduce confidence
rather than silently lowering the score.
"""
from __future__ import annotations

from typing import Any

WEIGHTS = {
    "soil": 0.30,       # drainage class / clay
    "terrain": 0.20,    # depressions and low relative ground
    "satellite": 0.35,  # wet-year NDVI penalty, unevenness, standing water, SAR
    "climate": 0.15,    # rain minus ET in the growing season
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score(soil: dict[str, Any], terrain: dict[str, Any], weather: dict[str, Any],
          sentinel: dict[str, Any]) -> dict[str, Any]:
    parts: dict[str, dict[str, Any]] = {}

    # Soil
    ss = soil.get("ssurgo")
    sg = (soil.get("soilgrids") or {}).get("values", {})
    if ss:
        v = ss["poorly_drained_component_share"]
        parts["soil"] = {"value": v, "basis": "SSURGO share of poorly drained components"}
    elif "clay_0-30cm" in sg:
        clay = sg["clay_0-30cm"]
        v = _clamp((clay - 20) / 25)  # 20% clay -> 0, 45% -> 1
        parts["soil"] = {"value": round(v, 2), "basis": f"SoilGrids topsoil clay {clay}% (proxy, no drainage class)"}

    # Terrain
    if terrain and "depression_share" in terrain:
        v = _clamp(terrain["depression_share"] * 2 + terrain["low_relative_share"]) \
            * (0.6 if terrain["mean_slope_pct"] < 0.5 else 1.0)  # flat ground: DEM less trustworthy
        parts["terrain"] = {"value": round(v, 2), "basis": "Copernicus 30 m depressions and low relative ground"}

    # Climate
    seasons = weather.get("seasons") or {}
    if seasons:
        bal = [s["balance_mm"] for s in seasons.values()]
        mean_bal = sum(bal) / len(bal)
        v = _clamp((mean_bal + 100) / 300)  # -100 mm -> 0, +200 mm -> 1
        parts["climate"] = {"value": round(v, 2), "basis": f"mean growing-season rain minus ET0 {mean_bal:.0f} mm"}

    # Satellite
    nd = sentinel.get("ndvi_seasons") or {}
    wet = set(weather.get("wet_seasons") or [])
    dry = set(weather.get("dry_seasons") or [])
    sat_components = []
    if nd and wet and dry:
        wet_ndvi = [nd[k]["peak_ndvi_mean"] for k in nd if k in wet]
        dry_ndvi = [nd[k]["peak_ndvi_mean"] for k in nd if k in dry]
        if wet_ndvi and dry_ndvi:
            penalty = (sum(dry_ndvi) / len(dry_ndvi)) - (sum(wet_ndvi) / len(wet_ndvi))
            sat_components.append(("wet_year_ndvi_penalty", _clamp(penalty / 0.15), round(penalty, 3)))
    if nd:
        spread = sum(s["peak_ndvi_spread"] for s in nd.values()) / len(nd)
        sat_components.append(("within_field_unevenness", _clamp((spread - 0.15) / 0.25), round(spread, 3)))
    water = sentinel.get("surface_water_months") or []
    if water:
        w = max(m["water_share"] for m in water)
        sat_components.append(("max_monthly_surface_water_share", _clamp(w / 0.1), w))
    s1 = sentinel.get("s1_low_backscatter_periods") or []
    if s1:
        w = max(m["low_vv_share"] for m in s1)
        sat_components.append(("max_sar_low_backscatter_share", _clamp(w / 0.2), w))
    if sat_components:
        v = sum(c[1] for c in sat_components) / len(sat_components)
        parts["satellite"] = {"value": round(v, 2), "basis": {c[0]: c[2] for c in sat_components}}

    total_w = sum(WEIGHTS[k] for k in parts)
    if total_w == 0:
        return {"score": None, "confidence": "none", "components": parts}
    s = sum(WEIGHTS[k] * parts[k]["value"] for k in parts) / total_w * 100
    confidence = "high" if total_w >= 0.95 and "ssurgo" in soil else "medium" if total_w >= 0.65 else "low"
    if terrain.get("mean_slope_pct", 1) < 0.5 and confidence == "high":
        confidence = "medium"  # very flat: DEM depressions unreliable
    label = "likely poorly drained" if s >= 60 else "possible drainage limitation" if s >= 35 else "no strong wetness signal"
    return {
        "score": round(s, 1),
        "label": label,
        "confidence": confidence,
        "components": parts,
        "weights_used": {k: WEIGHTS[k] for k in parts},
        "note": "Screening score from public data. Flags salinity, compaction and nutrient stress as wetness too; verify in the field before quoting yield impact.",
    }
