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
    "climate": 0.15,    # planting-window rain
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

    # Terrain. A depression only holds water if the soil under it is slow, so on soils
    # SSURGO calls well drained (glacial outwash kettles, sandy ridges) terrain counts for less.
    if terrain and "depression_share" in terrain:
        lidar = float(terrain.get("depression_threshold_m", 0.3)) <= 0.2
        if lidar:
            # LiDAR at 5 m with a 0.15 m threshold counts only real closed basins, so a small share is a
            # strong signal: 3.6% of a Des Moines Lobe pothole field (max depth 1.2 m) must not read 0.15.
            # 8% of the field in basins -> 1.0; low relative ground counts double. Flat ground is trusted.
            v = _clamp(terrain["depression_share"] / 0.08 + terrain["low_relative_share"] * 2)
        else:
            v = _clamp(terrain["depression_share"] * 2 + terrain["low_relative_share"]) \
                * (0.6 if terrain["mean_slope_pct"] < 0.5 else 1.0)  # flat ground: 30 m DEM less trustworthy
        soil_factor = 1.0
        basis = f"{terrain.get('source', 'DEM')}: depressions and low relative ground"
        if ss:
            poor = ss["poorly_drained_component_share"]
            soil_factor = 1.0 if poor >= 0.5 else 0.6 if poor > 0 else 0.3
            if soil_factor < 1.0:
                basis += f"; damped x{soil_factor} because SSURGO soils are mostly well drained"
        parts["terrain"] = {"value": round(v * soil_factor, 2), "basis": basis}

    # Climate: how wet the planting window is. Whole-season rain minus ET0 is negative across the
    # entire Corn Belt (summer ET wins), so it read 0.0 everywhere and only dragged scores down.
    # Spring (Apr-Jun north, May-Jul south) rain is what delays planting and drowns seedlings:
    # 150 mm -> 0, 350 mm -> 1 (central Iowa ~300 mm, western Nebraska ~180, central Illinois ~320).
    seasons = weather.get("seasons") or {}
    if seasons and weather.get("spring_rain_mean_mm") is not None:
        sr = float(weather["spring_rain_mean_mm"])
        v = _clamp((sr - 150.0) / 200.0)
        parts["climate"] = {"value": round(v, 2),
                            "basis": f"mean planting-window (months {weather.get('spring_months')}) rain {sr:.0f} mm; "
                                     f"spring rain minus ET0 {weather.get('spring_balance_mean_mm', 0):+.0f} mm"}
    elif seasons:
        bal = [s["balance_mm"] for s in seasons.values()]
        mean_bal = sum(bal) / len(bal)
        v = _clamp((mean_bal + 100) / 300)
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
    weights = dict(WEIGHTS)
    if sat_components:
        v = sum(c[1] for c in sat_components) / len(sat_components)
        # Four signals make up the satellite component; with fewer present it earns less weight.
        weights["satellite"] = WEIGHTS["satellite"] * len(sat_components) / 4
        parts["satellite"] = {"value": round(v, 2), "basis": {c[0]: c[2] for c in sat_components},
                              "signals_present": len(sat_components)}

    total_w = sum(weights[k] for k in parts)
    if total_w == 0:
        return {"score": None, "confidence": "none", "components": parts}
    s = sum(weights[k] * parts[k]["value"] for k in parts) / total_w * 100
    confidence = "high" if total_w >= 0.95 and "ssurgo" in soil else "medium" if total_w >= 0.65 else "low"
    if (terrain.get("mean_slope_pct", 1) < 0.5 and confidence == "high"
            and float(terrain.get("depression_threshold_m", 0.3)) > 0.2):
        confidence = "medium"  # very flat on a 30 m DEM: depressions unreliable (LiDAR is fine)
    label = "likely poorly drained" if s >= 60 else "possible drainage limitation" if s >= 35 else "no strong wetness signal"
    return {
        "score": round(s, 1),
        "label": label,
        "confidence": confidence,
        "components": parts,
        "weights_used": {k: round(weights[k], 3) for k in parts},
        "note": "Screening score from public data. Flags salinity, compaction and nutrient stress as wetness too; verify in the field before quoting yield impact.",
    }
