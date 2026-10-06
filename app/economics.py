"""Yield-loss and payback estimate for tile drainage.

Everything here is an assumption-driven screening estimate, not an appraisal. Every
default is overridable per request so a dealer can plug in local prices and bids. The
report must print the assumptions next to the numbers.

Evidence behind the defaults (Oct 2026):
- Corn: >10% yield gain in the first three years after tiling, 20-35% in year four, nine
  Illinois case-study fields (Illinois Extension / K. Brooks).
- Soybeans: +8% in experiments, +4% on producer fields across the North Central US,
  partly from earlier sowing (Agricultural Water Management, 2020).
- Wet spots inside a field commonly lose 30-60% of yield in wet years; in dry years they
  can out-yield the field. We use 40% expected recovery on persistent problem zones and
  15% on watch zones.
- Pattern tile by contractor: $800-1,500/ac in 2017 Midwest bids; 2026 default $1,200/ac.
  Own plow (Soil-Max): pipe plus mains plus fuel and time, default $550/ac. Override!
- Prices 2 Oct 2026 futures: corn $4.99, soybeans $12.77, wheat $6.86 per bu; cash
  defaults below take ~$0.40 basis off.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

HA_PER_AC = 0.404686

# bu/ac typical for drained Midwest ground, and cash $/bu
CROPS: dict[str, dict[str, float]] = {
    "corn": {"yield": 190.0, "price": 4.60, "response": 1.0},
    "soybeans": {"yield": 58.0, "price": 12.30, "response": 0.6},
    "spring wheat": {"yield": 55.0, "price": 6.40, "response": 0.6},
    "winter wheat": {"yield": 75.0, "price": 6.00, "response": 0.6},
    "durum wheat": {"yield": 50.0, "price": 7.50, "response": 0.6},
    "sugarbeets": {"yield": 28.0, "price": 55.0, "response": 0.8},   # tons/ac, $/ton
    "canola": {"yield": 40.0, "price": 10.0, "response": 0.6},
    "oats": {"yield": 80.0, "price": 3.30, "response": 0.5},
    "barley": {"yield": 70.0, "price": 5.00, "response": 0.5},
    "alfalfa": {"yield": 4.5, "price": 180.0, "response": 0.7},     # tons/ac
    "sorghum": {"yield": 90.0, "price": 4.20, "response": 0.6},
    "cotton": {"yield": 900.0, "price": 0.70, "response": 0.5},     # lb/ac
    "rice": {"yield": 7500.0, "price": 0.12, "response": 0.0},      # rice wants water
    "dry beans": {"yield": 22.0, "price": 38.0, "response": 0.7},   # cwt/ac
    "potatoes": {"yield": 420.0, "price": 10.0, "response": 0.7},   # cwt/ac
}
DEFAULT_ROTATION = ["corn", "soybeans"]


@dataclass
class Assumptions:
    crop: str = "rotation corn/soybeans"
    yield_per_ac: float = 0.0            # units/ac (bu unless noted)
    price_per_unit: float = 0.0          # $/unit
    gross_per_ac: float = 0.0            # yield x price
    install_cost_per_ac: float = 1200.0  # contractor pattern tile
    own_plow_cost_per_ac: float = 550.0  # pipe, mains, fuel, time with a Soil-Max plow
    discount_rate: float = 0.06
    horizon_years: int = 20
    problem_zone_gain: float = 0.40      # share of yield recovered on persistent wet spots
    watch_zone_gain: float = 0.15
    whole_field_gain: float = 0.0        # from the wetness score and crop response
    scenario_multipliers: tuple[float, float, float] = (0.6, 1.0, 1.4)


def _whole_field_gain(score: float, response: float) -> float:
    """Expected whole-field yield gain (fraction) outside the mapped zones, from the
    field-level wetness score: nothing below 20, 15% at 60, 25% at 80 and above, for corn
    (Illinois case study: 20-35% on fields that needed tile)."""
    if score <= 20:
        g = 0.0
    elif score <= 60:
        g = 0.15 * (score - 20) / 40
    else:
        g = 0.15 + 0.10 * min(score - 60, 20) / 20
    return round(g * response, 4)


def _crop_mix(crop_sequence: list[str] | None) -> list[str]:
    seq = [c for c in (crop_sequence or []) if c in CROPS]
    return seq or DEFAULT_ROTATION


def _annuity_pv(annual: float, rate: float, years: int) -> float:
    if rate <= 0:
        return annual * years
    return annual * (1 - (1 + rate) ** -years) / rate


def estimate(area_ha: float, score: float | None, zones: dict[str, Any] | None,
             crop_sequence: list[str] | None = None, overrides: dict[str, Any] | None = None,
             symptoms: dict[str, Any] | None = None) -> dict[str, Any]:
    """Screening economics for tiling this field. All money in USD, areas reported in both ha and ac.

    symptoms: {"satellite": 0-1 satellite component of the score, "stress_days": field mean stress
    days per season, "soil": 0-1 soil component, "pond": spring ponding share, "tile_label": AgTile-US
    label}. Poorly drained soil only pays for tile when the field is still undrained; the symptoms say
    whether it is, so the whole-field gain is scaled by them and by the public tile map. Stress on
    well-drained soil with no ponding is probably not water, so zone gains are capped there.
    """
    overrides = overrides or {}
    symptoms = symptoms or {}
    acres = area_ha / HA_PER_AC
    mix = _crop_mix(crop_sequence)
    # Rotation average of gross revenue and drainage response
    gross = sum(CROPS[c]["yield"] * CROPS[c]["price"] for c in mix) / len(mix)
    response = sum(CROPS[c]["response"] for c in mix) / len(mix)
    a = Assumptions(
        crop=" / ".join(dict.fromkeys(mix)),
        yield_per_ac=round(sum(CROPS[c]["yield"] for c in mix) / len(mix), 1),
        price_per_unit=round(sum(CROPS[c]["price"] for c in mix) / len(mix), 2),
        gross_per_ac=round(gross, 0),
    )
    for k, v in overrides.items():
        if hasattr(a, k) and v is not None:
            setattr(a, k, float(v) if k not in ("crop",) else v)
    if "gross_per_ac" in overrides:
        gross = a.gross_per_ac
    elif "yield_per_ac" in overrides or "price_per_unit" in overrides:
        gross = a.yield_per_ac * a.price_per_unit
        a.gross_per_ac = round(gross, 0)

    s = float(score or 0.0)
    sat = symptoms.get("satellite")
    sd = symptoms.get("stress_days")
    # Stress days from the 10 m zone map are the direct measure (every pass, every season); the field-level
    # satellite component mixes in unevenness and radar that read high on variable soils even when the
    # crop is fine, so when both exist the zone map carries most of the weight.
    sat_v = float(sat) if sat is not None else None
    sd_v = min(1.0, float(sd) / 20.0) if sd is not None else None
    if sat_v is not None and sd_v is not None:
        symptom = 0.3 * sat_v + 0.7 * sd_v
    else:
        symptom = sat_v if sat_v is not None else (sd_v or 0.0)
    symptom_known = sat is not None or sd is not None
    symptom_factor = max(0.15, min(1.0, symptom / 0.5)) if symptom_known else 1.0
    tile_label = str(symptoms.get("tile_label") or "")
    tile_factor = 0.5 if tile_label == "appears tiled" else 0.75 if tile_label == "partly tiled" else 1.0
    a.whole_field_gain = _whole_field_gain(s, response) * symptom_factor * tile_factor
    z = zones or {}
    ps = float(z.get("problem_share") or 0.0)
    ws = float(z.get("watch_share") or 0.0)
    rest = max(0.0, 1.0 - ps - ws)
    zones_known = "problem_share" in z

    # Expected yield gain as a fraction of the whole field's gross. Zone stress can come from
    # sand knobs, compaction or pests as well as water, so zone gains are scaled by how
    # plausible wetness is on this field: full weight from score 50 up, a quarter below 15.
    plausibility = max(0.25, min(1.0, (s - 15.0) / 35.0))
    soil_v0 = symptoms.get("soil")
    if soil_v0 is not None and float(soil_v0) < 0.15 and float(symptoms.get("pond") or 0.0) < 0.05:
        plausibility = min(plausibility, 0.1)  # well-drained soil, no ponding: zone stress is probably not water
    gain_frac = (ps * a.problem_zone_gain + ws * a.watch_zone_gain) * plausibility + rest * a.whole_field_gain
    if not zones_known:
        gain_frac = a.whole_field_gain  # no map: rely on the field score alone

    def scenario(mult: float, cost_per_ac: float) -> dict[str, Any]:
        annual = gain_frac * mult * gross * acres
        cost = cost_per_ac * acres
        payback = (cost / annual) if annual > 0 else None
        npv = _annuity_pv(annual, a.discount_rate, a.horizon_years) - cost
        return {
            "annual_benefit_usd": round(annual, 0),
            "annual_benefit_per_ac": round(annual / acres, 0) if acres else 0,
            "install_cost_usd": round(cost, 0),
            "simple_payback_years": round(payback, 1) if payback is not None else None,
            "npv_usd": round(npv, 0),
            "yield_gain_pct_whole_field": round(gain_frac * mult * 100, 1),
        }

    soil_v = symptoms.get("soil")
    pond = float(symptoms.get("pond") or 0.0)
    well_drained = soil_v is not None and float(soil_v) < 0.15 and pond < 0.05
    if well_drained and (ps >= 0.05 or symptom >= 0.5):
        status = ("crop stress shows on the imagery but the soil survey rates this ground well drained and no ponding was seen: "
                  "more likely compaction, fertility, sand or a crop difference than excess water; walk it before quoting tile")
    elif tile_label == "appears tiled" and ps < 0.03 and symptom < 0.4:
        status = "looks drained already: little symptom of excess water and the public tile map says tiled; any case for tile is the mapped wet spots"
    elif symptom_known and symptom < 0.25 and ps < 0.03:
        status = "few symptoms of excess water on the imagery; the soil may be wet by nature but the field is coping"
    elif ps >= 0.05 or symptom >= 0.5:
        status = "clear symptoms of excess water; tile has something to fix here"
    else:
        status = "some symptoms of excess water; worth a look in a wet spring"
    out = {
        "drainage_status": status,
        "symptom_strength": round(symptom, 2),
        "symptom_factor": round(symptom_factor, 2),
        "tile_factor": tile_factor,
        "area_ha": round(area_ha, 2),
        "area_ac": round(acres, 1),
        "crop_basis": a.crop,
        "crops": {c: {"yield_per_ac": CROPS[c]["yield"], "price_per_unit": CROPS[c]["price"]} for c in dict.fromkeys(mix)},
        "zones_used": zones_known,
        "zone_wetness_plausibility": round(plausibility, 2),
        "overridden": bool({"yield_per_ac", "price_per_unit", "gross_per_ac"} & set(overrides)),
        "expected_yield_gain_pct": round(gain_frac * 100, 1),
        "contractor": {k: scenario(m, a.install_cost_per_ac) for k, m in zip(("low", "mid", "high"), a.scenario_multipliers)},
        "own_plow": {k: scenario(m, a.own_plow_cost_per_ac) for k, m in zip(("low", "mid", "high"), a.scenario_multipliers)},
        "assumptions": asdict(a),
        "evidence": [
            "Illinois Extension case study (9 fields): corn >10% gain in years 1-3 after tiling, 20-35% in year 4",
            "Agricultural Water Management 2020, North Central US: soybeans +8% in trials, +4% on producer fields, "
            "partly from earlier sowing",
            "Persistent wet spots: 30-60% loss in wet years; 40% recovery assumed on problem zones, 15% on watch zones",
            f"Whole-field gain from the soil score is scaled by how much excess-water symptom the imagery shows "
            f"(x{symptom_factor:.2f}) and by the public tile map (x{tile_factor:.2f}): poorly drained soil only pays "
            f"for tile when the field is still undrained",
            f"Zone gains are weighted by how plausible wetness is on this field (x{plausibility:.2f}); on soil the survey "
            f"rates well drained with no ponding seen, crop stress is treated as probably not water",
            "Install cost: Midwest contractor bids $800-1,500/ac (2017 forum data, inflated to a $1,200/ac default). "
            "Own-plow cost is a placeholder for the dealer to replace",
            "Prices: 2 Oct 2026 futures corn $4.99, soybeans $12.77, wheat $6.86; cash defaults net of ~$0.40 basis",
        ],
        "caveat": "Screening estimate from public data. Yield response varies with outlet depth, spacing, soil and "
                  "management; the field's own yield maps or a check strip after install will replace these assumptions.",
    }
    return out
