"""Customer-facing PDF: field drainage assessment from one analysis result.

build_pdf(result) -> bytes. Pure Python (reportlab + Pillow), no browser needed. Three
pages: verdict and map; evidence; economics and assumptions. Written for a farmer and
the dealer sitting across a kitchen table, so plain words and the numbers that matter.
"""
from __future__ import annotations

import base64
import datetime as dt
import io
from typing import Any

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HA_TO_AC = 2.47105
GREEN, AMBER, RED, INK, MUTED = "#2E8B57", "#D9A400", "#B22222", "#1F2A37", "#5B6B7A"

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_ss["Title"], fontSize=20, leading=24, alignment=TA_LEFT, textColor=INK, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontSize=13, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontSize=10, leading=14, textColor=INK)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10, textColor=MUTED)
BIG = ParagraphStyle("big", parent=BODY, fontSize=30, leading=34, textColor=INK)
LABEL = ParagraphStyle("label", parent=BODY, fontSize=11, leading=14, textColor=INK)


def _fmt_money(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def _num(x: Any) -> float | None:
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


def _years(x: float | None) -> str:
    if x is None:
        return "never"
    return f"{x:.1f} yr" if x < 30 else "over 30 yr"


def _score_colour(score: float | None) -> str:
    if score is None:
        return MUTED
    return RED if score >= 60 else AMBER if score >= 35 else GREEN


def _map_image(zones: dict[str, Any], max_w: float, max_h: float) -> RLImage | None:
    """True-colour picture with the wet-zone overlay composited on top."""
    tc = zones.get("report_truecolor_png_base64")
    ov = zones.get("report_overlay_png_base64")
    if not ov:
        return None
    overlay = Image.open(io.BytesIO(base64.b64decode(ov))).convert("RGBA")
    if tc:
        base = Image.open(io.BytesIO(base64.b64decode(tc))).convert("RGBA")
        if base.size != overlay.size:
            base = base.resize(overlay.size, Image.BILINEAR)
    else:
        base = Image.new("RGBA", overlay.size, (235, 235, 230, 255))
    img = Image.alpha_composite(base, overlay).convert("RGB")
    # Upscale small fields so the PDF does not look blocky
    scale = max(1, int(600 / max(img.size)))
    if scale > 1:
        img = img.resize((img.size[0] * scale, img.size[1] * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    w, h = img.size
    ratio = min(max_w / w, max_h / h)
    return RLImage(buf, width=w * ratio, height=h * ratio)


def _findings(res: dict[str, Any]) -> list[str]:
    """Plain-language sentences from the component evidence."""
    out: list[str] = []
    wet = res.get("wetness") or {}
    comps = wet.get("components") or {}
    soil = res.get("soil") or {}
    ss = soil.get("ssurgo") or {}
    if ss:
        share = ss.get("poorly_drained_component_share", 0)
        soils = ss.get("soils") or []
        if soils:
            names = [f"{d['compname']} ({d.get('field_share', 0):.0%}, {(d.get('drainagecl') or '').lower()})" for d in soils[:3]]
            out.append(f"The soil survey puts most of this field on {', '.join(names)}. "
                       f"{share:.0%} of the field is on soils rated somewhat poorly drained or worse: ground that "
                       f"holds water in a wet spring unless it is tiled.")
        else:
            names = [f"{c['compname']} ({c['drainagecl'].lower()})" for c in (ss.get("components") or [])[:3]
                     if c.get("compname") and c.get("drainagecl")]
            if names:
                out.append(f"USDA soil survey maps this field as {', '.join(names)}. "
                           f"{share:.0%} of the mapped soil components are somewhat poorly drained or worse.")
    terr = res.get("terrain") or {}
    if "depression_share" in terr:
        out.append(f"Elevation data shows {terr['depression_share']:.0%} of the field sits in closed depressions "
                   f"(spots with no surface outlet) and the average slope is {terr['mean_slope_pct']:.1f}%.")
    w = res.get("weather") or {}
    seasons = w.get("seasons") or {}
    if seasons:
        wet_years = ", ".join(w.get("wet_seasons") or []) or "none"
        bal = [s.get("balance_mm", 0) for s in seasons.values()]
        mean_bal = sum(bal) / len(bal)
        out.append(f"Over {len(seasons)} growing seasons the field averaged {w.get('season_rain_mean_mm', 0):.0f} mm "
                   f"of rain, {'more' if mean_bal > 0 else 'less'} than crops could use by {abs(mean_bal):.0f} mm a season. "
                   f"Wet years: {wet_years}.")
    sat = comps.get("satellite") or {}
    basis = sat.get("basis") or {}
    if "wet_year_ndvi_penalty" in basis:
        pen = basis["wet_year_ndvi_penalty"]
        out.append(f"Satellite crop vigour at peak season was {abs(pen) * 100:.0f} points "
                   f"{'lower' if pen > 0 else 'higher'} in wet years than in dry years"
                   f"{' - the signature of a field that drowns when it rains.' if pen > 0.05 else '.'}")
    if "within_field_unevenness" in basis:
        out.append(f"Within-field unevenness at peak season: {basis['within_field_unevenness']:.2f} NDVI between the "
                   f"best and worst tenth of the field (under 0.15 is uniform; over 0.3 is patchy).")
    z = res.get("zones") or {}
    if "problem_share" in z:
        pz = z.get("problem_zones") or []
        where = ", ".join(f"{p['position']} ({p['area_ha'] * HA_TO_AC:.1f} ac)" for p in pz[:4])
        radar = " and radar" if z.get("radar_used") else ""
        out.append(f"Checking every clear satellite pass{radar} since 2019 at 10 m, {z['problem_share']:.0%} of the field "
                   f"({z['problem_ha'] * HA_TO_AC:.1f} ac) falls well behind the rest of the field for three weeks or more "
                   f"in a typical season, or ponds in spring{' - mainly in the ' + where if where else ''}. "
                   f"A further {z['watch_share']:.0%} is borderline. Across the whole field the crop averages "
                   f"{z.get('field_mean_stress_days_per_season', 0):.0f} stress days a season.")
    return out


def build_pdf(res: dict[str, Any], prepared_by: str = "Farm X") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                            title="Field drainage assessment", author=prepared_by)
    W = letter[0] - 1.6 * inch
    field = res.get("field") or {}
    wet = res.get("wetness") or {}
    econ = res.get("economics") or {}
    zones = res.get("zones") or {}
    boundary = res.get("boundary") or {}
    area_ha = field.get("area_ha") or 0.0
    lon, lat = (field.get("centroid") or [None, None])[:2]
    score = wet.get("score")
    story: list[Any] = []

    # Page 1: verdict
    story.append(Paragraph("Field drainage assessment", H1))
    sub = f"{field.get('name') or 'Field'} &nbsp;|&nbsp; {area_ha:.1f} ha ({area_ha * HA_TO_AC:.0f} ac)"
    if lat is not None:
        sub += f" &nbsp;|&nbsp; {lat:.4f}, {lon:.4f}"
    sub += f" &nbsp;|&nbsp; {dt.date.today():%d %b %Y}"
    story.append(Paragraph(sub, SMALL))
    story.append(Spacer(1, 10))

    col = _score_colour(score)
    verdict = Table([[
        [Paragraph(f'<font color="{col}">{score:.0f}</font>' if score is not None else "n/a", BIG),
         Paragraph("wetness score / 100", SMALL)],
        [Paragraph(f"<b>{(wet.get('label') or 'no result').capitalize()}</b>", LABEL),
         Paragraph(f"Confidence: {wet.get('confidence', 'n/a')}", BODY),
         Paragraph(("Expected yield gain from tile: <b>%.0f%%</b>" % econ.get("expected_yield_gain_pct", 0))
                   if econ else "", BODY)],
    ]], colWidths=[1.6 * inch, W - 1.6 * inch])
    verdict.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(col)),
                                 ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FA")),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.append(verdict)

    story.append(Paragraph("What the data shows", H2))
    for s in _findings(res):
        story.append(Paragraph(s, BODY))
        story.append(Spacer(1, 3))

    img = _map_image(zones, W, 3.1 * inch)
    if img is not None:
        story.append(Paragraph("Where the field struggles", H2))
        story.append(img)
        note = zones.get("report_truecolor_note")
        story.append(Paragraph(
            "Green: the crop here keeps pace with the rest of the field. Yellow to red: more and more days each season "
            "spent well below the field average, judged on every clear satellite pass since 2019 (red = 30 days or more a "
            "season). Picture: Sentinel-2" + (f", {note}" if note else "") + ". North is up.", SMALL))

    # Page 2: evidence
    story.append(PageBreak())
    story.append(Paragraph("Details for your agronomist", H1))
    story.append(Paragraph("How the verdict on page 1 was reached. Each signal is read 0 (dry) to 1 (wet) and weighted.", SMALL))
    comps = wet.get("components") or {}
    rows = [["Signal", "Reading (0 = dry, 1 = wet)", "Weight", "Basis"]]
    for k in ("soil", "terrain", "satellite", "climate"):
        c = comps.get(k)
        if not c:
            rows.append([k.capitalize(), "no data", "-", "-"])
            continue
        basis = c.get("basis")
        if isinstance(basis, dict):
            basis = "; ".join(f"{a.replace('_', ' ')} {b}" for a, b in basis.items())
        rows.append([k.capitalize(), f"{c['value']:.2f}", f"{(wet.get('weights_used') or {}).get(k, 0):.2f}",
                     Paragraph(str(basis), SMALL)])
    t = Table(rows, colWidths=[0.9 * inch, 1.5 * inch, 0.6 * inch, W - 3.0 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF2")), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3DA")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)

    ss = (res.get("soil") or {}).get("ssurgo") or {}
    soils = ss.get("soils") or []
    if soils:
        story.append(Paragraph("Soils in this field (USDA soil survey)", H2))
        rows = [["Soil", "Share of field", "Acres", "Drainage class", "Clay %", "Slowest layer Ksat (um/s)"]]
        for d in soils[:8]:
            fs, clay, ksat = _num(d.get("field_share")) or 0.0, _num(d.get("clay_pct")), _num(d.get("ksat_min_um_s"))
            rows.append([d.get("compname", ""), f"{fs:.0%}", f"{fs * area_ha * HA_TO_AC:.0f}", d.get("drainagecl") or "",
                         f"{clay:.0f}" if clay is not None else "", f"{ksat:.1f}" if ksat is not None else ""])
        t = Table(rows, colWidths=[1.3 * inch, 1.0 * inch, 0.6 * inch, 1.6 * inch, 0.6 * inch, W - 5.1 * inch])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF2")), ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3DA"))]))
        story.append(t)

    seasons = (res.get("weather") or {}).get("seasons") or {}
    nd = (res.get("sentinel") or {}).get("ndvi_seasons") or {}
    if seasons:
        story.append(Paragraph("Season by season", H2))
        rows = [["Season", "Rain (mm)", "Rain vs normal", "Rain minus ET0 (mm)", "Peak crop vigour (NDVI)", "Unevenness"]]
        for k, s in seasons.items():
            n = nd.get(k) or {}
            rows.append([k, f"{s.get('rain_mm', 0):.0f}", f"{s.get('rain_z', 0):+.1f} sd", f"{s.get('balance_mm', 0):+.0f}",
                         f"{n.get('peak_ndvi_mean'):.2f}" if n.get("peak_ndvi_mean") is not None else "",
                         f"{n.get('peak_ndvi_spread'):.2f}" if n.get("peak_ndvi_spread") is not None else ""])
        t = Table(rows, colWidths=[0.9 * inch, 0.8 * inch, 1.0 * inch, 1.3 * inch, 1.5 * inch, W - 5.5 * inch])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF2")), ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3DA"))]))
        story.append(t)

    pz = zones.get("problem_zones") or []
    if pz:
        story.append(Paragraph("Persistent problem zones", H2))
        rows = [["Where", "Area (ac)", "Area (ha)", "Stress days / season", "Worst season", "Spring ponding"]]
        for p in pz[:10]:
            rows.append([p["position"], f"{p['area_ha'] * HA_TO_AC:.1f}", f"{p['area_ha']:.2f}",
                         f"{p.get('mean_stress_days_per_season', 0):.0f}", f"{p.get('worst_season_stress_days', 0):.0f}",
                         f"{p.get('spring_ponding_share', 0):.0%} of passes"])
        t = Table(rows, colWidths=[0.9 * inch, 0.9 * inch, 0.9 * inch, 1.5 * inch, 1.1 * inch, 1.3 * inch], hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF2")), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3DA"))]))
        story.append(t)
    if boundary.get("used"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Field boundary: {boundary.get('method', '')}. Crop sequence "
                               f"{' / '.join(boundary.get('crop_sequence') or [])} ({', '.join(str(y) for y in boundary.get('years') or [])}). "
                               f"{boundary.get('note', '')}", SMALL))

    # Page 3: economics
    if econ:
        story.append(PageBreak())
        story.append(Paragraph("What tile would be worth", H1))
        a = econ.get("assumptions") or {}
        crops = econ.get("crops") or {}
        crop_txt = ", ".join(f"{c} {v['yield_per_ac']:g}/ac at ${v['price_per_unit']:.2f}" for c, v in crops.items())
        if econ.get("overridden"):
            crop_txt = f"dealer inputs: {a.get('yield_per_ac'):g}/ac at ${a.get('price_per_unit'):.2f}"
        story.append(Paragraph(
            f"Basis: {econ.get('crop_basis')} rotation, ${a.get('gross_per_ac', 0):,.0f}/ac gross revenue "
            f"({crop_txt}). Expected whole-field yield gain from tile: "
            f"<b>{econ.get('expected_yield_gain_pct', 0):.1f}%</b> (mid case), built from the mapped problem zones "
            f"({a.get('problem_zone_gain', 0):.0%} recovery), borderline zones ({a.get('watch_zone_gain', 0):.0%}) and the "
            f"rest of the field ({a.get('whole_field_gain', 0):.1%}).", BODY))
        story.append(Spacer(1, 6))

        def econ_table(title: str, block: dict[str, Any], cost_per_ac: float) -> Table:
            rows = [[title, "Low", "Mid", "High"],
                    ["Yield gain", *[f"{block[k]['yield_gain_pct_whole_field']:.1f}%" for k in ("low", "mid", "high")]],
                    ["Extra revenue per year", *[_fmt_money(block[k]["annual_benefit_usd"]) for k in ("low", "mid", "high")]],
                    ["Per acre per year", *[_fmt_money(block[k]["annual_benefit_per_ac"]) for k in ("low", "mid", "high")]],
                    [f"Install cost (${cost_per_ac:,.0f}/ac)", *[_fmt_money(block[k]["install_cost_usd"]) for k in ("low", "mid", "high")]],
                    ["Simple payback", *[_years(block[k]["simple_payback_years"]) for k in ("low", "mid", "high")]],
                    [f"20-year value at {a.get('discount_rate', 0.06):.0%}", *[_fmt_money(block[k]["npv_usd"]) for k in ("low", "mid", "high")]]]
            t = Table(rows, colWidths=[2.3 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch], hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF2")), ("FONTSIZE", (0, 0), (-1, -1), 9),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D3DA")),
                                   ("BACKGROUND", (2, 1), (2, -1), colors.HexColor("#F3F7E9")),
                                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
            return t

        story.append(KeepTogether([Paragraph("Installing with your own Soil-Max plow", H2),
                                   econ_table("Own plow", econ["own_plow"], a.get("own_plow_cost_per_ac", 0))]))
        story.append(KeepTogether([Paragraph("Hiring a contractor", H2),
                                   econ_table("Contractor", econ["contractor"], a.get("install_cost_per_ac", 0))]))
        story.append(Paragraph("How these numbers were built", H2))
        for e in econ.get("evidence") or []:
            story.append(Paragraph("- " + e, SMALL))
        story.append(Spacer(1, 4))
        story.append(Paragraph(econ.get("caveat", ""), SMALL))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"Prepared by {prepared_by} from public satellite, soil, elevation and weather records. "
        f"{wet.get('note', '')} Engine v{res.get('engine_version', '')}.", SMALL))

    doc.build(story)
    return buf.getvalue()
