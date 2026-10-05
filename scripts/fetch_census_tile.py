"""Build data/census_tile_2022.csv from USDA NASS Quick Stats: county tile-drained acres and
cropland acres from the 2022 Census of Agriculture.

Needs a free Quick Stats API key (https://quickstats.nass.usda.gov/api) in NASS_API_KEY.
Run every five years when a new Census comes out (next: 2027 data, published 2029):
    NASS_API_KEY=... python scripts/fetch_census_tile.py [year]
Commit the resulting CSV; the engine reads it at startup (app/sources/tile.py).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import httpx

API = "https://quickstats.nass.usda.gov/api/api_GET/"
OUT = Path(__file__).resolve().parents[1] / "data" / "census_tile_2022.csv"


def fetch(key: str, year: int, short_desc: str) -> dict[str, float]:
    params = {"key": key, "source_desc": "CENSUS", "year": year, "agg_level_desc": "COUNTY",
              "short_desc": short_desc, "format": "JSON"}
    r = httpx.get(API, params=params, timeout=180)
    r.raise_for_status()
    out: dict[str, float] = {}
    for rec in r.json().get("data", []):
        fips = f"{rec['state_fips_code']}{rec['county_code']}"
        val = str(rec.get("Value", "")).replace(",", "").strip()
        if val.startswith("("):  # (D) withheld, (Z) under half a unit
            continue
        try:
            out[fips] = float(val)
        except ValueError:
            continue
        out[fips + "_name"] = f"{rec.get('county_name', '')}|{rec.get('state_alpha', '')}"  # type: ignore[assignment]
    return out


def main() -> int:
    key = os.environ.get("NASS_API_KEY")
    if not key:
        print("set NASS_API_KEY (free at https://quickstats.nass.usda.gov/api)", file=sys.stderr)
        return 2
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
    tile = fetch(key, year, "AG LAND, CROPLAND, DRAINED BY TILE - ACRES")
    crop = fetch(key, year, "AG LAND, CROPLAND - ACRES")
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fips", "state", "county", "tile_acres", "cropland_acres"])
        for fips in sorted(k for k in crop if not k.endswith("_name")):
            name = str(crop.get(fips + "_name", "|")).split("|")
            w.writerow([fips, name[1], name[0].title(), tile.get(fips, 0), crop[fips]])
            n += 1
    print(f"wrote {OUT} with {n} counties ({sum(1 for k in tile if not k.endswith('_name'))} report tile acres)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
