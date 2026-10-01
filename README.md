# Farm X wetness engine

Takes a field boundary anywhere in the world and scores how likely it is to be chronically wet or poorly drained, using free public data. First piece of the Drainage Opportunity Report.

## What it does (v0.1)

For a GeoJSON boundary it pulls, in parallel:

| Source | What | Needs |
| --- | --- | --- |
| Open-Meteo historical (ERA5-Land) | daily rain, ET0, soil moisture since 2017; wet vs dry growing seasons; biggest rain events | `OPEN_METEO_API_KEY` for the commercial endpoint |
| SoilGrids 2.0 | clay, sand, silt, bulk density at three depths (global) | nothing |
| USDA SSURGO | drainage class, hydrologic group, Ksat, clay per component (USA only) | nothing |
| Copernicus DEM 30 m | relief, slope, depression share, low ground, lowest edge (outlet hint) | nothing |
| Sentinel-2 and Sentinel-1 via Sentinel Hub on CDSE | monthly NDVI/NDMI/standing-water share since 2017; 12-day SAR low-backscatter share for 2 years | `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET` |

Then `scoring.py` combines them into a 0-100 wetness score with a confidence label.

## Run

```
pip install -r requirements.txt
uvicorn app.main:app --reload
curl localhost:8000/selftest
```

Environment variables: see `app/config.py`.

## Deploy

Hosted on Railway (project `wetness-engine`, service `engine`). Pushes to `main` deploy automatically via the Railway GitHub app. Service variables hold all secrets.

## Endpoints

- `GET /health`
- `GET /selftest` runs the whole pipeline on a known Iowa field and reports which sources work; `?background=true` returns at once and the result lands in `GET /selftest/last`
- `POST /analyze` body `{"boundary": <GeoJSON>, "name": "Smith north 80"}`

Set `ENGINE_API_KEY` to require an `X-API-Key` header.

## Not yet

Within-field zone maps (needs the Sentinel Hub Process API, not Statistical), 1 m LiDAR where available, yield-loss and payback estimate, PDF report, drainage layout. See the research brief in the FarmX project.
