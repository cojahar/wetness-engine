"""One-off loader: fetch the AgTile-US tile-drainage map (Valayamkunnath et al. 2020, CC BY 4.0,
figshare 10.6084/m9.figshare.11825742), convert to a Cloud-Optimized GeoTIFF and upload it to
our bucket as agtile/2020.tif. The engine reads windows of it per field to report whether a
field already appears to be tiled.

Run exactly like scripts/load_cdl.py: a temporary Railway service from this repo with
start command `python scripts/load_agtile.py`, the S3_* variables of the engine, and
optionally WORK_DIR. Delete the service when the log says "done".
"""
from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

import rasterio
from boto3.s3.transfer import TransferConfig
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_cdl import _health_server, download, exists, log, s3  # noqa: E402

URL = "https://ndownloader.figshare.com/files/23079140"  # AgTile-US-TIFF.zip, ~126 MB
KEY = "agtile/2020.tif"


def main() -> int:
    _health_server()
    work = Path(os.environ.get("WORK_DIR", "/data"))
    work.mkdir(parents=True, exist_ok=True)
    client = s3()
    bucket = os.environ["S3_BUCKET"]
    if exists(client, bucket, KEY):
        log(f"{KEY} already in bucket; nothing to do")
    else:
        zpath = work / "AgTile-US-TIFF.zip"
        if not (zpath.exists() and zpath.stat().st_size > 100_000_000):
            download(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
            log(f"zip contents: {z.namelist()[:10]}")
            if not names:
                raise RuntimeError("no .tif in AgTile-US-TIFF.zip")
            z.extract(names[0], work)
            tif = work / names[0]
        with rasterio.open(tif) as ds:
            log(f"raster: crs={ds.crs} size={ds.width}x{ds.height} res={ds.res} dtype={ds.dtypes[0]} nodata={ds.nodata} "
                f"bounds={ds.bounds}")
            sample = ds.read(1, window=((ds.height // 2 - 500, ds.height // 2 + 500), (ds.width // 2 - 500, ds.width // 2 + 500)))
            import numpy as np
            vals, counts = np.unique(sample, return_counts=True)
            log(f"sample values near the centre: {dict(zip(vals.tolist(), counts.tolist()))}")
        cog = work / "agtile_cog.tif"
        profile = cog_profiles.get("deflate")
        profile.update({"blockxsize": 512, "blockysize": 512})
        log("converting to COG")
        cog_translate(str(tif), str(cog), profile, overview_level=5, overview_resampling="nearest", in_memory=False, quiet=True)
        log(f"uploading {cog.stat().st_size / 1e6:.0f} MB to {KEY}")
        client.upload_file(str(cog), bucket, KEY, Config=TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=4))
        log("done; delete this service")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
