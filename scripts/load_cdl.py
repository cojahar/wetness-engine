"""One-off loader: fetch USDA national CDL rasters, convert to Cloud-Optimized GeoTIFF,
upload to our bucket as cdl/<year>.tif. Run as a Railway job with a volume at /data.

Env: CDL_YEARS (e.g. "2025,2024,2023"), S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY_ID,
S3_SECRET_ACCESS_KEY, WORK_DIR (default /data)
"""
from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

import boto3
import httpx
from boto3.s3.transfer import TransferConfig
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

URL = "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{year}_30m_cdls.zip"


def log(msg: str) -> None:
    print(f"[load_cdl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("S3_REGION", "auto"),
    )


def exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:  # noqa: BLE001
        return False


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        log(f"already downloaded {dest.name}")
        return
    log(f"downloading {url}")
    with httpx.stream("GET", url, timeout=httpx.Timeout(None, connect=60), follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(8 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if done % (256 * 1024 * 1024) < 8 * 1024 * 1024:
                    log(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB")
    log(f"downloaded {dest.stat().st_size / 1e9:.2f} GB")


def unzip_tif(zpath: Path, outdir: Path) -> Path:
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".tif")]
        if not names:
            raise RuntimeError(f"no .tif in {zpath}")
        name = names[0]
        out = outdir / Path(name).name
        if out.exists():
            log(f"already extracted {out.name}")
            return out
        log(f"extracting {name}")
        z.extract(name, outdir)
        extracted = outdir / name
        if extracted != out:
            extracted.rename(out)
    return out


def main() -> int:
    years = [int(y) for y in os.environ.get("CDL_YEARS", "2025,2024,2023").split(",")]
    work = Path(os.environ.get("WORK_DIR", "/data"))
    work.mkdir(parents=True, exist_ok=True)
    client = s3()
    bucket = os.environ["S3_BUCKET"]
    for year in years:
        key = f"cdl/{year}.tif"
        if exists(client, bucket, key):
            log(f"{key} already in bucket, skipping")
            continue
        zpath = work / f"{year}_30m_cdls.zip"
        download(URL.format(year=year), zpath)
        tif = unzip_tif(zpath, work)
        zpath.unlink(missing_ok=True)
        cog = work / f"{year}_cog.tif"
        log(f"converting {tif.name} to COG")
        profile = cog_profiles.get("deflate")
        profile.update({"blockxsize": 512, "blockysize": 512, "predictor": 2})
        cog_translate(str(tif), str(cog), profile, overview_level=5, overview_resampling="nearest",
                      in_memory=False, quiet=True)
        tif.unlink(missing_ok=True)
        log(f"uploading {cog.stat().st_size / 1e9:.2f} GB to {key}")
        client.upload_file(str(cog), bucket, key,
                           Config=TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=4))
        cog.unlink(missing_ok=True)
        log(f"done {year}")
    log("all years loaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
