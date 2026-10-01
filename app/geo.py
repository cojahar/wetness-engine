"""Field geometry helpers: parse a boundary, measure it, project it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform


@dataclass
class Field:
    geom_wgs84: Polygon | MultiPolygon
    name: str | None = None

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.geom_wgs84.centroid
        return (c.x, c.y)  # lon, lat

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.geom_wgs84.bounds  # minx, miny, maxx, maxy

    @property
    def utm_crs(self) -> CRS:
        lon, lat = self.centroid
        zone = int((lon + 180) // 6) + 1
        epsg = (32600 if lat >= 0 else 32700) + zone
        return CRS.from_epsg(epsg)

    def to_utm(self):
        tr = Transformer.from_crs("EPSG:4326", self.utm_crs, always_xy=True)
        return transform(tr.transform, self.geom_wgs84)

    @property
    def area_ha(self) -> float:
        return self.to_utm().area / 10_000.0

    @property
    def wkt(self) -> str:
        return self.geom_wgs84.wkt

    def geojson(self) -> dict[str, Any]:
        return mapping(self.geom_wgs84)


def square_around(lon: float, lat: float, side_m: float, name: str | None = None) -> Field:
    """Axis-aligned square of side_m metres centred on a WGS84 point."""
    import math

    half_lat = (side_m / 2) / 111_320.0
    half_lon = (side_m / 2) / (111_320.0 * math.cos(math.radians(lat)))
    geom = Polygon([
        (lon - half_lon, lat - half_lat), (lon + half_lon, lat - half_lat),
        (lon + half_lon, lat + half_lat), (lon - half_lon, lat + half_lat),
        (lon - half_lon, lat - half_lat),
    ])
    return Field(geom_wgs84=geom, name=name)


def parse_field(payload: dict[str, Any], name: str | None = None) -> Field:
    """Accepts a GeoJSON Feature, FeatureCollection (first feature), or bare geometry."""
    t = payload.get("type")
    if t == "FeatureCollection":
        feats = payload.get("features") or []
        if not feats:
            raise ValueError("FeatureCollection has no features")
        payload = feats[0]
        t = payload.get("type")
    if t == "Feature":
        name = name or (payload.get("properties") or {}).get("name")
        geom = shape(payload["geometry"])
    else:
        geom = shape(payload)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Boundary must be a Polygon or MultiPolygon, got {geom.geom_type}")
    if not geom.is_valid:
        geom = geom.buffer(0)
    return Field(geom_wgs84=geom, name=name)
