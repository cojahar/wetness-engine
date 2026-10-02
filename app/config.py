from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All secrets come from environment variables (Railway service variables)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Open-Meteo commercial plan key. When set, the customer-* endpoints are used.
    open_meteo_api_key: str | None = None

    # Copernicus Data Space Ecosystem OAuth client (for Sentinel Hub APIs on CDSE).
    # Create at https://shapps.dataspace.copernicus.eu/dashboard/ -> User settings -> OAuth clients
    cdse_client_id: str | None = None
    cdse_client_secret: str | None = None

    # Years of history used for the wet-year / dry-year comparison.
    history_start_year: int = 2017

    # Optional shared secret so only our Supabase edge function can call /analyze.
    engine_api_key: str | None = None

    # Our own object storage (Railway bucket) holding hosted rasters such as cdl/<year>.tif.
    # When set, boundary snapping reads CDL windows from here instead of USDA's live service.
    s3_endpoint: str | None = None
    s3_bucket: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "auto"


settings = Settings()
