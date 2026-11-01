from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "image_create"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    max_upload_mb: int = 100
    max_images_per_analysis: int = 100
    max_image_pixels: int = 40_000_000
    near_duplicate_hash_distance: int = 6
    max_matching_pair_checks: int = 250
    matching_max_dimension: int = 1200
    matching_max_features: int = 3000
    upload_dir: Path = Path("data/uploads")
    api_key: str | None = Field(default=None, validation_alias="IMAGE_CREATE_API_KEY")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
