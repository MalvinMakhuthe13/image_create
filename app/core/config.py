from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "image_create"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    max_upload_mb: int = 100

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
