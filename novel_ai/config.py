from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "sqlite:///./novel.db"
    redis_url: str = "redis://localhost:6379/0"
    app_env: str = "development"
    dev_auth_enabled: bool = False
    dev_api_token: str = ""
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    model_base_url: str = "http://localhost:8001/v1"
    model_name: str = "native-novel-lm"
    model_api_key: str = ""
    model_timeout_seconds: int = 240
    max_chapters: int = 40
    max_words_per_chapter: int = 2500
    worker_lease_seconds: int = 900
    cors_origins: str = "http://localhost:8080"

@lru_cache
def settings() -> Settings:
    s = Settings()
    if s.app_env == "production" and s.dev_auth_enabled:
        raise RuntimeError("Development auth MUST be disabled in production")
    if s.dev_auth_enabled and len(s.dev_api_token) < 24:
        raise RuntimeError("DEV_API_TOKEN must contain at least 24 characters")
    return s
