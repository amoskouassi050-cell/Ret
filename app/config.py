from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    api_football_key: str | None = Field(default=None, env="API_FOOTBALL_KEY")
    api_football_host: str = Field(default="v3.football.api-sports.io")

    default_leagues: List[str] = Field(
        default_factory=lambda: [
            "EPL",
            "LaLiga",
            "SerieA",
            "Bundesliga",
            "Ligue1",
        ]
    )

    cache_ttl_minutes: int = Field(default=360)  # 6 hours
    database_url: str = Field(default="sqlite:///./data/app.db")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()