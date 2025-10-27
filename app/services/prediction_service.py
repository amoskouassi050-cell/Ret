from typing import Any, Dict, Optional
from app.clients.understat_client import UnderstatClient
from app.clients.weather import WeatherClient
from app.services.feature_engineering import FeatureEngineer
from app.services.poisson import PoissonModel


class PredictionService:
    def __init__(self) -> None:
        self.understat = UnderstatClient()
        self.weather = WeatherClient()
        self.features = FeatureEngineer(self.understat)
        self.model = PoissonModel()

    async def predict_match(
        self,
        home_team: str,
        away_team: str,
        league: Optional[str] = None,
        season: Optional[int] = None,
        match_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Fetch data
        team_context = await self.features.build_team_context(
            home_team=home_team, away_team=away_team, season=season
        )
        weather_context = await self.weather.get_weather_context(match_date=match_date, home_team=home_team)

        # Features
        engineered = self.features.make_features(team_context=team_context, weather_context=weather_context)

        # Model prediction
        prediction = self.model.predict(engineered)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "season": season,
            "match_date": match_date,
            "prediction": prediction,
        }

    async def explain_prediction(
        self,
        home_team: str,
        away_team: str,
        league: Optional[str] = None,
        season: Optional[int] = None,
        match_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        team_context = await self.features.build_team_context(
            home_team=home_team, away_team=away_team, season=season
        )
        weather_context = await self.weather.get_weather_context(match_date=match_date, home_team=home_team)
        engineered = self.features.make_features(team_context=team_context, weather_context=weather_context)
        prediction = self.model.predict(engineered)

        return {
            "input": {
                "home_team": home_team,
                "away_team": away_team,
                "league": league,
                "season": season,
                "match_date": match_date,
            },
            "team_context": team_context,
            "weather_context": weather_context,
            "features": engineered,
            "prediction": prediction,
        }