from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from app.clients.understat_client import UnderstatClient


@dataclass
class TeamForm:
    goals_for: float
    goals_against: float
    xg_for: float
    xg_against: float
    shots_on_target: float
    possession: float | None
    matches: int


@dataclass
class TeamContext:
    home: TeamForm
    away: TeamForm
    h2h_home_xg: float
    h2h_away_xg: float
    home_advantage: float


class FeatureEngineer:
    def __init__(self, understat: UnderstatClient) -> None:
        self.understat = understat

    async def build_team_context(self, home_team: str, away_team: str, season: Optional[int]) -> Dict[str, Any]:
        last_n = 10
        home_form = await self.understat.get_recent_form(team=home_team, last_n=last_n, season=season)
        away_form = await self.understat.get_recent_form(team=away_team, last_n=last_n, season=season)
        h2h = await self.understat.get_h2h(home_team=home_team, away_team=away_team, last_n=6, season=season)
        home_adv = await self.understat.estimate_home_advantage(home_team=home_team, season=season)

        context: Dict[str, Any] = {
            "home_form": home_form,
            "away_form": away_form,
            "h2h": h2h,
            "home_advantage": home_adv,
        }
        return context

    def make_features(self, team_context: Dict[str, Any], weather_context: Dict[str, Any]) -> Dict[str, Any]:
        home_form: Dict[str, float] = team_context["home_form"]
        away_form: Dict[str, float] = team_context["away_form"]
        h2h: Dict[str, float] = team_context["h2h"]
        home_adv: float = float(team_context["home_advantage"])  # multiplier

        min_matches_required = 5
        if home_form.get("matches", 0) < min_matches_required or away_form.get("matches", 0) < min_matches_required:
            # Insufficient data; return conservative low lambdas to avoid overclaiming
            return {
                "home_lambda": 0.05,
                "away_lambda": 0.05,
                "components": {
                    "insufficient_data": True,
                    "home_matches": home_form.get("matches", 0),
                    "away_matches": away_form.get("matches", 0),
                },
            }

        # Base rates from recent xG
        base_home_xg = max(0.05, home_form["xg_for"] / max(1, home_form["matches"]))
        base_away_xg = max(0.05, away_form["xg_for"] / max(1, away_form["matches"]))

        # Defensive adjustments
        def_adj_home = max(0.8, min(1.2, (away_form["xg_against"] / max(0.1, away_form["matches"])) / 1.2))
        def_adj_away = max(0.8, min(1.2, (home_form["xg_against"] / max(0.1, home_form["matches"])) / 1.2))

        # H2H tilt (xG differential per match)
        h2h_tilt = float(h2h.get("home_xg_for", 0) - h2h.get("away_xg_for", 0)) / max(1, h2h.get("matches", 1))
        h2h_factor_home = 1.0 + np.tanh(h2h_tilt) * 0.1
        h2h_factor_away = 1.0 - np.tanh(h2h_tilt) * 0.1

        # Weather adjustments (wind/heavy rain reduce xG)
        weather_adj = 1.0
        if weather_context:
            wind = float(weather_context.get("wind_speed", 0) or 0)
            rain = float(weather_context.get("precipitation", 0) or 0)
            temp = float(weather_context.get("temperature", 15) or 15)
            weather_adj *= (1.0 - min(0.15, wind / 100.0))
            weather_adj *= (1.0 - min(0.20, rain / 50.0))
            if temp < 0:
                weather_adj *= 0.95
            if temp > 32:
                weather_adj *= 0.96

        # Compose expected goals for the match
        home_lambda = base_home_xg * home_adv * def_adj_home * h2h_factor_home * weather_adj
        away_lambda = base_away_xg * def_adj_away * h2h_factor_away * weather_adj

        return {
            "home_lambda": float(max(home_lambda, 0.05)),
            "away_lambda": float(max(away_lambda, 0.05)),
            "components": {
                "base_home_xg": base_home_xg,
                "base_away_xg": base_away_xg,
                "home_advantage": home_adv,
                "def_adj_home": def_adj_home,
                "def_adj_away": def_adj_away,
                "h2h_factor_home": h2h_factor_home,
                "h2h_factor_away": h2h_factor_away,
                "weather_adj": weather_adj,
            },
        }