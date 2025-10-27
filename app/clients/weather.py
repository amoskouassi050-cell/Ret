from __future__ import annotations
from typing import Dict, Any, Optional
import httpx

# Minimal mapping for demo; ideally resolve stadium coordinates via a DB
TEAM_TO_COORDS = {
    # Examples
    "Manchester City": (53.4831, -2.2004),
    "Liverpool": (53.4308, -2.9608),
    "Arsenal": (51.5550, -0.1086),
}


class WeatherClient:
    async def get_weather_context(self, match_date: Optional[str], home_team: str) -> Dict[str, Any]:
        if not match_date:
            return {}
        coords = TEAM_TO_COORDS.get(home_team)
        if not coords:
            return {}
        lat, lon = coords
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "start_date": match_date.split("T")[0],
            "end_date": match_date.split("T")[0],
            "timezone": "UTC",
        }
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        hourly = data.get("hourly", {})
        # Select middle hour as proxy
        if not hourly:
            return {}
        idx = min(12, len(hourly.get("temperature_2m", [])) - 1)
        return {
            "temperature": hourly.get("temperature_2m", [None])[idx],
            "precipitation": hourly.get("precipitation", [None])[idx],
            "wind_speed": hourly.get("wind_speed_10m", [None])[idx],
        }