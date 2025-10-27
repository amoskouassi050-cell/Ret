from __future__ import annotations
from typing import Dict, Any, List, Optional
import re
import json
from datetime import datetime
import httpx


def _current_season_year() -> int:
    now = datetime.utcnow()
    # Season usually starts around July/August
    return now.year if now.month >= 7 else now.year - 1


def _decode_understat_json(encoded: str) -> Any:
    # The content inside JSON.parse('...') is a JS string literal, decode escapes then parse JSON
    decoded = json.loads(encoded.encode("utf-8").decode("unicode_escape"))
    return json.loads(decoded)


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


class UnderstatClient:
    def __init__(self) -> None:
        self.base_url = "https://understat.com"

    async def _fetch_team_matches(self, team: str, season: Optional[int] = None) -> List[Dict[str, Any]]:
        season_year = season or _current_season_year()
        team_slug = team.replace(" ", "%20")
        url = f"{self.base_url}/team/{team_slug}/{season_year}"
        try:
            async with httpx.AsyncClient(timeout=30, headers=DEFAULT_HEADERS) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return []
                html = r.text
        except Exception:
            return []
        m = re.search(r"var\s+matchesData\s*=\s*JSON.parse\('([^']+)'\)", html)
        if not m:
            return []
        try:
            matches = _decode_understat_json(m.group(1))
        except Exception:
            return []
        return matches  # list of dicts

    async def _fetch_recent_across_seasons(self, team: str, last_n: int, start_season: Optional[int] = None) -> List[Dict[str, Any]]:
        season_year = start_season or _current_season_year()
        collected: List[Dict[str, Any]] = []
        for year in [season_year, season_year - 1, season_year - 2]:
            matches = await self._fetch_team_matches(team=team, season=year)
            finished = [m for m in matches if m.get("isResult")]
            collected.extend(finished)
            if len(collected) >= last_n:
                break
        return collected[:last_n]

    async def get_recent_form(self, team: str, last_n: int = 10, season: Optional[int] = None) -> Dict[str, Any]:
        recent = await self._fetch_recent_across_seasons(team=team, last_n=last_n, start_season=season)
        if not recent:
            return {"goals_for": 0, "goals_against": 0, "xg_for": 0, "xg_against": 0, "shots_on_target": None, "possession": None, "matches": 0}
        gf = ga = xg_for = xg_against = 0.0
        for m in recent:
            is_home = (m.get("h_team") == team)
            h_goals = float(m.get("h_goals") or 0)
            a_goals = float(m.get("a_goals") or 0)
            h_xg = float(m.get("xG") or 0)
            a_xg = float(m.get("a_xG") or 0)
            if is_home:
                gf += h_goals
                ga += a_goals
                xg_for += h_xg
                xg_against += a_xg
            else:
                gf += a_goals
                ga += h_goals
                xg_for += a_xg
                xg_against += h_xg
        return {
            "goals_for": gf,
            "goals_against": ga,
            "xg_for": xg_for,
            "xg_against": xg_against,
            "shots_on_target": None,
            "possession": None,
            "matches": len(recent),
        }

    async def get_h2h(self, home_team: str, away_team: str, last_n: int = 6, season: Optional[int] = None) -> Dict[str, Any]:
        season_year = season or _current_season_year()
        combined: List[Dict[str, Any]] = []
        for year in [season_year, season_year - 1, season_year - 2]:
            matches = await self._fetch_team_matches(team=home_team, season=year)
            if not matches:
                continue
            h2h = [m for m in matches if (m.get("h_team") == home_team and m.get("a_team") == away_team) or (m.get("h_team") == away_team and m.get("a_team") == home_team)]
            combined.extend(h2h)
            if len(combined) >= last_n:
                break
        recent = combined[:last_n]
        if not recent:
            return {"matches": 0, "home_xg_for": 0.0, "away_xg_for": 0.0}
        home_xg_for = 0.0
        away_xg_for = 0.0
        for m in recent:
            if m.get("h_team") == home_team:
                home_xg_for += float(m.get("xG") or 0)
                away_xg_for += float(m.get("a_xG") or 0)
            else:
                home_xg_for += float(m.get("a_xG") or 0)
                away_xg_for += float(m.get("xG") or 0)
        return {"matches": len(recent), "home_xg_for": home_xg_for, "away_xg_for": away_xg_for}

    async def estimate_home_advantage(self, home_team: str, season: Optional[int] = None) -> float:
        season_year = season or _current_season_year()
        all_matches: List[Dict[str, Any]] = []
        for year in [season_year, season_year - 1]:
            all_matches.extend(await self._fetch_team_matches(team=home_team, season=year))
        if not all_matches:
            return 1.1
        home_xg = [float(m.get("xG") or 0) for m in all_matches if m.get("isResult") and m.get("h_team") == home_team]
        away_xg = [float(m.get("a_xG") or 0) for m in all_matches if m.get("isResult") and m.get("a_team") == home_team]
        if not home_xg or not away_xg:
            return 1.1
        hxg = sum(home_xg) / max(1, len(home_xg))
        axg = sum(away_xg) / max(1, len(away_xg))
        ratio = (hxg / axg) if axg > 0 else 1.2
        return float(max(1.02, min(1.25, ratio)))