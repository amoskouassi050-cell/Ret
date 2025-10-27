from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import httpx
import unicodedata

BASE = "https://api.sofascore.com/api/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()


class SofaScoreClient:
    async def search_team_id(self, query: str) -> Optional[int]:
        q = normalize_name(query)
        url = f"{BASE}/search/all?q={q}"
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return None
                data = r.json()
        except Exception:
            return None
        teams = data.get("teams", [])
        # Try exact normalized name match first, else take best candidate
        best_id: Optional[int] = None
        for t in teams:
            name = t.get("name") or ""
            if normalize_name(name) == q:
                return int(t.get("id"))
            if best_id is None:
                best_id = int(t.get("id"))
        return best_id

    async def get_event_statistics(self, event_id: int) -> Dict[str, Any]:
        url = f"{BASE}/event/{event_id}/statistics"
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return {}
                return r.json()
        except Exception:
            return {}

    def _extract_expected_goals(self, stats_json: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        # Find ALL period if present
        period = None
        if isinstance(stats_json, dict):
            for item in stats_json.get("statistics", []):
                if item.get("period") == "ALL":
                    period = item
                    break
        if period is None:
            # Some responses are a flat structure
            period = stats_json
        groups = period.get("groups", []) if isinstance(period, dict) else []
        for g in groups:
            for s in g.get("statisticsItems", []):
                name = (s.get("name") or "").lower()
                if "expected" in name and "goal" in name:
                    home_val = s.get("home")
                    away_val = s.get("away")
                    if home_val is not None and away_val is not None:
                        try:
                            return float(home_val), float(away_val)
                        except Exception:
                            pass
        return None

    async def get_team_recent_form(self, team_id: int, last_n: int = 10) -> Dict[str, Any]:
        # Fetch last events
        url = f"{BASE}/team/{team_id}/events/last/{last_n}"
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return {"goals_for": 0, "goals_against": 0, "xg_for": 0, "xg_against": 0, "shots_on_target": None, "possession": None, "matches": 0}
                data = r.json()
        except Exception:
            return {"goals_for": 0, "goals_against": 0, "xg_for": 0, "xg_against": 0, "shots_on_target": None, "possession": None, "matches": 0}
        events = data.get("events", [])
        if not events:
            return {"goals_for": 0, "goals_against": 0, "xg_for": 0, "xg_against": 0, "shots_on_target": None, "possession": None, "matches": 0}
        gf = ga = xgf = xga = 0.0
        cnt = 0
        for ev in events[:last_n]:
            cnt += 1
            home = ev.get("homeTeam", {})
            away = ev.get("awayTeam", {})
            home_id = int(home.get("id")) if home.get("id") is not None else None
            ev_home_score = (ev.get("homeScore") or {}).get("current") or 0
            ev_away_score = (ev.get("awayScore") or {}).get("current") or 0
            if home_id == team_id:
                gf += float(ev_home_score)
                ga += float(ev_away_score)
            else:
                gf += float(ev_away_score)
                ga += float(ev_home_score)
            stats = await self.get_event_statistics(int(ev.get("id")))
            xg_pair = self._extract_expected_goals(stats)
            if xg_pair:
                hxg, axg = xg_pair
                if home_id == team_id:
                    xgf += hxg
                    xga += axg
                else:
                    xgf += axg
                    xga += hxg
        return {
            "goals_for": gf,
            "goals_against": ga,
            "xg_for": xgf,
            "xg_against": xga,
            "shots_on_target": None,
            "possession": None,
            "matches": cnt,
        }

    async def get_h2h_xg(self, home_team_id: int, away_team_id: int, last_n: int = 6) -> Dict[str, Any]:
        url = f"{BASE}/team/{home_team_id}/h2h/teams/{away_team_id}"
        try:
            async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return {"matches": 0, "home_xg_for": 0.0, "away_xg_for": 0.0}
                data = r.json()
        except Exception:
            return {"matches": 0, "home_xg_for": 0.0, "away_xg_for": 0.0}
        events = data.get("events", [])
        home_xg_for = 0.0
        away_xg_for = 0.0
        cnt = 0
        for ev in events[:last_n]:
            cnt += 1
            home_id = int((ev.get("homeTeam") or {}).get("id") or 0)
            stats = await self.get_event_statistics(int(ev.get("id")))
            xg_pair = self._extract_expected_goals(stats)
            if not xg_pair:
                continue
            hxg, axg = xg_pair
            if home_id == home_team_id:
                home_xg_for += hxg
                away_xg_for += axg
            else:
                home_xg_for += axg
                away_xg_for += hxg
        return {"matches": cnt, "home_xg_for": home_xg_for, "away_xg_for": away_xg_for}