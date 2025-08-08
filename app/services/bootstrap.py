from __future__ import annotations
from typing import List

from app.services.prediction_service import PredictionService

POPULAR_FIXTURES: List[tuple[str, str]] = [
    ("Manchester City", "Arsenal"),
    ("Liverpool", "Chelsea"),
]


async def bootstrap_caches(service: PredictionService) -> None:
    for home, away in POPULAR_FIXTURES:
        try:
            await service.predict_match(home, away)
        except Exception:
            # Ignore warm errors
            pass