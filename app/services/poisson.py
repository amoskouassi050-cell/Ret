from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from math import exp, factorial


def poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * exp(-lam) / factorial(k)


class PoissonModel:
    def __init__(self, max_goals: int = 10) -> None:
        self.max_goals = max_goals

    def score_matrix(self, home_lambda: float, away_lambda: float) -> np.ndarray:
        home_probs = np.array([poisson_pmf(i, home_lambda) for i in range(self.max_goals + 1)])
        away_probs = np.array([poisson_pmf(j, away_lambda) for j in range(self.max_goals + 1)])
        return np.outer(home_probs, away_probs)

    def market_probs(self, mat: np.ndarray) -> Dict[str, float]:
        # 1X2
        home = float(np.tril(mat, -1).sum())
        draw = float(np.trace(mat))
        away = float(np.triu(mat, 1).sum())
        # BTTS
        btts_yes = float(mat[1:, 1:].sum())
        btts_no = 1.0 - btts_yes
        # over/under 2.5
        over25 = float(
            sum(
                mat[i, j]
                for i in range(mat.shape[0])
                for j in range(mat.shape[1])
                if i + j > 2
            )
        )
        under25 = 1.0 - over25
        return {
            "home": home,
            "draw": draw,
            "away": away,
            "btts_yes": btts_yes,
            "btts_no": btts_no,
            "over_2_5": over25,
            "under_2_5": under25,
        }

    def most_likely_score(self, mat: np.ndarray) -> Tuple[int, int, float]:
        idx = np.unravel_index(np.argmax(mat), mat.shape)
        return int(idx[0]), int(idx[1]), float(mat[idx])

    def suggest_bet(self, markets: Dict[str, float]) -> Dict[str, Any]:
        candidates = {
            "1": markets["home"],
            "X": markets["draw"],
            "2": markets["away"],
            "BTTS_Yes": markets["btts_yes"],
            "Under_2_5": markets["under_2_5"],
            "Over_2_5": markets["over_2_5"],
        }
        market, prob = max(candidates.items(), key=lambda kv: abs(kv[1] - 0.5))
        confidence = abs(prob - 0.5) * 2  # 0 to 1
        # Guardrails: avoid suggesting if draw dominates or confidence is weak
        if markets.get("draw", 0) > 0.38 or confidence < 0.15:
            return {"market": "NO_BET", "probability": None, "confidence": 0.0}
        return {"market": market, "probability": prob, "confidence": confidence}

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        home_lambda = float(features["home_lambda"])
        away_lambda = float(features["away_lambda"])
        mat = self.score_matrix(home_lambda, away_lambda)
        markets = self.market_probs(mat)
        hs, as_, p = self.most_likely_score(mat)
        suggestion = self.suggest_bet(markets)
        return {
            "expected_goals": {"home": home_lambda, "away": away_lambda},
            "most_likely_score": {"home": hs, "away": as_, "probability": p},
            "markets": markets,
            "suggestion": suggestion,
        }