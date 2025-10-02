from copy import deepcopy
import math
import numpy as np

from ._base import Battle, BattleOutcome, RatingSystem


class EloRatingSystem(RatingSystem):
    def __init__(self, models: list[str], *, initial_rating: float = 1500.0, k_factor: float = 96.0, draw_margin: float = 0.1):
        super().__init__(models)
        self.initial_rating = float(initial_rating)
        self.k = float(k_factor)
        self.draw_margin = float(draw_margin)
        self._ratings: dict[str, float] = {m: self.initial_rating for m in self.models.keys()}

    @staticmethod
    def _expected_score(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def _score_from_outcome(self, model: str, battle: Battle) -> float:
        if battle.outcome is BattleOutcome.WIN:
            return 1.0 if model == battle.model_a else 0.0
        if battle.outcome is BattleOutcome.LOSS:
            return 0.0 if model == battle.model_a else 1.0

        return 0.5

    def predict(self, model_a: str, model_b: str) -> np.ndarray:
        """Probability of model_a winning, drawing, and losing to model_b."""
        if model_a not in self._ratings or model_b not in self._ratings:
            missing = [m for m in (model_a, model_b) if m not in self._ratings]
            raise KeyError(f"Unknown model(s): {', '.join(missing)}")
        
        ra = self._ratings[model_a]
        rb = self._ratings[model_b]
        
        ea = self._expected_score(ra, rb)
        draw_margin = self.draw_margin

        if abs(ea - 0.5) < draw_margin:
            draw_prob = 1.0
        else:
            draw_prob = 0.0

        win_prob = ea * (1.0 - draw_prob)
        loss_prob = (1.0 - ea) * (1.0 - draw_prob)
        
        return np.array([win_prob, draw_prob, loss_prob])

    def update(self, battle: Battle) -> None:
        if battle.model_a not in self._ratings or battle.model_b not in self._ratings:
            missing = [m for m in (battle.model_a, battle.model_b) if m not in self._ratings]
            raise KeyError(f"Unknown model(s): {', '.join(missing)}")

        ra = self._ratings[battle.model_a]
        rb = self._ratings[battle.model_b]

        ea = self._expected_score(ra, rb)
        eb = 1.0 - ea

        sa = self._score_from_outcome(battle.model_a, battle)
        sb = 1.0 - sa

        self._ratings[battle.model_a] = ra + self.k * (sa - ea)
        self._ratings[battle.model_b] = rb + self.k * (sb - eb)

    def get_rating(self, model: str) -> float:
        try:
            return self._ratings[model]
        except KeyError:
            raise KeyError(f"Unknown model: {model}")
