import math

import numpy as np

from ._base import Battle, BattleOutcome, RatingSystem


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class OnlineBTRatingSystem(RatingSystem):
    def __init__(
        self,
        models: list[str],
        eta: float = 0.05,
        l2: float = 1e-4,
        rating_base: float = 1000.0,
        rating_scale: float = 400.0,
        draw_margin: float = 0.1,
    ):
        super().__init__(models)
        self.theta = {m: 0.0 for m in models}
        self.eta = eta
        self.l2 = l2
        self.rating_base = rating_base
        self.rating_scale = rating_scale
        self.draw_margin = draw_margin

    def _score_diff(self, a: str, b: str) -> float:
        return self.theta[a] - self.theta[b]

    def predict(self, model_a: str, model_b: str) -> tuple[float, float, float]:
        p_win = _sigmoid(self._score_diff(model_a, model_b))
        p_loss = 1.0 - p_win
        p_draw = 0.0

        if abs(p_win - 0.5) < self.draw_margin:
            p_draw = 1.0
            p_win = 0.0
            p_loss = 0.0

        return (float(p_win), float(p_draw), float(p_loss))

    def update(self, battle: Battle) -> None:
        a, b = battle.model_a, battle.model_b

        def _prob(a: str, b: str) -> float:
            z = self._score_diff(a, b)
            return _sigmoid(z)

        def _sgd_step(y, apply_decay=False):
            if apply_decay and self.l2 > 0.0:
                decay = (1.0 - self.eta * self.l2)
                self.theta[a] *= decay
                self.theta[b] *= decay

            p = _prob(a, b)
            g = (p - y)
            self.theta[a] -= self.eta * g
            self.theta[b] += self.eta * g

        if battle.outcome == BattleOutcome.WIN:
            _sgd_step(y=1.0, apply_decay=True)
        elif battle.outcome == BattleOutcome.LOSS:
            _sgd_step(y=0.0, apply_decay=True)
        elif battle.outcome == BattleOutcome.DRAW:
            _sgd_step(y=1.0, apply_decay=True)
            _sgd_step(y=0.0, apply_decay=False)
        else:
            raise ValueError(f"Invalid outcome: {battle.outcome}")

    def get_rating(self, model: str) -> float:
        return self.rating_base + self.rating_scale * self.theta[model]
