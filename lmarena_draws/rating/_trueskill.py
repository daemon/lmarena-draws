import math

from scipy.stats import norm
from trueskill import TrueSkill, Rating, rate_1vs1

from ._base import Battle, BattleOutcome, RatingSystem


class TrueSkillRatingSystem(RatingSystem):
    """Thin wrapper around the `trueskill` library."""
    def __init__(
        self,
        models: list[str],
        mu0: float = 25.0,
        sigma0: float = 25.0 / 3.0,
        beta: float = 25.0 / 6.0,
        tau: float = 25.0 / 300.0,
        draw_margin: float = 0.10,
    ):
        super().__init__(models)
        self.env = TrueSkill(mu=mu0, sigma=sigma0, beta=beta, tau=tau, draw_probability=draw_margin)
        self.ratings: dict[str, Rating] = {m: self.env.create_rating() for m in models}

        if 0.0 < draw_margin < 1.0:
            self._epsilon = math.sqrt(2.0) * beta * norm.ppf((draw_margin + 1.0) / 2.0)
        else:
            self._epsilon = 0.0

    def _mu_sigma(self, model: str) -> tuple[float, float]:
        r = self.ratings[model]
        return float(r.mu), float(r.sigma)

    def predict(self, model_a: str, model_b: str) -> tuple[float, float, float]:
        mu_a, sigma_a = self._mu_sigma(model_a)
        mu_b, sigma_b = self._mu_sigma(model_b)

        beta = float(self.env.beta)
        c2 = sigma_a**2 + sigma_b**2 + 2.0 * beta**2
        c = math.sqrt(max(1e-18, c2))
        z = (mu_a - mu_b) / c

        if self._epsilon > 0.0:
            m = self._epsilon / c
            p_draw = norm.cdf(m - z) - norm.cdf(-m - z)
            p_win  = 1.0 - norm.cdf(m - z)
            p_loss = norm.cdf(-m - z)
        else:
            p_draw = 0.0
            p_win  = 1.0 - norm.cdf(-z)
            p_loss = norm.cdf(-z)

        s = p_win + p_draw + p_loss

        if s <= 0:
            return (1/3, 1/3, 1/3)
        
        return (p_win / s, p_draw / s, p_loss / s)

    def update(self, battle: Battle) -> None:
        a, b = battle.model_a, battle.model_b
        ra, rb = self.ratings[a], self.ratings[b]

        if battle.outcome == BattleOutcome.WIN:
            ra_new, rb_new = rate_1vs1(ra, rb, env=self.env)
        elif battle.outcome == BattleOutcome.LOSS:
            rb_new, ra_new = rate_1vs1(rb, ra, env=self.env)
        elif battle.outcome == BattleOutcome.DRAW:
            ra_new, rb_new = rate_1vs1(ra, rb, drawn=True, env=self.env)
        else:
            raise ValueError("Invalid BattleOutcome")

        self.ratings[a] = ra_new
        self.ratings[b] = rb_new

    def get_rating(self, model: str) -> float:
        mu, sigma = self._mu_sigma(model)
        return mu - 3.0 * sigma
