import math

from ._base import Battle, BattleOutcome, RatingSystem


GLICKO2_SCALE = 173.7178
Q = math.log(10) / 400.0


def _to_g2_mu(R: float) -> float:
    return (R - 1500.0) / GLICKO2_SCALE


def _to_g2_phi(RD: float) -> float:
    return RD / GLICKO2_SCALE


def _from_g2_R(mu: float) -> float:
    return 1500.0 + GLICKO2_SCALE * mu


def _from_g2_RD(phi: float) -> float:
    return GLICKO2_SCALE * phi


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * (Q**2) * (phi**2) / (math.pi**2))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    g = _g(phi_j)
    return 1.0 / (1.0 + math.exp(-g * (mu - mu_j)))


def _volatility_update(phi: float, sigma: float, v: float, delta: float, tau: float) -> float:
    a = math.log(sigma**2)
    
    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta**2 - phi**2 - v - ex)
        den = 2.0 * (phi**2 + v + ex)**2
        return (num / den) - ((x - a) / (tau**2))

    A = a
    if delta**2 > (phi**2 + v):
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        B = a - k * tau
        while f(B) < 0:
            k += 1
            B = a - k * tau

    fa, fb = f(A), f(B)

    for _ in range(60):
        C = A + (A - B) * fa / (fb - fa)
        if not (min(A, B) < C < max(A, B)):
            C = 0.5 * (A + B)
        fc = f(C)
        if abs(fc) < 1e-12:
            a_new = C
            break
        if fc * fb < 0:
            A, fa = B, fb
        else:
            fa *= 0.5
        B, fb = C, fc
    else:
        a_new = C

    sigma_prime = math.exp(0.5 * a_new)

    return sigma_prime


class Glicko2RatingSystem(RatingSystem):
    def __init__(
        self,
        models: list[str],
        R0: float = 1500.0,
        RD0: float = 350.0,
        sigma0: float = 0.06,
        tau: float = 0.5,
        draw_margin: float = 0.1,
    ):
        super().__init__(models)
        self.tau = float(tau)
        mu0 = _to_g2_mu(R0)
        phi0 = _to_g2_phi(RD0)
        self.params: dict[str, tuple[float, float, float]] = {
            m: (mu0, phi0, float(sigma0)) for m in models
        }
        self.draw_margin = draw_margin

    def _get(self, model: str) -> tuple[float, float, float]:
        return self.params[model]

    def _set(self, model: str, mu: float, phi: float, sigma: float) -> None:
        self.params[model] = (mu, phi, sigma)

    def predict(self, model_a: str, model_b: str) -> tuple[float, float, float]:
        mu_a, _, _ = self._get(model_a)
        mu_b, phi_b, _ = self._get(model_b)
        p_win = _E(mu_a, mu_b, phi_b)
        p_loss = 1.0 - p_win

        if abs(p_win - 0.5) < self.draw_margin:
            p_draw = 1.0
            p_win = 0.0
            p_loss = 0.0
        else:
            p_draw = 0.0

        p_loss = 1.0 - p_win

        return (float(p_win), float(p_draw), float(p_loss))

    def update(self, battle: Battle) -> None:
        a, b = battle.model_a, battle.model_b
        mu_a, phi_a, sigma_a = self._get(a)
        mu_b, phi_b, sigma_b = self._get(b)

        if battle.outcome == BattleOutcome.WIN:
            s_a = 1.0
        elif battle.outcome == BattleOutcome.LOSS:
            s_a = 0.0
        elif battle.outcome == BattleOutcome.DRAW:
            s_a = 0.5
        else:
            raise ValueError("Invalid BattleOutcome")

        s_b = 1.0 - s_a

        g_b = _g(phi_b)
        E_ab = 1.0 / (1.0 + math.exp(-g_b * (mu_a - mu_b)))
        v_a = 1.0 / (Q**2 * (g_b**2) * E_ab * (1.0 - E_ab))
        delta_a = v_a * Q * g_b * (s_a - E_ab)

        sigma_a_prime = _volatility_update(phi_a, sigma_a, v_a, delta_a, self.tau)
        phi_a_star = math.sqrt(phi_a**2 + sigma_a_prime**2)
        phi_a_prime = 1.0 / math.sqrt((1.0 / (phi_a_star**2)) + (1.0 / v_a))
        mu_a_prime = mu_a + (phi_a_prime**2) * Q * g_b * (s_a - E_ab)

        g_a = _g(phi_a)
        E_ba = 1.0 / (1.0 + math.exp(-g_a * (mu_b - mu_a)))
        v_b = 1.0 / (Q**2 * (g_a**2) * E_ba * (1.0 - E_ba))
        delta_b = v_b * Q * g_a * (s_b - E_ba)

        sigma_b_prime = _volatility_update(phi_b, sigma_b, v_b, delta_b, self.tau)
        phi_b_star = math.sqrt(phi_b**2 + sigma_b_prime**2)
        phi_b_prime = 1.0 / math.sqrt((1.0 / (phi_b_star**2)) + (1.0 / v_b))
        mu_b_prime = mu_b + (phi_b_prime**2) * Q * g_a * (s_b - E_ba)

        self._set(a, mu_a_prime, phi_a_prime, sigma_a_prime)
        self._set(b, mu_b_prime, phi_b_prime, sigma_b_prime)

    def get_rating(self, model: str) -> float:
        mu, phi, _ = self._get(model)
        R = _from_g2_R(mu)
        RD = _from_g2_RD(phi)
        return float(R - 2.0 * RD)
