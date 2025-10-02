import enum
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
from tqdm import tqdm


class BattleOutcome(enum.Enum):
    WIN = 0
    DRAW = 1
    LOSS = 2


@dataclass
class Battle:
    model_a: str
    model_b: str
    outcome: BattleOutcome
    sub_outcome: str | None = None
    user_id: str | None = None
    attributes: dict[str, Any] | None = None


@dataclass
class PrequentialResults:
    accuracies: list[float]
    macro_accuracy: float
    addl_stats: dict[str, list[float]] | None = None


class RatingSystem:
    def __init__(self, models: list[str]):
        self.models = {model: idx for idx, model in enumerate(models)}

    def update(self, battle: Battle) -> None:
        raise NotImplementedError

    def get_rating(self, model: str) -> float:
        raise NotImplementedError

    def predict(self, model_a: str, model_b: str) -> np.ndarray:
        """Probability of model_a winning, drawing, and losing to model_b."""
        raise NotImplementedError

    def prequential_losses(
        self,
        battles: list[Battle],
        no_ties: bool = False,
        burn_in: int = 0,
        eval_categories: list[BattleOutcome] | None = None,
        update_categories: list[BattleOutcome] | None = None,
        disable_tqdm: bool = False,
        return_stats: bool = False,
        dropout_rate: float = 0.0,
    ) -> PrequentialResults:
        macro_accuracies = []
        accuracies = []
        stats = {}
        addl_stats = {}

        for t, battle in enumerate(tqdm(battles, disable=disable_tqdm)):
            p_win, p_draw, p_loss = self.predict(battle.model_a, battle.model_b)

            idx = battle.outcome.value
            probs = [p_win, p_draw, p_loss]
            addl_stats.setdefault("is_draw", []).append(float(battle.outcome == BattleOutcome.DRAW))
            addl_stats.setdefault("rating_diff", []).append(abs(self.get_rating(battle.model_a) - self.get_rating(battle.model_b)))

            if no_ties:
                probs[1] = -1.0

            uid = battle.user_id

            if eval_categories is None or battle.outcome in eval_categories:
                correct = float(np.argmax(probs) == idx)
                accuracies.append(correct)

                if uid is not None:
                    if uid not in stats:
                        stats[uid] = [0, 0]

                    stats[uid][0] += int(correct)
                    stats[uid][1] += 1

                if stats and t > len(battles) - 100:
                    per_user_acc = [c / n for (c, n) in stats.values() if n > 0]
                    macro = float(np.mean(per_user_acc)) if per_user_acc else 0.0
                else:
                    macro = 0.0

                macro_accuracies.append(macro)

            if (update_categories is None or battle.outcome in update_categories) and random.random() > dropout_rate:
                self.update(battle)

            if t + 1 == burn_in:
                stats = {}

        if return_stats:
            return PrequentialResults(accuracies, macro_accuracies[-1], addl_stats)

        return PrequentialResults(accuracies, macro_accuracies[-1])
