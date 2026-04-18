from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import Random

from .types import ActionType, RouterDecision, RouterFeatures


class FutureValueEstimator:
    def __init__(self, rng: Random | None = None, stochastic_init: bool = False) -> None:
        rng = rng or Random(0)
        self._weights = {
            ActionType.PARAM_ONLY: [0.0, 0.0, 0.1, 0.2, -0.1],
            ActionType.READ_MEMORY: [0.1, 0.4, 0.6, -0.2, -0.1],
            ActionType.RETRIEVE: [0.2, 0.1, -0.1, 0.2, -0.2],
            ActionType.WRITE_MEMORY: [0.3, 0.6, 0.7, -0.3, -0.1],
            ActionType.FAST_ADAPT: [0.2, 0.5, 0.4, -0.4, -0.2],
            ActionType.CONSOLIDATE: [0.4, 0.8, 0.9, -0.5, -0.3],
        }
        if stochastic_init:
            for action, weights in self._weights.items():
                self._weights[action] = [weight + rng.uniform(-0.025, 0.025) for weight in weights]
        self._lr = 0.05

    def estimate(self, action: ActionType, features: RouterFeatures) -> float:
        recurrence = features.recurrence_estimate
        stability = features.stability_score
        utility = [
            recurrence,
            stability,
            features.source_agreement_count,
            features.volatility_score,
            features.contradiction_risk,
        ]
        weights = self._weights[action]
        return sum(w * x for w, x in zip(weights, utility))

    def update(self, action: ActionType, features: RouterFeatures, observed_future_utility: float) -> None:
        current = self.estimate(action, features)
        error = observed_future_utility - current
        utility = [
            features.recurrence_estimate,
            features.stability_score,
            features.source_agreement_count,
            features.volatility_score,
            features.contradiction_risk,
        ]
        self._weights[action] = [
            weight + self._lr * error * feature
            for weight, feature in zip(self._weights[action], utility)
        ]


@dataclass(slots=True)
class BanditState:
    a: list[list[float]]
    b: list[float]


class ContextualBanditRouter:
    def __init__(
        self,
        alpha: float = 0.6,
        use_future_value: bool = True,
        seed: int = 0,
        stochastic_init: bool = False,
        future_value_scale: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.use_future_value = use_future_value
        self.future_value_scale = future_value_scale
        self._rng = Random(seed)
        self.stochastic_init = stochastic_init
        self.actions = list(ActionType)
        dims = len(RouterFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).to_vector())
        self._state = {
            action: BanditState(
                a=[[1.0 if i == j else 0.0 for j in range(dims)] for i in range(dims)],
                b=[0.0 for _ in range(dims)],
            )
            for action in self.actions
        }
        self.future_value_estimator = FutureValueEstimator(rng=self._rng, stochastic_init=stochastic_init)

    def _action_prior(self, action: ActionType, features: RouterFeatures) -> float:
        if action == ActionType.PARAM_ONLY:
            return 0.6 * features.model_confidence - 0.3 * features.volatility_score
        if action == ActionType.READ_MEMORY:
            return (
                1.0 * features.memory_hit_score
                + 0.8 * features.memory_alignment_score
                + 0.5 * features.recurrence_estimate
                - 0.2 * features.contradiction_risk
            )
        if action == ActionType.RETRIEVE:
            return 0.6 * features.retrieval_quality_estimate + 0.3 * features.volatility_score
        if action == ActionType.WRITE_MEMORY:
            return (
                0.8 * features.retrieval_quality_estimate
                + 0.7 * features.recurrence_estimate
                + 0.2 * features.memory_alignment_score
                + 0.4 * features.stability_score
            )
        if action == ActionType.FAST_ADAPT:
            return (
                0.7 * features.recurrence_estimate
                + 0.5 * features.domain_change_rate
                + 0.25 * features.volatility_score
                - 0.15 * features.contradiction_risk
            )
        return (
            0.9 * features.recurrence_estimate
            + 0.8 * features.stability_score
            + 0.35 * features.source_agreement_count
            - 0.7 * features.volatility_score
        )

    def _solve(self, matrix: list[list[float]], vector: list[float]) -> list[float]:
        n = len(vector)
        aug = [row[:] + [value] for row, value in zip(matrix, vector)]
        for i in range(n):
            pivot = aug[i][i] or 1e-9
            for j in range(i, n + 1):
                aug[i][j] /= pivot
            for k in range(n):
                if k == i:
                    continue
                factor = aug[k][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]
        return [aug[i][-1] for i in range(n)]

    def _quadratic_form(self, matrix: list[list[float]], vector: list[float]) -> float:
        solved = self._solve(matrix, vector)
        return sum(v * s for v, s in zip(vector, solved))

    def decide(
        self,
        features: RouterFeatures,
        action_mask: dict[ActionType, bool],
    ) -> RouterDecision:
        x = features.to_vector()
        scores: dict[str, float] = {}
        rationale = "bandit score with uncertainty bonus"
        best_action = ActionType.PARAM_ONLY
        best_score = float("-inf")
        best_future = 0.0
        for action in self.actions:
            if not action_mask[action]:
                scores[action.value] = float("-inf")
                continue
            state = self._state[action]
            theta = self._solve(state.a, state.b)
            mean_reward = sum(t * f for t, f in zip(theta, x))
            uncertainty = self.alpha * sqrt(max(self._quadratic_form(state.a, x), 0.0))
            future_value = (
                self.future_value_scale * self.future_value_estimator.estimate(action, features)
                if self.use_future_value
                else 0.0
            )
            score = mean_reward + uncertainty + future_value + self._action_prior(action, features)
            if self.stochastic_init:
                score += self._rng.uniform(-1e-6, 1e-6)
            scores[action.value] = score
            if score > best_score:
                best_score = score
                best_action = action
                best_future = future_value
        return RouterDecision(
            action=best_action,
            action_scores=scores,
            action_mask={action.value: action_mask[action] for action in self.actions},
            immediate_reward_estimate=best_score - best_future,
            future_value_estimate=best_future,
            rationale=rationale,
        )

    def update(self, action: ActionType, features: RouterFeatures, reward: float, observed_future_utility: float) -> None:
        x = features.to_vector()
        state = self._state[action]
        for row in range(len(x)):
            for col in range(len(x)):
                state.a[row][col] += x[row] * x[col]
        for idx, value in enumerate(x):
            state.b[idx] += reward * value
        if self.use_future_value:
            self.future_value_estimator.update(action, features, observed_future_utility)
