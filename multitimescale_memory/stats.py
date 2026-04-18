from __future__ import annotations

import math
import random
from typing import Iterable


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap_mean_diff(
    baseline: list[float],
    candidate: list[float],
    *,
    resamples: int = 1000,
    seed: int = 2026,
) -> dict[str, float]:
    if len(baseline) != len(candidate):
        raise ValueError("paired bootstrap requires arrays of equal length")
    if not baseline:
        return {"mean_diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "count": 0}
    diffs = [cand - base for base, cand in zip(baseline, candidate)]
    rng = random.Random(seed)
    sample_means: list[float] = []
    for _ in range(resamples):
        draws = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        sample_means.append(mean(draws))
    sample_means.sort()
    lower_index = max(0, int(0.025 * (len(sample_means) - 1)))
    upper_index = min(len(sample_means) - 1, int(0.975 * (len(sample_means) - 1)))
    return {
        "mean_diff": mean(diffs),
        "ci95_low": sample_means[lower_index],
        "ci95_high": sample_means[upper_index],
        "count": len(diffs),
    }


def effect_size_dz(seed_level_diffs: list[float]) -> dict[str, float]:
    if not seed_level_diffs:
        return {"mean_diff": 0.0, "std_diff": 0.0, "dz": 0.0, "count": 0}
    mu = mean(seed_level_diffs)
    if len(seed_level_diffs) == 1:
        return {"mean_diff": mu, "std_diff": 0.0, "dz": 0.0, "count": 1}
    variance = sum((value - mu) ** 2 for value in seed_level_diffs) / (len(seed_level_diffs) - 1)
    std = math.sqrt(max(variance, 0.0))
    dz = mu / std if std > 0 else 0.0
    return {"mean_diff": mu, "std_diff": std, "dz": dz, "count": len(seed_level_diffs)}


def paired_sign_test(diffs: list[float]) -> dict[str, float]:
    nonzero = [value for value in diffs if value != 0]
    wins = sum(1 for value in nonzero if value > 0)
    losses = sum(1 for value in nonzero if value < 0)
    ties = len(diffs) - len(nonzero)
    n = len(nonzero)
    if n == 0:
        return {
            "wins": 0,
            "losses": 0,
            "ties": ties,
            "nonzero_count": 0,
            "p_value_two_sided": 1.0,
            "test": "paired_sign_test",
        }

    k = min(wins, losses)
    if n <= 200:
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p_value = min(1.0, 2.0 * tail)
    else:
        mu = n / 2.0
        sigma = math.sqrt(n / 4.0)
        # Continuity-corrected normal approximation for the two-sided sign test.
        z = (abs(wins - mu) - 0.5) / sigma if sigma > 0 else 0.0
        p_value = math.erfc(z / math.sqrt(2.0))

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "nonzero_count": n,
        "p_value_two_sided": p_value,
        "test": "paired_sign_test",
    }
