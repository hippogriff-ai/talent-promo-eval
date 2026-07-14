"""Scoring: accuracy/flip-rate summaries, exact McNemar, Wilson CI. Pure functions, no I/O."""
import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Summary:
    n: int
    accuracy: float  # mean pair_score; ties score 0.5 — read alongside tie_rate
    flip_rate: float
    gepa_metric: float
    tie_rate: float = 0.0  # fraction of pairs with any tie/split verdict (score not in {0,1})
    by_lens: dict = field(default_factory=dict)
    by_severity: dict = field(default_factory=dict)
    by_type: dict = field(default_factory=dict)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results: list) -> Summary:
    scores = [r.pair_score for r in results]
    flips = [1.0 if r.flipped else 0.0 for r in results]
    by_sev, by_type = defaultdict(list), defaultdict(list)
    by_lens = {"ats_signal": [], "human_skim": []}
    for r in results:
        for lens in by_lens:
            by_lens[lens].append(r.lens_score(lens))
        tag = r.pair.tag
        if tag is not None:
            by_sev[tag.severity].append(r.pair_score)
            by_type[tag.name].append(r.pair_score)
    acc, flip = _mean(scores), _mean(flips)
    ties = [1.0 if s not in (0.0, 1.0) else 0.0 for s in scores]
    return Summary(
        n=len(results), accuracy=acc, flip_rate=flip,
        gepa_metric=acc - 0.25 * flip, tie_rate=_mean(ties),
        by_lens={k: _mean(v) for k, v in by_lens.items()},
        by_severity={k: _mean(v) for k, v in by_sev.items()},
        by_type={k: _mean(v) for k, v in by_type.items()},
    )


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar: b = top-correct/bottom-wrong, c = bottom-correct/top-wrong."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. `successes` may be fractional (ties contribute halves);
    the formula only needs p = successes/n, so no rounding — the CI stays centered
    on the same rate the caller reports."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
