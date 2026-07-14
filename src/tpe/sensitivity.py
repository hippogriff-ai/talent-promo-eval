"""Tier-sensitivity gate: the same frozen prompt must do better on smarter models.
A flat ladder means the rubric is a checklist any model can pattern-match — fail."""
from dataclasses import dataclass

from tpe.metrics import mcnemar_exact, summarize
from tpe.models import TIERS  # capability order owned by the ladder definition

MONOTONE_TOLERANCE = 0.02


@dataclass
class GateReport:
    per_tier: dict
    monotone: bool
    subtle_spread: float
    spread_ok: bool
    mcnemar_p: float
    significant: bool
    passed: bool
    spread_min: float = 0.10


def _subtle_scores(results: list) -> list[float]:
    return [r.pair_score for r in results
            if r.pair.tag is not None and r.pair.tag.severity == "subtle"]


def _subtle_accuracy(results: list) -> float:
    subtle = _subtle_scores(results)
    return sum(subtle) / len(subtle) if subtle else 0.0


def gate(tier_results: dict[str, list], spread_min: float = 0.10) -> GateReport:
    unknown = set(tier_results) - set(TIERS)
    if unknown:
        raise ValueError(f"tiers not in the ladder: {sorted(unknown)} — "
                         f"add them to models._DEFAULT_LADDER, which owns tier order")
    missing = set(TIERS) - set(tier_results)
    if missing:
        # Fail closed: a partial sweep (e.g. nano+top only) must not produce a PASS
        # that never checked the intermediate rungs for monotonicity.
        raise ValueError(f"gate requires results for every ladder tier; missing: "
                         f"{sorted(missing)}")
    tiers = [t for t in TIERS if t in tier_results]
    # Every tier must have judged the SAME pairs: mixed splits/limits/partial caches
    # would make accuracy, spread, and McNemar non-comparable across tiers.
    id_sets = {t: frozenset(r.pair.pair_id for r in tier_results[t]) for t in tiers}
    if len(set(id_sets.values())) > 1:
        counts = {t: len(ids) for t, ids in id_sets.items()}
        raise ValueError(f"tiers judged different pair sets {counts}; "
                         f"re-run the sweep so every tier covers identical pairs")
    summaries = {t: summarize(tier_results[t]) for t in tiers}
    accs = [summaries[t].accuracy for t in tiers]
    monotone = all(accs[i + 1] >= accs[i] - MONOTONE_TOLERANCE for i in range(len(accs) - 1))
    subtle = {t: _subtle_accuracy(tier_results[t]) for t in tiers}
    subtle_spread = subtle[tiers[-1]] - subtle[tiers[0]]
    # discordant pairs, bottom vs top tier (a pair counts correct when pair_score == 1.0)
    bottom = {r.pair.pair_id: r.pair_score == 1.0 for r in tier_results[tiers[0]]}
    top = {r.pair.pair_id: r.pair_score == 1.0 for r in tier_results[tiers[-1]]}
    b = sum(1 for pid in bottom if top.get(pid, False) and not bottom[pid])
    c = sum(1 for pid in bottom if bottom[pid] and not top.get(pid, False))
    p = mcnemar_exact(b, c)
    spread_ok = subtle_spread >= spread_min
    significant = p < 0.05
    return GateReport(
        per_tier={t: {"accuracy": summaries[t].accuracy, "subtle": subtle[t],
                      "subtle_n": len(_subtle_scores(tier_results[t])),
                      "flip_rate": summaries[t].flip_rate} for t in tiers},
        monotone=monotone, subtle_spread=subtle_spread, spread_ok=spread_ok,
        mcnemar_p=p, significant=significant,
        passed=monotone and spread_ok and significant,
        spread_min=spread_min,
    )


def render_report(g: GateReport) -> str:
    lines = ["# Tier-sensitivity report", "",
             "| tier | accuracy | subtle-slice | flip rate |", "|---|---|---|---|"]
    for t, row in g.per_tier.items():
        lines.append(f"| {t} | {row['accuracy']:.3f} | {row['subtle']:.3f} (n={row['subtle_n']}) | {row['flip_rate']:.3f} |")
    lines += ["",
              f"- monotone ladder: **{g.monotone}**",
              f"- subtle-slice spread (top − bottom): **{g.subtle_spread:+.3f}** (gate ≥ {g.spread_min:+.2f}: {g.spread_ok})",
              f"- McNemar bottom vs top: **p = {g.mcnemar_p:.4f}** (gate < 0.05: {g.significant})",
              f"- **GATE {'PASSED' if g.passed else 'FAILED'}**", ""]
    if not g.passed:
        lines.append("A failed gate means the rubric does not require model capability: "
                     "harvest the subtle pairs the top tier missed and feed them to the "
                     "next GEPA round.")
    return "\n".join(lines)
