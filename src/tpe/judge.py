"""Pairwise judge runner: render -> call -> parse, always both A/B orders, disk-cached."""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tpe.cache import DiskCache, cache_key
from tpe.models import RETRYABLE, complete_json
from tpe.schema import JudgeVerdict, KnownPair, PairJudgement

SYSTEM_PROMPT = (
    "You are a meticulous resume-screening judge. You compare two versions of a resume "
    "for the same job posting and decide which serves the candidate better. "
    "Base every verdict only on evidence in the materials provided. "
    "Respond only with JSON matching the required schema."
)

_VERDICT_SCHEMA = JudgeVerdict.strict_json_schema()  # static; built once


def render_prompt(template: str, job: str, original: str, a: str, b: str) -> str:
    return (template
            .replace("{{JOB_POSTING}}", job)
            .replace("{{ORIGINAL_RESUME}}", original)
            .replace("{{RESUME_A}}", a)
            .replace("{{RESUME_B}}", b))


def judge_pair(template: str, model: str, pair: KnownPair, order: str,
               cache: DiskCache) -> PairJudgement:
    a, b = (pair.better, pair.worse) if order == "BW" else (pair.worse, pair.better)
    user = render_prompt(template, pair.job_text, pair.original_resume, a, b)
    key = cache_key(model, SYSTEM_PROMPT, user)
    hit = cache.get(key)
    if hit is not None:
        return PairJudgement(pair_id=pair.pair_id, order=order, model=model,
                             verdict=JudgeVerdict.model_validate(hit), cached=True)
    raw = complete_json(model, SYSTEM_PROMPT, user, _VERDICT_SCHEMA)
    verdict = JudgeVerdict.model_validate(raw)
    cache.put(key, verdict.model_dump())
    return PairJudgement(pair_id=pair.pair_id, order=order, model=model,
                         verdict=verdict, cached=False)


def _order_score(winner: str, order: str) -> float:
    return {"better": 1.0, "worse": 0.0, "tie": 0.5}[_unswap(winner, order)]


def _unswap(winner: str, order: str) -> str:
    """Map an A/B verdict back to which SIDE won: the known-better or known-worse slot."""
    if winner == "tie":
        return "tie"
    if order == "BW":
        return "better" if winner == "A" else "worse"
    return "better" if winner == "B" else "worse"


@dataclass(frozen=True)
class BothOrders:
    pair: KnownPair
    bw: PairJudgement
    wb: PairJudgement

    @property
    def flipped(self) -> bool:
        u1 = _unswap(self.bw.verdict.overall.winner, "BW")
        u2 = _unswap(self.wb.verdict.overall.winner, "WB")
        return u1 != u2 and "tie" not in (u1, u2)

    @property
    def pair_score(self) -> float:
        return (_order_score(self.bw.verdict.overall.winner, "BW")
                + _order_score(self.wb.verdict.overall.winner, "WB")) / 2

    def lens_score(self, lens: str) -> float:
        w1 = getattr(self.bw.verdict, lens).winner
        w2 = getattr(self.wb.verdict, lens).winner
        return (_order_score(w1, "BW") + _order_score(w2, "WB")) / 2


def judge_both_orders(template: str, model: str, pair: KnownPair,
                      cache: DiskCache) -> BothOrders:
    return BothOrders(pair=pair,
                      bw=judge_pair(template, model, pair, "BW", cache),
                      wb=judge_pair(template, model, pair, "WB", cache))


def run_pairs(template: str, model: str, pairs: list[KnownPair], cache: DiskCache,
              max_workers: int = 4) -> list["BothOrders"]:
    """Judge all pairs. Transient API failures that outlive the pool get one serial
    retry round; non-transient errors (schema bugs, refusals) surface immediately with
    their original type. Returns complete results in input order, or raises."""
    def attempt(p: KnownPair) -> "BothOrders | Exception":
        try:
            return judge_both_orders(template, model, p, cache)
        except RETRYABLE as exc:
            return exc  # transient: eligible for the serial retry round

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(attempt, pairs))
    failed = [i for i, r in enumerate(results) if isinstance(r, Exception)]
    if failed:
        time.sleep(2.0)  # one breather for the API, then a low-concurrency pass
        for i in failed:
            results[i] = attempt(pairs[i])
    still_failed = [(pairs[i].pair_id, r) for i, r in enumerate(results)
                    if isinstance(r, Exception)]
    if still_failed:
        pair_ids = [pid for pid, _ in still_failed]
        raise RuntimeError(
            f"{len(still_failed)} pairs failed after retries (e.g. {pair_ids[:3]}); "
            f"completed calls are cached — re-run to resume") from still_failed[0][1]
    return results
