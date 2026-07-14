"""GEPA adapter: candidate = the judge prompt text; score = pair accuracy minus flip
penalty; reflective feedback names the degradation each miss failed to catch."""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from tpe.cache import DiskCache
from tpe.judge import BothOrders, judge_both_orders
from tpe.schema import KnownPair


@dataclass
class JudgeAdapter(GEPAAdapter):
    model: str
    cache: DiskCache
    max_workers: int = 4

    def evaluate(self, batch: list[KnownPair], candidate: dict[str, str],
                 capture_traces: bool = False) -> EvaluationBatch:
        template = candidate["judge_prompt"]

        def one(pair: KnownPair) -> tuple[KnownPair, BothOrders | None, float, dict]:
            from tpe.models import RETRYABLE
            for attempt in (1, 2):
                try:
                    r = judge_both_orders(template, self.model, pair, self.cache)
                    return pair, r, r.pair_score - (0.25 if r.flipped else 0.0), r.bw.verdict.model_dump()
                except RETRYABLE as exc:  # transient API noise must not become a
                    if attempt == 2:      # fake "rubric miss" that GEPA learns from
                        return pair, None, 0.0, {"error": f"transient after retries: {exc}"}
                    time.sleep(2.0)
                except Exception as exc:  # per-example failure -> fallback, never raise
                    return pair, None, 0.0, {"error": str(exc)}

        # This is the system's dominant wall-clock path: every GEPA rollout lands here.
        # pool.map preserves batch order, which the EvaluationBatch contract requires.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            evaluated = list(pool.map(one, batch))
        results = [(pair, r, score) for pair, r, score, _ in evaluated]
        outputs = [output for *_, output in evaluated]
        scores = [score for _, _, score, _ in evaluated]
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=results if capture_traces else None)

    def make_reflective_dataset(self, candidate: dict[str, str],
                                eval_batch: EvaluationBatch,
                                components_to_update: list[str]) -> dict[str, list[dict]]:
        entries = []
        for pair, r, score in eval_batch.trajectories:
            tag = pair.tag
            problem = (f"The known-worse version was produced by degradation "
                       f"'{tag.name}' (lens: {tag.lens}, severity: {tag.severity})."
                       if tag else "The known-better side is the professionally drafted version.")
            if r is None:
                feedback = ("Score 0.00 due to an INFRASTRUCTURE failure (API error), "
                            "not a rubric miss — draw no conclusions about the judge "
                            "prompt from this example.")
            else:
                feedback = (
                    f"Score {score:.2f}. {problem} "
                    f"Judge verdicts: order-1 -> {r.bw.verdict.overall.winner} "
                    f"({r.bw.verdict.overall.rationale!r}); "
                    f"order-2 -> {r.wb.verdict.overall.winner} "
                    f"({r.wb.verdict.overall.rationale!r})."
                    + (" The verdict FLIPPED with presentation order — the rubric is not "
                       "grounding the decision in content." if r.flipped else "")
                    + ("" if score >= 1.0 else
                       " The rubric failed to consistently catch this quality difference; "
                       "strengthen criteria that would detect it WITHOUT rewarding raw "
                       "keyword counts (keyword-stuffed versions must still lose).")
                )
            entries.append({
                "Inputs": {"pair_id": pair.pair_id, "job_excerpt": pair.job_text[:400]},
                "Generated Outputs": (r.bw.verdict.overall.model_dump() if r else "ERROR"),
                "Feedback": feedback,
            })
        return {c: entries for c in components_to_update}
