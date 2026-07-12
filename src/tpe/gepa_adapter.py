"""GEPA adapter: candidate = the judge prompt text; score = pair accuracy minus flip
penalty; reflective feedback names the degradation each miss failed to catch."""
from dataclasses import dataclass

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from tpe.cache import DiskCache
from tpe.judge import BothOrders, judge_both_orders
from tpe.schema import KnownPair


@dataclass
class JudgeAdapter(GEPAAdapter):
    model: str
    cache: DiskCache

    def evaluate(self, batch: list[KnownPair], candidate: dict[str, str],
                 capture_traces: bool = False) -> EvaluationBatch:
        template = candidate["judge_prompt"]
        outputs, scores, results = [], [], []
        for pair in batch:
            try:
                r: BothOrders | None = judge_both_orders(template, self.model, pair, self.cache)
                score = r.pair_score - (0.25 if r.flipped else 0.0)
                output = r.bw.verdict.model_dump()
            except Exception as exc:  # per-example failure -> fallback score, not a raise
                r, score, output = None, 0.0, {"error": str(exc)}
            results.append((pair, r, score))
            outputs.append(output)
            scores.append(score)
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
                feedback = f"Score 0.00. Judge call failed. {problem}"
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
