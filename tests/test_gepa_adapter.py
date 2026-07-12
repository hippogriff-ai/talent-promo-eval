from pathlib import Path
from unittest.mock import patch

from tpe.cache import DiskCache
from tpe.gepa_adapter import JudgeAdapter
from tpe.schema import DegradationTag, KnownPair

TEMPLATE = "A: {{RESUME_A}} B: {{RESUME_B}} {{JOB_POSTING}} {{ORIGINAL_RESUME}}"


def _pair(i: int) -> KnownPair:
    return KnownPair(pair_id=f"p{i}:dequantify:subtle", job_text="J", original_resume="O",
                     better=f"GOOD{i}", worse=f"BAD{i}",
                     tag=DegradationTag(name="dequantify", lens="ats", severity="subtle"),
                     source="synthetic", split="train")


def _verdict(winner: str) -> dict:
    lens = {"winner": winner, "evidence": "e"}
    return {"ats_signal": lens, "human_skim": lens,
            "overall": {"winner": winner, "margin": "clear", "rationale": "because"}}


def test_adapter_evaluate_scores_and_traces(tmp_path: Path):
    adapter = JudgeAdapter(model="test-model", cache=DiskCache(tmp_path))
    with patch("tpe.judge.complete_json",
               side_effect=lambda model, system, user, schema:
               _verdict("A" if "A: GOOD" in user else "B")):
        batch = adapter.evaluate([_pair(1), _pair(2)], {"judge_prompt": TEMPLATE},
                                 capture_traces=True)
    assert batch.scores == [1.0, 1.0]
    assert len(batch.trajectories) == 2


def test_adapter_survives_per_example_failure(tmp_path: Path):
    adapter = JudgeAdapter(model="test-model", cache=DiskCache(tmp_path))
    with patch("tpe.judge.complete_json", side_effect=RuntimeError("api down")):
        batch = adapter.evaluate([_pair(1)], {"judge_prompt": TEMPLATE}, capture_traces=True)
    assert batch.scores == [0.0]
    assert "error" in batch.outputs[0]


def test_adapter_reflective_dataset_names_the_degradation(tmp_path: Path):
    adapter = JudgeAdapter(model="test-model", cache=DiskCache(tmp_path))
    with patch("tpe.judge.complete_json", return_value=_verdict("A")):  # A both orders: flip
        batch = adapter.evaluate([_pair(1)], {"judge_prompt": TEMPLATE}, capture_traces=True)
    refl = adapter.make_reflective_dataset({"judge_prompt": "x"}, batch, ["judge_prompt"])
    entry = refl["judge_prompt"][0]
    assert "dequantify" in entry["Feedback"]
    assert "FLIPPED" in entry["Feedback"]
