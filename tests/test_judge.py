from pathlib import Path
from unittest.mock import patch

from tpe.cache import DiskCache
from tpe.judge import judge_both_orders, judge_pair, render_prompt
from tpe.schema import DegradationTag, KnownPair

PAIR = KnownPair(
    pair_id="t1:keyword_strip:subtle", job_text="JD", original_resume="ORIG",
    better="GOOD RESUME", worse="BAD RESUME",
    tag=DegradationTag(name="keyword_strip", lens="ats", severity="subtle"),
    source="synthetic", split="train",
)
TEMPLATE = "Job: {{JOB_POSTING}}\nOriginal: {{ORIGINAL_RESUME}}\nA: {{RESUME_A}}\nB: {{RESUME_B}}"


def _verdict(winner: str) -> dict:
    lens = {"winner": winner, "evidence": "e"}
    return {"ats_signal": lens, "human_skim": lens,
            "overall": {"winner": winner, "margin": "clear", "rationale": "r"}}


def test_render_prompt_places_sides():
    text = render_prompt(TEMPLATE, "JD", "ORIG", "GOOD RESUME", "BAD RESUME")
    assert "A: GOOD RESUME" in text and "B: BAD RESUME" in text


def test_judge_pair_bw_puts_better_as_a(tmp_path: Path):
    with patch("tpe.judge.complete_json", return_value=_verdict("A")) as mock:
        j = judge_pair(TEMPLATE, "m", PAIR, "BW", DiskCache(tmp_path))
    prompt_sent = mock.call_args.args[2]
    assert "A: GOOD RESUME" in prompt_sent
    assert j.verdict.overall.winner == "A" and j.cached is False


def test_judge_pair_uses_cache_on_second_call(tmp_path: Path):
    cache = DiskCache(tmp_path)
    with patch("tpe.judge.complete_json", return_value=_verdict("A")) as mock:
        judge_pair(TEMPLATE, "m", PAIR, "BW", cache)
        j2 = judge_pair(TEMPLATE, "m", PAIR, "BW", cache)
    assert mock.call_count == 1 and j2.cached is True


def test_both_orders_consistent_correct(tmp_path: Path):
    # BW: better is A -> "A" correct. WB: better is B -> "B" correct.
    responses = iter([_verdict("A"), _verdict("B")])
    with patch("tpe.judge.complete_json", side_effect=lambda *a, **k: next(responses)):
        both = judge_both_orders(TEMPLATE, "m", PAIR, DiskCache(tmp_path))
    assert both.pair_score == 1.0 and both.flipped is False


def test_both_orders_flip_detected(tmp_path: Path):
    responses = iter([_verdict("A"), _verdict("A")])  # A both times = position bias
    with patch("tpe.judge.complete_json", side_effect=lambda *a, **k: next(responses)):
        both = judge_both_orders(TEMPLATE, "m", PAIR, DiskCache(tmp_path))
    assert both.flipped is True and both.pair_score == 0.5


def test_lens_score_independent_of_overall(tmp_path: Path):
    # ats correct both orders; human_skim wrong both orders
    def respond(order_a_good: bool) -> dict:
        return {
            "ats_signal": {"winner": "A" if order_a_good else "B", "evidence": "e"},
            "human_skim": {"winner": "B" if order_a_good else "A", "evidence": "e"},
            "overall": {"winner": "A" if order_a_good else "B", "margin": "clear", "rationale": "r"},
        }
    responses = iter([respond(True), respond(False)])
    with patch("tpe.judge.complete_json", side_effect=lambda *a, **k: next(responses)):
        both = judge_both_orders(TEMPLATE, "m", PAIR, DiskCache(tmp_path))
    assert both.lens_score("ats_signal") == 1.0
    assert both.lens_score("human_skim") == 0.0
