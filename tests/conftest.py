"""Shared test helpers."""


def make_verdict(winner: str, margin: str = "clear") -> dict:
    """A full JudgeVerdict dict with the same winner on every lens."""
    lens = {"winner": winner, "evidence": "e"}
    return {"ats_signal": lens, "human_skim": lens,
            "overall": {"winner": winner, "margin": margin, "rationale": "r"}}
