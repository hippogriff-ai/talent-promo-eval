import json

import pytest
from pydantic import ValidationError

from tpe.schema import DegradationTag, JudgeVerdict, KnownPair

VERDICT = {
    "ats_signal": {"winner": "A", "evidence": "keywords intact"},
    "human_skim": {"winner": "tie", "evidence": "both scannable"},
    "overall": {"winner": "A", "margin": "clear", "rationale": "keyword coverage"},
}


def test_verdict_roundtrip():
    v = JudgeVerdict.model_validate(VERDICT)
    assert v.overall.winner == "A"
    assert v.human_skim.winner == "tie"


def test_verdict_rejects_unknown_winner():
    bad = json.loads(json.dumps(VERDICT))
    bad["overall"]["winner"] = "C"
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate(bad)


def test_strict_schema_forbids_extras():
    schema = JudgeVerdict.strict_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"ats_signal", "human_skim", "overall"}


def test_known_pair_construction():
    p = KnownPair(
        pair_id="t1:keyword_strip:subtle", job_text="j", original_resume="o",
        better="<p>good</p>", worse="<p>bad</p>",
        tag=DegradationTag(name="keyword_strip", lens="ats", severity="subtle"),
        source="synthetic", split="train",
    )
    assert p.tag.lens == "ats"
