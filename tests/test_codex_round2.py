"""Regression tests for the second codex review round."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tests.conftest import make_verdict
from tpe.cache import DiskCache
from tpe.cli import app
from tpe.degrade import DegradeContext, bland_leads, bury_relevant
from tpe.gepa_adapter import JudgeAdapter
from tpe.schema import DegradationTag, KnownPair
from tpe.sensitivity import gate

runner = CliRunner()


def test_bury_relevant_severe_never_promotes_hot_role():
    # hot role is LAST already; a blind reversal would promote it to first
    ctx = DegradeContext(jd_keywords=["python"])
    html = ("<h2>Experience</h2>"
            "<h3>Old Role</h3><ul><li>a</li></ul>"
            "<h3>Older Role</h3><ul><li>b</li></ul>"
            "<h3>Python Role</h3><ul><li>python work</li></ul>")
    out = bury_relevant(html, ctx, "severe")
    assert out.rstrip().endswith("<li>python work</li></ul>")  # hot role stays buried last


def test_bland_leads_noop_when_all_bullets_quantified():
    ctx = DegradeContext(jd_keywords=[])
    html = "<ul><li>Cut costs 10%.</li><li>Grew revenue 20%.</li></ul>"
    assert bland_leads(html, ctx, "moderate") == html


def test_bland_leads_noop_when_bland_already_leads():
    ctx = DegradeContext(jd_keywords=[])
    html = "<ul><li>Maintained tooling.</li><li>Cut latency 43%.</li></ul>"
    assert bland_leads(html, ctx, "moderate") == html


def test_bland_leads_preserves_within_class_order():
    ctx = DegradeContext(jd_keywords=[])
    html = ("<ul><li>Cut latency 43%.</li><li>Long unquantified bullet here.</li>"
            "<li>Short one.</li><li>Grew usage 2x.</li></ul>")
    out = bland_leads(html, ctx, "moderate")
    # non-digit bullets first in ORIGINAL relative order, then digit bullets in theirs
    assert out == ("<ul><li>Long unquantified bullet here.</li><li>Short one.</li>"
                   "<li>Cut latency 43%.</li><li>Grew usage 2x.</li></ul>")


def _pair(i: int) -> KnownPair:
    return KnownPair(pair_id=f"p{i}:dequantify:subtle", job_text="J", original_resume="O",
                     better=f"GOOD{i}", worse=f"BAD{i}",
                     tag=DegradationTag(name="dequantify", lens="ats", severity="subtle"),
                     source="synthetic", split="train")


def test_adapter_retries_transient_then_neutral_feedback(tmp_path: Path):
    from openai import RateLimitError
    import httpx
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x"), json={})
    err = RateLimitError("throttled", response=resp, body=None)
    adapter = JudgeAdapter(model="m", cache=DiskCache(tmp_path))
    with patch("tpe.judge.complete_json", side_effect=err) as mock, patch("tpe.gepa_adapter.time.sleep"):
        batch = adapter.evaluate([_pair(1)], {"judge_prompt": "{{RESUME_A}}{{RESUME_B}}{{JOB_POSTING}}{{ORIGINAL_RESUME}}"},
                                 capture_traces=True)
    assert mock.call_count == 2  # adapter retried the transient failure once
    assert batch.scores == [0.0]
    refl = adapter.make_reflective_dataset({"judge_prompt": "x"}, batch, ["judge_prompt"])
    assert "INFRASTRUCTURE" in refl["judge_prompt"][0]["Feedback"]  # not framed as a rubric miss


def _tier_results(pair_ids: list[str]):
    out = []
    for pid in pair_ids:
        r = MagicMock()
        r.pair_score = 1.0
        r.flipped = False
        r.lens_score = lambda lens: 1.0
        r.pair.tag = None
        r.pair.pair_id = pid
        out.append(r)
    return out


def test_gate_rejects_mismatched_pair_sets():
    tiers = {t: _tier_results(["a", "b"]) for t in ("nano", "mini", "mid")}
    tiers["top"] = _tier_results(["a", "c"])  # different pair set
    with pytest.raises(ValueError, match="different pair sets"):
        gate(tiers)


def test_compare_runs_rejects_duplicate_ids(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    row = {"id": "x", "job": "J", "original": "O", "resume": "R"}
    a.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    b.write_text(json.dumps(row) + "\n")
    result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                 "--prompt", "prompts/seed_judge.md",
                                 "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 1
    assert "duplicate ids" in result.output


def test_compare_runs_rejects_mismatched_job_or_original(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "x", "job": "JOB ONE", "original": "O", "resume": "R1"}) + "\n")
    b.write_text(json.dumps({"id": "x", "job": "JOB TWO", "original": "O", "resume": "R2"}) + "\n")
    result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                 "--prompt", "prompts/seed_judge.md",
                                 "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 1
    assert "mismatched job/original" in result.output
