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


# --- round 3 ---

def test_compare_runs_rejects_blank_original(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "x", "job": "J", "resume": "R1"}) + "\n")
    b.write_text(json.dumps({"id": "x", "job": "J", "resume": "R2"}) + "\n")
    result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                 "--prompt", "prompts/seed_judge.md",
                                 "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 1
    assert "grounding" in result.output


def test_gate_rejects_duplicate_pair_ids_within_tier():
    tiers = {t: _tier_results(["a", "b"]) for t in ("nano", "mini", "mid")}
    tiers["top"] = _tier_results(["a", "b", "b"])  # concatenated retry output
    with pytest.raises(ValueError, match="duplicate pair_ids"):
        gate(tiers)


def test_wall_of_text_noop_on_single_bullet_list():
    from tpe.degrade import wall_of_text
    ctx = DegradeContext(jd_keywords=[])
    html = "<ul><li>Only bullet without trailing period</li></ul>"
    assert wall_of_text(html, ctx, "severe") == html


# --- round 4 ---

def test_dequantify_preserves_versioned_technologies():
    from tpe.degrade import dequantify
    ctx = DegradeContext(jd_keywords=[])
    html = "<ul><li>Built OAuth2 flows on S3 and EC2, cutting latency by 43%.</li></ul>"
    out = dequantify(html, ctx, "severe")
    assert "OAuth2" in out and "S3" in out and "EC2" in out  # tech names untouched
    assert "43%" not in out  # the actual metric is stripped


def test_metric_detector_ignores_embedded_digits():
    from tpe.degrade import _has_metric
    assert not _has_metric("Built OAuth2 flows on S3 and EC2")
    assert _has_metric("Reduced cost by 43%")
    assert _has_metric("Led a team of 4 engineers")


def test_bland_leads_ignores_versioned_tool_bullets():
    ctx = DegradeContext(jd_keywords=[])
    html = "<ul><li>Built OAuth2 integration.</li><li>Maintained tooling.</li></ul>"
    assert bland_leads(html, ctx, "moderate") == html  # no quantified bullet exists


def test_keyword_density_requires_token_boundaries():
    from tpe.degrade import _keyword_density
    assert _keyword_density("understands capitalization laws", ["api", "aws"]) == 0
    assert _keyword_density("built an api on aws", ["api", "aws"]) == 2


def test_keyword_stuff_skips_profile_grounded_terms():
    from tpe.degrade import keyword_stuff
    ctx = DegradeContext(jd_keywords=["kubernetes", "terraform"],
                         source_text="Deep kubernetes production experience.")
    html = "<ul><li>Backend work.</li></ul>"
    out = keyword_stuff(html, ctx, "subtle")
    # kubernetes is grounded in the profile -> adding it is not a trap; terraform is not
    blob = out[len(html):]
    assert "Terraform" in blob and "kubernetes" not in blob.lower().replace("terraform", "")


# --- round 5 ---

def test_dequantify_preserves_numbered_standards():
    from tpe.degrade import dequantify
    ctx = DegradeContext(jd_keywords=[])
    html = "<ul><li>Managed SOC 2 controls and ISO 27001 readiness, cutting audit time by 30%.</li></ul>"
    out = dequantify(html, ctx, "severe")
    assert "SOC 2" in out and "ISO 27001" in out  # standards are names, not metrics
    assert "30%" not in out


def test_short_tech_tokens_extracted_case_aware():
    from tpe.degrade import extract_keywords
    kws = extract_keywords("Senior engineer writing Go services with R analytics and ML pipelines.")
    assert "go" in kws and "r" in kws and "ml" in kws
    kws2 = extract_keywords("We go fast and ship things, come r us.")
    assert "go" not in kws2  # lowercase verb, not the language


def test_slash_delimited_keywords_split():
    from tpe.degrade import extract_keywords, _kw_present
    kws = extract_keywords("Expert in Python/Java development. Python/Java daily. Python/Java stack.")
    assert "python" in kws and "java" in kws
    assert _kw_present("python", "I write Python services")


def test_tie_to_side_counts_as_flip(tmp_path: Path):
    from tpe.judge import judge_both_orders
    responses = iter([make_verdict("tie"), make_verdict("A")])  # tie, then worse-side win
    with patch("tpe.judge.complete_json", side_effect=lambda *a, **k: next(responses)):
        both = judge_both_orders("{{RESUME_A}}{{RESUME_B}}{{JOB_POSTING}}{{ORIGINAL_RESUME}}",
                                 "m", _pair(1), DiskCache(tmp_path))
    assert both.flipped is True  # verdict changed with presentation order


def test_double_tie_is_not_a_flip(tmp_path: Path):
    from tpe.judge import judge_both_orders
    with patch("tpe.judge.complete_json", return_value=make_verdict("tie")):
        both = judge_both_orders("{{RESUME_A}}{{RESUME_B}}{{JOB_POSTING}}{{ORIGINAL_RESUME}}",
                                 "m", _pair(2), DiskCache(tmp_path))
    assert both.flipped is False


def test_compare_runs_rejects_blank_job(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "x", "job": " ", "original": "O", "resume": "R1"}) + "\n")
    b.write_text(json.dumps({"id": "x", "job": " ", "original": "O", "resume": "R2"}) + "\n")
    result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                 "--prompt", "prompts/seed_judge.md",
                                 "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 1
    assert "blank 'job'" in result.output


def test_mcnemar_gate_requires_top_to_win_more():
    # bottom tier wins MORE discordants than top: significant p must NOT pass
    tiers = {}
    for t, correct_ids in (("nano", set("abcdefghij")), ("mini", set("abcdefghij")),
                           ("mid", set("abcdefghij")), ("top", set("ab"))):
        results = []
        for pid in "abcdefghij":
            r = MagicMock()
            r.pair_score = 1.0 if pid in correct_ids else 0.0
            r.flipped = False
            r.lens_score = lambda lens: 1.0
            r.pair.tag = None
            r.pair.pair_id = pid
            results.append(r)
        tiers[t] = results
    g = gate(tiers)
    assert g.mcnemar_c > g.mcnemar_b
    assert not g.significant  # p may be small, but the direction is wrong


def test_bland_leads_budget_skips_unchangeable_first_list():
    ctx = DegradeContext(jd_keywords=[])
    skills = "<ul><li>Python.</li><li>Kafka.</li></ul>"  # wait: digits? none. unchangeable
    exp = "<ul><li>Cut latency 43%.</li><li>Maintained tooling.</li></ul>"
    out = bland_leads(skills + exp, ctx, "subtle")
    assert out == skills + "<ul><li>Maintained tooling.</li><li>Cut latency 43%.</li></ul>"


def test_wall_of_text_budget_skips_single_bullet_first_list():
    from tpe.degrade import wall_of_text
    ctx = DegradeContext(jd_keywords=[])
    single = "<ul><li>Lone bullet.</li></ul>"
    multi = "<ul><li>One thing.</li><li>Another thing.</li></ul>"
    out = wall_of_text(single + multi, ctx, "moderate")
    assert out.startswith(single)
    assert out.count("<li>") == 2  # multi list merged into one bullet


# --- rounds 10-11 (found via paginated re-check) ---

def test_csharp_survives_keyword_extraction():
    from tpe.degrade import extract_keywords, _kw_present
    kws = extract_keywords("Senior C# engineer. C# and .NET daily. C# services.")
    assert "c#" in kws
    assert _kw_present("c#", "<li>Built C# microservices</li>")
    assert not _kw_present("c#", "<li>Built C++ services</li>")


def test_slash_delimited_short_tokens_survive():
    from tpe.degrade import extract_keywords
    kws = extract_keywords("Statistical modeling in R/Python required. R/Python daily work.")
    assert "r" in kws and "python" in kws
    kws2 = extract_keywords("Systems programming in C/C++ environments. C/C++ expertise.")
    assert "c" in kws2 and "c++" in kws2


def test_acronym_guard_spares_scale_metrics():
    from tpe.degrade import _has_metric, _metric_spans
    assert _has_metric("Scaled API 10M requests/day")     # magnitude suffix = metric
    assert not _has_metric("Managed SOC 2 controls")      # bare number after acronym = name
    assert not _has_metric("Led ISO 27001 readiness")


def test_mcnemar_one_sided_directional():
    from tpe.metrics import mcnemar_exact
    # 9 top-wins vs 2 bottom-wins: two-sided ~0.065 fails, one-sided ~0.033 passes
    assert mcnemar_exact(9, 2) > 0.05
    assert mcnemar_exact(9, 2, one_sided=True) < 0.05
    # wrong direction can never be significant one-sided
    assert mcnemar_exact(2, 9, one_sided=True) > 0.5


def test_compare_runs_rejects_non_dict_meta(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "x", "job": "J", "original": "O", "resume": "R1"}) + "\n")
    b.write_text(json.dumps({"id": "x", "job": "J", "original": "O", "resume": "R2",
                             "meta": "oops-a-string"}) + "\n")
    result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                 "--prompt", "prompts/seed_judge.md",
                                 "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 1
    assert "non-object 'meta'" in result.output


# --- round 13 ---

def test_versioned_tools_are_not_metrics():
    from tpe.degrade import _has_metric, dequantify
    assert not _has_metric("Skills: Python 3, Vue 2, Angular 2")
    assert _has_metric("Led 3 engineers")           # capitalized verb, real metric
    assert _has_metric("cut latency by 3")          # by-prefix, real metric
    ctx = DegradeContext(jd_keywords=[])
    html = "<ul><li>Migrated services to Python 3, cutting build time by 40%.</li></ul>"
    out = dequantify(html, ctx, "severe")
    assert "Python 3" in out and "40%" not in out
