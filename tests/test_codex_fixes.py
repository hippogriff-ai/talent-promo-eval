"""Regression tests for the codex-review fixes."""
import json
from pathlib import Path

import pytest

from tpe.corpus import CorpusRecord, load_corpus
from tpe.dataset import build_pairs
from tpe.degrade import DEGRADATIONS, DegradeContext, bury_relevant, keyword_strip, keyword_stuff
from tpe.judge import render_prompt
from tpe.sensitivity import gate

BY_NAME = {d.name: d for d in DEGRADATIONS}


def test_corpus_loads_nested_record_shape(tmp_path: Path):
    line = json.dumps({"trace_id": "t1",
                       "inputs": {"source_profile": "P", "job_posting": "J text"},
                       "outputs": {"generated_html": "<p>" + "H" * 20 + "</p>"}})
    f = tmp_path / "c.jsonl"
    f.write_text(line + "\n")
    rec = load_corpus(f)[0]
    assert rec.profile_text == "P" and rec.job_text == "J text" and "H" in rec.generated_html


def test_corpus_rejects_blank_fields_instead_of_silent_empty(tmp_path: Path):
    f = tmp_path / "c.jsonl"
    f.write_text(json.dumps({"trace_id": "t1", "unknown_key": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        load_corpus(f)


def test_keyword_matching_handles_symbolic_tokens():
    ctx = DegradeContext(jd_keywords=["c++", "python"])
    html = "<li>Wrote c++ services and Python tools.</li>"
    out = keyword_strip(html, ctx, "severe")
    assert "c++" not in out  # symbolic token actually stripped
    stuffed = keyword_stuff("<li>Java only.</li>", ctx, "subtle")
    assert "C++" in stuffed  # counted as missing, so added to the stuffing blob


def test_bury_relevant_noop_without_unique_relevant_role():
    ctx = DegradeContext(jd_keywords=["golang"])  # matches neither role
    html = ("<h2>Experience</h2><h3>Role One</h3><ul><li>a</li></ul>"
            "<h3>Role Two</h3><ul><li>b</li></ul>")
    assert bury_relevant(html, ctx, "moderate") == html
    assert bury_relevant(html, ctx, "severe") == html


def test_bland_leads_noop_on_digit_free_bullets():
    ctx = DegradeContext(jd_keywords=[])
    html = "<ul><li>Led the platform redesign.</li><li>Mentored the team.</li></ul>"
    assert BY_NAME["bland_leads"].fn(html, ctx, "moderate") == html


def test_build_pairs_dedupes_identical_degradation_outputs():
    resume = Path("tests/fixtures/resume.html").read_text()
    rec = CorpusRecord(trace_id="t1", profile_text="p", generated_html=resume,
                       job_text="Python role. Python required. Python.")
    pairs = build_pairs([rec], human_codes={})
    worse_texts = [p.worse for p in pairs]
    assert len(worse_texts) == len(set(worse_texts))  # no byte-identical leakage


def test_render_prompt_does_not_expand_placeholders_inside_documents():
    template = "A: {{RESUME_A}} B: {{RESUME_B}}"
    out = render_prompt(template, "j", "o", "content mentions {{RESUME_B}} literally", "B-DOC")
    assert out.count("B-DOC") == 1  # not spliced into A's content
    assert "content mentions {{RESUME_B}} literally" in out


def test_gate_fails_closed_on_partial_ladder():
    with pytest.raises(ValueError, match="missing"):
        gate({"nano": [], "top": []})
