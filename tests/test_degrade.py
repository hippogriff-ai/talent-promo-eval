import re
from pathlib import Path

from tpe.degrade import DEGRADATIONS, DegradeContext, extract_keywords

RESUME = Path("tests/fixtures/resume.html").read_text()
JD = """Staff Engineer, Agent Infrastructure. You will build LLM agent infrastructure
in Python on Kubernetes. Experience with Kafka, distributed systems, and observability
required. Python and Kubernetes expertise essential. LLM experience preferred."""
CTX = DegradeContext(jd_keywords=extract_keywords(JD))
BY_NAME = {d.name: d for d in DEGRADATIONS}


def test_extract_keywords_finds_tech_terms():
    kws = extract_keywords(JD)
    assert "python" in kws and "kubernetes" in kws
    assert "the" not in kws and "you" not in kws


def test_all_degradations_deterministic_and_modifying():
    for d in DEGRADATIONS:
        for sev in d.severities:
            out1 = d.fn(RESUME, CTX, sev)
            out2 = d.fn(RESUME, CTX, sev)
            assert out1 == out2, f"{d.name}:{sev} not deterministic"
            assert out1 != RESUME, f"{d.name}:{sev} was a no-op on the fixture"


def test_keyword_strip_removes_jd_terms():
    out = BY_NAME["keyword_strip"].fn(RESUME, CTX, "severe")
    assert "Python" not in out and "Kubernetes" not in out


def test_dequantify_severe_removes_digits_from_bullets():
    out = BY_NAME["dequantify"].fn(RESUME, CTX, "severe")
    bullets = re.findall(r"<li>(.*?)</li>", out, re.S)
    assert bullets and not any(re.search(r"\d", b) for b in bullets)


def test_header_flatten_severe_removes_h2():
    out = BY_NAME["header_flatten"].fn(RESUME, CTX, "severe")
    assert "<h2>" not in out and "Experience" in out


def test_bury_relevant_moves_keyword_dense_role_last():
    out = BY_NAME["bury_relevant"].fn(RESUME, CTX, "moderate")
    assert out.find("Data Platform") > out.find("Payments")


def test_wall_of_text_merges_bullets():
    out = BY_NAME["wall_of_text"].fn(RESUME, CTX, "severe")
    assert out.count("<li>") < RESUME.count("<li>")


def test_keyword_stuff_adds_unsupported_terms():
    out = BY_NAME["keyword_stuff"].fn(RESUME, CTX, "severe")
    assert "observability" in out.lower()  # in JD, not in original resume
    assert len(out) > len(RESUME)


def test_bland_leads_sinks_quantified_bullets():
    out = BY_NAME["bland_leads"].fn(RESUME, CTX, "moderate")
    first_bullet = re.search(r"<li>(.*?)</li>", out, re.S).group(1)
    assert not re.search(r"\d", first_bullet)


def test_drop_summary_removes_summary_section():
    out = BY_NAME["drop_summary"].fn(RESUME, CTX, "moderate")
    assert "<h2>Summary</h2>" not in out and "Experience" in out
