from tpe.grounding import FACTS_HEADER, compose_grounding


def test_no_facts_returns_original_unchanged():
    assert compose_grounding("resume text") == "resume text"
    assert compose_grounding("resume text", []) == "resume text"
    assert compose_grounding("resume text", ["", "  "]) == "resume text"


def test_facts_appended_under_header():
    out = compose_grounding("resume text", ["Led migration of 12 services", "AWS certified 2024"])
    assert out.startswith("resume text")
    assert FACTS_HEADER in out
    assert "- Led migration of 12 services" in out
    assert "- AWS certified 2024" in out


def test_facts_deduplicated_order_preserved():
    out = compose_grounding("r", ["fact A", "fact B", "fact A"])
    assert out.count("fact A") == 1
    assert out.index("fact A") < out.index("fact B")


def test_facts_header_authorizes_facts_as_evidence():
    out = compose_grounding("resume", ["AWS certified 2024"])
    assert "treat them as evidence equal to the resume above" in out
    assert "NOT as unsupported additions" in out
