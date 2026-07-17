from pathlib import Path

SEED = Path("prompts/seed_judge.md")


def test_seed_prompt_has_all_placeholders():
    text = SEED.read_text()
    for ph in ("{{JOB_POSTING}}", "{{ORIGINAL_RESUME}}", "{{RESUME_A}}", "{{RESUME_B}}"):
        assert ph in text


def test_seed_prompt_mentions_both_lenses_and_stuffing():
    text = SEED.read_text().lower()
    assert "ats" in text or "scanner" in text
    assert "skim" in text or "seconds" in text
    assert "stuff" in text  # anti-gaming clause present
    assert "tie" in text  # tie-legitimizing clause present
