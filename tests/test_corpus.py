from pathlib import Path

import pytest

from tpe.corpus import load_corpus, load_human_codes

CORPUS = Path("data/corpus/corpus.jsonl")


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus snapshot not present")
def test_load_corpus_full():
    records = load_corpus(CORPUS)
    assert len(records) == 28
    r = records[0]
    assert r.trace_id and r.job_text and len(r.generated_html) > 1000
    assert "—" in r.job_text or "-" in r.job_text  # title header composed


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus snapshot not present")
def test_load_human_codes_full():
    codes = load_human_codes()
    assert len(codes) == 28
    sample = next(iter(codes.values()))
    assert 0 <= sample["F"] <= 100


def test_load_corpus_handles_dict_job_posting(tmp_path: Path):
    line = ('{"trace_id": "t1", "source_profile": "P", '
            '"job_posting": {"title": "Staff Eng", "company": "Acme", "text": "Build things"}, '
            '"generated_html": "<p>H</p>"}')
    f = tmp_path / "c.jsonl"
    f.write_text(line + "\n")
    records = load_corpus(f)
    assert records[0].profile_text == "P"
    assert "Staff Eng" in records[0].job_text and "Build things" in records[0].job_text
    assert records[0].generated_html == "<p>H</p>"
