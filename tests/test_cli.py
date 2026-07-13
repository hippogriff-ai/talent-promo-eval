from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tpe.cli import app

runner = CliRunner()


def _verdict(winner: str) -> dict:
    lens = {"winner": winner, "evidence": "e"}
    return {"ats_signal": lens, "human_skim": lens,
            "overall": {"winner": winner, "margin": "clear", "rationale": "r"}}


def test_eval_prompt_runs_on_val(tmp_path: Path):
    # Build a tiny self-contained pairs dir: data/ is local-only (not in checkouts/CI)
    from tpe.corpus import CorpusRecord
    from tpe.dataset import build_pairs, write_splits

    resume = Path("tests/fixtures/resume.html").read_text()
    rec = CorpusRecord(trace_id="t1", profile_text="profile", generated_html=resume,
                       job_text="Staff Engineer. Python, Kubernetes, Kafka, observability.")
    pairs_dir = tmp_path / "pairs"
    write_splits(build_pairs([rec], human_codes={}), pairs_dir)
    with patch("tpe.judge.complete_json", return_value=_verdict("A")):
        result = runner.invoke(app, ["eval-prompt", "--prompt", "prompts/seed_judge.md",
                                     "--split", "val", "--limit", "2",
                                     "--pairs-dir", str(pairs_dir),
                                     "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    assert "accuracy" in result.output.lower()


def test_judge_one_reports_consistent_verdict(tmp_path: Path):
    for name, text in (("orig.txt", "ORIG"), ("a.html", "GOOD X"), ("b.html", "BAD Y")):
        (tmp_path / name).write_text(text)
    # judge always prefers whatever slot holds "GOOD"
    with patch("tpe.judge.complete_json",
               side_effect=lambda model, system, user, schema:
               _verdict("A" if user.find("GOOD") < user.find("BAD") else "B")):
        result = runner.invoke(app, ["judge-one", "--job", "JD text",
                                     "--original", str(tmp_path / "orig.txt"),
                                     "--a", str(tmp_path / "a.html"),
                                     "--b", str(tmp_path / "b.html"),
                                     "--prompt", "prompts/seed_judge.md",
                                     "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    assert "a.html wins both orders" in result.output
