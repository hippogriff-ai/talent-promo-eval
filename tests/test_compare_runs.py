import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tpe.cli import app

runner = CliRunner()


def _verdict(winner: str) -> dict:
    lens = {"winner": winner, "evidence": "e"}
    return {"ats_signal": lens, "human_skim": lens,
            "overall": {"winner": winner, "margin": "clear", "rationale": "r"}}


def _write_run(path: Path, resume_text: str):
    rows = [{"id": f"i{k}", "job": "JD", "original": "ORIG", "resume": f"{resume_text}{k}"}
            for k in range(3)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_compare_runs_reports_win_rate(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_run(a, "OLD")
    _write_run(b, "NEW")
    # B (NEW) always wins: pick whichever slot holds NEW
    with patch("tpe.judge.complete_json",
               side_effect=lambda model, system, user, schema:
               _verdict("A" if user.find("NEW") < user.find("OLD") else "B")):
        result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                     "--prompt", "prompts/seed_judge.md",
                                     "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    assert "3/3" in result.output
    assert "100" in result.output


def test_compare_runs_errors_on_disjoint_ids(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "x", "job": "J", "resume": "R"}) + "\n")
    b.write_text(json.dumps({"id": "y", "job": "J", "resume": "R"}) + "\n")
    result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                 "--prompt", "prompts/seed_judge.md",
                                 "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 1
    assert "no shared ids" in result.output
