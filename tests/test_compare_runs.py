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


def test_compare_runs_appends_discovered_facts_to_grounding(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "x", "job": "JD", "original": "ORIG", "resume": "OLD",
                             "discovered_facts": ["Fact from session A"]}) + "\n")
    b.write_text(json.dumps({"id": "x", "job": "JD", "original": "ORIG", "resume": "NEW",
                             "discovered_facts": ["Fact from session B"]}) + "\n")
    seen_prompts: list[str] = []

    def capture(model, system, user, schema):
        seen_prompts.append(user)
        return _verdict("tie")

    with patch("tpe.judge.complete_json", side_effect=capture):
        result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                     "--prompt", "prompts/seed_judge.md",
                                     "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    assert seen_prompts and all("Fact from session A" in p and "Fact from session B" in p
                                and "Additional facts confirmed by the candidate" in p
                                for p in seen_prompts)


def test_compare_runs_slices_by_meta(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows_a, rows_b = [], []
    for k in range(4):
        skipped = "yes" if k < 2 else "no"
        rows_a.append({"id": f"i{k}", "job": "JD", "original": "O", "resume": f"OLD{k}"})
        rows_b.append({"id": f"i{k}", "job": "JD", "original": "O", "resume": f"NEW{k}",
                       "meta": {"discovery_skipped": skipped}})
    a.write_text("".join(json.dumps(r) + "\n" for r in rows_a))
    b.write_text("".join(json.dumps(r) + "\n" for r in rows_b))
    # NEW wins only on ids i2/i3 (discovery not skipped); OLD wins i0/i1
    def respond(model, system, user, schema):
        import re
        k = int(re.search(r"(?:NEW|OLD)(\d)", user).group(1))
        new_first = user.find(f"NEW{k}") < user.find(f"OLD{k}")
        winner_is_new = k >= 2
        return _verdict("A" if (new_first == winner_is_new) else "B")

    with patch("tpe.judge.complete_json", side_effect=respond):
        result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                     "--prompt", "prompts/seed_judge.md",
                                     "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    assert "## by discovery_skipped" in result.output
    assert "discovery_skipped=no: 2/2 (100.0%)" in result.output
    assert "discovery_skipped=yes: 0/2 (0.0%)" in result.output


def test_compare_runs_errors_on_disjoint_ids(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "x", "job": "J", "resume": "R"}) + "\n")
    b.write_text(json.dumps({"id": "y", "job": "J", "resume": "R"}) + "\n")
    result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                 "--prompt", "prompts/seed_judge.md",
                                 "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 1
    assert "no shared ids" in result.output
