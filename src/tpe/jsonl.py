"""JSONL I/O. Every reader shares the same blank-line and error semantics
instead of hand-rolling the pattern per module."""
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: malformed JSONL line") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
