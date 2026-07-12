"""Read the snapshotted corpus. Field mapping matches the talent-promo corpus builder output."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusRecord:
    trace_id: str
    profile_text: str
    job_text: str
    generated_html: str


def _job_text(raw: dict) -> str:
    job = raw.get("job_posting", raw.get("job_text", ""))
    if isinstance(job, str):
        return job
    title, company, text = job.get("title", ""), job.get("company", ""), job.get("text", "")
    header = " — ".join(x for x in (title, company) if x)
    return f"{header}\n\n{text}".strip() if header else text


def load_corpus(path: Path = Path("data/corpus/corpus.jsonl")) -> list[CorpusRecord]:
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        records.append(CorpusRecord(
            trace_id=raw["trace_id"],
            profile_text=raw.get("source_profile", raw.get("profile_text", "")),
            job_text=_job_text(raw),
            generated_html=raw.get("generated_html", ""),
        ))
    return records


def load_human_codes(path: Path = Path("data/corpus/judge_alignment_from_coding.json")) -> dict[str, dict]:
    """trace_id -> {"F": faithfulness score, "R": job-relevance score}."""
    data = json.loads(Path(path).read_text())
    out = {}
    for entry in data.get("records", []):
        f = entry.get("faithfulness_score_suggested")
        r = entry.get("job_relevance_score")
        if isinstance(f, (int, float)):
            out[entry["trace_id"]] = {"F": int(f), "R": int(r) if isinstance(r, (int, float)) else None}
    return out
