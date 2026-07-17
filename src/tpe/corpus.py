"""Read the snapshotted corpus. Field mapping matches the talent-promo corpus builder output."""
import json
from dataclasses import dataclass
from pathlib import Path

from tpe.jsonl import read_jsonl


@dataclass(frozen=True)
class CorpusRecord:
    trace_id: str
    profile_text: str
    job_text: str
    generated_html: str


def _job_text(raw: dict) -> str:
    job = (raw.get("job_posting") or raw.get("job_text")
           or (raw.get("inputs") or {}).get("job_posting") or "")
    if isinstance(job, str):
        return job
    title, company, text = job.get("title", ""), job.get("company", ""), job.get("text", "")
    header = " — ".join(x for x in (title, company) if x)
    return f"{header}\n\n{text}".strip() if header else text


def _first(raw: dict, *paths: tuple[str, ...]):
    for path in paths:
        node = raw
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if node:
            return node
    return None


def load_corpus(path: Path = Path("data/corpus/corpus.jsonl")) -> list[CorpusRecord]:
    records = []
    for raw in read_jsonl(path):
        record = CorpusRecord(
            trace_id=raw["trace_id"],
            profile_text=_first(raw, ("source_profile",), ("profile_text",),
                                ("inputs", "source_profile")) or "",
            job_text=_job_text(raw),
            generated_html=_first(raw, ("generated_html",),
                                  ("outputs", "generated_html")) or "",
        )
        # Blank inputs would silently build a corrupt pair dataset downstream.
        missing = [f for f in ("profile_text", "job_text", "generated_html")
                   if not getattr(record, f).strip()]
        if missing:
            raise ValueError(f"corpus record {record.trace_id!r} is missing {missing}; "
                             f"unrecognized record shape? keys: {sorted(raw)}")
        records.append(record)
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
