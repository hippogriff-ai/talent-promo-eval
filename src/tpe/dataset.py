"""Manufacture known-order pairs from the corpus + degradations, and split them."""
import hashlib
from pathlib import Path

from tpe.corpus import CorpusRecord
from tpe.degrade import DEGRADATIONS, DegradeContext, extract_keywords
from tpe.schema import DegradationTag, KnownPair

HELDOUT_TYPES = {"header_flatten", "drop_summary"}  # never trained on: generalization guard
ANCHOR_F_MIN = 85  # human-coded faithfulness floor for original-vs-generated anchors


def _split_for(pair_id: str, tag_name: str | None, heldout_types: set[str]) -> str:
    if tag_name in heldout_types:
        return "test"
    h = int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else ("val" if h < 80 else "test")


def build_pairs(records: list[CorpusRecord], human_codes: dict[str, dict],
                heldout_types: set[str] = HELDOUT_TYPES) -> list[KnownPair]:
    pairs: list[KnownPair] = []
    for rec in records:
        ctx = DegradeContext(jd_keywords=extract_keywords(rec.job_text))
        for deg in DEGRADATIONS:
            for sev in deg.severities:
                worse = deg.fn(rec.generated_html, ctx, sev)
                if worse == rec.generated_html:
                    continue  # not applicable to this record
                pair_id = f"{rec.trace_id}:{deg.name}:{sev}"
                pairs.append(KnownPair(
                    pair_id=pair_id, job_text=rec.job_text,
                    original_resume=rec.profile_text,
                    better=rec.generated_html, worse=worse,
                    tag=DegradationTag(name=deg.name, lens=deg.lens, severity=sev),
                    source="synthetic",
                    split=_split_for(pair_id, deg.name, heldout_types),
                ))
        codes = human_codes.get(rec.trace_id)
        if codes and codes.get("F", 0) >= ANCHOR_F_MIN:
            pairs.append(KnownPair(
                pair_id=f"{rec.trace_id}:human_anchor", job_text=rec.job_text,
                original_resume=rec.profile_text,
                better=rec.generated_html, worse=rec.profile_text,
                tag=None, source="human_anchor", split="anchor",
            ))
    return pairs


def write_splits(pairs: list[KnownPair], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test", "anchor"):
        subset = [p for p in pairs if p.split == split]
        (out_dir / f"{split}.jsonl").write_text(
            "".join(p.model_dump_json() + "\n" for p in subset))
        counts[split] = len(subset)
    return counts


def load_pairs(path: Path) -> list[KnownPair]:
    return [KnownPair.model_validate_json(line)
            for line in Path(path).read_text().splitlines() if line.strip()]
