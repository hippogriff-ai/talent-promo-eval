from pathlib import Path

from tpe.corpus import CorpusRecord
from tpe.dataset import build_pairs, load_pairs, write_splits

RESUME = Path("tests/fixtures/resume.html").read_text()
JD = "Staff Engineer. Python, Kubernetes, Kafka, LLM agent infrastructure, observability."
REC = CorpusRecord(trace_id="t1", profile_text="profile", job_text=JD, generated_html=RESUME)


def test_build_pairs_generates_tagged_known_order_pairs():
    pairs = build_pairs([REC], human_codes={})
    assert len(pairs) >= 10
    p = pairs[0]
    assert p.better == RESUME and p.worse != RESUME and p.tag is not None


def test_split_is_deterministic_and_heldout_types_go_to_test():
    pairs1 = build_pairs([REC], human_codes={})
    pairs2 = build_pairs([REC], human_codes={})
    assert [(p.pair_id, p.split) for p in pairs1] == [(p.pair_id, p.split) for p in pairs2]
    for p in pairs1:
        if p.tag and p.tag.name in {"header_flatten", "drop_summary"}:
            assert p.split == "test"


def test_human_anchor_pairs_only_for_high_f_traces():
    codes = {"t1": {"F": 90, "R": 80}}
    pairs = build_pairs([REC], human_codes=codes)
    anchors = [p for p in pairs if p.source == "human_anchor"]
    assert len(anchors) == 1
    assert anchors[0].better == RESUME and anchors[0].worse == "profile"
    assert anchors[0].split == "anchor"


def test_no_anchor_for_low_f_traces():
    codes = {"t1": {"F": 60, "R": 80}}
    pairs = build_pairs([REC], human_codes=codes)
    assert not [p for p in pairs if p.source == "human_anchor"]


def test_write_and_load_roundtrip(tmp_path: Path):
    pairs = build_pairs([REC], human_codes={})
    write_splits(pairs, tmp_path)
    train = load_pairs(tmp_path / "train.jsonl")
    assert train and all(p.split == "train" for p in train)
