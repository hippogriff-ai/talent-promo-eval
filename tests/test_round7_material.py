"""Tests for the round-7 improvements: bootstrap-CI spread gate + shared jsonl I/O."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tpe.jsonl import read_jsonl, write_jsonl
from tpe.metrics import paired_bootstrap_diff_ci
from tpe.sensitivity import gate


def test_paired_bootstrap_ci_excludes_zero_for_clear_diff():
    top = [1.0] * 30
    bottom = [1.0] * 10 + [0.0] * 20
    lo, hi = paired_bootstrap_diff_ci(top, bottom)
    assert 0 < lo < hi <= 1.0


def test_paired_bootstrap_ci_straddles_zero_for_noise():
    top = [1.0, 0.0] * 15
    bottom = [0.0, 1.0] * 15  # same mean, pure noise
    lo, hi = paired_bootstrap_diff_ci(top, bottom)
    assert lo < 0 < hi


def test_paired_bootstrap_ci_is_deterministic():
    top, bottom = [1.0, 0.5, 1.0, 0.0] * 8, [0.5, 0.5, 0.0, 0.0] * 8
    assert paired_bootstrap_diff_ci(top, bottom) == paired_bootstrap_diff_ci(top, bottom)


def test_paired_bootstrap_requires_alignment():
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap_diff_ci([1.0], [1.0, 0.0])


def _tier(pair_scores: dict[str, float], subtle_ids: set[str]):
    out = []
    for pid, score in pair_scores.items():
        r = MagicMock()
        r.pair_score = score
        r.flipped = False
        r.lens_score = lambda lens: 1.0
        r.pair.pair_id = pid
        if pid in subtle_ids:
            r.pair.tag.severity = "subtle"
            r.pair.tag.name = "keyword_strip"
            r.pair.tag.lens = "ats"
        else:
            r.pair.tag = None
        out.append(r)
    return out


def test_gate_spread_requires_ci_above_zero():
    # spread point estimate passes 0.10 but is pure noise on a tiny slice
    ids = [f"p{i}" for i in range(8)]
    subtle = set(ids[:4])
    base = {pid: 1.0 for pid in ids}
    bottom = dict(base); bottom["p0"] = 0.0  # subtle acc 0.75
    top = dict(base)                          # subtle acc 1.00 -> spread +0.25, n=4
    tiers = {"nano": _tier(bottom, subtle), "mini": _tier(base, subtle),
             "mid": _tier(base, subtle), "top": _tier(top, subtle)}
    g = gate(tiers)
    assert g.subtle_spread >= 0.10
    # with n=4 the bootstrap CI must straddle zero -> spread_ok False
    assert g.spread_ci[0] <= 0
    assert not g.spread_ok


def test_read_jsonl_reports_line_numbers(tmp_path: Path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"ok": 1}\n\n{broken\n')
    with pytest.raises(ValueError, match="x.jsonl:3"):
        read_jsonl(f)


def test_jsonl_roundtrip(tmp_path: Path):
    f = tmp_path / "r.jsonl"
    rows = [{"a": 1}, {"b": [1, 2]}]
    write_jsonl(f, rows)
    assert read_jsonl(f) == rows
