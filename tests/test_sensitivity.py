from unittest.mock import MagicMock

from tpe.sensitivity import gate, render_report


def _results(acc: float, subtle_acc: float, n: int = 20):
    out = []
    for i in range(n):
        r = MagicMock()
        severity = "subtle" if i < n // 2 else "severe"
        target = subtle_acc if severity == "subtle" else acc
        # deterministic: first target*count of each slice correct
        idx_in_slice = i if severity == "subtle" else i - n // 2
        r.pair_score = 1.0 if idx_in_slice < round(target * (n // 2)) else 0.0
        r.flipped = False
        r.lens_score = lambda lens: 1.0
        r.pair.tag.severity = severity
        r.pair.tag.name = "keyword_strip"
        r.pair.tag.lens = "ats"
        r.pair.pair_id = f"p{i}"
        out.append(r)
    return out


def test_gate_passes_on_rising_ladder():
    tiers = {"nano": _results(0.6, 0.4), "mini": _results(0.75, 0.6),
             "mid": _results(0.85, 0.8), "top": _results(1.0, 1.0)}
    g = gate(tiers)
    assert g.monotone and g.spread_ok
    assert g.subtle_spread >= 0.10
    assert g.significant  # nano->top: many discordant pairs, one-sided
    assert g.passed


def test_gate_fails_on_flat_ladder():
    tiers = {t: _results(0.7, 0.7) for t in ("nano", "mini", "mid", "top")}
    g = gate(tiers)
    assert not g.spread_ok and not g.passed


def test_render_report_names_the_verdict():
    tiers = {t: _results(0.7, 0.7) for t in ("nano", "mini", "mid", "top")}
    text = render_report(gate(tiers))
    assert "GATE FAILED" in text and "| nano |" in text
