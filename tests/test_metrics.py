import math
from unittest.mock import MagicMock

from tpe.metrics import mcnemar_exact, summarize, wilson_ci


def _mock_result(pair_score: float, flipped: bool, severity="subtle", name="keyword_strip"):
    r = MagicMock()
    r.pair_score = pair_score
    r.flipped = flipped
    r.lens_score = lambda lens: pair_score
    r.pair.tag.severity = severity
    r.pair.tag.name = name
    r.pair.tag.lens = "ats"
    return r


def test_summarize_accuracy_and_flip_rate():
    results = [_mock_result(1.0, False), _mock_result(0.0, False),
               _mock_result(0.5, True), _mock_result(1.0, False)]
    s = summarize(results)
    assert s.n == 4
    assert math.isclose(s.accuracy, (1.0 + 0.0 + 0.5 + 1.0) / 4)
    assert math.isclose(s.flip_rate, 0.25)
    assert math.isclose(s.gepa_metric, s.accuracy - 0.25 * s.flip_rate)
    assert "subtle" in s.by_severity
    assert "keyword_strip" in s.by_type
    assert "ats_signal" in s.by_lens


def test_summarize_handles_untagged_pairs():
    r = _mock_result(1.0, False)
    r.pair.tag = None
    s = summarize([r])
    assert s.n == 1 and s.by_severity == {}


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    # b=8, c=2: two-sided exact p = 2 * P(X<=2 | n=10, p=.5) = 2*(1+10+45)/1024
    assert math.isclose(mcnemar_exact(8, 2), 2 * 56 / 1024)
    assert mcnemar_exact(5, 5) == 1.0


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(85, 100)
    assert 0.75 < lo < 0.85 < hi < 0.92
    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == 0.0 and hi0 > 0.2
