"""Regression tests for the code-review fixes."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from openai import BadRequestError

from tests.conftest import make_verdict
from tpe.cache import DiskCache, cache_key
from tpe.gepa_adapter import JudgeAdapter
from tpe.judge import run_pairs
from tpe.metrics import wilson_ci
from tpe.schema import DegradationTag, KnownPair
from tpe.sensitivity import gate


def _pair(i: int) -> KnownPair:
    return KnownPair(pair_id=f"p{i}:dequantify:subtle", job_text="J", original_resume="O",
                     better=f"GOOD{i}", worse=f"BAD{i}",
                     tag=DegradationTag(name="dequantify", lens="ats", severity="subtle"),
                     source="synthetic", split="train")


def test_cache_self_heals_corrupt_entry(tmp_path: Path):
    c = DiskCache(tmp_path)
    key = cache_key("m", "p")
    (tmp_path / f"{key}.json").write_text('{"truncated": ')  # killed mid-write
    assert c.get(key) is None  # heals instead of raising
    assert not (tmp_path / f"{key}.json").exists()
    c.put(key, {"ok": 1})
    assert c.get(key) == {"ok": 1}


def test_cache_concurrent_same_key_puts_are_safe(tmp_path: Path):
    c = DiskCache(tmp_path)
    key = cache_key("m", "same-prompt")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: c.put(key, {"v": 1}), range(64)))
    assert c.get(key) == {"v": 1}
    assert list(tmp_path.glob("*.tmp")) == []  # no leftover temp files


def test_run_pairs_surfaces_nontransient_error_immediately(tmp_path: Path):
    import httpx
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"),
                          json={"error": {"message": "bad schema"}})
    err = BadRequestError("bad schema", response=resp, body=None)
    with patch("tpe.judge.complete_json", side_effect=err):
        with pytest.raises(BadRequestError):  # original type, no RuntimeError laundering
            run_pairs("t {{RESUME_A}}{{RESUME_B}}{{JOB_POSTING}}{{ORIGINAL_RESUME}}",
                      "m", [_pair(1)], DiskCache(tmp_path))


def test_refusal_surfaces_without_retry():
    from tpe.models import JudgeRefusal, complete_json

    class Msg:
        content = None
        refusal = "policy"

    class Resp:
        choices = [type("C", (), {"message": Msg()})()]

    with patch("tpe.models.client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = Resp()
        with pytest.raises(JudgeRefusal, match="policy"):
            complete_json("m", "s", "u", {})
    assert mock_client.return_value.chat.completions.create.call_count == 1  # no retries


def test_wilson_ci_accepts_fractional_wins():
    lo, hi = wilson_ci(7.5, 10)
    assert lo < 0.75 < hi
    # CI is centered on the same rate the caller reports (no rounding to 8/10)
    lo8, hi8 = wilson_ci(8, 10)
    assert (lo, hi) != (lo8, hi8)


def test_adapter_parallel_preserves_batch_order(tmp_path: Path):
    adapter = JudgeAdapter(model="m", cache=DiskCache(tmp_path), max_workers=4)
    batch = [_pair(i) for i in range(8)]
    with patch("tpe.judge.complete_json",
               side_effect=lambda model, system, user, schema:
               make_verdict("A" if "A: GOOD" in user or user.find("GOOD") < user.find("BAD")
                            else "B")):
        result = adapter.evaluate(batch, {"judge_prompt":
                                          "A: {{RESUME_A}} B: {{RESUME_B}} {{JOB_POSTING}} {{ORIGINAL_RESUME}}"},
                                  capture_traces=True)
    assert len(result.scores) == 8
    assert [t[0].pair_id for t in result.trajectories] == [p.pair_id for p in batch]


def test_gate_rejects_unknown_tier():
    with pytest.raises(ValueError, match="not in the ladder"):
        gate({"warp-drive": []})
