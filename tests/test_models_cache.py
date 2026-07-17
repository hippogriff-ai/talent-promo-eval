from pathlib import Path

from tpe.cache import DiskCache, cache_key


def test_cache_key_stable_and_order_sensitive():
    assert cache_key("a", "b") == cache_key("a", "b")
    assert cache_key("a", "b") != cache_key("b", "a")
    assert cache_key("ab", "c") != cache_key("a", "bc")


def test_disk_cache_roundtrip(tmp_path: Path):
    c = DiskCache(tmp_path)
    key = cache_key("m", "p")
    assert c.get(key) is None
    c.put(key, {"x": 1})
    assert c.get(key) == {"x": 1}


def test_ladder_env_override(monkeypatch):
    monkeypatch.setenv("TPE_MODEL_MINI", "test-model-id")
    from tpe import models
    assert models.ladder()["mini"] == "test-model-id"
    assert set(models.ladder()) == {"nano", "mini", "mid", "top"}
