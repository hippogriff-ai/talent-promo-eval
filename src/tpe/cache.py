"""Content-addressed disk cache for judge calls. Reruns and GEPA re-evaluations are free."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x1f")  # separator so ("ab","c") != ("a","bc")
    return h.hexdigest()


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            # A truncated entry (killed mid-write) must self-heal, not poison the key.
            p.unlink(missing_ok=True)
            return None

    def put(self, key: str, value: dict) -> None:
        # Atomic write: an interrupted put leaves the old state, never a partial file.
        # tmp name must be unique per WRITER (threads share a pid): concurrent puts of
        # the same key would otherwise collide on the tmp path and os.replace twice.
        p = self._path(key)
        tmp = p.with_name(f"{p.stem}.{uuid4().hex}.tmp")
        tmp.write_text(json.dumps(value))
        os.replace(tmp, p)
