"""Content-addressed disk cache for judge calls. Reruns and GEPA re-evaluations are free."""
import hashlib
import json
from pathlib import Path


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
        return json.loads(p.read_text())

    def put(self, key: str, value: dict) -> None:
        self._path(key).write_text(json.dumps(value))
