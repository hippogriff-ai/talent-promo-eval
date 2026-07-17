# talent-promo-eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone eval harness with a GEPA-optimized pairwise LLM judge that decides which of two resume versions better fits a job posting (ATS lens + human-skim lens), validated by a model-tier sensitivity gate.

**Architecture:** Known-order pairs are manufactured by applying tagged degradations to real resumes from the talent-promo corpus snapshot. A single pairwise judge prompt (OpenAI structured output, both A/B orders, disk-cached) is scored on those pairs; GEPA evolves the prompt against that score; a frozen prompt must then pass accuracy, flip-rate, and tier-sensitivity gates. See `docs/design.html` for the approved design.

**Tech Stack:** Python 3.12 via uv · pydantic v2 · openai SDK · gepa · typer · httpx · pytest

## Global Constraints

- Repo: the repo root — standalone; NEVER import from the talent-promo app; corpus arrives via one-time file snapshot.
- Judge provider: OpenAI only (generator is Anthropic; cross-provider by design). Workhorse judge model: mini tier; reflection + top rung of ladder: top tier. Exact model IDs pinned from the tech-facts research report in `src/tpe/models.py` (single source of truth).
- Every judge call goes through the disk cache (`runs/cache/`); cache key = sha256(model, prompt text, pair content, order).
- All degradations deterministic (no LLM, no randomness) so pairs are reproducible; any randomness anywhere must be seeded.
- Unit tests never hit the network; live calls only in explicitly `@pytest.mark.live` tests gated on `OPENAI_API_KEY`.
- All repo artifacts written as product engineering documentation (no meta-narrative).
- Commits: never amend.

---

### Task 1: Scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `src/tpe/__init__.py`, `tests/__init__.py`, `tests/test_scaffold.py`

**Interfaces:**
- Produces: importable package `tpe`; `uv run pytest` works.

- [ ] **Step 1: Write files**

`pyproject.toml`:
```toml
[project]
name = "talent-promo-eval"
version = "0.1.0"
description = "Standalone eval harness: GEPA-optimized pairwise LLM judge for resume optimization quality"
requires-python = ">=3.12"
dependencies = [
    "openai>=1.60",
    "gepa>=0.4",
    "pydantic>=2.7",
    "typer>=0.12",
    "httpx>=0.27",
    "python-dotenv>=1.0",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["live: hits the real OpenAI API (requires OPENAI_API_KEY)"]
addopts = "-m 'not live'"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tpe"]
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
runs/cache/
.pytest_cache/
```

`.env.example`:
```
OPENAI_API_KEY=
```

`tests/test_scaffold.py`:
```python
def test_package_imports():
    import tpe
    assert tpe is not None
```

- [ ] **Step 2: Verify**

Run: `cd . && uv sync && uv run pytest -q`
Expected: 1 passed. (If `gepa` wheel needs an adjusted version pin, take the current PyPI version from the tech-facts report.)

- [ ] **Step 3: Commit** — `chore: scaffold uv project`

---

### Task 2: Schema

**Files:**
- Create: `src/tpe/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces (used by every later task):
  - `LensVerdict(winner: Literal["A","B","tie"], evidence: str)`
  - `JudgeVerdict(ats_signal: LensVerdict, human_skim: LensVerdict, overall: OverallVerdict)` with `OverallVerdict(winner, margin: Literal["slight","clear","decisive"], rationale: str)`; all models `extra="forbid"`; `JudgeVerdict.strict_json_schema()` returns an OpenAI-strict JSON schema.
  - `DegradationTag(name: str, lens: Literal["ats","human_skim","trap"], severity: Literal["subtle","moderate","severe"])`
  - `KnownPair(pair_id: str, job_text: str, original_resume: str, better: str, worse: str, tag: DegradationTag | None, source: Literal["synthetic","human_anchor"], split: Literal["train","val","test","anchor"])`
  - `PairJudgement(pair_id: str, order: Literal["BW","WB"], model: str, verdict: JudgeVerdict, cached: bool)`

- [ ] **Step 1: Failing test**

```python
# tests/test_schema.py
import json
import pytest
from pydantic import ValidationError
from tpe.schema import JudgeVerdict, KnownPair, DegradationTag


VERDICT = {
    "ats_signal": {"winner": "A", "evidence": "keywords intact"},
    "human_skim": {"winner": "tie", "evidence": "both scannable"},
    "overall": {"winner": "A", "margin": "clear", "rationale": "keyword coverage"},
}


def test_verdict_roundtrip():
    v = JudgeVerdict.model_validate(VERDICT)
    assert v.overall.winner == "A"
    assert v.human_skim.winner == "tie"


def test_verdict_rejects_unknown_winner():
    bad = json.loads(json.dumps(VERDICT))
    bad["overall"]["winner"] = "C"
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate(bad)


def test_strict_schema_forbids_extras():
    schema = JudgeVerdict.strict_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"ats_signal", "human_skim", "overall"}


def test_known_pair_construction():
    p = KnownPair(
        pair_id="t1:keyword_strip:subtle", job_text="j", original_resume="o",
        better="<p>good</p>", worse="<p>bad</p>",
        tag=DegradationTag(name="keyword_strip", lens="ats", severity="subtle"),
        source="synthetic", split="train",
    )
    assert p.tag.lens == "ats"
```

- [ ] **Step 2: Run to fail** — `uv run pytest tests/test_schema.py -q` → ImportError.

- [ ] **Step 3: Implement**

```python
# src/tpe/schema.py
"""Core data models. Everything the pipeline passes around is defined here."""
from typing import Literal

from pydantic import BaseModel, ConfigDict

Winner = Literal["A", "B", "tie"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LensVerdict(_Strict):
    winner: Winner
    evidence: str


class OverallVerdict(_Strict):
    winner: Winner
    margin: Literal["slight", "clear", "decisive"]
    rationale: str


class JudgeVerdict(_Strict):
    ats_signal: LensVerdict
    human_skim: LensVerdict
    overall: OverallVerdict

    @classmethod
    def strict_json_schema(cls) -> dict:
        """OpenAI strict-mode schema: all fields required, no extras, no $defs indirection issues."""
        schema = cls.model_json_schema()
        # pydantic emits $defs + required lists already; strict mode needs additionalProperties False everywhere
        def harden(node: dict) -> None:
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}).keys())
            for child in node.get("properties", {}).values():
                harden(child)
            for child in node.get("$defs", {}).values():
                harden(child)
        harden(schema)
        return schema


class DegradationTag(_Strict):
    name: str
    lens: Literal["ats", "human_skim", "trap"]
    severity: Literal["subtle", "moderate", "severe"]


class KnownPair(_Strict):
    pair_id: str
    job_text: str
    original_resume: str
    better: str
    worse: str
    tag: DegradationTag | None = None
    source: Literal["synthetic", "human_anchor"]
    split: Literal["train", "val", "test", "anchor"]


class PairJudgement(_Strict):
    pair_id: str
    order: Literal["BW", "WB"]  # BW: A=better, B=worse. WB: swapped.
    model: str
    verdict: JudgeVerdict
    cached: bool = False
```

- [ ] **Step 4: Run to pass** — `uv run pytest tests/test_schema.py -q` → 4 passed.
- [ ] **Step 5: Commit** — `feat: core schema (verdicts, known-order pairs)`

---

### Task 3: Model client wrapper + disk cache

**Files:**
- Create: `src/tpe/models.py`, `src/tpe/cache.py`
- Test: `tests/test_models_cache.py`

**Interfaces:**
- Produces:
  - `LADDER: dict[str, str]` — `{"nano": ..., "mini": ..., "mid": ..., "top": ...}` (IDs pinned from tech-facts report; env var `TPE_MODEL_<TIER>` overrides).
  - `complete_json(model: str, system: str, user: str, schema: dict) -> dict` — one OpenAI structured-output call, retries once on transient error, drops unsupported params automatically.
  - `DiskCache(root: Path)` with `.get(key: str) -> dict | None`, `.put(key: str, value: dict)`, and `cache_key(*parts: str) -> str`.

- [ ] **Step 1: Failing test**

```python
# tests/test_models_cache.py
import json
from pathlib import Path

from tpe.cache import DiskCache, cache_key


def test_cache_key_stable_and_order_sensitive():
    assert cache_key("a", "b") == cache_key("a", "b")
    assert cache_key("a", "b") != cache_key("b", "a")


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
```

- [ ] **Step 2: Run to fail** — ImportError.

- [ ] **Step 3: Implement**

```python
# src/tpe/cache.py
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
```

```python
# src/tpe/models.py
"""OpenAI client wrapper + the model ladder. Model IDs live HERE and nowhere else."""
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

load_dotenv()

# Pinned from docs/research-notes.md (tech-facts report). Env override: TPE_MODEL_<TIER>.
_DEFAULT_LADDER = {
    "nano": "gpt-5-nano",
    "mini": "gpt-5-mini",
    "mid": "gpt-5",       # confirm against research report at execution
    "top": "gpt-5.6",     # confirm against research report at execution
}


def ladder() -> dict[str, str]:
    return {t: os.getenv(f"TPE_MODEL_{t.upper()}", m) for t, m in _DEFAULT_LADDER.items()}


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def complete_json(model: str, system: str, user: str, schema: dict) -> dict:
    """One structured-output call; returns the parsed JSON dict."""
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "verdict", "strict": True, "schema": schema}},
    )
    for attempt in (1, 2):
        try:
            resp = client().chat.completions.create(**kwargs)
            return json.loads(resp.choices[0].message.content)
        except BadRequestError:
            raise  # schema/param problem: not transient, surface it
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0)
```

- [ ] **Step 4: Run to pass** — `uv run pytest tests/test_models_cache.py -q` → 3 passed.
- [ ] **Step 5: Commit** — `feat: model ladder, structured-output wrapper, disk cache`

---

### Task 4: Job fetcher

**Files:**
- Create: `src/tpe/job_fetch.py`
- Test: `tests/test_job_fetch.py`

**Interfaces:**
- Produces: `job_text(source: str, cache_dir: Path = Path("data/jobs")) -> str` — if `source` starts with http(s), fetch → extract text → cache to `data/jobs/<sha16>.txt`; if it's an existing file path, read it; otherwise treat as raw text and return unchanged. `html_to_text(html: str) -> str` exported for reuse.

- [ ] **Step 1: Failing test**

```python
# tests/test_job_fetch.py
from pathlib import Path

from tpe.job_fetch import html_to_text, job_text

HTML = """<html><head><style>p{color:red}</style><script>var x=1;</script></head>
<body><h1>Staff Engineer</h1><p>Build   agent infrastructure.</p></body></html>"""


def test_html_to_text_strips_script_style_tags():
    text = html_to_text(HTML)
    assert "Staff Engineer" in text
    assert "Build agent infrastructure." in text
    assert "var x" not in text
    assert "color:red" not in text
    assert "<p>" not in text


def test_job_text_passthrough_for_raw_text():
    assert job_text("Just a plain JD body") == "Just a plain JD body"


def test_job_text_reads_file(tmp_path: Path):
    f = tmp_path / "jd.txt"
    f.write_text("JD from file")
    assert job_text(str(f)) == "JD from file"
```

- [ ] **Step 2: Run to fail** — ImportError.

- [ ] **Step 3: Implement**

```python
# src/tpe/job_fetch.py
"""Job posting input: URL, file path, or raw text -> plain text (URLs cached on disk)."""
import hashlib
import re
from pathlib import Path

import httpx


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", html)
    html = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)\b[^>]*>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def job_text(source: str, cache_dir: Path = Path("data/jobs")) -> str:
    if source.startswith(("http://", "https://")):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / (hashlib.sha256(source.encode()).hexdigest()[:16] + ".txt")
        if cached.exists():
            return cached.read_text()
        resp = httpx.get(source, follow_redirects=True, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0 (resume-eval)"})
        resp.raise_for_status()
        text = html_to_text(resp.text)
        cached.write_text(text)
        return text
    if len(source) < 512 and Path(source).exists():
        return Path(source).read_text()
    return source
```

- [ ] **Step 4: Run to pass** — 3 passed.
- [ ] **Step 5: Commit** — `feat: job posting fetcher (url/file/raw, cached)`

---

### Task 5: Corpus snapshot

**Files:**
- Create: `scripts/snapshot_corpus.py`, `data/corpus/` (snapshotted content), `tests/test_corpus.py`
- Create: `src/tpe/corpus.py`

**Interfaces:**
- Produces: `load_corpus(path: Path = Path("data/corpus/corpus.jsonl")) -> list[CorpusRecord]` where `CorpusRecord(trace_id: str, profile_text: str, job_text: str, generated_html: str)`. Also `load_human_codes(path) -> dict[str, dict]` mapping trace_id → `{"F": int, "R": int}` from `judge_alignment_from_coding.json`.

- [ ] **Step 1: Locate source files in the app repo (read-only)**

Run: `ls $TALENT_PROMO_DIR/apps/api/evals/coding/ | head -30` and `ls $TALENT_PROMO_DIR/apps/api/evals/coding/*.jsonl`
Expected: `corpus.jsonl` (28 records), `judge_alignment_from_coding.json`, `job_postings_raw/`, `source_profile.md`, `source_profile_b.md`. Inspect the first record with `head -c 2000 corpus.jsonl` to confirm actual field names, then adapt the loader's field mapping if they differ from `profile_text`/`job_text`/`generated_html` (they may be nested under `inputs`/`outputs`).

- [ ] **Step 2: Write snapshot script and run it**

```python
# scripts/snapshot_corpus.py
"""One-time snapshot of the talent-promo eval corpus into this repo. Read-only on the source."""
import json
import shutil
from pathlib import Path

SRC = Path(os.environ.get("TALENT_PROMO_DIR", str(Path.home() / "talent-promo"))) / "apps/api/evals/coding"
DST = Path(__file__).resolve().parent.parent / "data/corpus"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name in ["corpus.jsonl", "judge_alignment_from_coding.json",
                 "source_profile.md", "source_profile_b.md", "taxonomy.md"]:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, DST / name)
            print(f"copied {name}")
        else:
            print(f"MISSING {name}")
    jobs_dst = DST / "job_postings_raw"
    jobs_dst.mkdir(exist_ok=True)
    for jd in sorted((SRC / "job_postings_raw").glob("*.txt")):
        shutil.copy2(jd, jobs_dst / jd.name)
    print(f"copied {len(list(jobs_dst.glob('*.txt')))} job postings")


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/snapshot_corpus.py`
Expected: all files copied, 14 job postings. If `corpus.jsonl` lives elsewhere, find it with `find $TALENT_PROMO_DIR/apps/api/evals -name "corpus*.jsonl"` and update SRC handling.

- [ ] **Step 3: Failing test for the loader**

```python
# tests/test_corpus.py
from pathlib import Path

import pytest

from tpe.corpus import load_corpus, load_human_codes

CORPUS = Path("data/corpus/corpus.jsonl")


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus snapshot not present")
def test_load_corpus_full():
    records = load_corpus(CORPUS)
    assert len(records) == 28
    r = records[0]
    assert r.trace_id and r.job_text and len(r.generated_html) > 1000


def test_load_corpus_tolerates_nested_fields(tmp_path: Path):
    line = ('{"trace_id": "t1", "inputs": {"source_profile": "P", "job_posting": "J"}, '
            '"outputs": {"generated_html": "<p>H</p>"}}')
    f = tmp_path / "c.jsonl"
    f.write_text(line + "\n")
    records = load_corpus(f)
    assert records[0].profile_text == "P"
    assert records[0].generated_html == "<p>H</p>"
```

- [ ] **Step 4: Implement loader**

```python
# src/tpe/corpus.py
"""Read the snapshotted corpus. Tolerant to flat or nested record shapes."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusRecord:
    trace_id: str
    profile_text: str
    job_text: str
    generated_html: str


def _pick(record: dict, *paths: tuple[str, ...]) -> str:
    for path in paths:
        node = record
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, str) and node:
            return node
    return ""


def load_corpus(path: Path = Path("data/corpus/corpus.jsonl")) -> list[CorpusRecord]:
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        records.append(CorpusRecord(
            trace_id=_pick(raw, ("trace_id",), ("id",)),
            profile_text=_pick(raw, ("profile_text",), ("inputs", "source_profile"), ("source_profile",)),
            job_text=_pick(raw, ("job_text",), ("inputs", "job_posting"), ("job_posting",)),
            generated_html=_pick(raw, ("generated_html",), ("outputs", "generated_html")),
        ))
    return records


def load_human_codes(path: Path = Path("data/corpus/judge_alignment_from_coding.json")) -> dict[str, dict]:
    data = json.loads(Path(path).read_text())
    # expected shape: {trace_id: {..."faithfulness": int, "job_relevance": int...}} — adapt at execution
    out = {}
    for trace_id, entry in (data.items() if isinstance(data, dict) else []):
        if isinstance(entry, dict):
            f = entry.get("F", entry.get("faithfulness"))
            r = entry.get("R", entry.get("job_relevance"))
            if isinstance(f, (int, float)):
                out[trace_id] = {"F": int(f), "R": int(r) if isinstance(r, (int, float)) else None}
    return out
```

- [ ] **Step 5: Run to pass** — `uv run pytest tests/test_corpus.py -q` → 2 passed (adapt `_pick` paths / `load_human_codes` shape to the real files inspected in Step 1).
- [ ] **Step 6: Commit** — `feat: corpus snapshot + loader` (data files included: they are redacted already).

---

### Task 6: Degradation engine

**Files:**
- Create: `src/tpe/degrade.py`
- Test: `tests/test_degrade.py` (+ fixture `tests/fixtures/resume.html`)

**Interfaces:**
- Produces:
  - `extract_keywords(job_text: str, top_n: int = 25) -> list[str]`
  - `DEGRADATIONS: list[Degradation]` where `Degradation(name, lens, severities, fn)`; `fn(html: str, ctx: DegradeContext, severity: str) -> str`; `DegradeContext(jd_keywords: list[str])`.
  - Guarantee: fn is pure/deterministic; returns input unchanged when not applicable (builder skips those).
  - Names (used in split logic + reports): `keyword_strip`, `header_flatten`, `dequantify`, `bury_relevant`, `bland_leads`, `wall_of_text`, `drop_summary`, `keyword_stuff`.

- [ ] **Step 1: Write fixture** — `tests/fixtures/resume.html`, shaped like the drafter's real output (h2 sections, h3 roles, ul/li bullets):

```html
<h2>Summary</h2>
<p>Engineer with 6 years building distributed systems and LLM agent infrastructure in Python and TypeScript.</p>
<h2>Experience</h2>
<h3>Senior Software Engineer, Data Platform</h3>
<ul>
<li>Reduced pipeline latency by 43% by rewriting the ingestion layer in Python with async batching.</li>
<li>Led a team of 4 engineers migrating 12 services to Kubernetes.</li>
<li>Responsible for maintaining internal tooling.</li>
</ul>
<h3>Software Engineer, Payments</h3>
<ul>
<li>Built fraud-detection features processing 2M transactions daily using Kafka.</li>
<li>Collaborated with stakeholders on roadmap planning.</li>
</ul>
<h2>Skills</h2>
<p>Python, TypeScript, Kubernetes, Kafka, PostgreSQL</p>
```

- [ ] **Step 2: Failing tests**

```python
# tests/test_degrade.py
import re
from pathlib import Path

import pytest

from tpe.degrade import DEGRADATIONS, DegradeContext, extract_keywords

RESUME = Path("tests/fixtures/resume.html").read_text()
JD = """Staff Engineer, Agent Infrastructure. You will build LLM agent infrastructure
in Python on Kubernetes. Experience with Kafka, distributed systems, and observability
required. Python and Kubernetes expertise essential. LLM experience preferred."""
CTX = DegradeContext(jd_keywords=extract_keywords(JD))
BY_NAME = {d.name: d for d in DEGRADATIONS}


def test_extract_keywords_finds_tech_terms():
    kws = extract_keywords(JD)
    assert "python" in kws and "kubernetes" in kws
    assert "the" not in kws and "you" not in kws


def test_all_degradations_deterministic_and_modifying():
    for d in DEGRADATIONS:
        for sev in d.severities:
            out1 = d.fn(RESUME, CTX, sev)
            out2 = d.fn(RESUME, CTX, sev)
            assert out1 == out2, f"{d.name}:{sev} not deterministic"
            assert out1 != RESUME, f"{d.name}:{sev} was a no-op on the fixture"


def test_keyword_strip_removes_jd_terms():
    out = BY_NAME["keyword_strip"].fn(RESUME, CTX, "severe")
    assert "Python" not in out and "Kubernetes" not in out


def test_dequantify_severe_removes_digits_from_bullets():
    out = BY_NAME["dequantify"].fn(RESUME, CTX, "severe")
    bullets = re.findall(r"<li>(.*?)</li>", out, re.S)
    assert bullets and not any(re.search(r"\d", b) for b in bullets)


def test_header_flatten_severe_removes_h2():
    out = BY_NAME["header_flatten"].fn(RESUME, CTX, "severe")
    assert "<h2>" not in out and "Experience" in out


def test_bury_relevant_moves_keyword_dense_role_last():
    out = BY_NAME["bury_relevant"].fn(RESUME, CTX, "moderate")
    assert out.find("Data Platform") > out.find("Payments")


def test_wall_of_text_merges_bullets():
    out = BY_NAME["wall_of_text"].fn(RESUME, CTX, "severe")
    assert out.count("<li>") < RESUME.count("<li>")


def test_keyword_stuff_adds_unsupported_terms():
    out = BY_NAME["keyword_stuff"].fn(RESUME, CTX, "severe")
    assert "observability" in out.lower()  # in JD, not in original resume
    assert len(out) > len(RESUME)


def test_bland_leads_sinks_quantified_bullets():
    out = BY_NAME["bland_leads"].fn(RESUME, CTX, "moderate")
    first_bullet = re.search(r"<li>(.*?)</li>", out, re.S).group(1)
    assert not re.search(r"\d", first_bullet)


def test_drop_summary_removes_summary_section():
    out = BY_NAME["drop_summary"].fn(RESUME, CTX, "moderate")
    assert "<h2>Summary</h2>" not in out and "Experience" in out
```

- [ ] **Step 3: Run to fail** — ImportError.

- [ ] **Step 4: Implement**

```python
# src/tpe/degrade.py
"""Deterministic resume degradations. Each produces a strictly-worse version along one lens,
tagged so the judge's misses can be attributed to a failure kind."""
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

STOPWORDS = frozenset(
    "the a an and or for with to of in on at by from as is are was were be been being "
    "this that these those you your we our it its they their will can may should must "
    "have has had do does did not no yes if then than more most other others such only "
    "who whom whose which what when where how all any both each few own same so too very "
    "required preferred experience years work team role including strong ability".split()
)


def extract_keywords(job_text: str, top_n: int = 25) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", job_text)
    counts = Counter(w.lower().strip("./-") for w in words)
    return [w for w, _ in counts.most_common(top_n * 2)
            if w not in STOPWORDS][:top_n]


@dataclass(frozen=True)
class DegradeContext:
    jd_keywords: list[str]


# --- html section helpers (tolerant regex splitting; resumes are simple h2/h3/ul html) ---

def split_h2(html: str) -> list[str]:
    """Split into blocks, each starting at an <h2> (first block may be preamble)."""
    parts = re.split(r"(?i)(?=<h2\b)", html)
    return [p for p in parts if p.strip()]


def h2_title(block: str) -> str:
    m = re.search(r"(?is)<h2\b[^>]*>(.*?)</h2>", block)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip().lower() if m else ""


def split_h3(block: str) -> tuple[str, list[str]]:
    """Within a section: (header part, list of h3 role blocks)."""
    parts = re.split(r"(?i)(?=<h3\b)", block)
    if len(parts) == 1:
        return block, []
    return parts[0], [p for p in parts[1:] if p.strip()]


def _keyword_density(text: str, keywords: list[str]) -> int:
    low = text.lower()
    return sum(low.count(kw) for kw in keywords)


# --- degradations ---

def keyword_strip(html: str, ctx: DegradeContext, severity: str) -> str:
    n = {"subtle": 2, "moderate": 5, "severe": 12}[severity]
    present = [kw for kw in ctx.jd_keywords if re.search(rf"(?i)\b{re.escape(kw)}\b", html)]
    out = html
    for kw in present[:n]:
        out = re.sub(rf"(?i)\s*\b{re.escape(kw)}\b", " relevant technologies", out)
        out = re.sub(r"(relevant technologies)(,?\s+relevant technologies)+", r"\1", out)
    return out


def header_flatten(html: str, ctx: DegradeContext, severity: str) -> str:
    out = re.sub(r"(?is)<h3\b[^>]*>(.*?)</h3>", r"<p><b>\1</b></p>", html)
    if severity == "severe":
        out = re.sub(r"(?is)<h2\b[^>]*>(.*?)</h2>", r"<p><b>\1</b></p>", out)
    return out


def dequantify(html: str, ctx: DegradeContext, severity: str) -> str:
    frac = {"subtle": 0.34, "moderate": 0.67, "severe": 1.0}[severity]
    pattern = re.compile(r"(?:\bby\s+|\bto\s+|\bof\s+)?\d[\d,.]*\s*(?:%|x|k|K|M|MM|\+)?\s*")
    matches = list(pattern.finditer(html))
    keep_after = int(len(matches) * frac)
    out, last = [], 0
    for i, m in enumerate(matches):
        if i < keep_after:
            out.append(html[last:m.start()])
            out.append("significantly " if html[m.start():m.end()].strip()[0].isdigit() is False else "")
            last = m.end()
    out.append(html[last:])
    text = "".join(out)
    return re.sub(r"\s{2,}", " ", text)


def bury_relevant(html: str, ctx: DegradeContext, severity: str) -> str:
    sections = split_h2(html)
    for idx, sec in enumerate(sections):
        if "experience" not in h2_title(sec):
            continue
        head, roles = split_h3(sec)
        if len(roles) < 2:
            return html
        dens = [_keyword_density(r, ctx.jd_keywords) for r in roles]
        hot = dens.index(max(dens))
        reordered = [r for i, r in enumerate(roles) if i != hot] + [roles[hot]]
        if severity == "severe":
            reordered = list(reversed(roles))  # full inversion
        sections[idx] = head + "".join(reordered)
        return "".join(sections)
    return html


def _sort_bullets(ul_inner: str) -> str:
    items = re.findall(r"(?is)<li>.*?</li>", ul_inner)
    if len(items) < 2:
        return ul_inner
    ranked = sorted(items, key=lambda li: (1 if re.search(r"\d", li) else 0, len(li)))
    return "".join(ranked)


def bland_leads(html: str, ctx: DegradeContext, severity: str) -> str:
    uls = list(re.finditer(r"(?is)<ul>(.*?)</ul>", html))
    if not uls:
        return html
    limit = 1 if severity == "subtle" else len(uls)
    out, last = [], 0
    for i, m in enumerate(uls):
        out.append(html[last:m.start(1)])
        out.append(_sort_bullets(m.group(1)) if i < limit else m.group(1))
        last = m.end(1)
    out.append(html[last:])
    return "".join(out)


def wall_of_text(html: str, ctx: DegradeContext, severity: str) -> str:
    uls = list(re.finditer(r"(?is)<ul>(.*?)</ul>", html))
    if not uls:
        return html
    limit = 1 if severity == "moderate" else len(uls)
    out, last = [], 0
    for i, m in enumerate(uls):
        out.append(html[last:m.start(1)])
        if i < limit:
            items = re.findall(r"(?is)<li>(.*?)</li>", m.group(1))
            out.append("<li>" + " ".join(x.strip().rstrip(".") + "." for x in items) + "</li>")
        else:
            out.append(m.group(1))
        last = m.end(1)
    out.append(html[last:])
    return "".join(out)


def drop_summary(html: str, ctx: DegradeContext, severity: str) -> str:
    sections = split_h2(html)
    kept = [s for s in sections
            if not any(k in h2_title(s) for k in ("summary", "profile", "objective"))]
    return "".join(kept) if len(kept) < len(sections) else html


def keyword_stuff(html: str, ctx: DegradeContext, severity: str) -> str:
    n = {"subtle": 5, "moderate": 10, "severe": 15}[severity]
    missing = [kw for kw in ctx.jd_keywords
               if not re.search(rf"(?i)\b{re.escape(kw)}\b", html)][:n]
    if not missing:
        return html
    blob = "<h2>Core Competencies</h2><p>" + ", ".join(
        kw.title() for kw in (ctx.jd_keywords[:n] + missing)) + "</p>"
    out = html + blob
    if severity != "subtle":
        out = re.sub(r"(?is)</li>", lambda m, it=iter(missing * 3):
                     f", leveraging {next(it, 'modern tooling')} expertise.</li>", out, count=3)
    return out


@dataclass(frozen=True)
class Degradation:
    name: str
    lens: str  # "ats" | "human_skim" | "trap"
    severities: tuple[str, ...]
    fn: Callable[[str, DegradeContext, str], str]


DEGRADATIONS: list[Degradation] = [
    Degradation("keyword_strip", "ats", ("subtle", "moderate", "severe"), keyword_strip),
    Degradation("header_flatten", "ats", ("moderate", "severe"), header_flatten),
    Degradation("dequantify", "ats", ("subtle", "moderate", "severe"), dequantify),
    Degradation("bury_relevant", "human_skim", ("moderate", "severe"), bury_relevant),
    Degradation("bland_leads", "human_skim", ("subtle", "moderate"), bland_leads),
    Degradation("wall_of_text", "human_skim", ("moderate", "severe"), wall_of_text),
    Degradation("drop_summary", "human_skim", ("moderate",), drop_summary),
    Degradation("keyword_stuff", "trap", ("subtle", "moderate", "severe"), keyword_stuff),
]
```

- [ ] **Step 5: Run to pass** — `uv run pytest tests/test_degrade.py -q` → all passed. Iterate on regexes against the fixture until green; then also spot-check against one REAL corpus record: `uv run python -c "from tpe.corpus import load_corpus; from tpe.degrade import *; r=load_corpus()[0]; ctx=DegradeContext(extract_keywords(r.job_text)); [print(d.name, len(d.fn(r.generated_html, ctx, d.severities[-1])) != len(r.generated_html)) for d in DEGRADATIONS]"`
- [ ] **Step 6: Commit** — `feat: deterministic degradation engine (8 tagged degradations)`

---

### Task 7: Pair builder + splits

**Files:**
- Create: `src/tpe/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `load_corpus`, `load_human_codes`, `DEGRADATIONS`, `KnownPair`.
- Produces:
  - `build_pairs(records, human_codes, heldout_types={"header_flatten", "drop_summary"}) -> list[KnownPair]`
  - `write_splits(pairs, out_dir: Path)` → `data/pairs/{train,val,test,anchor}.jsonl`
  - `load_pairs(path) -> list[KnownPair]`
  - Split rule (deterministic): held-out degradation types → `test`; human anchors → `anchor`; else sha256(pair_id) % 100: `<60 train, <80 val, else test`.

- [ ] **Step 1: Failing test**

```python
# tests/test_dataset.py
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


def test_write_and_load_roundtrip(tmp_path: Path):
    pairs = build_pairs([REC], human_codes={})
    write_splits(pairs, tmp_path)
    train = load_pairs(tmp_path / "train.jsonl")
    assert train and all(p.split == "train" for p in train)
```

- [ ] **Step 2: Run to fail** — ImportError.

- [ ] **Step 3: Implement**

```python
# src/tpe/dataset.py
"""Manufacture known-order pairs from the corpus + degradations, and split them."""
import hashlib
from pathlib import Path

from tpe.corpus import CorpusRecord
from tpe.degrade import DEGRADATIONS, DegradeContext, extract_keywords
from tpe.schema import DegradationTag, KnownPair

HELDOUT_TYPES = {"header_flatten", "drop_summary"}  # never trained on: generalization guard
ANCHOR_F_MIN = 85  # human-coded faithfulness floor for original-vs-generated anchors


def _split_for(pair_id: str, tag_name: str | None, source: str,
               heldout_types: set[str]) -> str:
    if source == "human_anchor":
        return "anchor"
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
                    split=_split_for(pair_id, deg.name, "synthetic", heldout_types),
                ))
        codes = human_codes.get(rec.trace_id)
        if codes and codes.get("F", 0) >= ANCHOR_F_MIN:
            pair_id = f"{rec.trace_id}:human_anchor"
            pairs.append(KnownPair(
                pair_id=pair_id, job_text=rec.job_text,
                original_resume=rec.profile_text,
                better=rec.generated_html, worse=rec.profile_text,
                tag=None, source="human_anchor",
                split="anchor",
            ))
    return pairs


def write_splits(pairs: list[KnownPair], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test", "anchor"):
        subset = [p for p in pairs if p.split == split]
        path = out_dir / f"{split}.jsonl"
        path.write_text("".join(p.model_dump_json() + "\n" for p in subset))
        counts[split] = len(subset)
    return counts


def load_pairs(path: Path) -> list[KnownPair]:
    from tpe.schema import KnownPair as KP
    return [KP.model_validate_json(line)
            for line in Path(path).read_text().splitlines() if line.strip()]
```

- [ ] **Step 4: Run to pass** — 4 passed.
- [ ] **Step 5: Build the real dataset**

Run: `uv run python -c "
from pathlib import Path
from tpe.corpus import load_corpus, load_human_codes
from tpe.dataset import build_pairs, write_splits
pairs = build_pairs(load_corpus(), load_human_codes())
print(write_splits(pairs, Path('data/pairs')))"`
Expected: roughly `{'train': 120±, 'val': 40±, 'test': 80±, 'anchor': ≤10}` (28 records × ~9-14 applicable degradation-severity combos). Inspect 2-3 pairs by eye for sanity.

- [ ] **Step 6: Commit** — `feat: known-order pair builder with deterministic splits` (include `data/pairs/`).

---

### Task 8: Pairwise judge runner

**Files:**
- Create: `src/tpe/judge.py`, `prompts/` (placeholder seed used by tests only)
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `complete_json`, `DiskCache`, `cache_key`, schema models.
- Produces:
  - `render_prompt(template: str, job: str, original: str, a: str, b: str) -> str` — replaces `{{JOB_POSTING}}`, `{{ORIGINAL_RESUME}}`, `{{RESUME_A}}`, `{{RESUME_B}}`.
  - `judge_pair(template, model, pair, order, cache) -> PairJudgement`
  - `judge_both_orders(template, model, pair, cache) -> BothOrders` where `BothOrders(bw: PairJudgement, wb: PairJudgement)` with properties: `.unswapped(order) -> "better"|"worse"|"tie"` per overall winner, `.flipped -> bool` (both non-tie and disagree), `.pair_score -> float` (mean of per-order scores: correct=1, tie=0.5, wrong=0).
  - `SYSTEM_PROMPT` constant: fixed, non-evolvable ("You are a meticulous resume-screening judge... respond only with JSON").
  - `run_pairs(template, model, pairs, cache, max_workers=4) -> list[BothOrders]` (ThreadPoolExecutor).

- [ ] **Step 1: Failing test**

```python
# tests/test_judge.py
from pathlib import Path
from unittest.mock import patch

from tpe.cache import DiskCache
from tpe.judge import judge_both_orders, judge_pair, render_prompt
from tpe.schema import DegradationTag, KnownPair

PAIR = KnownPair(
    pair_id="t1:keyword_strip:subtle", job_text="JD", original_resume="ORIG",
    better="GOOD RESUME", worse="BAD RESUME",
    tag=DegradationTag(name="keyword_strip", lens="ats", severity="subtle"),
    source="synthetic", split="train",
)
TEMPLATE = "Job: {{JOB_POSTING}}\nOriginal: {{ORIGINAL_RESUME}}\nA: {{RESUME_A}}\nB: {{RESUME_B}}"


def _verdict(winner: str) -> dict:
    lens = {"winner": winner, "evidence": "e"}
    return {"ats_signal": lens, "human_skim": lens,
            "overall": {"winner": winner, "margin": "clear", "rationale": "r"}}


def test_render_prompt_places_sides():
    text = render_prompt(TEMPLATE, "JD", "ORIG", "GOOD RESUME", "BAD RESUME")
    assert "A: GOOD RESUME" in text and "B: BAD RESUME" in text


def test_judge_pair_bw_puts_better_as_a(tmp_path: Path):
    with patch("tpe.judge.complete_json", return_value=_verdict("A")) as mock:
        j = judge_pair(TEMPLATE, "m", PAIR, "BW", DiskCache(tmp_path))
    prompt_sent = mock.call_args.kwargs.get("user") or mock.call_args.args[2]
    assert "A: GOOD RESUME" in prompt_sent
    assert j.verdict.overall.winner == "A" and j.cached is False


def test_judge_pair_uses_cache_on_second_call(tmp_path: Path):
    cache = DiskCache(tmp_path)
    with patch("tpe.judge.complete_json", return_value=_verdict("A")) as mock:
        judge_pair(TEMPLATE, "m", PAIR, "BW", cache)
        j2 = judge_pair(TEMPLATE, "m", PAIR, "BW", cache)
    assert mock.call_count == 1 and j2.cached is True


def test_both_orders_consistent_correct(tmp_path: Path):
    # BW: better is A -> "A" correct. WB: better is B -> "B" correct.
    responses = iter([_verdict("A"), _verdict("B")])
    with patch("tpe.judge.complete_json", side_effect=lambda *a, **k: next(responses)):
        both = judge_both_orders(TEMPLATE, "m", PAIR, DiskCache(tmp_path))
    assert both.pair_score == 1.0 and both.flipped is False


def test_both_orders_flip_detected(tmp_path: Path):
    responses = iter([_verdict("A"), _verdict("A")])  # A both times = position bias
    with patch("tpe.judge.complete_json", side_effect=lambda *a, **k: next(responses)):
        both = judge_both_orders(TEMPLATE, "m", PAIR, DiskCache(tmp_path))
    assert both.flipped is True and both.pair_score == 0.5
```

- [ ] **Step 2: Run to fail** — ImportError.

- [ ] **Step 3: Implement**

```python
# src/tpe/judge.py
"""Pairwise judge runner: render -> call -> parse, always both A/B orders, disk-cached."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tpe.cache import DiskCache, cache_key
from tpe.models import complete_json
from tpe.schema import JudgeVerdict, KnownPair, PairJudgement

SYSTEM_PROMPT = (
    "You are a meticulous resume-screening judge. You compare two versions of a resume "
    "for the same job posting and decide which serves the candidate better. "
    "Base every verdict only on evidence in the materials provided. "
    "Respond only with JSON matching the required schema."
)


def render_prompt(template: str, job: str, original: str, a: str, b: str) -> str:
    return (template
            .replace("{{JOB_POSTING}}", job)
            .replace("{{ORIGINAL_RESUME}}", original)
            .replace("{{RESUME_A}}", a)
            .replace("{{RESUME_B}}", b))


def judge_pair(template: str, model: str, pair: KnownPair, order: str,
               cache: DiskCache) -> PairJudgement:
    a, b = (pair.better, pair.worse) if order == "BW" else (pair.worse, pair.better)
    user = render_prompt(template, pair.job_text, pair.original_resume, a, b)
    key = cache_key(model, SYSTEM_PROMPT, user)
    hit = cache.get(key)
    if hit is not None:
        return PairJudgement(pair_id=pair.pair_id, order=order, model=model,
                             verdict=JudgeVerdict.model_validate(hit), cached=True)
    raw = complete_json(model, SYSTEM_PROMPT, user, JudgeVerdict.strict_json_schema())
    verdict = JudgeVerdict.model_validate(raw)
    cache.put(key, verdict.model_dump())
    return PairJudgement(pair_id=pair.pair_id, order=order, model=model,
                         verdict=verdict, cached=False)


def _order_score(winner: str, order: str) -> float:
    if winner == "tie":
        return 0.5
    correct = "A" if order == "BW" else "B"
    return 1.0 if winner == correct else 0.0


def _unswap(winner: str, order: str) -> str:
    if winner == "tie":
        return "tie"
    if order == "BW":
        return "better" if winner == "A" else "worse"
    return "better" if winner == "B" else "worse"


@dataclass(frozen=True)
class BothOrders:
    pair: KnownPair
    bw: PairJudgement
    wb: PairJudgement

    @property
    def flipped(self) -> bool:
        u1 = _unswap(self.bw.verdict.overall.winner, "BW")
        u2 = _unswap(self.wb.verdict.overall.winner, "WB")
        return u1 != u2 and "tie" not in (u1, u2)

    @property
    def pair_score(self) -> float:
        return (_order_score(self.bw.verdict.overall.winner, "BW")
                + _order_score(self.wb.verdict.overall.winner, "WB")) / 2

    def lens_score(self, lens: str) -> float:
        w1 = getattr(self.bw.verdict, lens).winner
        w2 = getattr(self.wb.verdict, lens).winner
        return (_order_score(w1, "BW") + _order_score(w2, "WB")) / 2


def judge_both_orders(template: str, model: str, pair: KnownPair,
                      cache: DiskCache) -> BothOrders:
    return BothOrders(pair=pair,
                      bw=judge_pair(template, model, pair, "BW", cache),
                      wb=judge_pair(template, model, pair, "WB", cache))


def run_pairs(template: str, model: str, pairs: list[KnownPair], cache: DiskCache,
              max_workers: int = 4) -> list[BothOrders]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(
            lambda p: judge_both_orders(template, model, p, cache), pairs))
```

- [ ] **Step 4: Run to pass** — 5 passed.
- [ ] **Step 5: Commit** — `feat: pairwise judge runner (order-swap, cache, thread pool)`

---

### Task 9: Metrics

**Files:**
- Create: `src/tpe/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `BothOrders`.
- Produces:
  - `summarize(results: list[BothOrders]) -> Summary` — `Summary(accuracy, flip_rate, gepa_metric, by_lens: dict, by_severity: dict, by_type: dict, n)`; `gepa_metric = accuracy - 0.25 * flip_rate`.
  - `mcnemar_exact(b: int, c: int) -> float` — exact two-sided binomial McNemar p-value.
  - `wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]`.

- [ ] **Step 1: Failing test**

```python
# tests/test_metrics.py
import math
from unittest.mock import MagicMock

import pytest

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
```

- [ ] **Step 2: Run to fail** — ImportError.

- [ ] **Step 3: Implement**

```python
# src/tpe/metrics.py
"""Scoring: accuracy/flip-rate summaries, exact McNemar, Wilson CI. Pure functions, no I/O."""
import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Summary:
    n: int
    accuracy: float
    flip_rate: float
    gepa_metric: float
    by_lens: dict = field(default_factory=dict)
    by_severity: dict = field(default_factory=dict)
    by_type: dict = field(default_factory=dict)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results: list) -> Summary:
    scores = [r.pair_score for r in results]
    flips = [1.0 if r.flipped else 0.0 for r in results]
    by_sev, by_type = defaultdict(list), defaultdict(list)
    by_lens = {"ats_signal": [], "human_skim": []}
    for r in results:
        for lens in by_lens:
            by_lens[lens].append(r.lens_score(lens))
        tag = r.pair.tag
        if tag is not None:
            by_sev[tag.severity].append(r.pair_score)
            by_type[tag.name].append(r.pair_score)
    acc, flip = _mean(scores), _mean(flips)
    return Summary(
        n=len(results), accuracy=acc, flip_rate=flip,
        gepa_metric=acc - 0.25 * flip,
        by_lens={k: _mean(v) for k, v in by_lens.items()},
        by_severity={k: _mean(v) for k, v in by_sev.items()},
        by_type={k: _mean(v) for k, v in by_type.items()},
    )


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar: b = hi-correct/lo-wrong, c = lo-correct/hi-wrong."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
```

- [ ] **Step 4: Run to pass** — 3 passed.
- [ ] **Step 5: Commit** — `feat: metrics (summary, exact mcnemar, wilson ci)`

---

### Task 10: Taste guide + seed judge prompt (research-dependent)

**Files:**
- Create: `docs/research-notes.md`, `docs/taste-guide.md`, `prompts/seed_judge.md`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: the deep-research workflow report (resume screening evidence) and tech-facts report (model IDs — also update `src/tpe/models.py` `_DEFAULT_LADDER` now).
- Produces: `prompts/seed_judge.md` containing all four placeholders and the two-lens rubric; this file is GEPA's seed candidate.

- [ ] **Step 1: Wait for / collect the deep-research report.** Write `docs/research-notes.md`: the cited findings organized by audience (ATS mechanics / recruiter skim / hiring-manager read), each claim with its source URL and a confidence note; include a "folklore vs evidence" section.
- [ ] **Step 2: Distill `docs/taste-guide.md`.** Structure: (1) how each audience actually reads; (2) discriminators — ordered list of things that make version X better than version Y for the same candidate+job, each traceable to a research-notes claim; (3) anti-signals (keyword stuffing, unsupported claims, formatting traps); (4) explicit non-goals (visual design, candidate strength itself).
- [ ] **Step 3: Write `prompts/seed_judge.md`.** Contract: contains `{{JOB_POSTING}}`, `{{ORIGINAL_RESUME}}`, `{{RESUME_A}}`, `{{RESUME_B}}`; instructs comparison on the two lenses with the taste-guide discriminators; requires evidence citations in verdicts; explicitly warns that added keywords unsupported by the original resume are a NEGATIVE; under ~2,500 tokens.
- [ ] **Step 4: Test**

```python
# tests/test_prompts.py
from pathlib import Path

SEED = Path("prompts/seed_judge.md")


def test_seed_prompt_has_all_placeholders():
    text = SEED.read_text()
    for ph in ("{{JOB_POSTING}}", "{{ORIGINAL_RESUME}}", "{{RESUME_A}}", "{{RESUME_B}}"):
        assert ph in text


def test_seed_prompt_mentions_both_lenses_and_stuffing():
    text = SEED.read_text().lower()
    assert "ats" in text or "scanner" in text
    assert "skim" in text or "seconds" in text
    assert "stuff" in text  # anti-gaming clause present
```

- [ ] **Step 5: Run to pass; update `_DEFAULT_LADDER` in `src/tpe/models.py` with pinned IDs.**
- [ ] **Step 6: Commit** — `docs: research notes, taste guide, seed judge prompt; pin model ladder`

---

### Task 11: Baseline eval CLI + live smoke

**Files:**
- Create: `src/tpe/cli.py`
- Test: `tests/test_cli.py` (+ live smoke, marked)

**Interfaces:**
- Produces typer app `tpe` (register `[project.scripts] tpe = "tpe.cli:app"` in pyproject) with commands:
  - `tpe eval-prompt --prompt prompts/seed_judge.md --split val --model-tier mini [--limit N]` → prints Summary as a table, writes JSON to `runs/eval_<split>_<tier>_<timestamp>.json`.
  - `tpe judge-one --job <url|file|text> --original <file> --a <file> --b <file>` → single verdict, pretty-printed.

- [ ] **Step 1: Test (mocked run through the CLI plumbing)**

```python
# tests/test_cli.py
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tpe.cli import app

runner = CliRunner()


def _verdict(winner: str) -> dict:
    lens = {"winner": winner, "evidence": "e"}
    return {"ats_signal": lens, "human_skim": lens,
            "overall": {"winner": winner, "margin": "clear", "rationale": "r"}}


def test_eval_prompt_runs_on_val(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path.cwd())  # repo root: data/pairs must exist by now
    with patch("tpe.models.complete_json", return_value=_verdict("A")):
        result = runner.invoke(app, ["eval-prompt", "--prompt", "prompts/seed_judge.md",
                                     "--split", "val", "--limit", "2",
                                     "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "accuracy" in result.output.lower()
```

Note: patch target is `tpe.models.complete_json`? No — `tpe.judge` imports it by name, so patch `tpe.judge.complete_json`. Use that in the test.

- [ ] **Step 2: Implement `cli.py`** — thin typer wiring over `load_pairs` + `run_pairs` + `summarize`; `--model-tier` resolves via `models.ladder()`; timestamped output path; `judge-one` builds a transient `KnownPair` (better=a, worse=b — labels arbitrary here, report winners by filename not by "better").
- [ ] **Step 3: Run mocked test to pass.**
- [ ] **Step 4: Live smoke (requires OPENAI_API_KEY in .env):** `uv run tpe eval-prompt --prompt prompts/seed_judge.md --split val --limit 5 --model-tier mini`
Expected: real accuracy printed (anything ≥0.6 plausible for seed); calls cached under `runs/cache/`. Record the number in CONTINUITY.md.
- [ ] **Step 5: Full val baseline:** drop `--limit`. Record baseline accuracy/flip-rate → this is the bar GEPA must beat.
- [ ] **Step 6: Commit** — `feat: eval CLI + seed baseline on val`

---

### Task 12: GEPA adapter + optimization run

**Files:**
- Create: `src/tpe/gepa_adapter.py`, `scripts/run_gepa.py`
- Test: `tests/test_gepa_adapter.py`

**Interfaces:**
- Consumes: `run_pairs`, `summarize`, tech-facts report for exact gepa API (verify against installed package source: `uv run python -c "import gepa, inspect; print(inspect.signature(gepa.optimize))"`).
- Produces: `JudgeAdapter(GEPAAdapter)` with candidate component `{"judge_prompt": <template text>}`; `scripts/run_gepa.py` runs `gepa.optimize(seed_candidate, trainset, valset, adapter, reflection_lm, max_metric_calls)` and writes best candidate to `prompts/optimized_judge.md` + full result log to `runs/gepa_<timestamp>/`.

- [ ] **Step 1: Verify installed gepa API** (signatures may differ from memory — trust the installed source):

Run: `uv run python -c "import gepa; help(gepa.optimize)" | head -50` and read `GEPAAdapter` protocol (`evaluate`, `make_reflective_dataset`, `EvaluationBatch`).

- [ ] **Step 2: Failing test (offline, fake judge)**

```python
# tests/test_gepa_adapter.py
from pathlib import Path
from unittest.mock import patch

from tpe.cache import DiskCache
from tpe.gepa_adapter import JudgeAdapter
from tpe.schema import DegradationTag, KnownPair


def _pair(i: int) -> KnownPair:
    return KnownPair(pair_id=f"p{i}:dequantify:subtle", job_text="J", original_resume="O",
                     better=f"GOOD{i}", worse=f"BAD{i}",
                     tag=DegradationTag(name="dequantify", lens="ats", severity="subtle"),
                     source="synthetic", split="train")


def _verdict(winner: str) -> dict:
    lens = {"winner": winner, "evidence": "e"}
    return {"ats_signal": lens, "human_skim": lens,
            "overall": {"winner": winner, "margin": "clear", "rationale": "because"}}


def test_adapter_evaluate_scores_and_traces(tmp_path: Path):
    adapter = JudgeAdapter(model="test-model", cache=DiskCache(tmp_path))
    # correct on both orders for every pair
    with patch("tpe.judge.complete_json",
               side_effect=lambda model, system, user, schema:
               _verdict("A" if "A: GOOD" in user else "B")):
        batch = adapter.evaluate([_pair(1), _pair(2)],
                                 {"judge_prompt": "A: {{RESUME_A}} B: {{RESUME_B}} {{JOB_POSTING}} {{ORIGINAL_RESUME}}"},
                                 capture_traces=True)
    assert batch.scores == [1.0, 1.0]
    assert len(batch.trajectories) == 2


def test_adapter_reflective_dataset_names_the_degradation(tmp_path: Path):
    adapter = JudgeAdapter(model="test-model", cache=DiskCache(tmp_path))
    with patch("tpe.judge.complete_json", return_value=_verdict("A")):  # A always: wrong on WB order
        batch = adapter.evaluate([_pair(1)], {"judge_prompt": "{{RESUME_A}}{{RESUME_B}}{{JOB_POSTING}}{{ORIGINAL_RESUME}}"},
                                 capture_traces=True)
    refl = adapter.make_reflective_dataset({"judge_prompt": "x"}, batch, ["judge_prompt"])
    entry = refl["judge_prompt"][0]
    assert "dequantify" in str(entry)  # feedback names the failure kind
```

- [ ] **Step 3: Implement**

```python
# src/tpe/gepa_adapter.py
"""GEPA adapter: candidate = the judge prompt text; score = accuracy - flip penalty;
reflective feedback names the degradation each miss failed to catch."""
from dataclasses import dataclass

from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # verify import path in Step 1

from tpe.cache import DiskCache
from tpe.judge import BothOrders, judge_both_orders
from tpe.schema import KnownPair


@dataclass
class JudgeAdapter(GEPAAdapter):
    model: str
    cache: DiskCache

    def evaluate(self, batch: list[KnownPair], candidate: dict[str, str],
                 capture_traces: bool = False) -> EvaluationBatch:
        template = candidate["judge_prompt"]
        results: list[BothOrders] = [
            judge_both_orders(template, self.model, pair, self.cache) for pair in batch
        ]
        scores = [r.pair_score - (0.25 if r.flipped else 0.0) for r in results]
        outputs = [r.bw.verdict.model_dump() for r in results]
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=results if capture_traces else None)

    def make_reflective_dataset(self, candidate: dict[str, str],
                                eval_batch: EvaluationBatch,
                                components_to_update: list[str]) -> dict[str, list[dict]]:
        entries = []
        for r, score in zip(eval_batch.trajectories, eval_batch.scores):
            tag = r.pair.tag
            problem = (f"Known-worse version was produced by degradation "
                       f"'{tag.name}' (lens: {tag.lens}, severity: {tag.severity})."
                       if tag else "Known-better side is the professionally drafted version.")
            feedback = (
                f"Score {score:.2f}. {problem} "
                f"The judge's verdicts: order BW -> {r.bw.verdict.overall.winner} "
                f"({r.bw.verdict.overall.rationale!r}); "
                f"order WB -> {r.wb.verdict.overall.winner} "
                f"({r.wb.verdict.overall.rationale!r}). "
                + ("Verdict flipped with presentation order — the rubric is not "
                   "grounding the decision in content." if r.flipped else "")
                + ("" if score >= 1.0 else
                   " The rubric failed to catch this quality difference; strengthen the "
                   "criteria that would detect it, without rewarding surface keyword counts.")
            )
            entries.append({
                "Inputs": {"job_excerpt": r.pair.job_text[:500], "pair_id": r.pair.pair_id},
                "Generated Outputs": r.bw.verdict.overall.model_dump(),
                "Feedback": feedback,
            })
        return {"judge_prompt": entries}
```

- [ ] **Step 4: Run adapter tests to pass** (adjust `EvaluationBatch` import/shape to installed gepa).

- [ ] **Step 5: Write `scripts/run_gepa.py`**

```python
# scripts/run_gepa.py
"""GEPA optimization entry. Budget-capped; all judge calls disk-cached."""
import json
import sys
import time
from pathlib import Path

import gepa

from tpe.cache import DiskCache
from tpe.dataset import load_pairs
from tpe.gepa_adapter import JudgeAdapter
from tpe.models import ladder

ROOT = Path(__file__).resolve().parent.parent


def main(max_metric_calls: int = 400) -> None:
    seed = (ROOT / "prompts/seed_judge.md").read_text()
    train = load_pairs(ROOT / "data/pairs/train.jsonl")
    val = load_pairs(ROOT / "data/pairs/val.jsonl")
    adapter = JudgeAdapter(model=ladder()["mini"], cache=DiskCache(ROOT / "runs/cache"))
    result = gepa.optimize(
        seed_candidate={"judge_prompt": seed},
        trainset=train, valset=val,
        adapter=adapter,
        reflection_lm=f"openai/{ladder()['top']}",  # litellm-style id; verify in Step 1
        max_metric_calls=max_metric_calls,
    )
    out_dir = ROOT / f"runs/gepa_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True)
    best = result.best_candidate["judge_prompt"]
    (ROOT / "prompts/optimized_judge.md").write_text(best)
    (out_dir / "result.json").write_text(json.dumps({
        "best_score": getattr(result, "val_aggregate_scores", None),
        "num_candidates": len(getattr(result, "candidates", []) or []),
    }, default=str))
    print(f"best candidate written to prompts/optimized_judge.md; logs in {out_dir}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
```

- [ ] **Step 6: Run GEPA (live, budget-capped):** `uv run python scripts/run_gepa.py 400`
Expected: several reflection rounds; `prompts/optimized_judge.md` written. Then re-run baseline eval on val with the optimized prompt and confirm `gepa_metric` > seed baseline. If not, run once more with a higher budget before concluding.
- [ ] **Step 7: Commit** — `feat: GEPA adapter + optimized judge prompt`

---

### Task 13: Tier-sensitivity gate

**Files:**
- Create: `src/tpe/sensitivity.py`; add `tpe sensitivity` command to `src/tpe/cli.py`
- Test: `tests/test_sensitivity.py`

**Interfaces:**
- Consumes: `run_pairs`, `summarize`, `mcnemar_exact`, `ladder()`.
- Produces:
  - `gate(tier_results: dict[str, list[BothOrders]], spread_min: float = 0.10) -> GateReport` — `GateReport(monotone: bool, subtle_spread: float, spread_ok: bool, mcnemar_p: float, significant: bool, passed: bool, per_tier: dict)`. Ladder order fixed: nano < mini < mid < top; monotone allows 2-point (0.02) tolerance dips.
  - `render_report(gate_report) -> str` (markdown) written to `runs/sensitivity_<timestamp>.md`.
  - CLI: `tpe sensitivity --prompt prompts/optimized_judge.md --split test`.

- [ ] **Step 1: Failing test**

```python
# tests/test_sensitivity.py
from unittest.mock import MagicMock

from tpe.sensitivity import gate


def _results(acc: float, subtle_acc: float, n: int = 20):
    out = []
    for i in range(n):
        r = MagicMock()
        severity = "subtle" if i < n // 2 else "severe"
        target = subtle_acc if severity == "subtle" else acc
        r.pair_score = 1.0 if (i % 100) < target * 100 else 0.0
        r.flipped = False
        r.lens_score = lambda lens: 1.0
        r.pair.tag.severity = severity
        r.pair.tag.name = "keyword_strip"
        r.pair.tag.lens = "ats"
        r.pair.pair_id = f"p{i}"
        out.append(r)
    return out


def test_gate_passes_on_rising_ladder():
    tiers = {"nano": _results(0.6, 0.5), "mini": _results(0.75, 0.65),
             "mid": _results(0.85, 0.8), "top": _results(0.95, 0.9)}
    g = gate(tiers)
    assert g.monotone and g.spread_ok
    assert g.subtle_spread >= 0.10


def test_gate_fails_on_flat_ladder():
    tiers = {t: _results(0.7, 0.7) for t in ("nano", "mini", "mid", "top")}
    g = gate(tiers)
    assert not g.spread_ok and not g.passed
```

- [ ] **Step 2: Run to fail; implement**

```python
# src/tpe/sensitivity.py
"""Tier-sensitivity gate: the same frozen prompt must do better on smarter models."""
import time
from dataclasses import dataclass
from pathlib import Path

from tpe.metrics import mcnemar_exact, summarize

TIER_ORDER = ["nano", "mini", "mid", "top"]
MONOTONE_TOLERANCE = 0.02


@dataclass
class GateReport:
    per_tier: dict
    monotone: bool
    subtle_spread: float
    spread_ok: bool
    mcnemar_p: float
    significant: bool
    passed: bool


def _subtle_accuracy(results: list) -> float:
    subtle = [r.pair_score for r in results
              if r.pair.tag is not None and r.pair.tag.severity == "subtle"]
    return sum(subtle) / len(subtle) if subtle else 0.0


def gate(tier_results: dict[str, list], spread_min: float = 0.10) -> GateReport:
    tiers = [t for t in TIER_ORDER if t in tier_results]
    summaries = {t: summarize(tier_results[t]) for t in tiers}
    accs = [summaries[t].accuracy for t in tiers]
    monotone = all(accs[i + 1] >= accs[i] - MONOTONE_TOLERANCE for i in range(len(accs) - 1))
    subtle = {t: _subtle_accuracy(tier_results[t]) for t in tiers}
    subtle_spread = subtle[tiers[-1]] - subtle[tiers[0]]
    # discordant pairs bottom vs top (pair correct = pair_score == 1.0)
    bottom = {r.pair.pair_id: r.pair_score == 1.0 for r in tier_results[tiers[0]]}
    top = {r.pair.pair_id: r.pair_score == 1.0 for r in tier_results[tiers[-1]]}
    b = sum(1 for pid in bottom if top.get(pid) and not bottom[pid])
    c = sum(1 for pid in bottom if bottom[pid] and not top.get(pid, False))
    p = mcnemar_exact(b, c)
    spread_ok = subtle_spread >= spread_min
    significant = p < 0.05
    return GateReport(
        per_tier={t: {"accuracy": summaries[t].accuracy, "subtle": subtle[t],
                      "flip_rate": summaries[t].flip_rate} for t in tiers},
        monotone=monotone, subtle_spread=subtle_spread, spread_ok=spread_ok,
        mcnemar_p=p, significant=significant,
        passed=monotone and spread_ok and significant,
    )


def render_report(g: GateReport) -> str:
    lines = ["# Tier-sensitivity report", "",
             "| tier | accuracy | subtle-slice | flip rate |", "|---|---|---|---|"]
    for t, row in g.per_tier.items():
        lines.append(f"| {t} | {row['accuracy']:.3f} | {row['subtle']:.3f} | {row['flip_rate']:.3f} |")
    lines += ["",
              f"- monotone ladder: **{g.monotone}**",
              f"- subtle-slice spread (top − bottom): **{g.subtle_spread:+.3f}** (gate ≥ +0.10: {g.spread_ok})",
              f"- McNemar bottom vs top: **p = {g.mcnemar_p:.4f}** (gate < 0.05: {g.significant})",
              f"- **GATE {'PASSED' if g.passed else 'FAILED'}**", ""]
    if not g.passed:
        lines.append("A failed gate means the rubric does not require model capability: "
                     "harvest the subtle pairs the top tier missed and feed them to the next GEPA round.")
    return "\n".join(lines)
```

- [ ] **Step 3: Tests pass; wire `tpe sensitivity` CLI command** (loop `ladder()` tiers × `run_pairs` on test split, then `gate` + `render_report`, write `runs/sensitivity_<ts>.md`).
- [ ] **Step 4: Live run:** `uv run tpe sensitivity --prompt prompts/optimized_judge.md`
Expected: 4 tiers × test pairs × 2 orders (cached where repeated). Read the report; record gate outcome in CONTINUITY.md. If the gate fails, that's a *finding*, not a bug — follow the report's remediation line.
- [ ] **Step 5: Commit** — `feat: tier-sensitivity gate + report`

---

### Task 14: compare-runs CLI

**Files:**
- Modify: `src/tpe/cli.py`
- Test: `tests/test_compare_runs.py`

**Interfaces:**
- Input format (documented in README): two JSONL files, each line `{"id": str, "job": str, "original": str, "resume": str}`; joined on `id`.
- Produces: `tpe compare-runs runA.jsonl runB.jsonl --prompt prompts/optimized_judge.md --model-tier mini` → per-lens win rates for B over A with Wilson CIs, plus per-id verdicts, written to `runs/compare_<timestamp>.md` and printed.

- [ ] **Step 1: Failing test**

```python
# tests/test_compare_runs.py
import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tpe.cli import app

runner = CliRunner()


def _verdict(winner: str) -> dict:
    lens = {"winner": winner, "evidence": "e"}
    return {"ats_signal": lens, "human_skim": lens,
            "overall": {"winner": winner, "margin": "clear", "rationale": "r"}}


def _write_run(path: Path, resume_text: str):
    rows = [{"id": f"i{k}", "job": "JD", "original": "ORIG", "resume": f"{resume_text}{k}"}
            for k in range(3)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_compare_runs_reports_win_rate(tmp_path: Path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_run(a, "OLD")
    _write_run(b, "NEW")
    # B always wins: judge says A when NEW is in slot A, B when NEW is in slot B
    with patch("tpe.judge.complete_json",
               side_effect=lambda model, system, user, schema:
               _verdict("A" if user.find("NEW") < user.find("OLD") else "B")):
        result = runner.invoke(app, ["compare-runs", str(a), str(b),
                                     "--prompt", "prompts/seed_judge.md",
                                     "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    assert "3/3" in result.output or "100" in result.output
```

- [ ] **Step 2: Implement.** For each joined id build `KnownPair(better=B.resume, worse=A.resume, tag=None, source="synthetic", split="test")` — "better" is just the slot for run B; `BothOrders.pair_score == 1.0` then means "B won both orders", 0.5 = split/tie, 0.0 = A won. Aggregate: `wins = sum(score for ...)`, report `wins/n` with `wilson_ci(round(wins), n)`, per-lens the same via `lens_score`. Emit markdown table of per-id outcomes + summary.
- [ ] **Step 3: Test passes.**
- [ ] **Step 4: Commit** — `feat: compare-runs cross-run win-rate report`

---

### Task 15: README + continuity + full suite

**Files:**
- Create: `README.md`, `CONTINUITY.md`
- Modify: none

- [ ] **Step 1: README.md** — what it is (one paragraph), setup (`uv sync`, `.env`), the five commands with examples, input format for compare-runs, pipeline diagram (ascii), pointers to `docs/design.html`, `docs/taste-guide.md`, gate reports in `runs/`.
- [ ] **Step 2: CONTINUITY.md** — ledger format per global CLAUDE.md (Goal/Constraints/Decisions/State/Working set) recording: seed baseline number, GEPA result, gate outcome.
- [ ] **Step 3: Full suite green:** `uv run pytest -q` — all tests pass, zero live calls.
- [ ] **Step 4: Commit** — `docs: README + continuity ledger`

---

## Self-review notes

- Spec coverage: design.html §01→Tasks 6/10 (lenses, taste guide), §02→Tasks 3/8/12 (pipeline), §03→Tasks 5/6/7 (data incl. traps + held-out types + anchors), §04→Task 8 (pairwise, swap, cache), §05→Task 13 (gate incl. McNemar), §06 phases→task order, §07 acceptance→Tasks 11/12/13/14 record each number.
- Deliberate deviations from bite-size TDD: Tasks 5 and 10 are artifact tasks (snapshot, docs) — their "tests" are existence/contract checks, which is appropriate.
- Types consistent: `KnownPair`/`BothOrders`/`Summary` signatures match across Tasks 7–14. `complete_json(model, system, user, schema)` used identically in Tasks 3/8. Patch target `tpe.judge.complete_json` everywhere a judge call is mocked.
- Known uncertainty, contained: exact gepa import paths/signatures (Task 12 Step 1 verifies against installed source) and exact model IDs (Task 10 pins from research; `models.ladder()` is the single indirection point). Neither blocks Tasks 1–9.
