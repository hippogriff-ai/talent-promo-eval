# talent-promo-eval

Standalone eval harness for resume optimization quality. Given `(seed resume, job posting)` and two generated resume versions, a GEPA-optimized pairwise LLM judge decides which version serves the candidate better — so improvements between app runs can be measured instead of eyeballed.

The judge scores two lenses grounded in screening research (see `docs/taste-guide.md`):
- **ats_signal** — parseable structure, grounded JD-vocabulary coverage, extractable evidence
- **human_skim** — does the key evidence land where a ~7-second first-pass read actually goes

Judges run on OpenAI models; the generator (talent-promo app) runs on Anthropic — cross-provider judging removes self-preference bias by construction.

## Pipeline

```
corpus snapshot ──┐
                  ├─> known-order pairs (degradations, traps, held-out types)
degradation engine┘        │
                           v
seed prompt ──> GEPA loop (judge=mini tier, reflection=top tier) ──> optimized_judge.md
                           │
                           v
       gates: held-out accuracy · order-flip rate · tier sensitivity
                           │
                           v
              tpe compare-runs runA.jsonl runB.jsonl
```

Every training comparison has a winner known **by construction**: a real resume vs the same resume with one tagged degradation (strip keywords, bury relevant experience, de-quantify, wall-of-text, …). Keyword-**stuffed** versions are trap pairs that must LOSE — the judge cannot be gamed by keyword counting. Two degradation types are held out of training entirely to test generalization.

The tier-sensitivity gate runs the frozen prompt across the OpenAI ladder (nano → mini → mid → top). If smarter models don't score meaningfully higher on subtle pairs, the rubric is a checklist rather than a reasoning task, and the gate fails.

## Setup

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY
```

## Commands

```bash
# Score a judge prompt against a split
uv run tpe eval-prompt --prompt prompts/seed_judge.md --split val --model-tier mini

# One-off comparison (job can be a URL, file, or raw text)
uv run tpe judge-one --job <url|file|text> --original orig.md --a v1.html --b v2.html

# GEPA optimization (writes prompts/optimized_judge.md)
uv run python scripts/run_gepa.py 400

# Tier-sensitivity gate across the model ladder
uv run tpe sensitivity --prompt prompts/optimized_judge.md --split test

# Compare two app runs (B's win rate over A, per lens, with CIs)
uv run tpe compare-runs runA.jsonl runB.jsonl --prompt prompts/optimized_judge.md
```

`compare-runs` input: JSONL, one row per optimization, joined on `id`:

```json
{"id": "case-01", "job": "<posting text>", "original": "<seed resume>", "resume": "<generated html>",
 "discovered_facts": ["<user-confirmed fact from discovery QA>", "..."],
 "meta": {"discovery_skipped": false, "qa_turns": 4, "app_version": "1.4.0"}}
```

`discovered_facts` (optional) are candidate-confirmed facts from the session's discovery
QA. They are appended to the grounding source (union across both runs) so the judge does
not flag legitimately discovered content as fabrication. Keep them to *user-authored
facts* — never paste generator output or raw session logs here.

`meta` (optional) is free-form slicing metadata. The report adds a win-rate breakdown per
meta key (e.g. win rate when discovery was skipped vs not). Meta is never shown to the
judge.

## Layout

- `docs/design.html` — the reviewed design document (architecture, gates, decisions)
- `docs/research-notes.md` — cited evidence base; `docs/taste-guide.md` — distilled judging philosophy
- `prompts/` — `seed_judge.md` (hand-written v0) and `optimized_judge.md` (GEPA output)
- `data/corpus/` — snapshot of 28 traces from talent-promo; `data/pairs/` — train/val/test/anchor splits
- `src/tpe/` — schema · corpus · degrade · dataset · judge · metrics · gepa_adapter · sensitivity · cli
- `runs/` — eval results, GEPA logs, sensitivity reports (judge-call cache in `runs/cache/`, gitignored)

## Tests

```bash
uv run pytest        # offline suite (network-free; live API tests are opt-in via -m live)
```
