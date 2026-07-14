# talent-promo-eval

**A resume-quality judge you can actually trust — because it's trained on comparisons with known answers and proven to require intelligence.**

Most LLM-as-judge setups are a hand-written prompt and a prayer: nobody knows whether the judge's verdicts track real quality or surface polish. This repo builds the judge differently, around two mechanisms:

### 1. GEPA evolves the rubric from examples with known answers

We never ask a human "which resume is better?" during training. Instead we *manufacture* comparisons where the answer is known by construction: take a real optimized resume, apply one tagged degradation (strip the job's keywords, bury the relevant role, de-quantify the bullets, merge bullets into walls of text…), and the original must win. Keyword-**stuffed** variants are trap pairs that must **lose** — so the judge cannot learn "more keywords = better."

[GEPA](https://arxiv.org/abs/2507.19457) then evolves the judge prompt against those pairs: a candidate rubric judges a batch; every miss produces reflective feedback that *names the degradation it failed to catch* ("this pair was `dequantify:subtle` and your verdict flipped with presentation order"); a top-tier model rewrites the rubric to fix exactly that; Pareto selection keeps the strongest variants. The rubric is **learned from failure, not authored from intuition** — the seed prompt (distilled from screening research, see `docs/taste-guide.md`) went from 0.784 → 0.935 accuracy and 0.358 → 0.123 order-flip rate on validation.

### 2. The tier-sensitivity gate proves the rubric requires reasoning

The core trust test: run the *same frozen prompt* across a model capability ladder (gpt-5.4-nano → gpt-5.4-mini → gpt-5.6-terra → gpt-5.6-sol). If the rubric encodes real judgment, smarter models must score higher — especially on *subtle* degradations. If every tier scores the same, the rubric is a checklist any model can pattern-match, and its verdicts carry no signal.

The gate demands: **monotone accuracy up the ladder** + **≥10-point spread on subtle pairs** + **directional McNemar p < 0.05**. Our optimized judge passed: 0.749 → 0.810 → 0.887 → 0.890, subtle-slice spread +21.6 points, p < 0.0001. A flat ladder would have failed the build.

> **Staleness caveat:** those numbers were measured on the v1 pair dataset (508 pairs).
> Subsequent label-integrity review rounds rebuilt the dataset (494 pairs, see
> `CONTINUITY.md`) and tightened the flip-rate definition; a re-run of GEPA + the gate
> on the current dataset is pending. Treat the exact figures as v1-dataset results.

```
corpus snapshot ──┐
                  ├─> known-order pairs (degradations, traps, held-out types)
degradation engine┘        │
                           v
seed prompt ──> GEPA loop (rollouts on mini, reflection on top) ──> optimized_judge.md
                           │
                           v
       gates: held-out accuracy · order-flip rate · TIER SENSITIVITY
                           │
                           v
              tpe compare-runs runA.jsonl runB.jsonl   (judged on an advanced model)
```

## Model-tier policy (deliberate, don't "optimize" it away)

| Tier | Model | Role |
|---|---|---|
| nano / mini | gpt-5.4-nano / -mini | Bottom rungs of the sensitivity ladder. **mini is also the GEPA rollout workhorse** (thousands of cheap calls during optimization) — it is NOT the production judge. |
| mid | gpt-5.6-terra | **Production judge** — default for `compare-runs` and `judge-one`. Test accuracy 0.887, flip rate 0.050, half the price of top. |
| top | gpt-5.6-sol | GEPA reflection (rewrites the rubric) + ceiling rung of the sensitivity ladder. |

The point of optimizing on mini and *judging* on mid/top: GEPA needs volume, verdicts need capability. The sensitivity gate is what proves the capability actually buys accuracy.

## Setup

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY
```

**Data is local-only by design.** `data/` (corpus snapshot + pairs) and `runs/` are
gitignored because they embed real job postings and the candidate profile. To
materialize them:

```bash
export TALENT_PROMO_DIR=~/path/to/talent-promo
uv run python scripts/snapshot_corpus.py
uv run python -c "
from pathlib import Path
from tpe.corpus import load_corpus, load_human_codes
from tpe.dataset import build_pairs, write_splits
print(write_splits(build_pairs(load_corpus(), load_human_codes()), Path('data/pairs')))"
```

Splits are deterministic (hash of pair_id), so every checkout rebuilds identical
train/val/test sets. Headline results are recorded in `CONTINUITY.md`.

## Commands

```bash
# Compare two app runs: B's win rate over A, per lens, with CIs (production judge = mid tier)
uv run tpe compare-runs runA.jsonl runB.jsonl --prompt prompts/optimized_judge.md

# One-off comparison (job can be a URL, file, or raw text)
uv run tpe judge-one --job <url|file|text> --original orig.md --a v1.html --b v2.html

# Score a judge prompt against a known-answer split (rubric development)
uv run tpe eval-prompt --prompt prompts/seed_judge.md --split val

# GEPA optimization (writes prompts/optimized_judge.md)
uv run python scripts/run_gepa.py 400

# The trust test: same prompt across the model ladder
uv run tpe sensitivity --prompt prompts/optimized_judge.md --split test
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

## What the judge scores

Two lenses, grounded in screening research (`docs/research-notes.md` has the cited evidence base; `docs/taste-guide.md` the distilled philosophy):

- **ats_signal** — parseable structure, grounded JD-vocabulary coverage inside dated experience, extractable quantified evidence
- **human_skim** — does the key evidence land where a ~7-second recruiter first pass actually goes (top third, titles, lead bullets)

One overriding rule: **grounding dominates**. Content the original resume (+ confirmed discovered facts) cannot support makes a version worse no matter how job-aligned it reads.

## Layout

- `AGENTS.md` — **onboarding for coding agents** (how it works, invariants, pitfalls)
- `docs/design.html` — the reviewed design document; `docs/superpowers/plans/` — the implementation plan
- `prompts/` — `seed_judge.md` (hand-written v0) and `optimized_judge.md` (GEPA output, frozen)
- `src/tpe/` — schema · corpus · degrade · dataset · judge · metrics · gepa_adapter · sensitivity · grounding · cli
- `runs/` — eval results, GEPA logs, sensitivity reports (gitignored; judge-call cache in `runs/cache/`)

## Tests

```bash
uv run pytest        # offline suite (network-free; live API tests are opt-in via -m live)
```
