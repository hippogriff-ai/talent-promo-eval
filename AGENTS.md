# Agent Onboarding — talent-promo-eval

You are in a standalone eval harness whose product is a **validated LLM-as-judge** for
resume optimization quality. Read this before changing anything; the design has several
deliberate choices that look like mistakes if you don't know why they're there.

## What this repo does, in one paragraph

Given `(job posting, original resume, resume version A, resume version B)`, the judge
(`prompts/optimized_judge.md`, executed by `src/tpe/judge.py`) decides which version
serves the candidate better, on two lenses: `ats_signal` (machine screening) and
`human_skim` (a recruiter's ~7-second first pass). The judge prompt was NOT hand-tuned:
it was **evolved by GEPA** against comparisons whose correct answer is known by
construction, and then **proven to require model capability** by the tier-sensitivity
gate. Those two mechanisms — known-answer training pairs and the sensitivity gate — are
the entire value of this repo. Everything else is plumbing.

## The two core mechanisms (do not weaken these)

1. **Known-order pairs** (`src/tpe/degrade.py` + `src/tpe/dataset.py`): a real optimized
   resume vs the same resume with ONE tagged degradation applied. The original must win.
   `keyword_stuff` pairs are **traps** — the stuffed side must LOSE, which is what stops
   the judge from learning "more keywords = better". Degradations are deterministic
   (same input → same output, no LLM, no randomness) so the dataset is reproducible.
   Two degradation types (`header_flatten`, `drop_summary`) are held out of training
   entirely and appear only in test — the generalization guard.

2. **Tier-sensitivity gate** (`src/tpe/sensitivity.py`): the same frozen prompt runs on
   a model capability ladder (nano → mini → mid → top). Gate = monotone accuracy +
   ≥10-point subtle spread with a paired-bootstrap CI excluding zero + directional
   exact McNemar p < 0.05 (top tier must WIN more discordants). A rubric that
   scores the same on every tier is a pattern-matchable checklist and FAILS. If you
   change the judge prompt, the gate result is stale until re-run.

## Model-tier policy (intentional; commonly "fixed" by mistake)

- **mini** (`gpt-5.4-mini`) = GEPA rollout workhorse + a sensitivity-ladder rung.
  It is deliberately cheap because optimization needs thousands of calls.
- **mid** (`gpt-5.6-terra`) = the **production judge** — default for `compare-runs`
  and `judge-one`. Chosen by the gate (test accuracy 0.883, flip rate 0.051 on dataset v5).
- **top** (`gpt-5.6-sol`) = GEPA reflection model + ladder ceiling.
- Model IDs live ONLY in `src/tpe/models.py` (`_DEFAULT_LADDER`), overridable via
  `TPE_MODEL_<TIER>` env vars. Do not scatter model IDs elsewhere.

## Map of the code (src/tpe/)

| Module | Responsibility | Key contract |
|---|---|---|
| `schema.py` | All pydantic data models | `KnownPair.better/worse` are ground-truth slots; `JudgeVerdict` is the judge's output shape |
| `models.py` | OpenAI wrapper + model ladder | `complete_json(model, system, user, schema) -> dict`; retries transient errors |
| `cache.py` | Content-addressed disk cache | key = sha256(model, system prompt, rendered user prompt); a cache hit replaces an API call |
| `judge.py` | Pairwise judging | EVERY pair is judged twice (A/B orders swapped); `BothOrders.pair_score`: 1.0 = known-better won both orders, 0.5 = split/tie, 0.0 = lost both; `flipped` = verdict changed with presentation order |
| `degrade.py` | 8 deterministic degradations | pure functions; return input unchanged when not applicable |
| `dataset.py` | Pair builder + splits | split by sha256(pair_id)%100 (<60 train, <80 val, else test) — deterministic across checkouts |
| `corpus.py` | Load the snapshotted corpus | tolerant to flat/nested record shapes |
| `metrics.py` | Summaries, exact McNemar, Wilson CI | pure functions, no I/O |
| `gepa_adapter.py` | GEPA integration | candidate = `{"judge_prompt": <text>}`; score = pair_score − 0.25×flip; reflective feedback names the missed degradation |
| `sensitivity.py` | The tier gate | consumes per-tier `run_pairs` results |
| `grounding.py` | Grounding composition | original resume + user-confirmed discovered facts = what "grounded" means |
| `cli.py` | typer CLI | `eval-prompt`, `judge-one`, `sensitivity`, `compare-runs` |

## Invariants a change must not break

- **Order-swap always.** Never judge a pair in one order only. Position bias is worst
  exactly on close calls (see `docs/research-notes.md` §4).
- **Grounding dominates.** The judge must punish content unsupported by the original
  resume + confirmed facts, even when it improves job-alignment. Keyword-stuffed trap
  pairs must keep losing.
- **Judge input stays minimal**: `(job, grounding, A, B)`. Session logs/process
  metadata never reach the judge — `meta` in compare-runs is report-layer slicing only.
- **Determinism**: degradations and splits are seed-free deterministic; any new
  randomness must be seeded and justified.
- **Unit tests are network-free.** Live API tests are opt-in via `pytest -m live`.
  Mock at `tpe.judge.complete_json` (the name judge.py imported), not `tpe.models`.
- **`prompts/optimized_judge.md` is a frozen GEPA artifact.** Don't hand-edit it;
  regenerate via `scripts/run_gepa.py` and re-run the gate. Calibration numbers in
  CONTINUITY.md describe THIS prompt on fact-free inputs.

## Data is local-only (privacy) — you will not find data/ in the checkout

`data/` and `runs/` are gitignored: they embed real job postings and the candidate's
profile. Tests that need corpus data skip when it's absent. To materialize locally:
set `TALENT_PROMO_DIR` to a talent-promo checkout, run `scripts/snapshot_corpus.py`,
then the pair-builder snippet in README "Setup". Never commit anything under `data/`
or `runs/`, and never embed job-posting text or profile text in code, tests, docs, or
commit messages — company-identifying content must stay out of git history. Note that
pair_ids embed trace ids (which name companies): error messages and eval logs that
print pair_ids are LOCAL-ONLY artifacts; never paste them into issues, PRs, or docs.

## Working on this repo

- `uv sync` then `uv run pytest -q` (offline, must stay green).
- Live smoke: `uv run tpe eval-prompt --prompt prompts/seed_judge.md --split val --limit 5 --model-tier mini` (needs `OPENAI_API_KEY` in `.env`, and `data/pairs/` materialized).
- All judge calls are disk-cached under `runs/cache/` — re-runs are free; delete a
  cache entry only if you know its content is corrupt.
- Read `CONTINUITY.md` for current state/results before starting; append your changes
  there when done (ledger format).
- Costs: a full 4-tier sensitivity sweep ≈ $35; a 400-call GEPA run ≈ $10–15. Don't
  start either without the operator asking for it.

## Known caveats (honest edges, documented in CONTINUITY.md)

- Human-anchor pairs (profile-vs-generated) score poorly by construction — the raw
  profile is the grounding superset. Fix is better anchor design, not judge changes.
- `drop_summary` generalizes poorly (0.482 on mid tier) — held out of training by
  design; whether summary-presence should be enforced is an open product decision.
- Gate numbers were measured on fact-free inputs; rows carrying `discovered_facts`
  eventually need a re-gate.
