# Continuity Ledger

## Goal (incl. success criteria)
Standalone eval harness: GEPA-optimized pairwise LLM judge for resume optimization quality (design: docs/design.html, plan: docs/superpowers/plans/2026-07-12-talent-promo-eval.md).
Success criteria (design §07): held-out accuracy ≥85% and above seed baseline; order-flip rate ≤10%; tier gate (monotone ladder, ≥10pt subtle-slice spread, McNemar p<0.05); accuracy holds within ~5pts on unseen degradation types; directional agreement with human anchors; compare-runs CLI with CIs.

## Constraints/Assumptions
- Standalone repo; never imports from talent-promo app (corpus arrived via one-time snapshot).
- Judge = OpenAI, generator = Anthropic (cross-provider, anti self-preference).
- Model ladder pinned 2026-07-12: nano=gpt-5.4-nano, mini=gpt-5.4-mini, mid=gpt-5.6-terra, top=gpt-5.6-sol (env-overridable via TPE_MODEL_<TIER>).
- All degradations deterministic; unit tests network-free; live tests behind `-m live`.
- Judge calls disk-cached at runs/cache/ (key: model+system+user sha256).

## Key decisions
- Pairwise verdicts (not absolute scores); both A/B orders always; flip penalty 0.25 in GEPA metric.
- gepa 0.1.1 custom adapter (candidate = {"judge_prompt": text}); reflection_lm = openai/gpt-5.6-sol via litellm string.
- Hybrid ground truth: synthetic known-order pairs (8 degradation types, 3 severities) + human anchors (F≥85 traces, validation-only).
- Held-out degradation types {header_flatten, drop_summary} → test split only (generalization guard).
- Splits deterministic by sha256(pair_id) % 100: <60 train, <80 val, else test.

## State
### Done (2026-07-12)
- Tasks 1–15 of the plan implemented, TDD, all committed to main of ~/Hanalei/talent-promo-eval.
- 46 offline tests green (`uv run pytest -q`).
- Corpus snapshot: 28 traces + human codes + 14 JDs from talent-promo (read-only copy).
- Dataset built: data/pairs = train 244 / val 81 / test 179 / anchor 4 (508 pairs).
- Research: docs/research-notes.md (deep-research workflow; 2 claims adversarially VERIFIED, rest SOURCED with quotes — verification pass hit session limits; 1 claim REFUTED and recorded). Model IDs + gepa API live-verified by tech-facts agent.
- docs/taste-guide.md + prompts/seed_judge.md written from the research.
- Design doc artifact: https://claude.ai/code/artifact/d5e0e214-5329-4272-8c35-da6ce85d4336

### Now
- BLOCKED on OPENAI_API_KEY (.env): seed baseline eval → GEPA run → sensitivity sweep.

### Next
1. `cp .env.example .env` + key; smoke: `uv run tpe eval-prompt --prompt prompts/seed_judge.md --split val --limit 5`
2. Full val baseline with seed prompt (record accuracy/flip-rate here).
3. `uv run python scripts/run_gepa.py 400` (~$10-15 at mini judge + sol reflection); confirm val gepa_metric beats baseline.
4. `uv run tpe sensitivity --prompt prompts/optimized_judge.md --split test` (~$35 all 4 tiers); record gate outcome.
5. If gate fails: harvest subtle misses → next GEPA round (report tells you).

## Open questions (UNCONFIRMED if needed)
- None blocking besides the API key.

## Working set (files/ids/commands)
- src/tpe/{schema,models,cache,job_fetch,corpus,degrade,dataset,judge,metrics,gepa_adapter,sensitivity,cli}.py
- scripts/{snapshot_corpus,run_gepa}.py · prompts/seed_judge.md · data/pairs/*.jsonl
- `uv run pytest -q` · `uv run tpe --help`
