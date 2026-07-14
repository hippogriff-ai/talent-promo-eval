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
- Tasks 1–15 of the plan implemented, TDD, all committed to main.
- 51 offline tests green (`uv run pytest -q`).
- Corpus snapshot: 28 traces + human codes + 14 JDs from talent-promo (read-only copy).
- Dataset built: data/pairs = train 244 / val 81 / test 179 / anchor 4 (508 pairs).
- Research: docs/research-notes.md (deep-research workflow; 2 claims adversarially VERIFIED, rest SOURCED with quotes — verification pass hit session limits; 1 claim REFUTED and recorded). Model IDs + gepa API live-verified by tech-facts agent.
- docs/taste-guide.md + prompts/seed_judge.md written from the research.
- Design doc artifact: https://claude.ai/code/artifact/d5e0e214-5329-4272-8c35-da6ce85d4336

### Live results (2026-07-12, all runs cached under runs/)
- **Seed baseline (val, mini)**: acc 0.784, flip 0.358, gepa_metric 0.694. Weakest: bland_leads 0.333, keyword_stuff traps 0.725, subtle 0.667.
- **GEPA (400 metric calls, mini judge / sol reflection)**: best val 0.904. Verified independently: **acc 0.935, flip 0.123, gepa_metric 0.904** (bland_leads→0.889, stuff traps→0.863, subtle→0.898). Frozen at prompts/optimized_judge.md.
- **Tier gate (test 179 pairs × 4 tiers): PASSED** — nano 0.749 / mini 0.810 / terra 0.887 / sol 0.890 (monotone); subtle spread +0.216 (gate ≥0.10); McNemar p<0.0001. Report: runs/sensitivity_20260712_233341.md.
- **Held-out degradation types (never trained)**: mini generalizes with a gap (header_flatten 0.661, drop_summary 0.777); terra nails header_flatten 0.951 but drop_summary 0.482 — summary-presence is a contested signal the optimized rubric doesn't encode (GEPA never saw it, by design).
- **Human anchors (4 pairs, terra): 0.25 acc, 0 flips — judge prefers raw profile on ats lens.** Root cause is anchor construction, not the judge: the raw profile is the grounding superset (cannot lose on grounding, keyword-rich), and on a high-F anchor the judge correctly flagged a projected-metric-stated-as-realized inflation plus one unsupported skill claim — the same failure class the Phase B human coding identified as the drafter's signature. human_skim lens agrees with humans 0.75. (Specific wording redacted per the no-profile-text rule; see the local run log for the verbatim rationale.)
- **Recommended production judge tier: mid (gpt-5.6-terra)** — acc 0.887 / flip 0.050 on test at half sol's price. mini stays the GEPA-loop workhorse; its test flip rate (0.335) makes it unsuitable as the production verdict-giver.
- Ops notes: OpenAI returns 401 "insufficient permissions" under sustained concurrency (not transient, not model-gating — all 4 models pass single-threaded). Mitigated: exponential backoff in complete_json + self-healing serial retry round in run_pairs; sweep ran clean at max_workers=4.

### Grounding extension + metadata slicing (2026-07-13)
- Decision: judge input stays minimal (screeners never see session logs); the one legitimate session-log payload is candidate-confirmed discovery facts, which extend the GROUNDING source, not the judge context.
- `src/tpe/grounding.py`: `compose_grounding(original, discovered_facts)` appends deduped facts under "## Additional facts confirmed by the candidate".
- `compare-runs`: rows accept optional `discovered_facts` (union across both runs — a fact confirmed in either session is true of the candidate) and optional `meta` dict; report now includes per-meta-key win-rate slices with Wilson CIs (meta never reaches the judge).
- 56 tests green. Calibration caveat: gate numbers were measured on fact-free inputs; when production rows start carrying discovered_facts, build a small facts-bearing pair set and re-check the gate.

### Self-review + doc/tier/onboarding pass (2026-07-14)
- 8-angle code review (parallel finders + 1-vote verify): 8 findings confirmed and fixed — atomic/self-healing cache, refusal surfacing, transient-only retry classification (no SDK stacking), run_pairs exception fidelity, fractional Wilson CIs, parallel GEPA adapter (dominant-cost path), tier-order single source of truth, tie_rate visibility. 63 tests green.
- README rewritten to lead with the two core mechanisms (GEPA-from-known-answer-pairs; tier-sensitivity gate as trust test) + explicit model-tier policy table.
- Production judge default now MID tier (gpt-5.6-terra) for compare-runs and judge-one; mini remains GEPA rollout workhorse only.
- AGENTS.md (canonical agent onboarding: mechanisms, invariants, module map, pitfalls, privacy rules) + CLAUDE.md (imports AGENTS.md).

### Codex review round (2026-07-14)
- Codex (freshly enabled) returned 7 findings; all adopted: corpus loader now supports nested record shapes and REJECTS blank fields (was silent ""), keyword regexes handle symbolic tokens (c++/c#), bury_relevant/bland_leads no-op when no genuinely worse reorder exists (were creating mislabeled pairs), build_pairs dedupes byte-identical degradation outputs (train/test leakage), render_prompt is single-pass (document content can no longer expand placeholders), gate fails closed on partial ladders.
- Dataset rebuilt with fixed builder: 508 → 500 pairs (train 240/val 78/test 178/anchor 4) — 8 duplicate/mislabeled pairs removed. Published GEPA/gate numbers were measured on the v1 dataset; deltas are marginal (1.6% of pairs) but a re-run on v2 would tighten the claim. 71 tests green.

### Codex round 2 (2026-07-14)
- Codex auto-re-reviewed the fix commit; 7 more findings, all adopted: P1 privacy (CONTINUITY quoted profile-derived wording — redacted in tree; NOTE: the verbatim phrases remain in earlier public git history, see open question), bury_relevant severe now guarantees the hot role sinks (blind reversal could promote it), bland_leads only fires when a quantified bullet is genuinely demoted (stable within-class order), GEPA adapter retries transient API errors and marks infra failures as non-rubric feedback, gate validates identical pair_id sets across tiers, compare-runs rejects duplicate ids and mismatched job/original rows.
- Dataset rebuilt (counts unchanged: 240/78/178/4 — round-2 guards didn't fire on this corpus). 79 tests green.

### Now
- Pipeline complete and gated. Ready for real use via `tpe compare-runs`.

### Next (optional follow-ups)
1. Better human anchors: replace profile-vs-generated with human-ranked generated-vs-generated pairs (removes the grounding-superset confound).
2. drop_summary/header_flatten: add to a GEPA round 2 if summary-presence should be an enforced signal (currently held out by design).
3. Anthropic-ladder cross-check of the tier gate (design open item).
4. Re-check gate on facts-bearing pairs once production sessions supply discovered_facts.

## Open questions (UNCONFIRMED if needed)
- Two short profile-derived phrases (a metric-inflation quote and one skill name) existed in CONTINUITY.md in public git history before the 2026-07-14 redaction. Low identifiability (no names/companies), but full removal would require another fresh-repo migration — user's call.
- Whether summary-presence should count as a quality signal for this product (drop_summary 0.482 on terra says the rubric doesn't enforce it; eye-tracking evidence says it helps the skim).
- Gate power caveat (verifier finding, 2026-07-14): the subtle-slice spread compares two ~29-pair means against a 0.10 threshold without a CI — SE(diff) ≈ 0.13, so spreads near the threshold are noise-sensitive. Our +0.216 clears the noise band; a future borderline result should add a bootstrap CI on the spread before trusting PASS/FAIL. McNemar is unaffected (runs over all 179 pairs).

## Working set (files/ids/commands)
- src/tpe/{schema,models,cache,job_fetch,corpus,degrade,dataset,judge,metrics,gepa_adapter,sensitivity,cli}.py
- scripts/{snapshot_corpus,run_gepa}.py · prompts/seed_judge.md · data/pairs/*.jsonl
- `uv run pytest -q` · `uv run tpe --help`
