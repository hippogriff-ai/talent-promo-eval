# talent-promo-eval — Claude Code instructions

@AGENTS.md

The imported AGENTS.md is the canonical onboarding: what the repo does, the two core
mechanisms (known-answer pairs + tier-sensitivity gate), the model-tier policy, module
map, invariants, and privacy rules. Follow it exactly — in particular:

- Never commit `data/` or `runs/` content, or any job-posting/profile text.
- Never hand-edit `prompts/optimized_judge.md`; regenerate via GEPA + re-gate.
- Keep unit tests network-free; mock `tpe.judge.complete_json`.
- Production judging defaults to the mid tier; mini is the GEPA workhorse only.
