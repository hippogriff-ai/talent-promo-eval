You are comparing two versions of the same candidate's resume, optimized for the same job posting. Decide which version serves the candidate better. You are NOT rating the candidate — both versions describe the same person. You are rating the resume *as an artifact* passing through two gates: an AI screening system, then a human recruiter's ~7-second skim.

## Materials

### Job posting
{{JOB_POSTING}}

### Candidate's original resume (the grounding source — the only facts that exist)
{{ORIGINAL_RESUME}}

### Version A
{{RESUME_A}}

### Version B
{{RESUME_B}}

## How to judge

Evaluate each lens independently, then combine.

### Lens 1 — ats_signal (machine screening)
Modern screeners parse sections, extract skills with recency/duration from dated roles, infer seniority from titles and dates, and rank by semantic similarity to the job description — not keyword counts. Ask:
- Which version names the job's actual technologies, skills, and vocabulary *inside dated experience bullets with supporting evidence*? Skills in a context-free list count far less.
- Which version has cleaner extractable structure: recognizable section headers (Summary, Experience, Skills), dated roles with clear titles, single-column bulleted content?
- Which version preserves quantified, machine-extractable evidence (metrics, scale, dates)?

### Lens 2 — human_skim (~7 seconds, top-third first)
Recruiters fixate on job titles, read the top third, then bold titles and first bullets (F-pattern). Ask:
- In the first screenful, which version lands the most job-relevant titles and evidence?
- Which version leads each role with its strongest, most quantified, most job-relevant bullet — rather than generic duty statements?
- Which version is more scannable: short evidence-led bullets, a tight summary, clear headers — versus merged wall-of-text bullets or missing sections?

### Overriding rule — grounding (checked before both lenses)
Compare every claim against the ORIGINAL resume. Content the original cannot support — new or inflated metrics, longer tenure, transplanted accomplishments, invented scope or seniority — makes a version WORSE regardless of how job-aligned it reads. **More job keywords is NOT better if the added keywords lack supporting evidence in the original.** A version with a stuffed skills blob, repeated job-description terms without evidence, or "leveraging X expertise" boilerplate must lose to its honest counterpart. Keyword stuffing measurably hurts with both machines and humans.

### Ties
If the versions do not genuinely differ on a lens, say "tie" for that lens. Do not manufacture a preference. The overall verdict may also be "tie".

## Verdict

Return JSON only:
- `ats_signal`: {winner: "A"|"B"|"tie", evidence: cite the specific difference that decided it}
- `human_skim`: {winner: "A"|"B"|"tie", evidence: cite the specific difference that decided it}
- `overall`: {winner, margin: "slight"|"clear"|"decisive", rationale}

For `overall`: grounding violations dominate; otherwise weigh both lenses, preferring the lens with the more severe, better-evidenced difference. Cite concrete evidence (quote or name the exact bullet/section/keyword), never generic impressions.
