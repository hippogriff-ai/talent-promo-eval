# Taste Guide: Judging Resume Optimization Quality

The judging philosophy behind `prompts/seed_judge.md`. Every discriminator traces to a
finding in `research-notes.md`. The judge's job is narrow: **given the same candidate and
the same job posting, which of two resume versions serves the candidate better?** It does
not rate the candidate, and it does not reward decoration.

## How the three audiences actually read

1. **AI scanner (first gate).** Parses sections, extracts skills with recency/duration
   metadata from dated work history, infers seniority from titles + dates, and ranks by
   *semantic* similarity to the job description — not keyword counts. Unrecognizable
   section structure breaks extraction; skills evidenced inside dated roles count more
   than skills in a bare list; contextual JD vocabulary beats stuffed vocabulary
   (stuffing measurably *lowers* modern rankings).

2. **Recruiter skim (~6–7.4 seconds).** Fixates on job titles more than anything else,
   reads the top third first, follows F/E patterns (bold titles → first bullets), and
   rewards clear section headers, white space, and scannability. Clutter, wall-of-text
   bullets, and visible keyword stuffing are the classic reject-triggers. Time spent on
   the Experience section predicts advancing.

3. **Hiring manager (deeper read).** Wants responsibilities AND accomplishments;
   quantified, specific outcomes; most-relevant experience surfaced first. Adjective-led
   self-description without evidence is a red flag.

## Discriminators, in priority order

When comparing version X vs version Y for the same job:

1. **Grounding beats everything.** Content unsupported by the candidate's original
   materials (new metrics, inflated tenure, transplanted scope, invented seniority) makes
   a version WORSE no matter how job-aligned it reads. A resume is a claim under audit.
2. **Evidence density where the eyes land.** The version that puts the most job-relevant,
   quantified evidence in the top third and in lead bullets of the most recent role wins
   the skim. Same facts buried below the fold = worse version.
3. **Grounded JD-vocabulary coverage.** Naming the job's actual technologies and skills —
   inside dated experience bullets, with evidence — is a real ranking signal. Coverage in
   a context-free list is weak; coverage the original resume cannot support is a trap
   (see 1).
4. **Specific outcomes over duties; duties over adjectives.** "Reduced latency 43% by
   rewriting X" > "responsible for X" > "passionate engineer". De-quantified bullets are
   a strictly worse version of the same fact.
5. **Extractable structure.** Recognizable section headers (Summary/Experience/Skills…),
   single-column flow, dated roles with titles, bulleted accomplishments. Flattened or
   unconventional structure degrades both machine parsing and the human skim.
6. **Scannability.** Short evidence-led bullets beat merged wall-of-text paragraphs;
   a tight summary up top beats none.
7. **Title/seniority signal.** Titles are the single most-fixated element and drive
   machine-inferred seniority — but only as grounded by the original materials.

## Anti-signals (version-losers)

- **Keyword stuffing** — competency blobs, repeated JD terms without evidence,
  "leveraging X expertise" boilerplate. Hurts with humans AND modern semantic rankers;
  a stuffed version must LOSE to its unstuffed counterpart.
- **Fabrication/inflation** — anything the original resume can't support.
- **Generic filler** — adjectives without facts.
- **Skim-hostile layout** — buried relevance, missing headers, run-on bullets.

## Explicit non-goals

- **Not judged:** visual design/fonts/colors (we compare content HTML), candidate
  strength itself (same person by construction), resume length per se (only what the
  length displaces from the skim zone).
- **Ties are legitimate.** If two versions genuinely don't differ on the discriminators,
  say tie — forced arbitrary picks are noise (LLMs are known-bad at abstaining; the
  rubric must make abstaining respectable).

## Two lenses, one overall verdict

- **ats_signal** = discriminators 1, 3, 5 (+ 4's quantification as extractable evidence).
- **human_skim** = discriminators 2, 4, 6, 7 under a 7-second, top-third-first read.
- **overall** weighs both; grounding violations (1) dominate; when lenses disagree,
  prefer the verdict backed by the more severe, better-evidenced difference.
