You are comparing two optimized versions of the same candidate’s resume for the same job. Decide which resume is the stronger artifact—not which candidate is stronger. Both versions describe one person, and the ORIGINAL_RESUME is the sole source of truth.

The resume must pass two gates:
1. a modern ATS/AI screening system; and
2. a recruiter’s approximately seven-second skim.

## Inputs

### Job posting
{{JOB_POSTING}}

### Candidate’s original resume
{{ORIGINAL_RESUME}}

### Version A
{{RESUME_A}}

### Version B
{{RESUME_B}}

Optional metadata such as `pair_id` may also be present. Never use metadata, version labels, presentation order, or assumptions about how a version was generated to choose the winner.

## Required comparison procedure

Before judging either lens, perform a claim-level comparison in this order.

### Step 1: Establish the grounded fact set

Treat the original resume as the only evidence that exists. It determines the candidate’s:
- employers, titles, dates, and tenure;
- technologies and skills;
- metrics, team sizes, datasets, scale, and outcomes;
- ownership, leadership, scope, and seniority;
- education, certifications, publications, and projects.

A rewritten claim may use reasonable paraphrasing, but it may not add or inflate facts.

Grounding violations include:
- invented technologies, metrics, outcomes, responsibilities, or domains;
- increased team size, scale, tenure, ownership, or seniority;
- moving an accomplishment or technology from one role to another without support;
- implying production use when the original only supports experimentation;
- converting participation into leadership or support into ownership;
- presenting a job-posting phrase as candidate experience without evidence.

Grounding violations dominate the verdict. A better-aligned fabrication must lose to an honest counterpart.

Do not mislabel simple information loss as fabrication. Replacing “100-case golden dataset” with “several-case,” omitting “3-engineer team,” or corrupting “Mar 2025” into “Mar several” is a dequantification or precision defect, not necessarily an invented claim. It is still harmful because it destroys machine-extractable evidence and recruiter credibility.

### Step 2: Identify only the substantive differences

Create an internal delta list of facts, wording, ordering, structure, and formatting that differ between A and B. Judge those differences rather than rewarding content both versions share.

For every difference, ask:
1. Is it supported by the original?
2. Is it more or less specific than the original?
3. Is it placed where ATS systems can associate it with a dated role?
4. Is it visible and persuasive during a top-third/F-pattern skim?
5. Is it useful evidence, or merely job-description vocabulary?

Run the comparison symmetrically: mentally swap A and B and confirm that the same content—not the label or order—would still win.

### Step 3: Detect degradation and quality traps

Explicitly check for the following.

#### A. Dequantification or malformed substitutions
Penalize a version that replaces grounded facts with vague, broken, or placeholder-like language, including:
- exact counts changed to “several,” “multiple,” or similar vagueness;
- malformed constructions such as “several -case” or “several -engineer”;
- corrupted dates such as “Mar several”;
- deletion of metrics, scale, team size, dates, or outcomes.

Preserving exact evidence such as “100-case golden dataset,” “Led 3-engineer team,” or “Mar 2025” is valuable to both ATS extraction and human trust. Do not call a degradation harmless merely because the surrounding keywords remain.

#### B. Keyword stuffing
Do not award points for raw keyword count.

A keyword or skill is valuable primarily when:
- it is supported by the original;
- it appears naturally in a dated experience or project bullet;
- the bullet explains what the candidate did with it;
- it is relevant to the target job.

A conventional, concise Skills section may help parsing when it lists concrete, grounded tools or languages. It remains weaker evidence than dated experience.

Penalize a “Core Competencies,” “Expertise,” summary, or skills line when it:
- copies clusters of phrases from the posting;
- repeats terms already stated elsewhere without adding evidence;
- contains generic abstractions such as “agent systems,” “evaluation,” “scalability,” or “cross-functional leadership” without support;
- uses boilerplate such as “leveraging X expertise”;
- adds a dense comma-separated blob primarily to increase semantic overlap;
- makes unsupported proficiency or expertise claims.

This remains a quality defect even if some individual words loosely overlap with the original. If two versions have the same substantive experience and one merely adds a stuffed competencies blob, the stuffed version must not win for ATS alignment and should generally lose overall. Do not reinterpret keyword stuffing as a “legitimate readable Skills section” solely because it has a section header.

Distinguish:
- legitimate Skills: concise, concrete, grounded technologies such as named languages, frameworks, or tools;
- stuffed competencies: broad job-description phrases, duplicated concepts, unsupported expertise, or unnatural semantic-overlap text.

#### C. Structural degradation
Penalize:
- missing or unconventional section headers;
- unclear title/employer/date relationships;
- merged wall-of-text bullets;
- important evidence buried late in a role;
- overly long summaries;
- duplicated content;
- multi-column or decorative layouts that impede parsing, when visible in the supplied text.

Do not invent a formatting difference that is not observable from the input.

## Evaluation lenses

Evaluate the lenses independently after completing the grounding and quality checks.

### Lens 1 — `ats_signal`

Modern screeners parse sections, associate skills with dated roles, estimate recency and duration, infer seniority from titles and dates, and rank semantic relevance. They do not simply count keywords.

Prefer the version that:
- places relevant, grounded technologies and skills inside dated experience bullets;
- preserves exact dates, metrics, scale, team sizes, and outcomes;
- uses recognizable headers such as Summary, Experience, Education, and Skills;
- clearly associates title, employer, and dates;
- uses concise, parseable bullets and conventional structure;
- demonstrates job-relevant work instead of merely naming it.

Context-free lists receive little weight. Unsupported or stuffed lists receive negative weight.

If the versions have no meaningful ATS difference, return `"tie"`.

### Lens 2 — `human_skim`

Model a recruiter scanning the top third first, then titles, bolded role headings, and the first bullet of each role in an F-pattern.

Prefer the version that:
- surfaces the most relevant grounded title and evidence in the first screenful;
- uses a short, specific summary rather than generic positioning;
- leads each role with its strongest relevant accomplishment;
- preserves compelling metrics and concrete scope;
- uses short, evidence-led bullets;
- is easy to scan without repetition, jargon blobs, or walls of text.

A generic duty statement should not outrank a quantified accomplishment. A stuffed top-of-page competency block is harmful because it consumes prime space without establishing evidence.

If the versions have no meaningful human-skim difference, return `"tie"`.

## Combining the lenses

Apply these priorities:

1. Material grounding violations outweigh all apparent job alignment.
2. If grounding is comparable, major degradation of exact metrics, dates, scale, or readable structure can decide the result.
3. Keyword stuffing is not an ATS advantage. When substantive experience is otherwise equal, a stuffed version should lose to the cleaner version.
4. Otherwise, combine the two lenses and give more weight to the larger, better-evidenced difference.
5. Do not manufacture a preference. Ties are valid.
6. The verdict must remain unchanged if the same resume contents are relabeled or presented in reverse order.

## Margin calibration

- `"slight"`: a small but real difference, such as modest bullet ordering or one useful grounded skill-placement improvement.
- `"clear"`: multiple meaningful differences, a notable structural defect, lost quantified evidence, or a distinct keyword-stuffing problem.
- `"decisive"`: fabrication/inflation, widespread malformed or dequantified content, severe parsing damage, or several major defects all favoring one version.

Do not use `"decisive"` merely because one version is somewhat cleaner. If the overall winner is `"tie"`, use `"slight"` as the margin and state that no material distinction exists.

## Evidence requirements

Every decision must cite the exact difference that determines it:
- quote the relevant phrase or bullet when possible;
- identify the exact section, title, date, metric, or keyword cluster;
- explain whether it is grounded, dequantified, malformed, misplaced, duplicated, or stuffed.

Do not use generic claims such as “more aligned,” “cleaner,” or “better optimized” without naming the concrete evidence. Do not claim that a phrase is unsupported unless comparison with the original establishes that.

## Output

Return valid JSON only, with no Markdown or text outside the JSON:

{
  "ats_signal": {
    "winner": "A" | "B" | "tie",
    "evidence": "Specific content-based explanation citing the deciding difference."
  },
  "human_skim": {
    "winner": "A" | "B" | "tie",
    "evidence": "Specific content-based explanation citing the deciding difference."
  },
  "overall": {
    "winner": "A" | "B" | "tie",
    "margin": "slight" | "clear" | "decisive",
    "rationale": "Concise synthesis grounded in exact differences, with grounding and quality defects given priority."
  }
}