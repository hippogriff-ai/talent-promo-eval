You are comparing two versions of the same candidate’s resume, both tailored to the same job posting. Judge only which resume is the better artifact for passing:

1. a modern ATS/AI screening system, and
2. a recruiter’s approximately seven-second skim.

You are not judging the candidate. The ORIGINAL_RESUME is the sole source of truth; both versions describe the same person.

## Inputs

You may receive metadata such as `pair_id` or a job excerpt in addition to the materials below. Do not infer the winner from metadata, version order, labels, or a degradation name. Base the verdict only on the actual documents.

### Job posting
{{JOB_POSTING}}

### Candidate’s original resume
{{ORIGINAL_RESUME}}

### Version A
{{RESUME_A}}

### Version B
{{RESUME_B}}

## Required evaluation procedure

Perform the following comparison internally before producing the JSON. Apply the same tests to A and B symmetrically so the result would not change if their labels or presentation order were swapped.

### Step 1 — Ground every claim

For each version, compare titles, employers, dates, technologies, responsibilities, metrics, scope, seniority, and accomplishments against the ORIGINAL_RESUME.

Treat these as serious grounding violations:

- invented technologies, responsibilities, projects, or accomplishments;
- new or inflated metrics, scale, revenue, user counts, performance gains, or team size;
- longer tenure or altered dates;
- inflated titles or seniority;
- moving an accomplishment from one role to another when the original does not support that attribution;
- converting weak/general exposure into claimed expertise or production ownership;
- job-description language inserted as if it were demonstrated experience;
- boilerplate such as “leveraging X expertise” when the original supplies no evidence.

A context-free Skills section cannot legitimize an unsupported claim. Repetition of job-posting terms is not evidence.

Distinguish unsupported additions from omissions:

- An unsupported addition is a grounding violation.
- Removing a metric or technology is not a grounding violation, but it can materially weaken ATS signal or human credibility.
- Mangled replacements such as missing-number fragments, placeholder-like text, or grammatically broken metrics are worse than clean omission because they impair extraction and credibility.

Grounding violations override job alignment. A version that appears more tailored only because it fabricates or inflates evidence must lose to an honest counterpart.

### Step 2 — Evaluate `ats_signal`

Modern screeners do more than count keywords. They parse sections and dated roles, infer recency and duration, connect skills to evidence, infer seniority from titles and dates, and rank semantic similarity to the posting.

Prefer the version that:

- places the posting’s actual technologies, methods, and domain vocabulary inside dated experience or project bullets with support from the original;
- preserves concrete, machine-extractable evidence such as metrics, latency, throughput, scale, dates, deployment scope, and named systems;
- uses recognizable headers such as Summary, Experience, Skills, Projects, and Education;
- presents titles, employers, and dates consistently;
- keeps bullets and role boundaries structurally distinct;
- makes it easy to associate each skill or accomplishment with the correct role and date.

Apply these limits:

- A dedicated Skills section can help extraction, but it is a secondary signal. Do not award a meaningful ATS win merely because one version has a longer skills list.
- Skills such as Python, AWS, LLMs, structured outputs, MCP, orchestration, or other job-specific terms count most when supported in dated bullets; a stuffed skills blob counts little and can hurt.
- Do not reward raw keyword frequency, repeated job-description phrases, or synonyms added without evidence.
- Do not claim an ATS parsing penalty unless the visible structure actually makes roles, dates, sections, or evidence harder to associate.
- Dequantification is an ATS weakness even when all remaining text is true: deleting supported numbers removes extractable evidence of impact and scale.
- Wall-of-text formatting may reduce ATS clarity when separate accomplishments or role associations are collapsed, but its strongest effect is usually on the human skim.

### Step 3 — Evaluate `human_skim`

Model a recruiter who spends about seven seconds, focuses first on the top third, fixates on titles and employers, and scans bold text and the first bullet of each role in an F-pattern.

Prefer the version that:

- places the most recent and job-relevant role, title, and evidence in the first screenful;
- uses a tight, factual summary rather than generic positioning language;
- leads each role with its strongest, most quantified, and most job-relevant accomplishment;
- uses short, distinct, evidence-led bullets;
- exposes metrics and named technologies where the eye can find them quickly;
- uses clear section headers and consistent chronology.

Penalize:

- burying a current or highly relevant role below older, less relevant roles;
- leading with generic duties while stronger evidence is hidden later;
- merging several bullets into dense paragraphs or wall-of-text blocks;
- compressed or malformed sections that obscure titles, dates, or accomplishments;
- keyword-heavy summaries or skills blobs that push dated evidence below the fold;
- broken numeric phrases or placeholder-like wording in high-salience bullets.

Important calibration:

- If two versions contain the same facts but one buries the newest, most relevant role, that is a real human-skim disadvantage even if ATS can still parse both. If the rest is nearly identical, the overall margin is usually slight.
- If one version turns distinct evidence-led bullets into dense wall-of-text paragraphs, treat that as a substantial human-skim difference. Do not downgrade it to a trivial preference merely because the words are unchanged. If this affects multiple roles or the top third, it commonly supports a clear overall margin.
- If one version removes several supported metrics or corrupts them into fragments, that harms both machine extraction and human credibility. Multiple high-salience instances can justify a clear or decisive margin.

### Step 4 — Compare, do not score in isolation

For each lens, identify the exact feature that differs between A and B. Do not give a version credit for a strength both versions share.

Before selecting a winner, internally verify:

- Would the same document still win if A and B were relabeled?
- Is the deciding evidence visible in the supplied text?
- Am I rewarding supported evidence rather than keyword count?
- Am I confusing a Skills section with dated proof?
- Am I overlooking bullet order, role order, deleted metrics, or wall-of-text formatting?
- Did I incorrectly call an omission a fabrication?
- Did I state that both versions are grounded without actually checking their claims against the original?

If the versions do not genuinely differ on a lens, return `"tie"` for that lens. Do not manufacture a preference.

## Overall decision

Grounding dominates. Otherwise combine the two lenses according to the severity and quality of the documented differences, not by mechanically counting lens wins.

Use margins consistently:

- `"slight"`: a localized or moderate advantage; underlying evidence is mostly the same, such as burying one relevant role while preserving the content.
- `"clear"`: a material advantage visible across a lens or in multiple high-value locations, such as widespread wall-of-text formatting, several removed metrics, or substantially better top-third prioritization.
- `"decisive"`: major fabrication/inflation, pervasive corruption, or severe high-salience damage affecting both ATS extraction and recruiter comprehension.

Do not use `"decisive"` merely because one version is somewhat cleaner. If the overall winner is `"tie"`, use `"slight"` as the margin solely to satisfy the required schema and explain that no substantive difference exists.

## Evidence requirements

Every verdict must cite concrete differences. Quote or name the exact bullet, role, date, metric, section, keyword placement, or formatting change that decided the lens.

Avoid unsupported generic statements such as:

- “A is more professional.”
- “B has better ATS optimization.”
- “A is cleaner.”
- “Both are grounded.”

Instead explain precisely, for example:

- one version places the May 2025–Present AI role before older positions;
- one preserves a supported latency or scale metric that the other deletes;
- one keeps three accomplishments as separate bullets while the other merges them into one paragraph;
- one lists a technology only in Skills while the other also demonstrates it in a dated role;
- one introduces a title, date, metric, or technology absent from the original.

## Output

Return one valid JSON object only. Do not use Markdown or add commentary outside the JSON.

{
  "ats_signal": {
    "winner": "A" | "B" | "tie",
    "evidence": "Specific, comparative evidence naming the exact difference that decided this lens."
  },
  "human_skim": {
    "winner": "A" | "B" | "tie",
    "evidence": "Specific, comparative evidence naming the exact difference that decided this lens."
  },
  "overall": {
    "winner": "A" | "B" | "tie",
    "margin": "slight" | "clear" | "decisive",
    "rationale": "Concise synthesis explaining how grounding and the two lenses determine the result, with concrete document references."
  }
}