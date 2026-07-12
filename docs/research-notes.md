# Research Notes: What Makes a Resume "Good" in Tech (2025–2026)

Evidence base for the judge's taste guide (`taste-guide.md`). Collected 2026-07-12 via a
fan-out research pass over 23 sources (5 search angles: ATS mechanics, eye-tracking
research, hiring-manager practice, myth-busting, LLM-judge research).

**Confidence labels:**
- **VERIFIED** — survived 3-vote adversarial verification (independent refutation attempts).
- **SOURCED** — extracted from the named source with verbatim quotes; adversarial
  verification did not complete (infrastructure limits), so treat as single-pass.
- **REFUTED** — killed in adversarial verification; recorded so we don't re-adopt it.

---

## 1. AI scanners / ATS — how ranking actually works

### Match signal is semantic, not keyword-count (VERIFIED)
Modern embedding-based resume ranking computes cosine similarity between
transformer-generated embeddings of the resume and the job description — not keyword
overlap counts. Semantically equivalent phrasing scores similarly even without exact
keyword matches. (Resume2Vec, six models incl. BERT/RoBERTa/GPT/Gemini/Llama.)
— https://www.mdpi.com/2079-9292/14/4/794

### Embedding ranking tracks human judgment better than keyword ATS (VERIFIED)
Embedding-based ranking aligned with human rankings better than a traditional
keyword ATS in most domains (nDCG up to +15.85%, RBO up to +15.94%). Keyword-match
score and human-judged fit are measurably different signals.
— https://www.mdpi.com/2079-9292/14/4/794

**REFUTED:** the inverse claim that keyword ATS retains an edge specifically in
software-adjacent domains (Software Testing nDCG 0.96 vs 0.90) did not survive
verification — do not build the rubric on "literal keyword coverage beats semantics
for tech roles."

### Parsers reward structure; layout traps are real (SOURCED)
Textkernel's production parser (a market-leading resume parser):
- Multi-column layouts are flagged as fatal-level machine-readability findings
  (code 433); single-column top-to-bottom is the parser-safe format.
- Parsing begins by splitting the document into *recognized sections*; unrecognizable
  section headers trigger fatal findings (codes 412–416, e.g. no WORK HISTORY found).
  Conventional headers directly determine extraction success.
- Extracted skills carry *months-of-use, last-used date, and position linkage*: a skill
  evidenced inside dated work history yields recency/duration metadata that a skill
  in a bare skills list does not.
- Seniority is machine-inferred per position (8-level enum) from titles + date ranges.
— https://developer.textkernel.com/tx-platform/v9/resume-parser/overview/parser-output/

### Skill placement drives credited experience (SOURCED)
Some ATS score skill strength by frequency; others infer years-of-experience for a
skill from which job entry it appears under (a skill under a 3-year job is credited
~3 years). Keywords embedded in dated work history beat keywords in a skills blob.
— https://www.techinterviewhandbook.org/resume/

### Keyword stuffing backfires even against machines (SOURCED)
- Modern screeners prioritize keywords appearing in context within work descriptions
  over disconnected skill lists; stuffing can *reduce* ranking.
  — https://www.interviewpal.com/blog/your-resume-keywords-are-getting-you-rejected-heres-why
- Against LLM-based screeners, invisible keyword stuffing was the *least* effective
  manipulation (6.6% attack success on GPT-4o Mini) vs semantic manipulation (97.5%).
  LLM rankers respond to semantic content, not raw keyword frequency.
  — https://arxiv.org/html/2512.20164v1

### Length is a mechanical variable (SOURCED)
Embedding matchers are sensitive to document length independent of qualifications —
resume length itself affects retrieval-based selection.
— https://arxiv.org/abs/2407.20371

### File format (SOURCED, practitioner test)
In an 8-ATS parse test, DOCX out-parsed PDF in 6 of 8 systems; design-tool PDF
exports (Canva/Illustrator) were the worst failure mode (text embedded as objects).
— https://quickcv.io/blog/i-tested-8-ats-systems-to-see-how-they-actually-parse-resumes

---

## 2. Human recruiters / HR partners — the first-pass skim

### The 6–7.4 second skim (SOURCED, primary PDF)
Ladders' 2018 eye-tracking study: average initial screen 7.4 seconds (up from 6s in
the 2012 study). Methodology caveats: stage 1 timed recruiters unaware of timing;
stage 2 lab eye-tracking; no sample size stated in the report.
— https://www.theladders.com/static/images/basicSite/pdfs/TheLadders-EyeTracking-StudyC2.pdf

### What wins the skim (SOURCED, same study)
- Top-performing resumes: recruiters fixated on **job titles** more than any other
  element; clear simple layouts with clearly marked section/title headers.
- Layouts matching F-/E-pattern reading (bold job titles + bulleted accomplishment
  lists), an overview/mission at the top of page 1; even left-side scanning.
- Worst performers: cluttered look (long sentences, multiple columns, little white
  space), layout failing to draw the eye down the page, and **visible keyword
  stuffing** — the study explicitly warns stuffing that may help ATS hurts with
  humans; keywords should appear in context.

### Gaze behavior predicts screening outcomes (SOURCED, academic)
221 recruiters × 2,043 CS resumes: a Random Forest on eye-tracking features alone
predicted advance/reject with AUC 0.767 — screening decisions are systematically
encoded in gaze, not idiosyncratic. Most predictive features: fixations on
whitespace (deliberation), total viewing time, and time on **Experience and
Education** sections. Longer viewing time correlates with approval: resumes that
hold attention and prompt deliberation get moved forward.
— https://www.mdpi.com/2504-4990/5/3/38

**Judge-relevant reading:** the top-third of the document and the first bullets under
the most recent role are where the 7.4 seconds go; whether key evidence lands there
discriminates versions.

---

## 3. Hiring managers — the deeper read (software/AI roles)

### Responsibilities + accomplishments, most omit the latter (SOURCED)
An effective engineering resume needs both what the job was and what the results
were; the majority of candidates omit accomplishments. Version with concrete
outcomes > version with duties only.
— https://jacobian.org/tags/resumes/

### Adjectives without evidence are a red flag (SOURCED, practitioner forum)
Hiring managers treat self-descriptive adjectives ("passionate", "strong",
"self-motivated") combined with absent performance data as a negative signal — the
same candidate scores worse when bullets are adjective-led rather than evidence-led.
— https://news.ycombinator.com/item?id=16982575

### Practitioner canon (SOURCED)
The Tech Resume Inside Out (ex-Uber EM, input from 20+ tech recruiters/HMs): lead
with results, quantify impact, most-relevant experience first.
— https://thetechresume.com/

---

## 4. LLM-judge design implications (directly about our judge)

### Pairwise resume judging works, and capability scales with gap size (SOURCED)
LLMs constructing/judging directly comparable resumes with known ground-truth
superiority: Claude Sonnet 4 criterion validity 0.87 at a 1-qualification gap →
0.99 at 3 (avg 0.96); GPT-5: 0.83/0.93/0.98. **Weaker models fail at small gaps:
Llama 3.3 70B scored 44% (below chance) at 1-qualification gaps.** This is the
empirical basis for our tier-sensitivity gate: subtle pairs separate strong from
weak judge models.
— https://arxiv.org/abs/2602.18550

### Ties need explicit handling (SOURCED)
Most models' discriminant validity < 0.9 when candidates are equally qualified —
forced choices between equals are >10% arbitrary. The judge schema includes "tie"
and the rubric must legitimize choosing it.
— https://arxiv.org/abs/2602.18550

### Position bias is worst exactly on close calls (SOURCED)
Position bias in pairwise LLM judging is driven mostly by the quality gap between
candidates: when the two are close, positional preference dominates. Order-swapped
double evaluation matters most on subtle pairs — which is where our flip-rate
penalty bites.
— https://arxiv.org/abs/2406.07791

### Model disagreement without calibration (SOURCED)
Three LLM architectures fully agreed on only 20.7% of 463 job-candidate match
classifications (Fleiss' kappa 0.079). An uncalibrated judge's verdicts are
model-dependent — hence known-answer pairs + GEPA rather than trusting any prompt
as written.
— https://arxiv.org/html/2512.20164v1

---

## 5. Folklore vs evidence

| Claim | Status | Evidence |
|---|---|---|
| "75% of resumes are auto-rejected by ATS before a human sees them" | **Folklore** | Traced to a 2012 sales pitch by Preptel (defunct 2013); no methodology ever published. — https://unchartedcareer.com/blog/the-75-of-resumes-are-auto-rejected-myth-traced-to-its-source |
| "ATS auto-reject resumes on formatting/design" | **Mostly folklore** | 92% of 25 U.S. recruiters interviewed (2025) across Workday/iCIMS/Greenhouse/Bullhorn said their systems do NOT auto-reject on those grounds. Humans reject; parsers just garble. — https://enhancv.com/blog/does-ats-reject-resumes/ |
| "Beat the ATS by stuffing keywords" | **Backfires** | Hurts with humans (Ladders) and with modern semantic/LLM rankers (arXiv 2512.20164; interviewpal). |
| "Recruiters read resumes for 6–7.4 seconds" | **Supported with caveats** | Real study (Ladders 2012/2018) but measures the *initial skim*, not the full decision; methodology thin. The design implication (top-third placement matters) holds. |
| "Keyword coverage of the JD is what gets you ranked" | **Half-true** | Exact-match layers still exist in older ATS, but modern rankers are semantic; contextual evidence beats frequency. Grounded JD vocabulary still helps — stuffed vocabulary doesn't. |

## 6. Infrastructure facts (pinned 2026-07-12, live-verified)

**OpenAI judge ladder** (from developers.openai.com; all figures fetched live, per-1M-token
input/output pricing):

| Tier | Model ID | Price in/out | Context |
|---|---|---|---|
| nano | `gpt-5.4-nano` | $0.20 / $1.25 | 400K |
| mini | `gpt-5.4-mini` | $0.75 / $4.50 | 400K |
| mid | `gpt-5.6-terra` | $2.50 / $15 | 1.05M |
| top | `gpt-5.6-sol` (alias `gpt-5.6`) | $5 / $30 | 1.05M |

Notes: "GPT-5.6" is a family (Sol/Terra/Luna); there is no gpt-5.6-mini/nano — Terra/Luna
fill those roles. `gpt-5-mini` is superseded by `gpt-5.4-mini`. Long-context surcharge
(>272K input) bills 2×/1.5× — irrelevant at our ~10K-token calls.

**gepa package**: PyPI 0.1.1 (requires-python >=3.10,<3.15). Candidate = `dict[str,str]`;
`GEPAAdapter` protocol = `evaluate(batch, candidate, capture_traces) -> EvaluationBatch(outputs, scores, trajectories)`
+ `make_reflective_dataset(candidate, eval_batch, components_to_update) -> {component: [{"Inputs","Generated Outputs","Feedback"}]}`.
`gepa.optimize(seed_candidate, trainset, valset, adapter=..., reflection_lm="openai/<id>",
max_metric_calls=N)`; with a custom adapter, `task_lm`/`evaluator` must be None; scores
higher-is-better; never raise on a single-example failure (return 0.0 + error in trajectory).
Sources: gepa-ai/gepa README + src/gepa/{api.py,core/adapter.py} @ main, PyPI JSON.

## Corpus-side note

Claims about *our* generator's failure modes (tenure inflation, metric transplants,
scope conflation) come from the human-coded corpus in `data/corpus/` — see
`taxonomy.md` there. The judge's anti-fabrication clause exists because the
original resume is the only grounding source available at judge time.
