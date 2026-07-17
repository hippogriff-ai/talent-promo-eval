"""Deterministic resume degradations. Each produces a strictly-worse version along one lens,
tagged so a judge miss can be attributed to the quality difference it failed to catch."""
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

STOPWORDS = frozenset(
    "the a an and or for with to of in on at by from as is are was were be been being "
    "this that these those you your we our it its they their will can may should must "
    "have has had do does did not no yes if then than more most other others such only "
    "who whom whose which what when where how all any both each few own same so too very "
    "required preferred experience years work team role including strong ability".split()
)


# Short standalone tech tokens the length filter would drop. Matched CASE-SENSITIVELY
# against the JD ("Go" the language, not "go" the verb) and downstream in resumes.
SHORT_TECH = {
    "go": r"\bGo\b", "r": r"\bR\b(?!&)", "c": r"\bC\b(?![+#])",
    "c#": r"\bC#(?![A-Za-z0-9_+])",
    "ai": r"\bAI\b", "ml": r"\bML\b", "ci": r"\bCI\b", "cd": r"\bCD\b",
    "qa": r"\bQA\b", "ui": r"\bUI\b", "ux": r"\bUX\b", "k8s": r"\b[Kk]8s\b",
}


def extract_keywords(job_text: str, top_n: int = 25) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", job_text)
    # "Python/Java" is two skills, not one token the boundary matcher can never find.
    # Short whitelisted parts (R/Python, C/C++) survive the length filter.
    parts = [p for w in words for p in w.split("/")]
    cleaned = [p.strip("./-") for p in parts]
    counts = Counter(p.lower() for p in cleaned
                     if len(p) >= 3 or p.lower() in SHORT_TECH)
    kws = [w for w, _ in counts.most_common(top_n * 2)
           if w not in STOPWORDS and w not in SHORT_TECH][:top_n]
    kws += [tok for tok, pat in SHORT_TECH.items() if tok not in kws
            and (tok in counts or re.search(pat, job_text))]
    return kws


@dataclass(frozen=True)
class DegradeContext:
    jd_keywords: list[str]
    source_text: str = ""  # the original profile: the grounding source for trap pairs


def _kw_pattern(kw: str) -> str:
    """Keyword regex with boundary emulation that works for symbolic tokens (c++, c#):
    plain \\b fails after a trailing +/# because they are not word characters.
    SHORT_TECH tokens stay case-sensitive everywhere ("Go" the language != "go")."""
    if kw in SHORT_TECH:
        return SHORT_TECH[kw]
    return rf"(?<![A-Za-z0-9_])(?i:{re.escape(kw)})(?![A-Za-z0-9_])"


def _kw_present(kw: str, html: str) -> bool:
    return re.search(_kw_pattern(kw), html) is not None


# --- html section helpers (tolerant regex splitting; resumes are simple h2/h3/ul html) ---

def split_h2(html: str) -> list[str]:
    """Split into blocks, each starting at an <h2> (first block may be preamble)."""
    parts = re.split(r"(?i)(?=<h2\b)", html)
    return [p for p in parts if p.strip()]


def h2_title(block: str) -> str:
    m = re.search(r"(?is)<h2\b[^>]*>(.*?)</h2>", block)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip().lower() if m else ""


def split_h3(block: str) -> tuple[str, list[str]]:
    """Within a section: (header part, list of h3 role blocks)."""
    parts = re.split(r"(?i)(?=<h3\b)", block)
    if len(parts) == 1:
        return block, []
    return parts[0], [p for p in parts[1:] if p.strip()]


def _keyword_density(text: str, keywords: list[str]) -> int:
    # Boundary-aware: a substring count would score "api" inside "capitalization".
    return sum(len(re.findall(_kw_pattern(kw), text)) for kw in keywords)


# Metric-like numbers only: a digit embedded in a token (OAuth2, S3, EC2) is a
# technology name, not quantification — mangling it would corrupt pair labels.
_NUM = re.compile(r"(?:\b(?:by|to)\s+)?(?<![A-Za-z0-9.])\d[\d,.]*(?:\s?(?:%|x|k|K|M|MM|\+))?(?![A-Za-z0-9])")


def _metric_spans(text: str) -> list[tuple[int, int]]:
    """Metric matches, excluding numbers that NAME things: an all-caps acronym
    directly before the number (SOC 2, ISO 27001, PCI-DSS 4) is a standard/cert,
    and rewriting it would mangle a grounded keyword, not remove quantification."""
    spans = []
    for m in _NUM.finditer(text):
        # The acronym exemption applies only to BARE numbers: "SOC 2"/"ISO 27001"
        # name a standard, but "API 10M requests" is a scale metric — a magnitude
        # suffix means quantification regardless of what precedes it.
        bare = not re.search(r"(?:%|x|k|K|M|MM|\+)\s*$", m.group(0).rstrip())
        if bare and re.search(r"\b[A-Z]{2,}[ \-]?$", text[:m.start()]):
            continue
        spans.append(m.span())
    return spans


def _has_metric(text: str) -> bool:
    return bool(_metric_spans(text))


# --- degradations ---

def keyword_strip(html: str, ctx: DegradeContext, severity: str) -> str:
    n = {"subtle": 2, "moderate": 5, "severe": 12}[severity]
    present = [kw for kw in ctx.jd_keywords if _kw_present(kw, html)]
    out = html
    for kw in present[:n]:
        out = re.sub(rf"\s*(?:{_kw_pattern(kw)})", " relevant technologies", out)
        out = re.sub(r"(relevant technologies)(,?\s+relevant technologies)+", r"\1", out)
    return out


def header_flatten(html: str, ctx: DegradeContext, severity: str) -> str:
    out = re.sub(r"(?is)<h3\b[^>]*>(.*?)</h3>", r"<p><b>\1</b></p>", html)
    if severity == "severe":
        out = re.sub(r"(?is)<h2\b[^>]*>(.*?)</h2>", r"<p><b>\1</b></p>", out)
    return out


def dequantify(html: str, ctx: DegradeContext, severity: str) -> str:
    """Strip quantification from bullets only (dates in headers stay intact)."""
    frac = {"subtle": 0.34, "moderate": 0.67, "severe": 1.0}[severity]
    lis = list(re.finditer(r"(?is)<li>(.*?)</li>", html))
    numbered = [(i, span) for i, m in enumerate(lis) for span in _metric_spans(m.group(1))]
    n_remove = math.ceil(len(numbered) * frac)
    remove: dict[int, list[tuple[int, int]]] = {}
    for i, span in numbered[:n_remove]:
        remove.setdefault(i, []).append(span)
    out, last = [], 0
    for i, m in enumerate(lis):
        inner = m.group(1)
        if i in remove:
            buf, ilast = [], 0
            for s, e in remove[i]:
                buf.append(inner[ilast:s])
                matched = inner[s:e].lstrip().lower()
                buf.append("substantially " if matched.startswith(("by ", "to ")) else "several ")
                ilast = e
            buf.append(inner[ilast:])
            inner = re.sub(r"[ \t]{2,}", " ", "".join(buf))
        out.append(html[last:m.start(1)])
        out.append(inner)
        last = m.end(1)
    out.append(html[last:])
    return "".join(out)


def bury_relevant(html: str, ctx: DegradeContext, severity: str) -> str:
    sections = split_h2(html)
    for idx, sec in enumerate(sections):
        if "experience" not in h2_title(sec):
            continue
        head, roles = split_h3(sec)
        if len(roles) < 2:
            return html
        # Without one uniquely JD-relevant role there is nothing to "bury" — moving an
        # arbitrary role would create a pair that is not worse by construction.
        dens = [_keyword_density(r, ctx.jd_keywords) for r in roles]
        if max(dens) == 0 or dens.count(max(dens)) > 1:
            return html
        hot = dens.index(max(dens))
        rest = [r for i, r in enumerate(roles) if i != hot]
        if severity == "severe":
            rest = list(reversed(rest))  # additionally break the remaining chronology
        # Both severities GUARANTEE the hot role sinks to last (a blind full reversal
        # could promote a hot role that wasn't first, mislabeling the pair).
        reordered = rest + [roles[hot]]
        sections[idx] = head + "".join(reordered)
        return "".join(sections)
    return html


def _sort_bullets(ul_inner: str) -> str:
    items = re.findall(r"(?is)<li>.*?</li>", ul_inner)
    if len(items) < 2:
        return ul_inner
    # Emit a change ONLY when a quantified bullet is genuinely demoted below a
    # non-quantified one: all-quantified, all-unquantified, or already-bland-led
    # lists must pass through unchanged (reordering same-class bullets is not
    # worse by construction). Stable sort preserves within-class order.
    has_digit = [_has_metric(li) for li in items]
    if not (any(has_digit) and not all(has_digit)):
        return ul_inner
    first_digit = has_digit.index(True)
    if not any(not d for d in has_digit[first_digit:]):
        return ul_inner  # every non-quantified bullet already leads; nothing to demote
    ranked = sorted(items, key=lambda li: 1 if _has_metric(li) else 0)
    return "".join(ranked)


def bland_leads(html: str, ctx: DegradeContext, severity: str) -> str:
    uls = list(re.finditer(r"(?is)<ul>(.*?)</ul>", html))
    if not uls:
        return html
    budget = 1 if severity == "subtle" else len(uls)
    out, last, changed = [], 0, 0
    for m in uls:
        out.append(html[last:m.start(1)])
        inner = m.group(1)
        if changed < budget:  # budget counts CHANGED lists, not raw indices
            new = _sort_bullets(inner)
            if new != inner:
                changed += 1
                inner = new
        out.append(inner)
        last = m.end(1)
    out.append(html[last:])
    return "".join(out)


def wall_of_text(html: str, ctx: DegradeContext, severity: str) -> str:
    uls = list(re.finditer(r"(?is)<ul>(.*?)</ul>", html))
    if not uls:
        return html
    budget = 1 if severity == "moderate" else len(uls)
    out, last, merged = [], 0, 0
    for m in uls:
        out.append(html[last:m.start(1)])
        items = re.findall(r"(?is)<li>(.*?)</li>", m.group(1))
        if merged < budget and len(items) >= 2:  # budget counts MERGED lists only
            out.append("<li>" + " ".join(x.strip().rstrip(".") + "." for x in items) + "</li>")
            merged += 1
        else:
            out.append(m.group(1))
        last = m.end(1)
    out.append(html[last:])
    return "".join(out)


def drop_summary(html: str, ctx: DegradeContext, severity: str) -> str:
    sections = split_h2(html)
    kept = [s for s in sections
            if not any(k in h2_title(s) for k in ("summary", "profile", "objective"))]
    return "".join(kept) if len(kept) < len(sections) else html


def keyword_stuff(html: str, ctx: DegradeContext, severity: str) -> str:
    n = {"subtle": 5, "moderate": 10, "severe": 15}[severity]
    # A keyword grounded in the ORIGINAL profile is a legitimate addition, not a
    # trap: stuffing must use only terms the candidate's materials cannot support.
    missing = [kw for kw in ctx.jd_keywords
               if not _kw_present(kw, html)
               and not _kw_present(kw, ctx.source_text)][:n]
    if not missing:
        return html
    # The blob carries ONLY unsupported terms: mixing in grounded keywords would make
    # the "trap" partially legitimate and its known-worse label unreliable.
    blob = "<h2>Core Competencies</h2><p>" + ", ".join(
        kw.title() for kw in missing) + "</p>"
    out = html + blob
    if severity != "subtle":
        stuffer = iter(missing * 3)
        out = re.sub(r"(?is)</li>",
                     lambda m: f", leveraging {next(stuffer, 'modern tooling')} expertise.</li>",
                     out, count=3)
    return out


@dataclass(frozen=True)
class Degradation:
    name: str
    lens: str  # "ats" | "human_skim" | "trap"
    severities: tuple[str, ...]
    fn: Callable[[str, DegradeContext, str], str]


DEGRADATIONS: list[Degradation] = [
    Degradation("keyword_strip", "ats", ("subtle", "moderate", "severe"), keyword_strip),
    Degradation("header_flatten", "ats", ("moderate", "severe"), header_flatten),
    Degradation("dequantify", "ats", ("subtle", "moderate", "severe"), dequantify),
    Degradation("bury_relevant", "human_skim", ("moderate", "severe"), bury_relevant),
    Degradation("bland_leads", "human_skim", ("subtle", "moderate"), bland_leads),
    Degradation("wall_of_text", "human_skim", ("moderate", "severe"), wall_of_text),
    Degradation("drop_summary", "human_skim", ("moderate",), drop_summary),
    Degradation("keyword_stuff", "trap", ("subtle", "moderate", "severe"), keyword_stuff),
]
