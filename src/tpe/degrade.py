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


def extract_keywords(job_text: str, top_n: int = 25) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", job_text)
    counts = Counter(w.lower().strip("./-") for w in words)
    return [w for w, _ in counts.most_common(top_n * 2) if w not in STOPWORDS][:top_n]


@dataclass(frozen=True)
class DegradeContext:
    jd_keywords: list[str]


def _kw_pattern(kw: str) -> str:
    """Keyword regex with boundary emulation that works for symbolic tokens (c++, c#):
    plain \\b fails after a trailing +/# because they are not word characters."""
    return rf"(?<![A-Za-z0-9_]){re.escape(kw)}(?![A-Za-z0-9_])"


def _kw_present(kw: str, html: str) -> bool:
    return re.search(rf"(?i){_kw_pattern(kw)}", html) is not None


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
    low = text.lower()
    return sum(low.count(kw) for kw in keywords)


# --- degradations ---

def keyword_strip(html: str, ctx: DegradeContext, severity: str) -> str:
    n = {"subtle": 2, "moderate": 5, "severe": 12}[severity]
    present = [kw for kw in ctx.jd_keywords if _kw_present(kw, html)]
    out = html
    for kw in present[:n]:
        out = re.sub(rf"(?i)\s*{_kw_pattern(kw)}", " relevant technologies", out)
        out = re.sub(r"(relevant technologies)(,?\s+relevant technologies)+", r"\1", out)
    return out


def header_flatten(html: str, ctx: DegradeContext, severity: str) -> str:
    out = re.sub(r"(?is)<h3\b[^>]*>(.*?)</h3>", r"<p><b>\1</b></p>", html)
    if severity == "severe":
        out = re.sub(r"(?is)<h2\b[^>]*>(.*?)</h2>", r"<p><b>\1</b></p>", out)
    return out


_NUM = re.compile(r"(?:\b(?:by|to)\s+)?\d[\d,.]*\s*(?:%|x|k|K|M|MM|\+)?\s*")


def dequantify(html: str, ctx: DegradeContext, severity: str) -> str:
    """Strip quantification from bullets only (dates in headers stay intact)."""
    frac = {"subtle": 0.34, "moderate": 0.67, "severe": 1.0}[severity]
    lis = list(re.finditer(r"(?is)<li>(.*?)</li>", html))
    numbered = [(i, nm) for i, m in enumerate(lis) for nm in _NUM.finditer(m.group(1))]
    n_remove = math.ceil(len(numbered) * frac)
    remove: dict[int, list[tuple[int, int]]] = {}
    for i, nm in numbered[:n_remove]:
        remove.setdefault(i, []).append(nm.span())
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
    has_digit = [bool(re.search(r"\d", li)) for li in items]
    if not (any(has_digit) and not all(has_digit)):
        return ul_inner
    first_digit = has_digit.index(True)
    if not any(not d for d in has_digit[first_digit:]):
        return ul_inner  # every non-quantified bullet already leads; nothing to demote
    ranked = sorted(items, key=lambda li: 1 if re.search(r"\d", li) else 0)
    return "".join(ranked)


def bland_leads(html: str, ctx: DegradeContext, severity: str) -> str:
    uls = list(re.finditer(r"(?is)<ul>(.*?)</ul>", html))
    if not uls:
        return html
    limit = 1 if severity == "subtle" else len(uls)
    out, last = [], 0
    for i, m in enumerate(uls):
        out.append(html[last:m.start(1)])
        out.append(_sort_bullets(m.group(1)) if i < limit else m.group(1))
        last = m.end(1)
    out.append(html[last:])
    return "".join(out)


def wall_of_text(html: str, ctx: DegradeContext, severity: str) -> str:
    uls = list(re.finditer(r"(?is)<ul>(.*?)</ul>", html))
    if not uls:
        return html
    limit = 1 if severity == "moderate" else len(uls)
    out, last = [], 0
    for i, m in enumerate(uls):
        out.append(html[last:m.start(1)])
        items = re.findall(r"(?is)<li>(.*?)</li>", m.group(1))
        if i < limit and len(items) >= 2:  # merging one bullet degrades nothing
            out.append("<li>" + " ".join(x.strip().rstrip(".") + "." for x in items) + "</li>")
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
    missing = [kw for kw in ctx.jd_keywords if not _kw_present(kw, html)][:n]
    if not missing:
        return html
    blob = "<h2>Core Competencies</h2><p>" + ", ".join(
        kw.title() for kw in (ctx.jd_keywords[:n] + missing)) + "</p>"
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
