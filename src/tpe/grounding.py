"""Grounding-source composition. The judge's grounding input is the original resume
plus any candidate-confirmed facts discovered during the session (QA answers). Facts
are appended under a clear header so the judge treats them as legitimate grounding,
not as generator claims."""

FACTS_HEADER = ("## Additional facts confirmed by the candidate\n"
                "(Candidate-authored during a verification session. These facts are "
                "part of the grounding source: treat them as evidence equal to the "
                "resume above, NOT as unsupported additions.)")


def compose_grounding(original: str, discovered_facts: list[str] | None = None) -> str:
    facts = [f.strip() for f in (discovered_facts or []) if f and f.strip()]
    if not facts:
        return original
    seen: set[str] = set()
    unique = [f for f in facts if not (f in seen or seen.add(f))]
    lines = "\n".join(f"- {f}" for f in unique)
    return f"{original.rstrip()}\n\n{FACTS_HEADER}\n{lines}\n"
