"""Job posting input: URL, file path, or raw text -> plain text (URLs cached on disk)."""
import hashlib
import re
from pathlib import Path

import httpx


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", html)
    html = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)\b[^>]*>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&nbsp;", " ").replace("&#39;", "'"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def job_text(source: str, cache_dir: Path = Path("data/jobs")) -> str:
    if source.startswith(("http://", "https://")):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / (hashlib.sha256(source.encode()).hexdigest()[:16] + ".txt")
        if cached.exists():
            return cached.read_text()
        resp = httpx.get(source, follow_redirects=True, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0 (resume-eval)"})
        resp.raise_for_status()
        text = html_to_text(resp.text)
        cached.write_text(text)
        return text
    if len(source) < 512 and "\n" not in source and Path(source).exists():
        return Path(source).read_text()
    return source
