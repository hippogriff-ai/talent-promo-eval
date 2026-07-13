"""OpenAI client wrapper + the model ladder. Model IDs live HERE and nowhere else."""
import json
import os
import time

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

# Pinned 2026-07-12 from live developers.openai.com pages (see docs/research-notes.md §6).
# Env override: TPE_MODEL_<TIER>. Prices per 1M tokens (input/output):
# nano gpt-5.4-nano $0.20/$1.25 · mini gpt-5.4-mini $0.75/$4.50
# mid gpt-5.6-terra $2.50/$15 · top gpt-5.6-sol $5/$30
_DEFAULT_LADDER = {
    "nano": "gpt-5.4-nano",
    "mini": "gpt-5.4-mini",
    "mid": "gpt-5.6-terra",
    "top": "gpt-5.6-sol",
}


def ladder() -> dict[str, str]:
    return {t: os.getenv(f"TPE_MODEL_{t.upper()}", m) for t, m in _DEFAULT_LADDER.items()}


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def complete_json(model: str, system: str, user: str, schema: dict) -> dict:
    """One structured-output call; returns the parsed JSON dict."""
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "verdict", "strict": True, "schema": schema}},
    )
    attempts = 4
    for attempt in range(1, attempts + 1):
        try:
            resp = client().chat.completions.create(**kwargs)
            return json.loads(resp.choices[0].message.content)
        except BadRequestError:
            raise  # schema/param problem: not transient, surface it
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2.0 ** attempt)  # 2s, 4s, 8s
