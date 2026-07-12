"""OpenAI client wrapper + the model ladder. Model IDs live HERE and nowhere else."""
import json
import os
import time

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

# Pinned from docs/research-notes.md (tech-facts report). Env override: TPE_MODEL_<TIER>.
_DEFAULT_LADDER = {
    "nano": "gpt-5-nano",
    "mini": "gpt-5-mini",
    "mid": "gpt-5",
    "top": "gpt-5.6",
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
    for attempt in (1, 2):
        try:
            resp = client().chat.completions.create(**kwargs)
            return json.loads(resp.choices[0].message.content)
        except BadRequestError:
            raise  # schema/param problem: not transient, surface it
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0)
