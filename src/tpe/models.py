"""OpenAI client wrapper + the model ladder. Model IDs live HERE and nowhere else."""
import json
import os
import threading
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

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

TIERS: tuple[str, ...] = tuple(_DEFAULT_LADDER)  # capability order, low -> high


def ladder() -> dict[str, str]:
    return {t: os.getenv(f"TPE_MODEL_{t.upper()}", m) for t, m in _DEFAULT_LADDER.items()}


_client: OpenAI | None = None
_client_lock = threading.Lock()


def client() -> OpenAI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenAI()
    return _client


class JudgeRefusal(RuntimeError):
    """The model refused to produce a verdict (content policy). Not retryable."""


# The SDK itself already retries 408/429/5xx/connection errors (max_retries=2) with
# Retry-After support; this outer loop exists for failures that outlive the SDK's
# retries. AuthenticationError is included deliberately: the API has been observed
# returning transient 401s under sustained concurrency (see CONTINUITY ops notes).
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError,
             InternalServerError, AuthenticationError)


def complete_json(model: str, system: str, user: str, schema: dict) -> dict:
    """One structured-output call; returns the parsed JSON dict.

    Non-retryable errors (schema bugs, bad params, refusals) surface immediately.
    """
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "verdict", "strict": True, "schema": schema}},
    )
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            resp = client().chat.completions.create(**kwargs)
            message = resp.choices[0].message
            if message.content is None:
                refusal = getattr(message, "refusal", None) or "no content returned"
                raise JudgeRefusal(f"model {model} refused: {refusal}")
            return json.loads(message.content)
        except RETRYABLE:
            if attempt == attempts:
                raise
            time.sleep(2.0 ** attempt)  # 2s, 4s
