"""Small helpers shared by the agents: model calls with retries, reading model replies, building the OpenRouter models.

Same behaviour as in Homework 2 (copied so that this folder runs on its own).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1"
RETRY_SLEEP = [10, 20, 40, 60]          # seconds between retries of a transient provider error
STATS = {"model_calls": 0}              # every request sent to a provider, retries included (free tier: 20/min, 50/day)


def emit(msg: str = "") -> None:
    """A live line for the terminal (the runner shows the run as it happens)."""
    print(msg, flush=True)


# ----------------------------------------------------------------------------- model calls
def _status(e: Exception):
    return getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)


def daily_quota(e: Exception) -> bool:
    m = str(e).lower()
    return "per-day" in m or "per day" in m


def is_transient(e: Exception) -> bool:
    """Worth retrying? With an HTTP status the status decides (a 400/404 from a model that lacks a feature must not be
    retried: every retry spends a request of the free quota). Without a status, look at the message."""
    if daily_quota(e):
        return False
    s = _status(e)
    if s is not None:
        return s in (408, 409, 425, 429, 500, 502, 503, 504)
    if isinstance(e, TypeError) and "nonetype" in str(e).lower():   # provider returned an error body, no choices
        return True
    m = str(e).lower()
    return any(k in m for k in ("rate limit", "rate-limit", "timed out", "timeout", "overloaded",
                                "temporarily", "connection error", "provider returned error"))


def call(model, messages):
    """model.invoke with retries on rate limits / provider hiccups (OpenRouter free models have both)."""
    for attempt in range(len(RETRY_SLEEP) + 1):
        STATS["model_calls"] += 1
        try:
            return model.invoke(messages)
        except Exception as e:  # noqa: BLE001
            if daily_quota(e):
                raise RuntimeError(
                    "the provider says the DAILY quota is used up (OpenRouter free models: 50 requests/day, "
                    "1000/day after buying 10 credits). Try again tomorrow or add credits. Original error: "
                    f"{str(e)[:300]}") from e
            if attempt == len(RETRY_SLEEP) or not is_transient(e):
                raise
            wait = RETRY_SLEEP[attempt]
            emit(f"    [retry] {type(e).__name__}: {str(e)[:140]} - waiting {wait}s")
            time.sleep(wait)


def tools_unsupported(e: Exception) -> bool:
    m = str(e).lower()
    return ("tool" in m or "function" in m) and any(
        k in m for k in ("support", "endpoint", "unsupported", "not available", "not allowed", "invalid"))


# ----------------------------------------------------------------------------- reading replies
def text_of(msg: Any) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    parts = []
    for b in c or []:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def extract_json(text: str) -> dict:
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object in the model reply")


def extract_code(text: str) -> Optional[str]:
    """The script from a fenced block (used when a model prints code instead of calling the tool)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    long_ = [b for b in blocks if b.count("\n") >= 2]
    return max(long_, key=len) if long_ else None


_FN_RE = re.compile(r"<function=([\w.\-]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|$)", re.S)


def markup_tool_calls(text: str) -> list[dict]:
    """Some models print their tool call as text (<function=NAME><parameter=KEY>VALUE</parameter></function>) instead of
    returning it in the tool_calls field. Read those calls too (integers become integers, true/false booleans)."""
    out = []
    for i, m in enumerate(_FN_RE.finditer(text or "")):
        args: dict = {}
        for k, v in _PARAM_RE.findall(m.group(2)):
            v = v.strip()
            if re.fullmatch(r"-?\d+", v):
                args[k] = int(v)
            elif v.lower() in ("true", "false"):
                args[k] = v.lower() == "true"
            else:
                args[k] = v
        out.append({"name": m.group(1), "args": args, "id": f"markup{i}"})
    return out


# ----------------------------------------------------------------------------- models
ROLES = ("orchestrator", "scientist", "general", "hardware", "executor")


def default_models(model_name: str | None = None, provider: str | None = None, hardware_model: str | None = None):
    """OpenRouter (OPENROUTER_API_KEY) first, then Google (GOOGLE_API_KEY), then Anthropic.

    For OpenRouter `model_name` may be a comma-separated list: the first is the model, the whole list is sent as
    OpenRouter's `models` fallback array (a fallback inside one request costs one request of the free quota).
    `hardware_model` (optional) is a different model for the role that calls the camera tools.
    """
    name = model_name or os.environ.get("AGENT_MODEL")
    provider = provider or os.environ.get("AGENT_PROVIDER")
    if not provider:
        provider = ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else
                    "google" if os.environ.get("GOOGLE_API_KEY") else "anthropic")

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        if not name:
            raise SystemExit("OpenRouter needs a model: pass --model <id> (for example an id ending in :free) "
                             "or set AGENT_MODEL.")
        base_url = os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_URL)   # override only for tests / proxies

        def make(spec: str):
            ids = [m.strip() for m in spec.split(",") if m.strip()]
            extra = {"models": ids} if len(ids) > 1 else {}
            return ChatOpenAI(
                model=ids[0], api_key=os.environ["OPENROUTER_API_KEY"], base_url=base_url,
                temperature=0.2, timeout=180, max_retries=0,   # retries are done in call(), where they are counted
                # without a cap OpenRouter reserves the model's maximum answer length (65536 tokens for some models) and answers
                # 402 "can only afford N tokens" to an account with a small balance; our answers are short
                max_tokens=int(os.environ.get("HW3_MAX_TOKENS", "3000")),
                default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "hw3-hardware-in-the-loop"},
                extra_body=extra or None,
            )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        def make(spec: str):
            return ChatGoogleGenerativeAI(model=spec or "gemini-3.8-flash", temperature=0.2,
                                          convert_system_message_to_human=True)
    else:
        from langchain_anthropic import ChatAnthropic

        def make(spec: str):
            return ChatAnthropic(model=spec or "claude-sonnet-4-5-20250929", temperature=0.2)

    models = {role: make(name or "") for role in ROLES}
    if hardware_model:
        models["hardware"] = make(hardware_model)
    return models
