"""
LLM client — supports Ollama (local) and any OpenAI-compatible HTTP API.
Switch between them by setting LLM_PROVIDER in your .env file.

For cloud generation (LLM_PROVIDER != "ollama") the client uses a priority
fallback chain of OpenAI-compatible endpoints, tried in order:
  1. Groq        (GROQ_*   in config.py)  — primary
  2. OpenRouter  (OPENAI_* in config.py)  — fallback
A rate-limit / transient error on the primary automatically cascades to the
fallback. Because Groq, OpenRouter, NVIDIA NIM and OpenAI all speak the OpenAI
protocol, the same code path serves any of them — only base_url/key/model differ.

Public API
----------
  call_llm(prompt, system=None)            → plain-text response
  call_llm_json(prompt, system=None)       → dict parsed from a JSON response
                                             (Ollama format=json / OpenAI json_object)

Both paths share an in-process response cache (keyed on a hash of
provider+model+system+prompt+format) so the 8–15 sequential calls a full
ETHER→AIR→DER query can trigger on a local model are not all paid for twice.
The cache is process-local and bounded (LLM_CACHE_MAXSIZE); it is safe because
generation runs at temperature 0.1 (near-deterministic) and prompts are
self-contained.
"""
import hashlib
import json
import logging
import os
from collections import OrderedDict
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

# ── In-process bounded LRU response cache ─────────────────────────────────────
_CACHE: "OrderedDict[str, str]" = OrderedDict()


def _cache_key(prompt: str, system: Optional[str], want_json: bool) -> str:
    h = hashlib.sha256()
    h.update((settings.LLM_PROVIDER or "").encode())
    if settings.LLM_PROVIDER == "ollama":
        model, host = settings.OLLAMA_MODEL, settings.OLLAMA_BASE_URL
    else:
        # Key on the whole chain signature so a cached answer isn't tied to which
        # endpoint (Groq or OpenRouter) happened to serve it, but changing any
        # model/base URL / fallback setting still invalidates old entries.
        model = f"{settings.GROQ_MODEL}+{settings.OPENAI_MODEL}"
        host = (
            f"{settings.GROQ_BASE_URL}|{settings.OPENAI_BASE_URL}"
            f"|fb={settings.LLM_FALLBACK_ENABLED}"
        )
    h.update((model or "").encode())
    h.update((host or "").encode())
    h.update(b"\x00json" if want_json else b"\x00text")
    h.update(b"\x00")
    h.update((system or "").encode())
    h.update(b"\x00")
    h.update((prompt or "").encode())
    return h.hexdigest()


def _cache_get(key: str) -> Optional[str]:
    if settings.LLM_CACHE_MAXSIZE <= 0:
        return None
    val = _CACHE.get(key)
    if val is not None:
        _CACHE.move_to_end(key)
    return val


def _cache_put(key: str, value: str) -> None:
    if settings.LLM_CACHE_MAXSIZE <= 0:
        return
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > settings.LLM_CACHE_MAXSIZE:
        _CACHE.popitem(last=False)


def clear_cache() -> None:
    """Clear the response cache (used by tests / after re-ingestion)."""
    _CACHE.clear()


# ── Public API ────────────────────────────────────────────────────────────────

def call_llm(prompt: str, system: str = None) -> str:
    """
    Send a prompt to the configured LLM and return the text response.

    Raises:
        RuntimeError if the LLM call fails.
    """
    key = _cache_key(prompt, system, want_json=False)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    if settings.LLM_PROVIDER == "ollama":
        out = _call_ollama(prompt, system, want_json=False)
    else:
        out = _call_openai_chain(prompt, system, want_json=False)

    _cache_put(key, out)
    return out


def call_llm_json(prompt: str, system: str = None, default: Any = None) -> Any:
    """
    Send a prompt and parse the response as JSON.

    The underlying provider is asked to emit strict JSON (Ollama format="json",
    OpenAI response_format json_object). The raw text is still cached so repeated
    identical structured calls are free.

    Args:
        prompt:  instruction; should ask for a JSON object.
        system:  optional system message.
        default: value returned if the call fails or JSON cannot be parsed
                 (defaults to {}). This keeps AIR/DER loops robust on a local
                 model that occasionally emits malformed JSON.

    Returns:
        Parsed JSON (usually a dict) or `default` on failure.
    """
    if default is None:
        default = {}

    key = _cache_key(prompt, system, want_json=True)
    raw = _cache_get(key)
    if raw is None:
        try:
            if settings.LLM_PROVIDER == "ollama":
                raw = _call_ollama(prompt, system, want_json=True)
            else:
                raw = _call_openai_chain(prompt, system, want_json=True)
        except Exception as exc:
            logger.warning("call_llm_json failed (%s) — returning default", exc)
            return default
        _cache_put(key, raw)

    parsed = _parse_json_lenient(raw)
    if parsed is None:
        logger.warning("call_llm_json could not parse JSON from response — returning default")
        return default
    return parsed


def _parse_json_lenient(text: str) -> Optional[Any]:
    """Parse JSON, tolerating code fences or surrounding prose from weaker models."""
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # Strip ```json … ``` fences
    if "```" in t:
        inner = t.split("```")
        for seg in inner:
            seg = seg.strip()
            if seg.startswith("json"):
                seg = seg[4:].strip()
            try:
                return json.loads(seg)
            except Exception:
                continue
    # Fall back to the first {...} or [...] span
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = t.find(open_ch)
        end = t.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except Exception:
                continue
    return None


# ── Ollama ────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, system: str = None, want_json: bool = False) -> str:
    """Call a locally-running Ollama model."""
    try:
        import ollama

        # Create a client that points to the configured host.
        # The default ollama.chat() always uses localhost:11434, which
        # fails inside Docker where the host is reached via host.docker.internal.
        client = ollama.Client(host=settings.OLLAMA_BASE_URL, timeout=settings.OLLAMA_TIMEOUT_S)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": settings.OLLAMA_MODEL,
            "messages": messages,
            "options": {"temperature": 0.1},   # Low temperature → more factual
            "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        }
        if want_json:
            kwargs["format"] = "json"

        response = client.chat(**kwargs)
        return response["message"]["content"]

    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed: {e}\n"
            f"Ensure Ollama is running (https://ollama.com) and the model is pulled:\n"
            f"  ollama pull {settings.OLLAMA_MODEL}"
        )


# ── OpenAI-compatible API with a priority fallback chain ──────────────────────
# Both Groq and OpenRouter speak the OpenAI protocol, so a single code path
# serves both — only (base_url, key, model) differ. Endpoints are tried in
# priority order (Groq first, OpenRouter second); a transient failure or
# rate-limit on one cascades to the next instead of bubbling up to the
# raw-chunk "_fallback_answer" in the retriever.

# One cached OpenAI client per endpoint name.
_CLIENTS: dict = {}


def _openai_endpoints() -> list[dict]:
    """
    Build the ordered list of OpenAI-compatible endpoints to try.

    Priority:
      1. Groq       (settings.GROQ_*)      — primary, if GROQ_API_KEY is set
      2. OpenRouter (settings.OPENAI_*)    — fallback, if OPENAI_API_KEY is set

    When LLM_FALLBACK_ENABLED is False, only the highest-priority configured
    endpoint is returned. Raises if nothing is configured.

    Values are read ENVIRONMENT-FIRST, falling back to the frozen `settings`
    snapshot. This matters on Streamlit Community Cloud: `settings` is built once
    at import, but secrets (GROQ_API_KEY, …) may be injected into the environment
    slightly later (or the retriever is served from an @st.cache_resource that
    predates the secrets). Reading os.environ live here means a key added via the
    Secrets UI is picked up on the next query without needing a cold restart.
    """
    def _cfg(env_name: str, settings_val):
        v = os.environ.get(env_name)
        return v if v not in (None, "") else settings_val

    endpoints: list[dict] = []

    groq_key = _cfg("GROQ_API_KEY", settings.GROQ_API_KEY)
    if groq_key:
        endpoints.append({
            "name":        "groq",
            "api_key":     groq_key,
            "base_url":    _cfg("GROQ_BASE_URL", settings.GROQ_BASE_URL),
            "model":       _cfg("GROQ_MODEL", settings.GROQ_MODEL),
            "timeout":     settings.GROQ_TIMEOUT_S,
            "max_retries": settings.GROQ_MAX_RETRIES,
            "headers":     None,
        })

    openai_key = _cfg("OPENAI_API_KEY", settings.OPENAI_API_KEY)
    if openai_key:
        openai_base = _cfg("OPENAI_BASE_URL", settings.OPENAI_BASE_URL)
        headers = {}
        if "openrouter" in (openai_base or ""):
            if settings.OPENROUTER_SITE_URL:
                headers["HTTP-Referer"] = settings.OPENROUTER_SITE_URL
            if settings.OPENROUTER_APP_NAME:
                headers["X-Title"] = settings.OPENROUTER_APP_NAME
        endpoints.append({
            "name":        "openrouter",
            "api_key":     openai_key,
            "base_url":    openai_base,
            "model":       _cfg("OPENAI_MODEL", settings.OPENAI_MODEL),
            "timeout":     settings.OPENAI_TIMEOUT_S,
            "max_retries": settings.OPENAI_MAX_RETRIES,
            "headers":     headers or None,
        })

    if not endpoints:
        raise RuntimeError(
            "No generation endpoint configured. Set GROQ_API_KEY (primary) "
            "and/or OPENAI_API_KEY (fallback) in your .env / Space secrets.\n"
            "  Groq key:       https://console.groq.com/keys  (gsk_…)\n"
            "  OpenRouter key: https://openrouter.ai/keys     (sk-or-v1-…)"
        )

    fallback = _cfg("LLM_FALLBACK_ENABLED", str(settings.LLM_FALLBACK_ENABLED))
    if str(fallback).strip().lower() not in ("1", "true", "yes"):
        return endpoints[:1]
    return endpoints


def _get_client(ep: dict):
    """Build (once) and cache an OpenAI client for a given endpoint spec."""
    cached = _CLIENTS.get(ep["name"])
    if cached is not None:
        return cached

    from openai import OpenAI

    client = OpenAI(
        api_key=ep["api_key"],
        base_url=ep["base_url"] or None,
        timeout=ep["timeout"],
        max_retries=ep["max_retries"],
        default_headers=ep["headers"],
    )
    _CLIENTS[ep["name"]] = client
    return client


def _one_openai_call(
    client, model: str, prompt: str, system: Optional[str], want_json: bool
) -> str:
    """
    Single chat-completion call against one endpoint. Handles the JSON-mode
    retry (some models reject response_format) and reasoning-model empty-content.
    Raises on failure so the caller can cascade to the next endpoint.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model, "messages": messages, "temperature": 0.1}
    if want_json:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Not every model supports response_format — retry as plain text and let
        # _parse_json_lenient recover the object.
        if want_json:
            logger.info("JSON mode rejected by %s (%s) — retrying as plain text", model, e)
            kwargs.pop("response_format", None)
            kwargs["messages"] = [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt + "\n\nRespond with ONLY a valid JSON object, no prose."},
            ]
            response = client.chat.completions.create(**kwargs)  # raise → caller cascades
        else:
            raise

    text = response.choices[0].message.content
    if text is None:
        # Reasoning models can return the answer under a separate field with an
        # empty content string; surface whatever is there rather than crashing.
        text = getattr(response.choices[0].message, "reasoning_content", "") or ""
    return text


def _call_openai_chain(prompt: str, system: str = None, want_json: bool = False) -> str:
    """
    Try each configured OpenAI-compatible endpoint in priority order; return the
    first success. If every endpoint fails, raise a RuntimeError summarising why.
    """
    endpoints = _openai_endpoints()
    errors: list[str] = []

    for i, ep in enumerate(endpoints):
        try:
            client = _get_client(ep)
            return _one_openai_call(client, ep["model"], prompt, system, want_json)
        except Exception as e:
            errors.append(f"{ep['name']} ({ep['model']}): {e}")
            is_last = i == len(endpoints) - 1
            if is_last:
                break
            logger.warning(
                "Generation endpoint '%s' failed (%s) — falling back to '%s'",
                ep["name"], e, endpoints[i + 1]["name"],
            )

    raise RuntimeError("All generation endpoints failed → " + " | ".join(errors))
