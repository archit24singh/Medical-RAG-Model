"""
LLM client — supports Ollama (local) and OpenAI (cloud).
Switch between them by setting LLM_PROVIDER in your .env file.
"""
import logging
from config import settings

logger = logging.getLogger(__name__)


def call_llm(prompt: str, system: str = None) -> str:
    """
    Send a prompt to the configured LLM and return the text response.

    Args:
        prompt:  The user message / instruction.
        system:  Optional system message to set the assistant's role.

    Returns:
        The LLM's plain-text response.

    Raises:
        RuntimeError if the LLM call fails.
    """
    if settings.LLM_PROVIDER == "ollama":
        return _call_ollama(prompt, system)
    else:
        return _call_openai(prompt, system)


# ── Ollama ────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, system: str = None) -> str:
    """Call a locally-running Ollama model."""
    try:
        import ollama

        # Create a client that points to the configured host.
        # The default ollama.chat() always uses localhost:11434, which
        # fails inside Docker where the host is reached via host.docker.internal.
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat(
            model=settings.OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.1},   # Low temperature → more factual
        )
        return response["message"]["content"]

    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed: {e}\n"
            f"Ensure Ollama is running (https://ollama.com) and the model is pulled:\n"
            f"  ollama pull {settings.OLLAMA_MODEL}"
        )


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _call_openai(prompt: str, system: str = None) -> str:
    """Call the OpenAI chat completions API."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")
