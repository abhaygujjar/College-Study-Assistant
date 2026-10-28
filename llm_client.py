"""Talk to the LLM (Google Gemini by default, free tier works).

If there is no API key, we still return something useful: the best matching
passages from the notes. So the app never just crashes.
"""

from __future__ import annotations

import os
import re
import time

import requests

from text_chunker import Chunk

# "-latest" is an alias that Google keeps pointing at a current Flash model.
DEFAULT_MODEL = "gemini-flash-latest"
BACKUP_MODEL = "gemini-flash-lite-latest"  # lighter model, used when the main one is busy
RETRY_CODES = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = 25  # seconds to wait for one answer before retrying
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_URL = BASE_URL + "/models/{model}:generateContent"

_discovered_model: str | None = None  # remembered after the first auto-discovery

PROMPT = """You are a helpful study assistant for a college student.
Answer the question using ONLY the notes below. Use simple, clear English.
If the notes do not contain the answer, say: "I could not find this in your notes."
Add a short example if it helps the student understand.

NOTES:
{context}

QUESTION: {question}

ANSWER:"""


class LLMError(Exception):
    pass


class LLMConnectionError(LLMError):
    """Network problem: timeout, no internet, connection dropped."""


def api_key_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    parts = []
    for chunk, _score in hits:
        parts.append(f"[{chunk.source}, page {chunk.page}]\n{chunk.text}")
    return "\n\n".join(parts)


def find_flash_model(key: str) -> str | None:
    """Ask Google which models exist and return the newest plain 'flash' one.

    Model names change often (old ones get switched off), so we do not
    depend on one fixed name.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/models",
            headers={"x-goog-api-key": key},
            params={"pageSize": 200},
            timeout=30,
        )
        models = response.json().get("models", [])
    except (requests.RequestException, ValueError):
        return None

    best_version, best_name = -1.0, None
    for model in models:
        name = model.get("name", "").removeprefix("models/")
        if "generateContent" not in model.get("supportedGenerationMethods", []):
            continue
        match = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash", name)  # skips lite / image / tts
        if match and float(match.group(1)) > best_version:
            best_version, best_name = float(match.group(1)), name
    return best_name


def _generate(model: str, key: str, body: dict) -> requests.Response:
    try:
        return requests.post(
            API_URL.format(model=model),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise LLMConnectionError(f"Could not reach the LLM: {exc}") from exc


def _generate_with_retry(model: str, key: str, body: dict, attempts: int = 3) -> requests.Response:
    """Retry short problems with a pause: 503 'high demand', 429 rate limit, timeouts."""
    for attempt in range(attempts):
        last_try = attempt == attempts - 1
        try:
            response = _generate(model, key, body)
        except LLMConnectionError:
            if last_try:
                raise
        else:
            if response.status_code not in RETRY_CODES or last_try:
                return response
        time.sleep(1.5 * 2**attempt)  # 1.5s, then 3s
    raise LLMError("unreachable")  # the loop always returns or raises before this


def _try(model: str, key: str, body: dict, attempts: int = 3) -> requests.Response | None:
    """Like _generate_with_retry, but returns None if the network keeps failing."""
    try:
        return _generate_with_retry(model, key, body, attempts)
    except LLMConnectionError:
        return None


def _read_answer(response: requests.Response) -> str:
    try:
        data = response.json()
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError("LLM answer was in an unexpected format") from exc

    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        reason = candidate.get("finishReason", "unknown")
        raise LLMError(f"LLM returned an empty answer (reason: {reason})")
    return text


def ask_llm(question: str, hits: list[tuple[Chunk, float]]) -> str:
    global _discovered_model

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")

    model = _discovered_model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    prompt = PROMPT.format(context=build_context(hits), question=question)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500},
    }

    response = _try(model, key, body)

    # 404 = this model name no longer exists. Find a current one and try once more.
    if response is not None and response.status_code == 404:
        newer = find_flash_model(key)
        if newer and newer != model:
            _discovered_model = newer
            model = newer
            response = _try(newer, key, body)

    # Still busy or timing out after retries: try the lighter model once.
    if (response is None or response.status_code in RETRY_CODES) and model != BACKUP_MODEL:
        backup = _try(BACKUP_MODEL, key, body, attempts=1)
        if backup is not None and backup.status_code == 200:
            response = backup

    if response is None:
        raise LLMError("Could not reach Google (timeout). Check your internet and try again.")
    if response.status_code != 200:
        raise LLMError(f"LLM returned {response.status_code}: {response.text[:200]}")
    return _read_answer(response)


def fallback_answer(hits: list[tuple[Chunk, float]]) -> str:
    """No LLM: show the most relevant passages directly."""
    if not hits:
        return "I could not find this in your notes."
    lines = ["No LLM available, so here are the best matching passages from your notes:\n"]
    seen = set()
    for chunk, _score in hits:
        if chunk.text in seen:
            continue
        seen.add(chunk.text)
        text = chunk.text if len(chunk.text) <= 700 else chunk.text[:700].rsplit(" ", 1)[0] + " ..."
        lines.append(f"**{chunk.source} (page {chunk.page})**\n\n> {text}\n")
        if len(seen) == 3:
            break
    return "\n".join(lines)
