"""
ai_router.py
------------
Fallback-Kette für KI-Anfragen: Gemini -> Groq -> OpenRouter.
Wird von main.py importiert und anstelle des bisherigen Ollama-Aufrufs benutzt.

Alle drei Anbieter sind kostenlos nutzbar (kein Zahlungsmittel nötig):
- Google AI Studio (Gemini): https://aistudio.google.com/apikey
- Groq:                      https://console.groq.com/keys
- OpenRouter:                https://openrouter.ai/keys

Benötigte Umgebungsvariablen (in Render als "Environment Variables" eintragen,
NICHT im Code hardcoden):
    GEMINI_API_KEY
    GROQ_API_KEY
    OPENROUTER_API_KEY
"""

import os
import logging
from typing import List, Dict

import httpx

logger = logging.getLogger("ai_router")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

TIMEOUT = httpx.Timeout(60.0)


class AllProvidersFailedError(Exception):
    """Wird geworfen, wenn Gemini, Groq UND OpenRouter alle fehlschlagen."""
    pass


async def _call_gemini(messages: List[Dict[str, str]]) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY fehlt")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    # Gemini erwartet ein eigenes Nachrichtenformat -> umwandeln
    contents = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
    ]
    payload = {"contents": contents}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def _call_groq(messages: List[Dict[str, str]]) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY fehlt")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"model": GROQ_MODEL, "messages": messages}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _call_openrouter(messages: List[Dict[str, str]]) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY fehlt")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    payload = {"model": OPENROUTER_MODEL, "messages": messages}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def get_ai_response(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Versucht der Reihe nach Gemini -> Groq -> OpenRouter.
    Gibt {"provider": ..., "text": ...} zurück oder wirft AllProvidersFailedError.

    messages: Liste im Format [{"role": "user", "content": "..."}, ...]
    """
    providers = [
        ("gemini", _call_gemini),
        ("groq", _call_groq),
        ("openrouter", _call_openrouter),
    ]

    last_error = None
    for name, func in providers:
        try:
            text = await func(messages)
            logger.info(f"Antwort erfolgreich von: {name}")
            return {"provider": name, "text": text}
        except Exception as e:
            logger.warning(f"Anbieter '{name}' fehlgeschlagen: {e}")
            last_error = e
            continue

    raise AllProvidersFailedError(
        f"Alle Anbieter fehlgeschlagen. Letzter Fehler: {last_error}"
    )
