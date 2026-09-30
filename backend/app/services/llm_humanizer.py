import asyncio
import os
import json
import re
import logging

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
]

LANGUAGE_NAMES: dict[str, str] = {
    "it": "Italian",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
}

SYSTEM_PROMPT_TEMPLATE = """You are an expert writing humanizer. You rewrite AI-generated text to sound naturally human-written.

CRITICAL RULES:
- Respond ENTIRELY in {language_name}.
- Keep the original meaning and information intact.
- Make it sound like a real person wrote it: vary sentence length, use casual transitions, add minor imperfections.
- Remove AI-typical patterns: "furthermore", "it is important to note", "in conclusion", overly formal connectives.
- Break uniform sentence structures. Mix short punchy sentences with longer flowing ones.
- Use contractions, colloquial expressions, and natural phrasing.
- Output ONLY a JSON object with this exact format, nothing else:

{{
  "humanized_text": "the rewritten text in {language_name}",
  "changes_made": ["change 1 description in {language_name}", "change 2 description in {language_name}"]
}}
"""

USER_PROMPT_TEMPLATE = """Rewrite this text to sound naturally human-written. Keep the same meaning but make it sound like a real person wrote it.

Text to humanize:
{text}

The text is in {language_name}. Respond ENTIRELY in {language_name}."""


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _parse_response(content: str) -> dict | None:
    cleaned = _strip_thinking(content).strip()

    # Remove markdown fence only if it wraps the entire response
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence_match:
        candidate = fence_match.group(1).strip()
        # Only use if the fenced block looks like JSON, not embedded code inside humanized_text
        if candidate.startswith("{"):
            cleaned = candidate

    # Try full JSON parse
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict) and "humanized_text" in data:
                changes = data.get("changes_made", [])
                if not isinstance(changes, list):
                    changes = []
                return {
                    "humanized_text": str(data["humanized_text"]),
                    "changes_made": [str(c) for c in changes[:10]],
                }
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: JSON was truncated — extract humanized_text directly via regex
    text_match = re.search(r'"humanized_text"\s*:\s*"((?:[^"\\]|\\.)*)', cleaned)
    if text_match:
        partial_text = text_match.group(1).replace("\\n", "\n").replace('\\"', '"')
        if len(partial_text) > 50:
            logger.warning("JSON truncated, using partial humanized_text (%d chars)", len(partial_text))
            return {
                "humanized_text": partial_text,
                "changes_made": [],
            }

    return None


async def _call_model(model: str, messages: list[dict], headers: dict) -> tuple[dict | None, str | None]:
    """One attempt on one model. Returns (result, None) on success, (None, error_type) otherwise."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 4096,
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPError as e:
        return None, f"network_{type(e).__name__}"

    if response.status_code != 200:
        return None, f"http_{response.status_code}"

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None, "malformed"
    if not isinstance(content, str) or not content.strip():
        return None, "empty"

    parsed = _parse_response(content)
    if not parsed:
        return None, "malformed"
    if not parsed["humanized_text"].strip():
        return None, "empty"
    return parsed, None


async def humanize_text(text: str, language: str) -> tuple[dict | None, bool]:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "sk-or-v1-your-key-here":
        logger.warning("OpenRouter API key not configured")
        return None, False

    language_name = LANGUAGE_NAMES.get(language, "English")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language_name=language_name)
    user_prompt = USER_PROMPT_TEMPLATE.format(text=text[:5000], language_name=language_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://realtext.org",
        "X-Title": "RealText",
    }

    errors: list[str] = []
    for i, model in enumerate(FREE_MODELS):
        try:
            result, error = await _call_model(model, messages, headers)
        except Exception as e:
            result, error = None, f"error_{type(e).__name__}"

        if result:
            logger.info(
                "llm_request endpoint=humanize model=%s attempts=%d errors=%s",
                model, i + 1, ",".join(errors) or "-",
            )
            return result, True

        errors.append(f"{model}:{error}")
        if error == "http_429" and i < len(FREE_MODELS) - 1:
            await asyncio.sleep(3)

    logger.warning(
        "llm_request endpoint=humanize model=none attempts=%d errors=%s",
        len(FREE_MODELS), ",".join(errors) or "-",
    )
    return None, False
