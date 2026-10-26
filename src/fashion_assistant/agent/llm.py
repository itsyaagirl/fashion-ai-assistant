"""Gemini chat model factory + small helpers."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from ..config import settings


def get_llm(model: str | None = None, temperature: float | None = None):
    """ChatGoogleGenerativeAI reads GOOGLE_API_KEY (or GEMINI_API_KEY) from the environment."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError("Set GOOGLE_API_KEY (free key: https://aistudio.google.com/apikey).")
    return ChatGoogleGenerativeAI(
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_retries=int(os.environ.get("FA_LLM_RETRIES", "2")),   # a couple of retries for blips, then a clear error
        timeout=float(os.environ.get("FA_LLM_TIMEOUT", "60")),    # seconds per call, so a stuck call can't hang forever
    )


def message_text(msg) -> str:
    """Plain text of an AI message. Gemini 3 can return a list of content blocks
    (text + 'thinking' blocks); we keep only the text parts."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def load_image_b64(image, max_side: int = 768) -> tuple[str, str]:
    """Accept a path, bytes or PIL image -> (base64 JPEG, mime). Resizing keeps API calls small."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    else:
        img = image
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def b64_to_pil(b64: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
