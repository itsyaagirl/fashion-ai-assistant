"""Influencer personas.

A persona is just (1) a set of ranking weights and (2) words it likes / avoids.
The same candidate pool is re-ranked once per persona, so each "influencer"
gives its own pick. The LLM later writes the explanation in that persona's voice.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    voice: str                         # how the LLM should sound
    w_style: float
    w_trend: float
    w_color: float
    w_affinity: float
    likes: tuple = field(default_factory=tuple)
    avoids: tuple = field(default_factory=tuple)


PERSONAS: dict[str, Persona] = {
    "editor": Persona(
        key="editor", name="The Editor (balanced pick)",
        voice="a fashion magazine editor: confident, concise, explains why it works",
        w_style=0.45, w_trend=0.30, w_color=0.25, w_affinity=0.0,
    ),
    "minimalist": Persona(
        key="minimalist", name="Mia the Minimalist",
        voice="a calm minimalist creator: less is more, quality basics, clean lines",
        w_style=0.40, w_trend=0.10, w_color=0.30, w_affinity=0.20,
        likes=("leather", "thin", "simple", "plain", "small", "silver", "gold", "black", "white", "beige", "tote", "watch", "stud"),
        avoids=("glitter", "sequin", "print", "pattern", "neon", "fringe", "multicolour", "chunky", "slogan"),
    ),
    "streetwear": Persona(
        key="streetwear", name="Jay the Streetwear Creator",
        voice="a hype streetwear TikTok creator: energetic, trend-aware, uses a little slang",
        w_style=0.30, w_trend=0.40, w_color=0.10, w_affinity=0.20,
        likes=("cap", "beanie", "bucket", "chain", "chunky", "crossbody", "shoulder bag", "nylon", "logo", "sport", "bandana", "socks"),
        avoids=("pearl", "silk", "lace", "dainty", "formal"),
    ),
    "luxe": Persona(
        key="luxe", name="Lena the Luxe Stylist",
        voice="a polished luxury stylist: elegant, refined, talks about investment pieces",
        w_style=0.40, w_trend=0.15, w_color=0.25, w_affinity=0.20,
        likes=("leather", "gold", "pearl", "silk", "structured", "clutch", "scarf", "belt", "sunglasses", "satin", "suede"),
        avoids=("nylon", "plastic", "neon", "slogan", "sport", "glitter"),
    ),
    "boho": Persona(
        key="boho", name="Bea the Boho Influencer",
        voice="a warm boho lifestyle influencer: relaxed, earthy, festival-ready",
        w_style=0.35, w_trend=0.20, w_color=0.25, w_affinity=0.20,
        likes=("fringe", "beaded", "bead", "woven", "straw", "raffia", "shell", "wide", "layered", "brown", "suede", "hoop", "bandana"),
        avoids=("structured", "logo", "sport", "patent"),
    ),
}


def affinity(persona: Persona, text: str) -> float:
    """0..1: how much an item's text matches what the persona likes, minus what it avoids."""
    if not persona.likes and not persona.avoids:
        return 0.5
    t = text.lower()
    likes = sum(1 for w in persona.likes if w in t)
    avoids = sum(1 for w in persona.avoids if w in t)
    raw = 0.5 + 0.2 * min(likes, 2) - 0.35 * min(avoids, 2)
    return max(0.0, min(1.0, raw))
