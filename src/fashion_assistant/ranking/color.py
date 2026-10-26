"""Rule-based colour harmony.

We map colour names (H&M names like "Dark Blue", or whatever the LLM says,
like "navy") to a position on the colour wheel, then score pairs with classic
stylist rules:

    neutral (black/white/beige/grey/denim) ... goes with almost anything
    metallic (gold/silver)                  ... accent that works almost everywhere
    same hue (monochrome)                   ... good
    analogous (neighbours on the wheel)     ... good
    complementary (opposites)               ... bold but intentional
    anything in between                     ... risky
"""
from __future__ import annotations

NEUTRALS = {"black", "white", "off white", "offwhite", "cream", "ivory", "beige", "grey", "gray", "taupe",
            "mole", "nude", "camel", "denim", "transparent", "stone", "sand", "charcoal", "tan", "brown"}
METALLICS = {"gold", "silver", "bronze", "copper", "metal", "metallic", "rose gold", "champagne"}

# hue in degrees on the RYB-ish stylist wheel
HUES = {
    "red": 0, "burgundy": 350, "wine": 345, "maroon": 350, "coral": 15, "orange": 30, "rust": 25,
    "terracotta": 20, "mustard": 50, "yellow": 60, "lime": 80, "green": 120, "olive": 90, "khaki": 75,
    "sage": 110, "mint": 140, "emerald": 135, "turquoise": 175, "teal": 180, "aqua": 180, "blue": 220,
    "navy": 225, "cobalt": 220, "light blue": 210, "purple": 275, "lilac": 280, "lavender": 270,
    "violet": 275, "pink": 330, "fuchsia": 320, "magenta": 315,
}


def normalize_colour(name: str) -> str:
    n = (name or "").lower().strip()
    for prefix in ("dark ", "light ", "greyish ", "greenish ", "yellowish ", "other "):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n


def colour_kind(name: str) -> tuple[str, float | None]:
    """Return ("neutral"|"metallic"|"hue"|"unknown", hue_degrees_or_None)."""
    raw = (name or "").lower()
    n = normalize_colour(name)
    if any(m in raw for m in METALLICS):
        return "metallic", None
    if n in NEUTRALS or any(w in n.split() for w in NEUTRALS):
        return "neutral", None
    # Check the cleaned name first ("yellowish green" -> "green"), longest key first
    # so "light blue" beats "blue"; then fall back to the raw name.
    for text in (n, raw):
        for key in sorted(HUES, key=len, reverse=True):
            if key in text:
                return "hue", float(HUES[key])
    return "unknown", None


def pair_harmony(a: str, b: str) -> float:
    ka, ha = colour_kind(a)
    kb, hb = colour_kind(b)
    if "unknown" in (ka, kb):
        return 0.6
    if ka == "metallic" or kb == "metallic":
        return 0.85
    if ka == "neutral" or kb == "neutral":
        return 0.85 if ka == kb else 0.8
    d = abs(ha - hb) % 360
    d = min(d, 360 - d)
    if d <= 15:
        return 0.8      # monochrome
    if d <= 50:
        return 0.75     # analogous
    if d >= 150:
        return 0.7      # complementary
    return 0.35         # clash zone


def harmony_score(outfit_colours: list[str], item_colour: str) -> float:
    """Average harmony of one accessory colour against all outfit colours (0..1)."""
    cols = [c for c in (outfit_colours or []) if c]
    if not cols:
        return 0.6
    return sum(pair_harmony(c, item_colour) for c in cols) / len(cols)
