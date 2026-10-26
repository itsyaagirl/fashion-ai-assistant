"""The "Influencer Suggestion" ranker.

    influencer_score = w_style   * style_fit      (does it look right with this outfit?  CLIP similarity)
                     + w_trend   * trend_fit      (is it selling / rising right now?     H&M sales (+ TikTok))
                     + w_color   * color_fit      (do the colours work together?         colour-wheel rules)
                     + w_affinity* persona_fit    (is it this influencer's taste?        likes / avoids words)

All four parts are scaled 0..1, so the weights read like percentages. The
weights come from a Persona (see personas.py); "editor" is the default
balanced pick.

This step is deliberately *not* an LLM: it is fast, deterministic, cheap and
easy to explain. The LLM's job is understanding the photo and explaining the
result; the maths decides the order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings
from .color import harmony_score
from .personas import PERSONAS, Persona, affinity


def _minmax(x: pd.Series) -> pd.Series:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=x.index)
    return (x - lo) / (hi - lo)


def trend_fit(df: pd.DataFrame) -> pd.Series:
    base = 0.7 * df["trend_score"].astype(float) + 0.3 * df["popularity_score"].astype(float)
    if "tiktok_score" in df and float(df["tiktok_score"].max()) > 0:
        w = settings.tiktok_weight
        return (1 - w) * base + w * df["tiktok_score"].astype(float)
    return base


def score_candidates(cands: pd.DataFrame, outfit_colours: list[str], persona: Persona | str = "editor") -> pd.DataFrame:
    """cands needs: article_id, text, colour_group_name, trend_score, popularity_score, style_sim."""
    p = PERSONAS[persona] if isinstance(persona, str) else persona
    df = cands.copy()
    df["style_fit"] = _minmax(df["style_sim"].astype(float))
    df["trend_fit"] = trend_fit(df)
    df["color_fit"] = df["colour_group_name"].map(lambda c: harmony_score(outfit_colours, c))
    df["persona_fit"] = df["text"].map(lambda t: affinity(p, t))
    total_w = p.w_style + p.w_trend + p.w_color + p.w_affinity
    df["influencer_score"] = (
        p.w_style * df["style_fit"] + p.w_trend * df["trend_fit"]
        + p.w_color * df["color_fit"] + p.w_affinity * df["persona_fit"]
    ) / total_w
    return df.sort_values("influencer_score", ascending=False).reset_index(drop=True)


def diversify(df: pd.DataFrame, k: int, type_col: str = "product_type_name", max_per_type: int = 1) -> pd.DataFrame:
    """Take the best items but avoid five pairs of earrings: at most `max_per_type` per type first,
    then fill any remaining slots by score."""
    picked, counts = [], {}
    for i, row in df.iterrows():
        t = row.get(type_col, "")
        if counts.get(t, 0) < max_per_type:
            picked.append(i)
            counts[t] = counts.get(t, 0) + 1
        if len(picked) == k:
            break
    if len(picked) < k:
        picked += [i for i in df.index if i not in picked][: k - len(picked)]
    return df.loc[picked].reset_index(drop=True)


def explain_row(row: pd.Series) -> str:
    """Short, human-readable evidence string the LLM can turn into prose."""
    lift = row.get("trend_lift", np.nan)
    lift_txt = f", selling {lift:.1f}x its usual weekly rate" if pd.notna(lift) and lift > 1.2 else ""
    tk = row.get("tiktok_score", 0) or 0
    tk_txt = f", trending on TikTok Shop ({tk:.2f})" if tk > 0.6 else ""
    return (f"style {row['style_fit']:.2f}, trend {row['trend_fit']:.2f}{lift_txt}{tk_txt}, "
            f"colour {row['color_fit']:.2f}, persona {row['persona_fit']:.2f} -> score {row['influencer_score']:.2f}")
