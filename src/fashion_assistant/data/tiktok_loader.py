"""Optional TikTok Shop plug-in.

There is no free public TikTok Shop dataset. If you export products yourself
(for example with an Apify TikTok Shop scraper, keyword "fashion accessories"),
drop the CSV/JSON files in data/tiktok/ and this module turns them into a
*trend boost*:

    "what words show up in the titles of accessories that sell a lot on TikTok?"
     -> e.g. {"pearl": 0.92, "claw": 0.88, "bucket hat": 0.75, ...}

Each H&M catalog item then gets `tiktok_score` = the strongest trending phrase
its own text contains. The ranker blends it into the trend score
(settings.tiktok_weight). With no files present, everything still works and
`tiktok_score` is 0.

Column names differ between scrapers, so we look for the first match of each
alias list below.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..config import settings

TITLE_ALIASES = ["title", "name", "product_name", "productName", "product_title"]
SOLD_ALIASES = ["sold_count", "sold", "soldCount", "sales", "units_sold", "sold_num", "salesCount"]
CATEGORY_ALIASES = ["category", "category_name", "categoryName", "category_path"]

STOPWORDS = set("""a an and for the with of in on to women womens men mens new hot sale set pcs pc 1 2 3 4 5
fashion style trendy cute gift gifts girls girl lady ladies size free shipping 2024 2025 2026 korean y2k
""".split())

# Accessory words we care about. Titles that mention none of these are ignored.
ACCESSORY_VOCAB = [
    "necklace", "earring", "earrings", "hoop", "hoops", "bracelet", "bangle", "ring", "rings", "anklet",
    "pendant", "choker", "chain", "pearl", "pearls", "charm", "brooch", "bag", "tote", "crossbody",
    "shoulder bag", "clutch", "backpack", "purse", "wallet", "belt", "scarf", "bandana", "hat",
    "bucket hat", "cap", "beanie", "beret", "sunglasses", "glasses", "hair clip", "claw clip", "scrunchie",
    "headband", "hair band", "watch", "gloves", "tie", "keychain",
]


def _pick(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".json", ".jsonl"):
        txt = path.read_text()
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            data = [json.loads(l) for l in txt.splitlines() if l.strip()]
        return pd.json_normalize(data)
    return pd.DataFrame()


def load_tiktok_products(folder: Path | None = None) -> pd.DataFrame:
    folder = folder or settings.data_dir / "tiktok"
    frames = []
    for p in sorted(folder.glob("*")):
        if p.name.startswith("EXAMPLE"):      # format template, never real data
            continue
        df = _read_any(p)
        if df.empty:
            continue
        t, s = _pick(df, TITLE_ALIASES), _pick(df, SOLD_ALIASES)
        if not t:
            print(f"[tiktok] {p.name}: no title column, skipped")
            continue
        out = pd.DataFrame({"title": df[t].astype(str)})
        out["sold"] = pd.to_numeric(df[s], errors="coerce").fillna(0) if s else 1
        c = _pick(df, CATEGORY_ALIASES)
        out["category"] = df[c].astype(str) if c else ""
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["title", "sold", "category"])
    return pd.concat(frames, ignore_index=True)


def _phrases(text: str) -> set[str]:
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    words = [w for w in t.split() if w not in STOPWORDS and len(w) > 2]
    return set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}


def trending_terms(products: pd.DataFrame, min_products: int = 3) -> dict[str, float]:
    """Score each word/bigram by how much accessory sales volume it appears in (0..1)."""
    if products.empty:
        return {}
    vocab = set(ACCESSORY_VOCAB)
    weight, count = defaultdict(float), defaultdict(int)
    for title, sold in zip(products["title"], products["sold"]):
        ph = _phrases(title)
        if not ph & vocab:
            continue
        w = math.log1p(float(sold))
        for p in ph:
            weight[p] += w
            count[p] += 1
    scored = {p: v for p, v in weight.items() if count[p] >= min_products}
    if not scored:
        return {}
    ranked = pd.Series(scored).rank(pct=True)
    return ranked.to_dict()


def attach_tiktok_scores(catalog: pd.DataFrame, terms: dict[str, float]) -> pd.DataFrame:
    """Add catalog['tiktok_score'] = strongest trending phrase found in the item's text."""
    catalog = catalog.copy()
    if not terms:
        catalog["tiktok_score"] = 0.0
        return catalog
    catalog["tiktok_score"] = catalog["text"].map(lambda t: max((terms.get(p, 0.0) for p in _phrases(t)), default=0.0))
    return catalog
