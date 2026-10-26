"""LangChain tools the Gemini agent can call.

Tools are built by a factory (`make_tools`) so they can close over the loaded
catalog index and knowledge base without global variables. Every tool returns
JSON text: easy for the LLM to read, and easy for the `collect_candidates` node
to parse back out of the conversation.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from ..config import settings
from ..rag.catalog_index import CatalogIndex


def _item(r) -> dict:
    return {
        "article_id": int(r["article_id"]),
        "name": r["prod_name"],
        "type": r["product_type_name"],
        "colour": r["colour_group_name"],
        "match": round(float(r.get("sim", 0.0)), 3),
        "trend": round(float(r["trend_score"]), 2),
    }


def make_tools(index: CatalogIndex, kb=None) -> list:
    @tool
    def search_accessories(query: str, category: str = "", colour: str = "", k: int = 8) -> str:
        """Search the accessory catalog by description (multimodal CLIP search over product photos).

        Args:
            query: short visual description, e.g. "chunky gold chain necklace".
            category: optional accessory category, e.g. "earrings", "bag", "belt", "hat", "scarf", "sunglasses".
            colour: optional colour to prefer, e.g. "gold", "black".
            k: number of results (max 12).
        """
        k = max(1, min(int(k), settings.candidates_per_search))
        hits = index.search_text(query, k=k, category=category or None, colour=colour or None)
        return json.dumps({"query": query, "category": category, "results": [_item(r) for _, r in hits.iterrows()]})

    @tool
    def get_trending_accessories(category: str = "", k: int = 5) -> str:
        """Return the fastest-rising accessories (by recent sales growth), optionally within one category.

        Args:
            category: optional accessory category, e.g. "bag" or "earrings".
            k: number of results (max 10).
        """
        hits = index.trending(category or None, k=max(1, min(int(k), 10)))
        return json.dumps({"category": category, "results": [_item(r) for _, r in hits.iterrows()]})

    @tool
    def lookup_style_rules(question: str) -> str:
        """Retrieve styling rules from the knowledge base (necklines, colour harmony, metals, occasions, personas).

        Args:
            question: what you want to know, e.g. "what jewellery suits a v-neck black dress for a wedding".
        """
        if kb is None:
            return json.dumps({"rules": []})
        return json.dumps({"rules": kb.search(question, k=3)})

    return [search_accessories, get_trending_accessories, lookup_style_rules]
