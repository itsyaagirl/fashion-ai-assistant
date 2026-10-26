"""Public entry point.

    from fashion_assistant import FashionAssistant
    fa = FashionAssistant()                       # loads the saved index + knowledge base + Gemini
    out = fa.recommend("my_outfit.jpg", request="for a summer wedding", personas=["minimalist", "luxe"])
    print(out["final_answer"])
"""
from __future__ import annotations

from .agent.graph import build_graph
from .agent.llm import get_llm, load_image_b64
from .rag.catalog_index import CatalogIndex

DEFAULT_PERSONAS = ["minimalist", "streetwear", "luxe", "boho"]


class FashionAssistant:
    def __init__(self, index: CatalogIndex | None = None, kb=None, llm=None, encode_image=None):
        if kb is None:
            from .rag.style_kb import StyleKnowledgeBase

            kb = StyleKnowledgeBase()
        self.index = index or CatalogIndex.load()
        self.kb = kb
        self.llm = llm or get_llm()
        self.graph = build_graph(self.llm, self.index, self.kb, encode_image=encode_image)

    def recommend(self, image, request: str = "", personas: list[str] | None = None) -> dict:
        b64, mime = load_image_b64(image)
        return self.graph.invoke(
            {"image_b64": b64, "mime_type": mime, "user_request": request,
             "personas": DEFAULT_PERSONAS if personas is None else personas},
            config={"recursion_limit": 30},
        )

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
