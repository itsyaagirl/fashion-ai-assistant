"""Public entry point.

    from fashion_assistant import FashionAssistant
    fa = FashionAssistant()                       # loads the saved index + knowledge base + Gemini
    out = fa.recommend("my_outfit.jpg", request="for a summer wedding", personas=["minimalist", "luxe"])
    print(out["final_answer"])

    for node, update in fa.stream("my_outfit.jpg"):   # live progress, one event per graph node
        print(node)
"""
from __future__ import annotations

from typing import Iterator

from .agent.graph import build_graph
from .agent.llm import get_llm, load_image_b64
from .rag.catalog_index import CatalogIndex

DEFAULT_PERSONAS = ["minimalist", "streetwear", "luxe", "boho"]
RUN_CONFIG = {"recursion_limit": 30}


class FashionAssistant:
    def __init__(self, index: CatalogIndex | None = None, kb=None, llm=None, encode_image=None):
        if kb is None:
            from .rag.style_kb import StyleKnowledgeBase

            kb = StyleKnowledgeBase()
        self.index = index or CatalogIndex.load()
        self.kb = kb
        self.llm = llm or get_llm()
        self.graph = build_graph(self.llm, self.index, self.kb, encode_image=encode_image)

    @staticmethod
    def inputs(image, request: str = "", personas: list[str] | None = None) -> dict:
        b64, mime = load_image_b64(image)
        return {"image_b64": b64, "mime_type": mime, "user_request": request,
                "personas": DEFAULT_PERSONAS if personas is None else personas}

    def recommend(self, image, request: str = "", personas: list[str] | None = None) -> dict:
        return self.graph.invoke(self.inputs(image, request, personas), config=RUN_CONFIG)

    def stream(self, image, request: str = "", personas: list[str] | None = None) -> Iterator[tuple[str, dict, dict]]:
        """Yield (node_name, that node's update, state so far) as each graph node finishes."""
        state = self.inputs(image, request, personas)
        state["messages"] = []
        for chunk in self.graph.stream(state, config=RUN_CONFIG, stream_mode="updates"):
            for node, update in chunk.items():
                update = update or {}
                for key, value in update.items():
                    if key == "messages":          # mirror the add_messages reducer
                        state["messages"] = state["messages"] + list(value)
                    else:
                        state[key] = value
                yield node, update, state

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
