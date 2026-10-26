"""The LangGraph state: the single dict every node reads from and writes to.

Each node returns only the keys it changed; LangGraph merges them in. The
`messages` key uses the `add_messages` reducer, so nodes *append* messages
instead of overwriting the conversation.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # ---- inputs ----
    image_b64: str                      # the outfit photo, base64
    mime_type: str                      # e.g. image/jpeg
    user_request: str                   # optional text, e.g. "for a summer wedding"
    personas: list[str]                 # which influencer personas to run

    # ---- produced along the way ----
    messages: Annotated[list[AnyMessage], add_messages]   # LLM <-> tool conversation
    outfit: dict[str, Any]              # OutfitAnalysis as a dict
    outfit_embedding: list[float]       # CLIP vector of the outfit photo
    steps: int                          # agent loop counter (safety cap)
    candidates: list[dict[str, Any]]    # pooled search results
    style_notes: list[str]              # retrieved styling rules
    ranked: list[dict[str, Any]]        # editor's top picks (the Influencer Suggestion)
    persona_picks: dict[str, list[dict[str, Any]]]
    final_answer: str                   # markdown answer for the user
