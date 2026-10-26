"""Personalized AI Fashion Assistant: multimodal LangGraph agent + RAG for accessory recommendations."""

__version__ = "1.0.0"


def __getattr__(name):
    # Lazy import so data/eval utilities work without LangChain installed.
    if name == "FashionAssistant":
        from .assistant import FashionAssistant

        return FashionAssistant
    raise AttributeError(name)
