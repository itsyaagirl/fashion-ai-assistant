"""Text RAG over the styling guide (knowledge/style_guide.md).

Classic LangChain RAG: split the markdown into sections -> embed with a small
sentence-transformer -> store in LangChain's InMemoryVectorStore -> retrieve
top-k sections for a question. The agent calls this through the
`lookup_style_rules` tool so its advice is grounded in written rules rather than
whatever the LLM happens to remember.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..config import settings


def split_markdown(path: Path) -> list[tuple[str, str]]:
    """Return (heading, body) for every '## ' section."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## +", text, flags=re.MULTILINE)[1:]
    out = []
    for p in parts:
        heading, _, body = p.partition("\n")
        out.append((heading.strip(), body.strip()))
    return out


class StyleKnowledgeBase:
    def __init__(self, path: Path | None = None, embeddings=None):
        from langchain_core.documents import Document
        from langchain_core.vectorstores import InMemoryVectorStore

        path =path or settings.knowledge_dir / "style_guide.md"
        docs = [Document(page_content=f"{h}: {b}", metadata={"section": h}) for h, b in split_markdown(path)]
        if embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=settings.text_embed_model)
        self.store = InMemoryVectorStore.from_documents(docs, embeddings)
        self.n_sections = len(docs)

    def as_retriever(self, k: int = 3):
        return self.store.as_retriever(search_kwargs={"k": k})

    def search(self, question: str, k: int = 3) -> list[dict]:
        hits = self.store.similarity_search_with_score(question, k=k)
        return [{"section": d.metadata["section"], "text": d.page_content, "score": round(float(s), 3)} for d, s in hits]
