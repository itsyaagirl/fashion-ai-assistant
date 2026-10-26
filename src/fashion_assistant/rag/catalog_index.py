"""Vector index over the accessory catalog (the "R" in RAG).

Each product vector = 0.75 * CLIP(image) + 0.25 * CLIP(short text), normalised.
Mostly the photo (that is where style lives), plus a little text so colour and
type words in a query also land.

Search uses FAISS (inner product = cosine on normalised vectors). If FAISS is
not installed we fall back to plain NumPy, which is still instant for 3,000 items.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings
from . import embeddings as emb

IMAGE_WEIGHT = 0.75

# Free-text category words the LLM may use -> substrings of H&M product_type_name
CATEGORY_SYNONYMS = {
    "jewelry": ["earring", "necklace", "ring", "bracelet", "jewel"],
    "jewellery": ["earring", "necklace", "ring", "bracelet", "jewel"],
    "earring": ["earring"], "necklace": ["necklace"], "ring": ["ring"], "bracelet": ["bracelet", "bangle"],
    "bag": ["bag", "bumbag", "backpack", "wallet"], "purse": ["bag", "wallet"], "handbag": ["bag"],
    "hat": ["hat", "cap", "beanie"], "cap": ["cap"], "beanie": ["beanie"],
    "belt": ["belt"], "scarf": ["scarf"], "sunglasses": ["sunglasses"], "glasses": ["sunglasses"],
    "hair": ["hair", "headband", "alice"], "watch": ["watch"], "gloves": ["gloves"], "tie": ["tie"],
}


def _short_text(row: pd.Series) -> str:
    return f"{row['prod_name']}, {row['colour_group_name']} {row['product_type_name']}"


def category_mask(df: pd.DataFrame, category: str | None) -> pd.Series:
    if not category:
        return pd.Series(True, index=df.index)
    c = re.sub(r"[^a-z ]", "", category.lower()).strip()
    keys = []
    for word in c.split():
        stem = word[:-1] if word.endswith("s") and len(word) > 3 else word
        keys += CATEGORY_SYNONYMS.get(stem, CATEGORY_SYNONYMS.get(word, [stem]))
    types = df["product_type_name"].str.lower()
    mask = pd.Series(False, index=df.index)
    for k in set(keys):
        # "ring" is a substring of "earring", so match it exactly
        mask |= (types == k) if k == "ring" else types.str.contains(k, regex=False)
    return mask


class CatalogIndex:
    def __init__(self, catalog: pd.DataFrame, vectors: np.ndarray):
        self.catalog = catalog.reset_index(drop=True)
        self.vectors = emb.normalize(vectors)
        self._by_id = {int(a): i for i, a in enumerate(self.catalog["article_id"])}
        try:
            import faiss

            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
        except ImportError:
            self._faiss = None

    # ------------------------------------------------------------------ build / save / load
    @classmethod
    def build(cls, catalog: pd.DataFrame) -> "CatalogIndex":
        print(f"[index] encoding {len(catalog)} product photos with CLIP ...")
        img = emb.encode_images(catalog["image_path"])
        txt = emb.encode_texts(catalog.apply(_short_text, axis=1))
        return cls(catalog, IMAGE_WEIGHT * img + (1 - IMAGE_WEIGHT) * txt)

    def save(self, folder: Path | None = None) -> None:
        folder = folder or settings.index_dir
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        self.catalog.to_parquet(folder / "catalog.parquet")
        (folder / "meta.json").write_text(json.dumps({"clip_model": settings.clip_model, "n": len(self.catalog)}))

    @classmethod
    def load(cls, folder: Path | None = None) -> "CatalogIndex":
        folder = folder or settings.index_dir
        return cls(pd.read_parquet(folder / "catalog.parquet"), np.load(folder / "vectors.npy"))

    # ------------------------------------------------------------------ search
    def _scores(self, query_vec: np.ndarray) -> np.ndarray:
        q = emb.normalize(np.asarray(query_vec).reshape(1, -1))
        if self._faiss is not None:
            sims, idx = self._faiss.search(q, len(self.catalog))
            out = np.empty(len(self.catalog), dtype="float32")
            out[idx[0]] = sims[0]
            return out
        return (self.vectors @ q[0]).astype("float32")

    def search_vector(self, query_vec: np.ndarray, k: int = 10, category: str | None = None,
                      colour: str | None = None, exclude: set[int] | None = None) -> pd.DataFrame:
        df = self.catalog.copy()
        df["sim"] = self._scores(query_vec)
        mask = category_mask(df, category)
        if colour:
            cm = df["colour_group_name"].str.lower().str.contains(colour.lower().split()[-1], regex=False)
            if (mask & cm).sum() >= 3:      # only apply the colour filter if it leaves enough choice
                mask &= cm
        if exclude:
            mask &= ~df["article_id"].isin(exclude)
        res = df[mask]
        if res.empty:                        # category word we could not map -> search everything
            res = df
        return res.nlargest(k, "sim").reset_index(drop=True)

    def search_text(self, query: str, k: int = 10, **filters) -> pd.DataFrame:
        return self.search_vector(emb.encode_texts([query])[0], k=k, **filters)

    def vector_of(self, article_id: int) -> np.ndarray:
        return self.vectors[self._by_id[int(article_id)]]

    def rows(self, article_ids) -> pd.DataFrame:
        idx = [self._by_id[int(a)] for a in article_ids if int(a) in self._by_id]
        return self.catalog.iloc[idx].reset_index(drop=True)

    def similarity(self, query_vec: np.ndarray, article_ids) -> np.ndarray:
        q = emb.normalize(np.asarray(query_vec).reshape(1, -1))[0]
        idx = [self._by_id[int(a)] for a in article_ids]
        return self.vectors[idx] @ q

    def trending(self, category: str | None = None, k: int = 5) -> pd.DataFrame:
        df = self.catalog[category_mask(self.catalog, category)]
        if df.empty:
            df = self.catalog
        return df.nlargest(k, "trend_score").reset_index(drop=True)


def as_langchain_retriever(index: CatalogIndex, k: int = 8):
    """Wrap the index as a standard LangChain retriever (returns Documents),
    so it can be dropped into any LangChain chain as well as our agent."""
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict

    class CatalogRetriever(BaseRetriever):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        idx: CatalogIndex
        top_k: int = 8

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
            hits = self.idx.search_text(query, k=self.top_k)
            return [Document(page_content=r["text"],
                             metadata={"article_id": int(r["article_id"]), "sim": float(r["sim"]),
                                       "type": r["product_type_name"], "colour": r["colour_group_name"],
                                       "trend_score": float(r["trend_score"]), "image_path": r["image_path"]})
                    for _, r in hits.iterrows()]

    return CatalogRetriever(idx=index, top_k=k)
