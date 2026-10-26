"""CLIP encoder: puts product photos, outfit photos and text queries in ONE vector space.

That shared space is what makes the RAG "multimodal": the agent can search the
catalog with words ("gold hoop earrings") and the ranker can compare the
user's outfit photo directly with product photos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ..config import settings

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(settings.clip_model)
    return _MODEL


def _to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.open(Path(x)).convert("RGB")


def encode_images(images: Iterable, batch_size: int = 64) -> np.ndarray:
    pil = [_to_pil(x) for x in images]
    return _model().encode(pil, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=len(pil) > 200).astype("float32")


def encode_texts(texts: Iterable[str], batch_size: int = 128) -> np.ndarray:
    # CLIP's text encoder only reads ~77 tokens, so keep inputs short.
    short = [str(t)[:200] for t in texts]
    return _model().encode(short, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True).astype("float32")


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-9, None)
