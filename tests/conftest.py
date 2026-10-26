"""Shared fixtures: a tiny synthetic catalog and a fake CLIP, so tests run in seconds
with no downloads, no GPU and no API key."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DIM = 32
TYPES = ["Earring", "Necklace", "Bag", "Belt", "Hat/beanie", "Sunglasses", "Scarf", "Bracelet"]
COLOURS = ["Black", "Gold", "Silver", "Beige", "Dark Blue", "Red", "White", "Brown"]


def fake_vec(key: str) -> np.ndarray:
    seed = int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=DIM).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def fake_clip(monkeypatch):
    from fashion_assistant.rag import embeddings as emb

    monkeypatch.setattr(emb, "encode_texts", lambda texts, **k: np.stack([fake_vec(t) for t in texts]))
    monkeypatch.setattr(emb, "encode_images", lambda imgs, **k: np.stack([fake_vec(str(i)[:40]) for i in imgs]))


@pytest.fixture
def catalog() -> pd.DataFrame:
    rows = []
    for i in range(80):
        t, c = TYPES[i % len(TYPES)], COLOURS[(i * 3) % len(COLOURS)]
        rows.append({
            "article_id": 100000 + i, "prod_name": f"Item {i} {t.split('/')[0]}", "product_type_name": t,
            "product_group_name": "Accessories", "colour_group_name": c, "graphical_appearance_name": "Solid",
            "section_name": "Accessories", "detail_desc": "leather simple" if i % 5 == 0 else "chunky chain",
            "total_sales": 1000 - i * 10, "recent_sales": (i * 7) % 50, "baseline_sales": 30,
            "trend_lift": 1 + (i % 9) / 4, "popularity_score": 1 - i / 80, "trend_score": ((i * 13) % 80) / 80,
            "image_path": f"/nowhere/{i}.jpg",
        })
    df = pd.DataFrame(rows)
    df["text"] = df["prod_name"] + " | " + df["product_type_name"] + " | " + df["colour_group_name"] + ". " + df["detail_desc"]
    return df


@pytest.fixture
def index(catalog):
    from fashion_assistant.rag.catalog_index import CatalogIndex

    vecs = np.stack([fake_vec(a) for a in catalog["article_id"]])
    return CatalogIndex(catalog, vecs)
