"""Data pipeline + evaluation on a tiny synthetic dataset shaped exactly like H&M's files."""
import numpy as np
import pandas as pd
import pytest

from fashion_assistant.config import settings


@pytest.fixture
def fake_hm(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", tmp_path)
    monkeypatch.setattr(settings, "catalog_size", 30)
    settings.ensure_dirs()
    rng = np.random.default_rng(0)
    arts = []
    for i in range(60):
        acc = i < 40
        arts.append({
            "article_id": 500000 + i, "prod_name": f"P{i}",
            "product_type_name": ["Earring", "Bag", "Belt", "Necklace"][i % 4] if acc else ["Dress", "Blazer"][i % 2],
            "product_group_name": ("Accessories" if i % 4 != 1 else "Bags") if acc else "Garment Full body",
            "graphical_appearance_name": "Solid", "colour_group_name": ["Black", "Gold", "Beige"][i % 3],
            "perceived_colour_master_name": "x", "department_name": "d", "index_group_name": "Ladieswear",
            "section_name": "Womens", "garment_group_name": "g", "detail_desc": "nice item",
        })
    pd.DataFrame(arts).to_csv(settings.raw_dir / "articles.csv", index=False)
    days = pd.date_range("2020-05-01", "2020-09-22")
    tx = []
    for _ in range(4000):
        cust = f"c{rng.integers(0, 300)}"
        day = days[rng.integers(0, len(days))]
        tx.append({"t_dat": day.date(), "customer_id": cust, "article_id": 500000 + int(rng.integers(0, 60)),
                   "price": 0.03, "sales_channel_id": 2})
    pd.DataFrame(tx).to_csv(settings.raw_dir / "transactions_train.csv", index=False)
    return tmp_path


def test_popularity_catalog_evalset(fake_hm):
    from fashion_assistant.data import hm_loader
    from fashion_assistant.eval.build_eval_set import build_eval_set

    arts = hm_loader.load_articles()
    pop, recent = hm_loader.compute_popularity(chunksize=1000)
    assert pop["trend_score"].between(0, 1).all() and pop["total_sales"].sum() == 4000
    cat = hm_loader.build_catalog(arts, pop)
    assert len(cat) == 30 and set(cat["product_group_name"]) <= {"Accessories", "Bags"}
    assert cat["text"].str.len().min() > 5
    ev = build_eval_set(arts, cat, recent, n_queries=10)
    assert len(ev) > 0
    garments = set(arts.loc[arts["product_group_name"].str.startswith("Garment"), "article_id"])
    assert ev["garment_id"].isin(garments).all()
    assert all(set(r) <= set(cat["article_id"]) for r in ev["relevant_ids"])


def test_offline_eval_and_report(fake_hm, monkeypatch):
    from conftest import fake_vec

    try:
        import pyarrow  # noqa: F401  (Colab has it; minimal local envs may not)
    except ImportError:
        monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, p, **k: self.to_pickle(p))
        monkeypatch.setattr(pd, "read_parquet", lambda p, **k: pd.read_pickle(p))
    from fashion_assistant.data import hm_loader
    from fashion_assistant.eval.build_eval_set import build_eval_set
    from fashion_assistant.eval.run_eval import run_offline_eval, save_results
    from fashion_assistant.rag.catalog_index import CatalogIndex

    arts = hm_loader.load_articles()
    pop, recent = hm_loader.compute_popularity(chunksize=1000)
    cat = hm_loader.build_catalog(arts, pop)
    index = CatalogIndex(cat, np.stack([fake_vec(a) for a in cat["article_id"]]))
    index.save()
    index2 = CatalogIndex.load()
    assert len(index2.catalog) == len(cat)
    ev = build_eval_set(arts, cat, recent, n_queries=10)
    table, systems = run_offline_eval(index2, ev, k=10)
    assert set(table.index) == {"Keyword search (BM25)", "Popularity", "CLIP retrieval", "Influencer ranker (ours)"}
    assert table.to_numpy().min() >= 0
    save_results(table, {"n": 3, "win": 2, "tie": 1, "loss": 0, "win_rate": 0.667, "win_or_tie_rate": 1.0}, len(ev))
    md = (settings.docs_dir / "RESULTS.md").read_text()
    assert "Influencer ranker" in md and "Agent wins" in md


def test_style_kb_with_fake_embeddings():
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from fashion_assistant.rag.style_kb import StyleKnowledgeBase

    kb = StyleKnowledgeBase(embeddings=DeterministicFakeEmbedding(size=16))
    assert kb.n_sections >= 15
    hits = kb.search("what necklace for a v-neck", k=3)
    assert len(hits) == 3 and all("section" in h for h in hits)
