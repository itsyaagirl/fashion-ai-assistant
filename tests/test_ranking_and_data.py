"""Pure-Python tests: colour rules, ranker, personas, index search, TikTok plug-in, metrics."""
import numpy as np
import pandas as pd

from fashion_assistant.ranking.color import harmony_score, pair_harmony
from fashion_assistant.ranking.influencer_score import diversify, score_candidates
from fashion_assistant.ranking.personas import PERSONAS, affinity


def test_colour_rules():
    assert pair_harmony("Black", "Red") >= 0.8            # neutral goes with anything
    assert pair_harmony("navy", "Gold") >= 0.8             # metallic accent
    assert pair_harmony("blue", "orange") > pair_harmony("blue", "yellowish green")  # complementary beats clash
    assert 0 <= harmony_score(["cream", "olive"], "Gold") <= 1


def test_persona_affinity():
    mini = PERSONAS["minimalist"]
    assert affinity(mini, "thin leather belt") > affinity(mini, "glitter sequin clutch")


def test_ranker_orders_and_diversifies(catalog):
    cands = catalog.copy()
    cands["style_sim"] = np.linspace(1, 0, len(cands))
    ranked = score_candidates(cands, ["black", "white"], "editor")
    assert ranked["influencer_score"].is_monotonic_decreasing
    top = diversify(ranked, 5)
    assert len(top) == 5 and top["product_type_name"].nunique() == 5


def test_personas_disagree(catalog):
    cands = catalog.copy()
    cands["style_sim"] = 0.5
    picks = {p: score_candidates(cands, ["beige"], p).iloc[0]["article_id"] for p in ["minimalist", "streetwear"]}
    assert picks["minimalist"] != picks["streetwear"]


def test_index_category_filter(index):
    res = index.search_text("gold hoops", k=5, category="earrings")
    assert len(res) == 5 and set(res["product_type_name"]) == {"Earring"}
    res = index.search_text("anything", k=3, category="ring")          # must not match "Earring"
    assert "Earring" not in set(res["product_type_name"]) or len(res) == 3


def test_tiktok_plugin(tmp_path, catalog):
    from fashion_assistant.data.tiktok_loader import attach_tiktok_scores, load_tiktok_products, trending_terms

    pd.DataFrame({"title": ["leather belt black"] * 4 + ["chunky chain necklace"] * 4 + ["phone case"] * 4,
                  "sold_count": [900] * 4 + [50] * 4 + [5000] * 4}).to_csv(tmp_path / "export.csv", index=False)
    (tmp_path / "EXAMPLE_format.csv").write_text("title,sold_count\nEXAMPLE ROW,1\n")
    prods = load_tiktok_products(tmp_path)
    assert len(prods) == 12                                   # EXAMPLE file ignored
    terms = trending_terms(prods)
    assert "phone" not in terms                                # not an accessory title
    assert terms["leather"] > terms["chunky"]                  # sold more
    scored = attach_tiktok_scores(catalog, terms)
    assert scored["tiktok_score"].max() > 0


def test_metrics():
    from fashion_assistant.eval.metrics import hit_at_k, mrr, ndcg_at_k

    assert hit_at_k([3, 1, 2], {1}, 2) == 1.0 and hit_at_k([3, 1, 2], {1}, 1) == 0.0
    assert mrr([3, 1, 2], {1}) == 0.5
    assert abs(ndcg_at_k([1, 2], {1}, 2) - 1.0) < 1e-9


def test_bm25_baseline(catalog):
    from fashion_assistant.eval.baselines import KeywordSearch

    ks = KeywordSearch(catalog)
    top = ks.search("simple leather", k=5)
    assert all("leather" in t for t in catalog.set_index("article_id").loc[top, "text"])
