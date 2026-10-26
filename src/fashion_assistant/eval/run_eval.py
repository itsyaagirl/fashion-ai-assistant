"""Side-by-side evaluation: our system vs keyword search (and two simple baselines).

Part 1 - offline, free, all queries (no LLM):
    Keyword search (BM25)     garment text -> accessory text
    Popularity                best sellers, same list for everyone
    CLIP retrieval            garment photo -> accessory photos (our RAG step alone)
    Influencer ranker         CLIP retrieval + style/trend/colour re-ranking (our ranking step)
  scored against real bought-together accessories (hit@k, MRR, NDCG) plus
  quality of the top 5 (trend, colour harmony, variety).

Part 2 - blind LLM side-by-side, a sample of outfits (needs Gemini):
    full LangGraph agent's 5 picks  vs  keyword search's 5 picks
  A judge model sees the outfit photo and the two lists in random order and
  picks the better one. We report the agent's win / tie / loss rate.
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
import pandas as pd

from ..config import settings
from ..rag import embeddings as emb
from ..rag.catalog_index import CatalogIndex
from ..ranking.color import harmony_score
from ..ranking.influencer_score import score_candidates, trend_fit
from .baselines import KeywordSearch, popularity
from .metrics import hit_at_k, mrr, ndcg_at_k


# ----------------------------------------------------------------------------- our system, no LLM
def garment_query_vectors(eval_set: pd.DataFrame, image_paths: list[str]) -> np.ndarray:
    img = emb.encode_images(image_paths)
    txt = emb.encode_texts([f"an accessory to wear with a {c} {t}" for c, t in
                            zip(eval_set["garment_colour"], eval_set["garment_type"])])
    return emb.normalize(0.7 * img + 0.3 * txt)


def clip_retrieval(index: CatalogIndex, qvec: np.ndarray, k: int = 50) -> pd.DataFrame:
    return index.search_vector(qvec, k=k)


def influencer_rerank(hits: pd.DataFrame, garment_colour: str) -> pd.DataFrame:
    df = hits.rename(columns={"sim": "style_sim"})
    return score_candidates(df, [garment_colour], "editor")


# ----------------------------------------------------------------------------- part 1
def run_offline_eval(index: CatalogIndex, eval_set: pd.DataFrame, k: int = 10) -> tuple[pd.DataFrame, dict]:
    from ..data.hm_loader import local_image_path

    cat = index.catalog
    meta = cat.set_index("article_id")
    meta_tf = trend_fit(cat).set_axis(cat["article_id"])
    bm25 = KeywordSearch(cat)
    pop = popularity(cat, k)
    qvecs = garment_query_vectors(eval_set, [str(local_image_path(g)) for g in eval_set["garment_id"]])

    systems = {"Keyword search (BM25)": [], "Popularity": [], "CLIP retrieval": [], "Influencer ranker (ours)": []}
    for i, row in eval_set.reset_index(drop=True).iterrows():
        hits = clip_retrieval(index, qvecs[i])
        systems["Keyword search (BM25)"].append(bm25.search(row["garment_text"], k))
        systems["Popularity"].append(pop)
        systems["CLIP retrieval"].append(hits["article_id"].astype(int).tolist()[:k])
        systems["Influencer ranker (ours)"].append(
            influencer_rerank(hits, row["garment_colour"])["article_id"].astype(int).tolist()[:k])

    records = []
    for name, lists in systems.items():
        m = {"system": name}
        rel = [set(r) for r in eval_set["relevant_ids"]]
        m["hit@5"] = np.mean([hit_at_k(l, r, 5) for l, r in zip(lists, rel)])
        m[f"hit@{k}"] = np.mean([hit_at_k(l, r, k) for l, r in zip(lists, rel)])
        m[f"mrr@{k}"] = np.mean([mrr(l, r, k) for l, r in zip(lists, rel)])
        m[f"ndcg@{k}"] = np.mean([ndcg_at_k(l, r, k) for l, r in zip(lists, rel)])
        top5 = [l[:5] for l in lists]
        m["trend (top5)"] = np.mean([meta_tf.loc[l].mean() for l in top5])
        m["colour harmony (top5)"] = np.mean([
            np.mean([harmony_score([c], meta.loc[a, "colour_group_name"]) for a in l])
            for l, c in zip(top5, eval_set["garment_colour"])])
        m["variety (top5)"] = np.mean([meta.loc[l, "product_type_name"].nunique() / 5 for l in top5])
        records.append(m)
    table = pd.DataFrame(records).set_index("system").round(3)
    return table, systems


# ----------------------------------------------------------------------------- part 2
def _describe(ids, meta) -> str:
    return "\n".join(f"- {meta.loc[a, 'prod_name']} ({meta.loc[a, 'colour_group_name']} {meta.loc[a, 'product_type_name']})"
                     for a in ids)


def run_judge_eval(assistant, eval_set: pd.DataFrame, n: int = 20, seed: int = 0,
                   pause_s: float = 4.0) -> dict:
    """Blind A/B: full agent vs keyword search, judged by a separate Gemini model."""
    from langchain_core.messages import HumanMessage

    from ..agent.llm import get_llm, load_image_b64, message_text
    from ..agent.prompts import JUDGE_PROMPT
    from ..data.hm_loader import local_image_path

    judge = get_llm(settings.judge_model, temperature=0.0)
    cat = assistant.index.catalog
    meta = cat.set_index("article_id")
    bm25 = KeywordSearch(cat)
    rng = random.Random(seed)
    outcomes = []
    sample = eval_set.sample(min(n, len(eval_set)), random_state=seed)
    for _, row in sample.iterrows():
        img = str(local_image_path(row["garment_id"]))
        try:
            out = assistant.recommend(img, personas=[])
        except Exception as err:  # noqa: BLE001 - one failed call should not kill the eval
            print(f"[judge] agent failed on {row['garment_id']}: {err}")
            continue
        ours = [r["article_id"] for r in out["ranked"]][:5]
        kw = bm25.search(row["garment_text"], 5)
        ours_first = rng.random() < 0.5
        a, b = (ours, kw) if ours_first else (kw, ours)
        b64, mime = load_image_b64(img)
        msg = HumanMessage(content=[
            {"type": "text", "text": JUDGE_PROMPT.format(outfit=row["garment_text"], list_a=_describe(a, meta),
                                                         list_b=_describe(b, meta))},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ])
        verdict = message_text(judge.invoke([msg])).strip().upper()[:1]
        if verdict == "T" or verdict not in "AB":
            outcomes.append("tie")
        else:
            outcomes.append("win" if (verdict == "A") == ours_first else "loss")
        print(f"[judge] {len(outcomes)}/{len(sample)}: {outcomes[-1]}")
        time.sleep(pause_s)   # stay under free-tier rate limits
    c = pd.Series(outcomes).value_counts()
    total = max(len(outcomes), 1)
    return {"n": len(outcomes), "win": int(c.get("win", 0)), "tie": int(c.get("tie", 0)),
            "loss": int(c.get("loss", 0)), "win_rate": round(c.get("win", 0) / total, 3),
            "win_or_tie_rate": round((c.get("win", 0) + c.get("tie", 0)) / total, 3)}


# ----------------------------------------------------------------------------- report
def _md_table(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown()
    except ImportError:  # tabulate missing
        cols = ["system", *table.columns]
        rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        rows += ["| " + " | ".join([str(i), *[f"{v:.3f}" for v in r]]) + " |" for i, r in table.iterrows()]
        return "\n".join(rows)


def save_results(table: pd.DataFrame, judge: dict | None = None, n_queries: int | None = None) -> None:
    out = settings.docs_dir
    (out / "images").mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results_offline.csv")
    (out / "results.json").write_text(json.dumps({"offline": table.reset_index().to_dict(orient="records"),
                                                  "judge": judge, "n_queries": n_queries}, indent=2, default=float))
    try:
        import matplotlib.pyplot as plt

        cols = [c for c in table.columns if c.startswith(("hit@", "ndcg@", "mrr@"))]
        ax = table[cols].plot.bar(figsize=(9, 4), rot=0)
        ax.set_ylabel("score (higher is better)")
        ax.set_title("Accessory recommendation vs bought-together ground truth")
        plt.tight_layout()
        plt.savefig(out / "images" / "offline_metrics.png", dpi=150)
        plt.close()
    except Exception as err:  # noqa: BLE001
        print(f"[plot] skipped: {err}")

    md = ["# Results", "",
          "_Auto-generated by `run_eval.save_results()` - re-run the notebook to refresh._", "",
          f"Evaluation queries: **{n_queries or '?'}** real H&M baskets (garment + accessory bought together).", "",
          "## Offline comparison (all queries, no LLM)", "", _md_table(table), "",
          "![offline metrics](images/offline_metrics.png)", ""]
    if judge:
        md += ["## Blind side-by-side: full agent vs keyword search", "",
               f"Judge: `{settings.judge_model}`, {judge['n']} outfits, order randomised per outfit.", "",
               "| Agent wins | Ties | Keyword wins | Win rate | Win-or-tie rate |",
               "|---|---|---|---|---|",
               f"| {judge['win']} | {judge['tie']} | {judge['loss']} | {judge['win_rate']:.0%} | {judge['win_or_tie_rate']:.0%} |", ""]
    (out / "RESULTS.md").write_text("\n".join(md))
    print(f"[results] written to {out / 'RESULTS.md'}")
