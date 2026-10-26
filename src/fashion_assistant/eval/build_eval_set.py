"""Build an evaluation set from real H&M shopping baskets.

Idea: if a customer bought a garment AND an accessory on the same day, that
accessory is a real-world "goes with this" signal. For each such basket we use
the garment photo as the query ("the outfit") and the accessories bought with it
as the ground-truth relevant items.

This is a proxy, not perfect ground truth (people also buy unrelated things
together), which is why the evaluation also includes a blind LLM side-by-side
judgement of style quality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings


def build_eval_set(articles: pd.DataFrame, catalog: pd.DataFrame, recent_tx: pd.DataFrame,
                   n_queries: int = 200, seed: int = 7) -> pd.DataFrame:
    if recent_tx is None or recent_tx.empty:
        return pd.DataFrame(columns=["garment_id", "garment_text", "garment_colour", "garment_type", "relevant_ids"])

    garment_ids = set(articles.loc[articles["product_group_name"].isin(settings.garment_groups), "article_id"])
    catalog_ids = set(catalog["article_id"])

    key = ["customer", "t_dat"]
    garments = (recent_tx[recent_tx["article_id"].isin(garment_ids)]
                .groupby(key)["article_id"].agg(lambda s: sorted(set(s))).rename("garments"))
    accessories = (recent_tx[recent_tx["article_id"].isin(catalog_ids)]
                   .groupby(key)["article_id"].agg(lambda s: sorted(set(s))).rename("accessories"))
    # inner join = baskets that contain at least one garment AND one catalog accessory
    baskets = pd.concat([garments, accessories], axis=1, join="inner")

    rng = np.random.default_rng(seed)
    rows, used_garments = [], set()
    for _, b in baskets.sample(frac=1.0, random_state=seed).iterrows():
        g = int(rng.choice(b["garments"]))
        if g in used_garments:          # one query per garment keeps the set diverse
            continue
        used_garments.add(g)
        rows.append({"garment_id": g, "relevant_ids": [int(a) for a in b["accessories"]]})
        if len(rows) >= n_queries * 2:   # over-sample: some garment photos may be missing
            break

    ev = pd.DataFrame(rows)
    if ev.empty:
        return pd.DataFrame(columns=["garment_id", "garment_text", "garment_colour", "garment_type", "relevant_ids"])
    meta = articles.set_index("article_id")
    from ..data.hm_loader import item_text

    ev["garment_text"] = ev["garment_id"].map(lambda a: item_text(meta.loc[a]))
    ev["garment_colour"] = ev["garment_id"].map(lambda a: meta.loc[a, "colour_group_name"])
    ev["garment_type"] = ev["garment_id"].map(lambda a: meta.loc[a, "product_type_name"])
    return ev.head(n_queries * 2).reset_index(drop=True)
