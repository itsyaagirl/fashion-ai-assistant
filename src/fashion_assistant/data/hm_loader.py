"""H&M dataset pipeline.

Steps
-----
1. download_tables()      -> articles.csv + transactions_train.csv from Kaggle
2. compute_popularity()   -> per-article total sales, recent sales and a trend score
3. build_catalog()        -> top-N accessories (Accessories + Bags) with text + trend fields
4. download_images()      -> only the product photos we need (not the 30 GB image dump)

The H&M data is licensed for non-commercial / academic use via the Kaggle
competition rules. You must accept the rules once on kaggle.com before downloading.
"""
from __future__ import annotations

import random
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings

ARTICLE_COLS = [
    "article_id", "prod_name", "product_type_name", "product_group_name",
    "graphical_appearance_name", "colour_group_name", "perceived_colour_master_name",
    "department_name", "index_group_name", "section_name", "garment_group_name", "detail_desc",
]


# ----------------------------------------------------------------------------- helpers
def _kaggle_api():
    """Authenticated Kaggle client. Needs KAGGLE_USERNAME / KAGGLE_KEY env vars or ~/.kaggle/kaggle.json."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _unzip_if_needed(path: Path, dest_dir: Path) -> None:
    """Kaggle sometimes serves single files zipped (file.csv.zip). Unzip in place."""
    zipped = path.with_name(path.name + ".zip")
    if zipped.exists() and not path.exists():
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(dest_dir)
        zipped.unlink()


def article_image_relpath(article_id: int) -> str:
    """H&M stores images as images/0XX/0XXXXXXXXX.jpg (10-digit zero-padded id)."""
    aid = f"{int(article_id):010d}"
    return f"images/{aid[:3]}/{aid}.jpg"


def local_image_path(article_id: int) -> Path:
    return settings.image_dir / f"{int(article_id):010d}.jpg"


# ----------------------------------------------------------------------------- 1. download
def download_tables(force: bool = False) -> None:
    settings.ensure_dirs()
    api = _kaggle_api()
    for fname in ("articles.csv", "transactions_train.csv"):
        target = settings.raw_dir / fname
        if target.exists() and not force:
            print(f"[skip] {fname} already present")
            continue
        print(f"[download] {fname} ... (transactions is ~600 MB zipped, takes a few minutes)")
        api.competition_download_file(settings.kaggle_competition, fname, path=str(settings.raw_dir), quiet=False)
        _unzip_if_needed(target, settings.raw_dir)


def load_articles() -> pd.DataFrame:
    df = pd.read_csv(settings.raw_dir / "articles.csv", usecols=ARTICLE_COLS, dtype={"article_id": "int64"})
    df["detail_desc"] = df["detail_desc"].fillna("")
    return df


# ----------------------------------------------------------------------------- 2. popularity & trend
def compute_popularity(chunksize: int = 5_000_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream the 31M-row transactions file once.

    Returns
    -------
    pop : one row per article with
        total_sales      - all purchases 2018-09 .. 2020-09
        recent_sales     - last `trend_recent_weeks` weeks
        baseline_sales   - the `trend_baseline_weeks` weeks before that
        popularity_score - percentile rank of log(total_sales)          (0..1)
        trend_score      - percentile rank of the recent-vs-baseline lift (0..1)
    recent_tx : transactions from the last `eval_window_weeks` weeks
        (customer hashed to int64 to save memory) used to build the eval set.
    """
    path = settings.raw_dir / "transactions_train.csv"
    # First pass on a small slice is not enough to find the end date, so we use the known one
    # and verify it while streaming.
    end = pd.Timestamp("2020-09-22")
    recent_start = end - pd.Timedelta(weeks=settings.trend_recent_weeks)
    base_start = recent_start - pd.Timedelta(weeks=settings.trend_baseline_weeks)
    eval_start = end - pd.Timedelta(weeks=settings.eval_window_weeks)

    total, recent, base = {}, {}, {}
    recent_parts = []
    max_seen = pd.Timestamp("1900-01-01")

    reader = pd.read_csv(
        path, usecols=["t_dat", "customer_id", "article_id"],
        dtype={"article_id": "int64"}, parse_dates=["t_dat"], chunksize=chunksize,
    )
    for i, chunk in enumerate(reader):
        max_seen = max(max_seen, chunk["t_dat"].max())
        for store, mask in (
            (total, slice(None)),
            (recent, chunk["t_dat"] > recent_start),
            (base, (chunk["t_dat"] > base_start) & (chunk["t_dat"] <= recent_start)),
        ):
            counts = chunk.loc[mask, "article_id"].value_counts()
            for aid, c in counts.items():
                store[aid] = store.get(aid, 0) + int(c)
        ev = chunk[chunk["t_dat"] > eval_start].copy()
        if len(ev):
            ev["customer"] = pd.util.hash_pandas_object(ev["customer_id"], index=False).to_numpy()  # uint64
            recent_parts.append(ev[["t_dat", "customer", "article_id"]])
        print(f"  chunk {i + 1}: {len(chunk):,} rows")

    if max_seen != end:
        print(f"[warn] data ends {max_seen.date()}, expected {end.date()} - windows still computed from {end.date()}")

    pop = pd.DataFrame({"total_sales": pd.Series(total)}).fillna(0)
    pop["recent_sales"] = pd.Series(recent)
    pop["baseline_sales"] = pd.Series(base)
    pop = pop.fillna(0).astype("int64")
    pop.index.name = "article_id"

    # Weekly rates so windows of different length are comparable; +1 smoothing avoids /0
    recent_rate = pop["recent_sales"] / settings.trend_recent_weeks
    base_rate = pop["baseline_sales"] / settings.trend_baseline_weeks
    pop["trend_lift"] = (recent_rate + 1) / (base_rate + 1)
    pop["popularity_score"] = np.log1p(pop["total_sales"]).rank(pct=True)
    # Trend = growth, but damped for items with almost no recent sales (a 0 -> 2 jump is noise).
    pop["trend_score"] = (np.log(pop["trend_lift"]) * np.log1p(pop["recent_sales"])).rank(pct=True)
    recent_tx = pd.concat(recent_parts, ignore_index=True) if recent_parts else pd.DataFrame()
    return pop.reset_index(), recent_tx


# ----------------------------------------------------------------------------- 3. catalog
def item_text(row: pd.Series) -> str:
    """One readable line per product; used for CLIP text side, BM25 baseline and LLM prompts."""
    parts = [row["prod_name"], row["product_type_name"], row["colour_group_name"],
             row["graphical_appearance_name"], row["section_name"]]
    desc = str(row.get("detail_desc", "") or "")
    return " | ".join(str(p) for p in parts if isinstance(p, str) and p) + (f". {desc}" if desc else "")


def build_catalog(articles: pd.DataFrame, pop: pd.DataFrame, size: int | None = None) -> pd.DataFrame:
    size = size or settings.catalog_size
    acc = articles[articles["product_group_name"].isin(settings.accessory_groups)]
    acc = acc.merge(pop, on="article_id", how="left").fillna(
        {"total_sales": 0, "recent_sales": 0, "baseline_sales": 0, "trend_lift": 1.0,
         "popularity_score": 0.0, "trend_score": 0.0})
    # Keep what people actually buy: rank by a blend of recent and lifetime sales.
    acc["_rank"] = 0.6 * acc["recent_sales"].rank(pct=True) + 0.4 * acc["total_sales"].rank(pct=True)
    cat = acc.sort_values("_rank", ascending=False).head(size).drop(columns="_rank").reset_index(drop=True)
    # Re-rank scores *within* the catalog so 0..1 means "relative to other accessories".
    cat["popularity_score"] = cat["total_sales"].rank(pct=True)
    cat["trend_score"] = cat["trend_score"].rank(pct=True)
    cat["text"] = cat.apply(item_text, axis=1)
    cat["image_path"] = cat["article_id"].map(lambda a: str(local_image_path(a)))
    return cat


# ----------------------------------------------------------------------------- 4. images
_last_errors: list[str] = []
_rate = {"consecutive_429": 0, "stopped": False}
_rate_lock = threading.Lock()
RATE_LIMIT_STOP_AFTER = 12   # this many 429s in a row = Kaggle has blocked us for now; stop asking


def _is_rate_limit(msg: str) -> bool:
    return "429" in msg or "Too Many Requests" in msg


def _download_one(api, article_id: int, attempts: int = 3) -> bool:
    """Download one product photo. Retries briefly on hiccups; if Kaggle starts refusing
    requests (HTTP 429) repeatedly, the whole batch stops instead of hammering the API,
    which only makes the block last longer."""
    out = local_image_path(article_id)
    if out.exists():
        return True
    tmp_dir = settings.raw_dir / "_img_tmp" / f"{int(article_id):010d}"
    for attempt in range(attempts):
        if _rate["stopped"]:
            return False
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            api.competition_download_file(settings.kaggle_competition, article_image_relpath(article_id),
                                          path=str(tmp_dir), quiet=True)
            for f in tmp_dir.iterdir():
                if f.suffix == ".zip":
                    with zipfile.ZipFile(f) as zf:
                        zf.extractall(tmp_dir)
            jpgs = list(tmp_dir.rglob("*.jpg"))
            with _rate_lock:
                _rate["consecutive_429"] = 0
            if jpgs:
                shutil.move(str(jpgs[0]), out)
                return True
            return False                      # downloaded fine, but no photo inside
        except Exception as err:  # noqa: BLE001
            msg = str(err)
            if "404" in msg or "Not Found" in msg:
                return False                  # this article genuinely has no photo
            if len(_last_errors) < 5:
                _last_errors.append(f"{err.__class__.__name__}: {msg[:150]}")
            if _is_rate_limit(msg):
                with _rate_lock:
                    _rate["consecutive_429"] += 1
                    if _rate["consecutive_429"] >= RATE_LIMIT_STOP_AFTER:
                        _rate["stopped"] = True
            time.sleep(2 ** attempt + random.random())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return False


def download_images(article_ids, workers: int = 4) -> set[int]:
    """Fetch individual product photos in parallel. Returns the ids that are on disk afterwards."""
    settings.ensure_dirs()
    ids = [int(a) for a in article_ids]
    ok = {a for a in ids if local_image_path(a).exists()}
    todo = [a for a in ids if a not in ok]
    _last_errors.clear()
    _rate.update(consecutive_429=0, stopped=False)
    if todo:
        api = _kaggle_api()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_one, api, a): a for a in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                if fut.result():
                    ok.add(futures[fut])
                if n % 250 == 0:
                    print(f"  images: {n}/{len(todo)} new ({len(ok)} on disk)")
    print(f"[images] {len(ok)}/{len(ids)} photos available")
    if _rate["stopped"]:
        print("[images] Kaggle is rate-limiting this account (HTTP 429), so downloading stopped early. "
              "Everything still works; photos already saved are kept. Try again in an hour for more.")
    elif _last_errors:
        print("[images] some downloads failed, e.g.:\n   " + "\n   ".join(_last_errors[:3]))
    return ok


# ----------------------------------------------------------------------------- one-call pipeline
def _finish_eval_set(eval_set: pd.DataFrame, eval_queries: int, with_images: bool,
                     photos_wanted: int = 40) -> pd.DataFrame:
    """Keep every outfit (the offline eval can use the garment's text when there is no photo)
    and fetch photos for a small number of them (the blind LLM judge needs real photos)."""
    print(f"[eval] {len(eval_set)} candidate outfits found in real baskets")
    eval_set = eval_set.head(eval_queries).reset_index(drop=True)
    if with_images and len(eval_set):
        download_images(eval_set["garment_id"].head(photos_wanted))
    eval_set["has_photo"] = eval_set["garment_id"].map(lambda g: local_image_path(g).exists()) if len(eval_set) else []
    eval_set.to_parquet(settings.processed_dir / "eval_set.parquet")
    print(f"[eval] {len(eval_set)} outfit queries saved, {int(eval_set['has_photo'].sum()) if len(eval_set) else 0} with a garment photo")
    return eval_set


def prepare_all(with_images: bool = True, eval_queries: int = 200, seed: int = 7) -> pd.DataFrame:
    """Run the whole data pipeline and save parquet files in data/processed/."""
    from ..eval.build_eval_set import build_eval_set

    settings.ensure_dirs()
    download_tables()
    articles = load_articles()
    articles.to_parquet(settings.processed_dir / "articles.parquet")
    print(f"[articles] {len(articles):,} products")
    pop, recent_tx = compute_popularity()
    recent_tx.to_parquet(settings.processed_dir / "recent_transactions.parquet")
    # Over-select a little because some photos are missing, then trim.
    catalog = build_catalog(articles, pop, size=int(settings.catalog_size * 1.1))
    if with_images:
        ok = download_images(catalog["article_id"])
        catalog = catalog[catalog["article_id"].isin(ok)]
    catalog = catalog.head(settings.catalog_size).reset_index(drop=True)
    catalog.to_parquet(settings.processed_dir / "catalog.parquet")
    print(f"[catalog] {len(catalog):,} accessories saved")

    eval_set = build_eval_set(articles, catalog, recent_tx, n_queries=eval_queries, seed=seed)
    _finish_eval_set(eval_set, eval_queries, with_images)
    return catalog


def rebuild_eval_set(eval_queries: int = 200, seed: int = 7, with_images: bool = True) -> pd.DataFrame:
    """Rebuild only the evaluation set (no re-download of the big files, photos already
    on disk are skipped). Use it if the eval set came out empty or too small."""
    from ..eval.build_eval_set import build_eval_set

    articles = pd.read_parquet(settings.processed_dir / "articles.parquet")
    catalog = pd.read_parquet(settings.processed_dir / "catalog.parquet")
    cached = settings.processed_dir / "recent_transactions.parquet"
    if cached.exists():
        recent_tx = pd.read_parquet(cached)
    else:
        print("[eval] re-reading recent transactions (a few minutes) ...")
        _, recent_tx = compute_popularity()
        recent_tx.to_parquet(cached)
    print(f"[eval] {len(recent_tx):,} transactions in the last {settings.eval_window_weeks} weeks")
    eval_set = build_eval_set(articles, catalog, recent_tx, n_queries=eval_queries, seed=seed)
    return _finish_eval_set(eval_set, eval_queries, with_images)
