"""Command line: python -m fashion_assistant.cli <command>

    prepare     download H&M data, build catalog + eval set, fetch images
    index       build the CLIP vector index (adds TikTok scores if files are in data/tiktok/)
    recommend   python -m fashion_assistant.cli recommend outfit.jpg --request "for a wedding"
    eval        offline comparison (+ --judge N for the LLM side-by-side)
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import settings


def cmd_prepare(a):
    from .data.hm_loader import prepare_all

    prepare_all(with_images=not a.no_images, eval_queries=a.eval_queries)


def cmd_index(a):
    from .data.tiktok_loader import attach_tiktok_scores, load_tiktok_products, trending_terms
    from .rag.catalog_index import CatalogIndex

    cat = pd.read_parquet(settings.processed_dir / "catalog.parquet")
    terms = trending_terms(load_tiktok_products())
    print(f"[tiktok] {len(terms)} trending terms" if terms else "[tiktok] no exports found - skipping")
    cat = attach_tiktok_scores(cat, terms)
    CatalogIndex.build(cat).save()
    print(f"[index] saved to {settings.index_dir}")


def cmd_recommend(a):
    from .assistant import FashionAssistant

    out = FashionAssistant().recommend(a.image, request=a.request, personas=a.personas)
    print(out["final_answer"])


def cmd_eval(a):
    from .eval.run_eval import run_judge_eval, run_offline_eval, save_results
    from .rag.catalog_index import CatalogIndex

    index = CatalogIndex.load()
    ev = pd.read_parquet(settings.processed_dir / "eval_set.parquet")
    table, _ = run_offline_eval(index, ev)
    print(table.to_string())
    judge = None
    if a.judge:
        from .assistant import FashionAssistant

        judge = run_judge_eval(FashionAssistant(index=index), ev, n=a.judge)
        print(judge)
    save_results(table, judge, len(ev))


def main():
    p = argparse.ArgumentParser(prog="fashion_assistant")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("prepare"); s.add_argument("--no-images", action="store_true")
    s.add_argument("--eval-queries", type=int, default=200); s.set_defaults(fn=cmd_prepare)
    s = sub.add_parser("index"); s.set_defaults(fn=cmd_index)
    s = sub.add_parser("recommend"); s.add_argument("image"); s.add_argument("--request", default="")
    s.add_argument("--personas", nargs="*", default=["minimalist", "streetwear", "luxe", "boho"]); s.set_defaults(fn=cmd_recommend)
    s = sub.add_parser("eval"); s.add_argument("--judge", type=int, default=0); s.set_defaults(fn=cmd_eval)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
