# Results

_This file is overwritten automatically by section 5 of the notebook (`run_eval.save_results()`)._

**Not run yet.** Open `notebooks/fashion_assistant_colab.ipynb`, run all cells, then commit the regenerated `docs/RESULTS.md`, `docs/results.json`, `docs/results_offline.csv` and `docs/images/offline_metrics.png`.

What will appear here:

1. **Offline comparison** on ~200 real H&M outfits (garment + accessory bought together): Hit@5, Hit@10, MRR@10, NDCG@10, plus trend, colour harmony and variety of the top 5, for Keyword search (BM25), Popularity, CLIP retrieval and the Influencer ranker.
2. **Blind side-by-side**: full LangGraph agent vs keyword search on 20 outfits, judged by a separate Gemini model with randomised order. Win / tie / loss counts and win rate.
