# 👗 Personalized AI Fashion Assistant

**Upload an outfit photo → get accessory recommendations ranked as an "Influencer Suggestion".**
A multimodal **LangGraph** agent (Gemini vision + tool calling) searches a **RAG** index of 3,000 real best-selling H&M accessories, then ranks the candidates by **style fit, trend fit and colour harmony**, and explains the picks in the voice of different influencer personas.

`Python` · `LangChain` · `LangGraph` · `RAG` · `Gemini` · `CLIP` · `FAISS` · `FastAPI` · `Vercel`

**Try the web app: [finishing-touch.vercel.app](https://finishing-touch.vercel.app)** (connect it to a running model server; see [docs/WEB_APP.md](docs/WEB_APP.md))

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itsyaagirl/fashion-ai-assistant/blob/main/notebooks/fashion_assistant_colab.ipynb)

---

## What it does

```
📷 outfit photo ──► Gemini vision: "black slip dress, minimalist, date night, needs earrings + bag"
                     │
                     ▼
                   Agent loop (Gemini + tools)
                     ├─ lookup_style_rules("jewellery for black slip dress")   ← text RAG
                     ├─ search_accessories("thin gold drop earrings", "earrings") ← image RAG (CLIP)
                     └─ search_accessories("small black leather clutch", "bag")
                     │
                     ▼
                   Influencer score = style fit + trend fit + colour fit (+ persona taste)
                     │
                     ▼
⭐ Influencer Suggestion + one pick each from Mia the Minimalist, Jay the Streetwear Creator,
   Lena the Luxe Stylist, Bea the Boho Influencer — explained in their own voices
```

## Highlights

- **Multimodal RAG**: CLIP puts product photos, your outfit photo and text queries in one vector space (FAISS index), plus a text knowledge base of styling rules (LangChain vector store).
- **LangGraph agent with tool calling**: Gemini decides which categories to search and which rules to look up; a capped tool loop, a typed vision step (Pydantic structured output) and an early exit for non-outfit photos.
- **Real trend signal**: popularity and trend velocity come from **31M real H&M purchases**, not made-up scores. Optional **TikTok Shop** plug-in blends in your own export.
- **Explainable ranking**: the "Influencer Suggestion" is a transparent weighted score, so every pick comes with its evidence ("selling 2.3× its usual rate, gold warms up beige").
- **Side-by-side evaluation** against keyword search (BM25): offline metrics on real bought-together baskets + a blind LLM judge. See [docs/RESULTS.md](docs/RESULTS.md).
- **Web app on Vercel** ("Finishing Touch"): upload a photo, watch the agent work live (streamed graph steps), see the Influencer Suggestion with hang-tag scores, persona picks and the agent's tool trace. FastAPI backend runs in Colab behind a free Cloudflare tunnel.
- **19 tests** including a full graph run and the web API with a scripted fake LLM (no API key needed).

## Quick start (Colab, ~25 min end to end)

1. Get a free [Gemini API key](https://aistudio.google.com/apikey) and a [Kaggle API token](https://www.kaggle.com/settings).
2. Accept the [H&M competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules) once.
3. Open the notebook in Colab (badge above), choose a **T4 GPU** runtime, add secrets `GOOGLE_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`.
4. Run all cells. Section 6 prints a link to the web app, already connected to your Colab server.

## Local / command line

```bash
pip install -r requirements.txt
cp .env.example .env            # fill in keys, then export them
export PYTHONPATH=src

python -m fashion_assistant.cli prepare            # H&M data -> catalog + eval set + images
python -m fashion_assistant.cli index              # CLIP + FAISS index (adds TikTok scores if present)
python -m fashion_assistant.cli recommend my_outfit.jpg --request "summer wedding"
python -m fashion_assistant.cli eval --judge 20    # writes docs/RESULTS.md
python serve.py --share                            # web API + UI, public link
python app.py                                      # simple Gradio demo
pytest -q tests                                    # no keys or downloads needed
```

```python
from fashion_assistant import FashionAssistant
fa = FashionAssistant()
out = fa.recommend("my_outfit.jpg", request="for a summer wedding", personas=["minimalist", "luxe"])
print(out["final_answer"])        # markdown answer
out["ranked"]                     # top 5 with style/trend/colour scores + evidence
```

## Project structure

```
fashion-ai-assistant/
├── notebooks/fashion_assistant_colab.ipynb   # run everything here
├── src/fashion_assistant/
│   ├── config.py                 # every setting in one place (models, sizes, weights)
│   ├── data/hm_loader.py         # Kaggle download, popularity & trend scores, catalog, images
│   ├── data/tiktok_loader.py     # optional TikTok Shop trend boost
│   ├── rag/embeddings.py         # CLIP encoder (image + text, same space)
│   ├── rag/catalog_index.py      # FAISS index + LangChain retriever wrapper
│   ├── rag/style_kb.py           # styling-rules RAG (LangChain InMemoryVectorStore)
│   ├── agent/                    # LangGraph: state, schemas, prompts, tools, nodes, graph
│   ├── ranking/                  # influencer score, colour harmony, personas
│   ├── eval/                     # eval set from baskets, BM25 baseline, metrics, LLM judge
│   ├── assistant.py              # FashionAssistant (public API, incl. streaming)
│   ├── api.py                    # FastAPI server for the web app
│   ├── launch.py                 # start server + Cloudflare tunnel from Colab
│   └── cli.py
├── knowledge/style_guide.md      # the styling knowledge base (edit me!)
├── frontend/                     # Vercel web app (HTML/CSS/JS, no build step)
├── serve.py                      # run the web API outside Colab
├── app.py                        # simple Gradio demo
├── tests/                        # 19 tests, run offline
└── docs/
    ├── ARCHITECTURE.md           # diagrams, LangGraph logic, design decisions
    ├── HOW_WE_BUILT_IT.md        # step-by-step build story in plain language
    ├── WEB_APP.md                # front end, API, hosting
    └── RESULTS.md                # generated by the evaluation
```

## Results

Run section 5 of the notebook; it writes the tables and chart into [docs/RESULTS.md](docs/RESULTS.md). The evaluation compares four systems on the same ~200 real outfits (keyword search, popularity, CLIP retrieval, our Influencer ranker) and runs a blind A/B judge of the full agent vs keyword search.

## Data & license

- Code: MIT (see `LICENSE`).
- Data: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations), downloaded at run time, **non-commercial / academic use**; never committed to this repo.
- TikTok Shop: no public dataset exists; the plug-in only reads exports you create yourself, under your own responsibility for the platform's terms.

## Docs

- [Architecture & LangGraph logic](docs/ARCHITECTURE.md)
- [How we built it (plain-language walkthrough)](docs/HOW_WE_BUILT_IT.md)
- [Web app & API](docs/WEB_APP.md)
- [Results](docs/RESULTS.md)
