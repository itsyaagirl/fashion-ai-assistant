# How We Built It

A plain-language walkthrough of the whole project: what each piece is, why it's there, and how they connect. If you can explain this document, you can explain the project in an interview.

---

## Part 1 — The idea in one minute

You give the app a photo of an outfit. It answers: *"Wear these gold hoops and this black crossbody bag, and here's why."*

To do that well, three different skills are needed, and each one is done by the tool best at it:

| Skill | Who does it | Why that tool |
|---|---|---|
| **See** the outfit (garments, colours, style, occasion) | Gemini (a vision LLM) | LLMs are great at describing images in words |
| **Find** real products that could match | RAG: CLIP vectors + FAISS search | Search must return *real* products, not invented ones |
| **Decide** the order and **explain** it | A scoring formula + Gemini | Maths is consistent and uses sales data; the LLM writes nicely |

LangGraph is the "conductor" that runs these steps in the right order and lets the LLM call search tools as many times as it needs (up to a limit).

---

## Part 2 — Key concepts (glossary)

**LLM (Large Language Model).** A model that reads and writes text. Gemini is multimodal, so it can also read images.

**Embedding / vector.** A list of numbers (here 512) that represents the *meaning* of an image or text. Similar things get similar vectors. "Similar" is measured with **cosine similarity** (1 = identical direction, 0 = unrelated).

**CLIP.** A model trained on 400M image–caption pairs to put pictures and their descriptions close together. So the text "gold hoop earrings" lands near *photos* of gold hoops. That shared space is what makes our search **multimodal**.

**FAISS.** A library from Meta for searching vectors fast. We give it 3,000 product vectors; it returns the closest ones to any query vector.

**RAG (Retrieval-Augmented Generation).** Instead of letting the LLM answer from memory (where it might invent products), we first **retrieve** real data (products, styling rules) and give it to the LLM. We use RAG twice:
1. *Product RAG*: CLIP + FAISS over the catalog.
2. *Knowledge RAG*: a styling guide split into sections, embedded with a small text model, searched with LangChain's vector store.

**Tool calling.** We describe Python functions to the LLM (name, arguments, docstring). The LLM can reply "call `search_accessories(query='thin gold hoops', category='earrings')`". Our code runs it and sends the result back.

**LangChain.** The library that gives us standard building blocks: chat model wrappers (`ChatGoogleGenerativeAI`), messages, `@tool`, `Document`, vector stores, retrievers.

**LangGraph.** Built on LangChain. You define a **graph**: nodes are Python functions, edges say what runs next, and **conditional edges** let the code (or the LLM's answer) choose the path. A shared **state** dict is passed along. This is how you build *agents* that loop (think → call tool → think again) safely.

**Structured output.** Asking the LLM to fill a fixed schema (a Pydantic class) instead of free text, so the code can rely on `outfit.colors` being a list of strings.

**BM25.** The classic keyword-search formula used by search engines before vectors. It's our baseline: "what would a normal shop search box return?"

---

## Part 3 — Step-by-step build

### Step 1: Choose the data (`data/hm_loader.py`)

We wanted trendy, popular products. TikTok Shop has no free public dataset (only paid scrapers whose data can't be shared), so we chose the **H&M Personalized Fashion Recommendations** dataset on Kaggle:

- `articles.csv` — 105k products: name, type ("Earring", "Bag"…), colour, description.
- `transactions_train.csv` — **31 million real purchases**, Sept 2018 → Sept 2020.
- `images/` — one photo per product.

**Popularity & trend.** We stream the 31M rows in chunks (so it fits in Colab's RAM) and count, for every product:
- total sales (all time),
- recent sales (last 4 weeks of the data),
- baseline sales (the 12 weeks before that).

Then:
- **popularity score** = how its total sales rank against others (percentile, 0..1).
- **trend score** = is it selling *faster* now than before? We compare weekly rates (`lift`) and multiply by `log(1 + recent sales)` so that "0 → 2 sales" is not mistaken for a trend. Then we take the percentile.

**Catalog.** We keep the 3,000 best-selling products from the *Accessories* and *Bags* groups and download only their photos (one at a time from Kaggle, 8 in parallel), not the whole 30 GB image archive.

**Evaluation set.** From the last 8 weeks we find **baskets**: the same customer on the same day buying a garment *and* a catalog accessory. The garment photo becomes a test "outfit", and the accessories in that basket are the "right answers". We sample 200.

> Note: in this data, "trending" means trending in September 2020, when the dataset ends.

### Step 2: Optional TikTok boost (`data/tiktok_loader.py`)

If you export TikTok Shop products yourself, the loader reads any CSV/JSON (it recognises common column names like `title`, `sold_count`), keeps titles that mention accessories, and scores each word and word-pair by the sales volume it appears in. Every H&M product then gets a `tiktok_score` = its strongest trending word (e.g. a pearl necklace scores high if "pearl" sells a lot on TikTok). The ranker blends it 30% into the trend score. No export means the score is 0 and nothing breaks.

### Step 3: Build the RAG indexes (`rag/`)

**Product index.** For each product: `0.75 × CLIP(photo) + 0.25 × CLIP("name, colour type")`. Mostly the photo, because style lives in how a thing looks; a bit of text so colour and type words in queries land well. Stored in FAISS. We also wrap it as a standard **LangChain retriever** so it plugs into any LangChain chain.

**Style knowledge base.** `knowledge/style_guide.md` holds 16 short sections of styling rules (necklines, colour rules, metals, occasions, seasons, personas). It's split on headings, embedded with `all-MiniLM-L6-v2` and stored in `InMemoryVectorStore`. Editing that file changes what the agent "knows" with no retraining.

### Step 4: The agent graph (`agent/`)

Read alongside the diagram in [ARCHITECTURE.md](ARCHITECTURE.md#3-the-langgraph-logic).

1. **analyze_outfit** — the photo goes to Gemini with a prompt and the `OutfitAnalysis` schema. We get back garments, colours, style, occasion, season, accessories already worn, and 2–4 categories that would complete the look. We also compute the photo's CLIP vector for later. If Gemini says it isn't an outfit, a conditional edge ends the run politely.
2. **agent** — Gemini, now with three tools attached, reads the analysis and decides what to do. A typical run: one `lookup_style_rules` call plus `search_accessories` for earrings, bag and belt, often all in one turn.
3. **tools** — LangGraph's `ToolNode` runs the calls and adds the JSON results to the conversation. Then control goes back to **agent**, which can search again or say "DONE". A counter stops this loop after 4 rounds.
4. **collect_candidates** — reads all tool results out of the conversation into one pool (keeping each product's best match score), plus the 10 products that *look* most like the outfit photo. Safety net: if the LLM never searched, this node searches the suggested categories itself.
5. **rank** — the Influencer score (Step 5) is computed for every candidate, first with balanced "editor" weights (that's the **Influencer Suggestion**), then once per persona. We keep variety (max one per product type) and skip categories already worn.
6. **write_up** — Gemini gets the ranked items *with their evidence* and writes the answer: a one-line outfit summary, the Influencer Suggestion, then a short paragraph per persona in that persona's voice. It is told not to reorder or invent products.

### Step 5: The Influencer score (`ranking/`)

Each part is a number from 0 to 1:

- **Style fit** — half "how well it matches what the agent searched for", half "how similar its photo is to the outfit photo" (CLIP cosine), rescaled within the pool.
- **Trend fit** — 70% trend score + 30% popularity score (+ TikTok if loaded).
- **Colour fit** — stylist colour-wheel rules: neutrals and metallics go with almost anything; same hue or neighbouring hues are safe; opposites are bold but fine; in-between hues clash.
- **Persona fit** — each persona has words it likes and avoids (Mia the Minimalist likes "leather, thin, plain", avoids "glitter, sequin").

Final score = weighted average. Editor weights: 45% style, 30% trend, 25% colour. Personas shift these (Jay the Streetwear Creator puts 40% on trend).

**Why not let the LLM rank?** The formula is consistent (same input → same answer), free, instant, testable, uses sales numbers the LLM never sees, and gives the explanation for free.

### Step 6: Evaluation (`eval/`)

**Part 1 – offline, all ~200 outfits, no LLM cost.** Four systems recommend 10 accessories per garment photo:

| System | How it works |
|---|---|
| Keyword search (BM25) | garment's text as a search query over accessory texts |
| Popularity | the 10 best sellers, same for everyone |
| CLIP retrieval | garment photo → nearest accessory photos (our RAG step alone) |
| Influencer ranker (ours) | CLIP retrieval → re-ranked by the Influencer score |

Metrics:
- **Hit@5 / Hit@10** — was any really-bought-together accessory in the top 5 / 10?
- **MRR@10** — 1 ÷ position of the first correct item.
- **NDCG@10** — rewards correct items more when ranked higher.
- **Trend, colour harmony, variety of the top 5** — quality signals that match the resume claim "style, trend fit".

**Part 2 – blind side-by-side, 20 outfits.** The full agent's top 5 vs keyword search's top 5. A separate Gemini model (the judge) sees the garment photo and both lists labelled A and B **in random order**, and picks the better one or a tie. We report win / tie / loss rates.

**Being honest about the metrics.** Co-purchases are noisy (people buy unrelated items together), so all Hit@k numbers are low, including the baselines. Popularity can do surprisingly well on Hit@k because best sellers appear in many baskets; that's why we also measure trend, colour and variety, plus the blind judge. Report whatever the notebook produces, including where a baseline wins.

### Step 7: The web app (`frontend/`, `api.py`, `launch.py`)

The model is wrapped in a FastAPI server with an upload endpoint. A second endpoint **streams** the graph: LangGraph's `graph.stream(..., stream_mode="updates")` emits one update per node as it finishes, and the server forwards each one to the browser as a Server-Sent Event. That's how the page shows real progress ("Planned searches: gold drop earrings…").

The front end is plain HTML/CSS/JS hosted on **Vercel**. Vercel can't run the model itself: CLIP + PyTorch is too big for its serverless functions, and the H&M photos can't be published. So the model server runs in Colab, and a free **Cloudflare quick tunnel** gives it a public HTTPS address. The launch cell prints a Vercel link with `?api=<tunnel>` so the site connects in one click. Details in [WEB_APP.md](WEB_APP.md).

### Step 8: Tests (`tests/`)

19 tests run in about 2 seconds with **no API key or downloads**:
- a fake CLIP that turns text into repeatable random vectors;
- a synthetic dataset shaped exactly like H&M's files (to test the popularity maths, catalog and eval set);
- a **scripted fake Gemini** that returns a fixed analysis, makes three tool calls, then says DONE. It runs the *real* LangGraph graph end to end and checks routing, the ToolNode, candidate pooling, ranking, personas and the not-an-outfit exit.
- the **web API**: uploads, the streamed events, bad-file errors, the concurrency slot being released, and CORS for the Vercel origin.

---

## Part 4 — Mapping to the resume bullets

> *Built a multimodal LangGraph agent with RAG and tool calling to recommend accessories from outfit photos.*

- **Multimodal:** Gemini reads the photo; CLIP searches photos with text and photos with photos.
- **LangGraph agent:** 7-node `StateGraph` with conditional edges, a capped tool loop and an early exit.
- **RAG:** product retrieval (CLIP + FAISS, wrapped as a LangChain retriever) + styling-rules retrieval (LangChain vector store).
- **Tool calling:** three `@tool` functions bound to Gemini with `bind_tools`, executed by `ToolNode`.

> *Ranked "Influencer Suggestion" by style, trend fit and outperforming keyword search in side-by-side evaluation.*

- **Ranked by style, trend fit:** the Influencer score (style fit + trend fit from 31M real purchases + colour fit), plus four persona variants.
- **Side-by-side vs keyword search:** BM25 baseline on the same outfits, offline metrics + blind randomised LLM judge. Quote the numbers from your own `docs/RESULTS.md` once the notebook has run.

---

## Part 5 — Likely interview questions

**Why LangGraph instead of a single prompt?** The task has distinct stages with different tools: vision, a variable number of searches, deterministic ranking, writing. A graph makes each stage explicit and testable, lets the LLM loop over tools with a hard cap, and allows branches like "not an outfit".

**How do you stop the LLM inventing products?** It only ever sees products returned by the tools; the ranker chooses the final items, and the write-up prompt says to use exact names. The LLM explains; it doesn't pick.

**Why is CLIP able to search images with text?** It was trained contrastively on image–caption pairs, so matching images and texts get similar vectors in one shared space.

**What's weak, and what would you improve?**
1. Hand-set weights → learn them from basket data (e.g. logistic regression or LightGBM on bought-together pairs).
2. Generic CLIP → try FashionCLIP (fashion-tuned) by changing one setting.
3. The eval queries are product photos, not people wearing outfits → add street-style photos (e.g. DeepFashion2) with human labels.
4. Trends end in 2020 → plug in a live source (the TikTok loader is the hook).
5. Colour rules are simple → extract dominant colours from pixels instead of names.

**Why not host the model on Vercel too?** Vercel serverless functions have a ~250 MB size limit and no GPU; CLIP + PyTorch alone exceeds that. In production you'd put the API on a GPU host (Cloud Run, Modal, a small VM) and keep the static front end on Vercel, the same split as here.

**How would it scale?** Approximate FAISS indexes (HNSW / IVF-PQ) for millions of items, a nightly batch for trend scores, caching analyses by image hash, and serving the graph behind an API.

---

## Part 6 — Changing things

| I want to… | Change |
|---|---|
| use another Gemini model | `GEMINI_MODEL` env var (default `gemini-3.8-flash`; Google is retiring the 2.5 family) |
| a bigger catalog | `FA_CATALOG_SIZE` (e.g. 10000) |
| include shoes | add `"Shoes"` to `accessory_groups` in `config.py` |
| change ranking weights | `ranking/personas.py` |
| add a persona | add an entry to `PERSONAS` (weights, voice, likes, avoids) |
| teach new styling rules | edit `knowledge/style_guide.md`, re-run the index cell |
| try FashionCLIP | `FA_CLIP_MODEL` env var, then rebuild the index |
