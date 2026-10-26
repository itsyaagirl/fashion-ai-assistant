"""HTTP API for the web front end.

    GET  /api/health                  is the model server up? which model, catalog size
    GET  /api/personas                the influencer personas the UI can offer
    GET  /api/trending?category=&k=   fastest-rising accessories (for the home screen)
    GET  /api/items/{id}/image        product photo
    POST /api/recommend               multipart: image, request, personas -> full result (JSON)
    POST /api/recommend/stream        same, but Server-Sent Events: one "step" event per graph
                                      node, then a "result" event (lets the UI show real progress)

The front end can live on Vercel (any origin) or be served from this same
server at "/". CORS is open by default because the API is read-only and holds
no user data; set FA_CORS_ORIGINS to lock it to your Vercel domain.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .ranking.personas import PERSONAS

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


# ----------------------------------------------------------------------------- serialisation
def _item(it: dict) -> dict:
    keys = ["article_id", "prod_name", "product_type_name", "colour_group_name", "style_fit", "trend_fit",
            "color_fit", "persona_fit", "influencer_score", "evidence"]
    out = {k: it.get(k) for k in keys}
    out["image_url"] = f"/api/items/{int(it['article_id'])}/image"
    return out


def tool_trace(messages) -> list[dict]:
    """The agent's tool calls in order, with how many results each returned."""
    from langchain_core.messages import AIMessage, ToolMessage

    trace, by_id = [], {}
    for m in messages or []:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                by_id[tc.get("id")] = len(trace)
                trace.append({"tool": tc["name"], "args": tc.get("args", {}), "results": None})
        elif isinstance(m, ToolMessage) and m.tool_call_id in by_id:
            try:
                payload = json.loads(m.content)
                trace[by_id[m.tool_call_id]]["results"] = len(payload.get("results", payload.get("rules", [])))
            except (TypeError, ValueError):
                pass
    return trace


def serialize_result(state: dict, seconds: float | None = None) -> dict:
    return {
        "outfit": state.get("outfit", {}),
        "ranked": [_item(i) for i in state.get("ranked", [])],
        "persona_picks": {
            k: {"name": PERSONAS[k].name, "voice": PERSONAS[k].voice, "picks": [_item(i) for i in v]}
            for k, v in (state.get("persona_picks") or {}).items() if k in PERSONAS
        },
        "final_answer": state.get("final_answer", ""),
        "style_notes": state.get("style_notes", []),
        "trace": tool_trace(state.get("messages")),
        "candidates": len(state.get("candidates", []) or []),
        "agent_steps": state.get("steps", 0),
        "seconds": round(seconds, 1) if seconds is not None else None,
    }


def step_summary(node: str, update: dict) -> dict:
    """A short, human-readable line per finished graph node for the progress view."""
    if node == "analyze_outfit":
        o = update.get("outfit", {})
        return {"node": node, "label": "Read the outfit",
                "detail": o.get("vibe") or ", ".join(o.get("garments", [])), "outfit": o}
    if node == "agent":
        calls = getattr((update.get("messages") or [None])[-1], "tool_calls", None) or []
        if calls:
            return {"node": node, "label": "Planned searches",
                    "detail": "; ".join(c["args"].get("query") or c["args"].get("question") or c["name"] for c in calls)}
        return {"node": node, "label": "Finished searching", "detail": ""}
    if node == "tools":
        return {"node": node, "label": "Searched the catalog",
                "detail": f"{len(update.get('messages', []))} tool results"}
    if node == "collect_candidates":
        return {"node": node, "label": "Gathered candidates", "detail": f"{len(update.get('candidates', []))} products"}
    if node == "rank":
        top = (update.get("ranked") or [{}])[0]
        return {"node": node, "label": "Ranked by style, trend and colour", "detail": top.get("prod_name", "")}
    if node == "write_up":
        return {"node": node, "label": "Wrote the styling notes", "detail": ""}
    return {"node": node, "label": node.replace("_", " "), "detail": ""}


# ----------------------------------------------------------------------------- app
def create_app(assistant=None, max_concurrent: int = 2):
    if assistant is None:
        from .assistant import FashionAssistant

        assistant = FashionAssistant()

    app = FastAPI(title="Finishing Touch API", version="1.0.0")
    origins = [o.strip() for o in os.environ.get("FA_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])
    slots = threading.BoundedSemaphore(max_concurrent)   # Gemini free tier: keep parallel runs low
    catalog = assistant.index.catalog.set_index("article_id")

    async def read_image(image: UploadFile) -> bytes:
        data = await image.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "That photo is over 10 MB. Use a smaller image.")
        try:
            Image.open(io.BytesIO(data)).verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "That file isn't an image we can read. Use a JPG, PNG or WebP photo.")
        return data

    def parse_personas(personas: str) -> list[str]:
        return [p for p in (s.strip() for s in personas.split(",")) if p in PERSONAS and p != "editor"]

    def take_slot():
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "The stylist is busy with other outfits. Try again in a few seconds.")

    @app.get("/api/health")
    def health():
        from .config import settings

        return {"status": "ok", "model": settings.llm_model, "catalog_size": len(catalog)}

    @app.get("/api/personas")
    def personas():
        return [{"key": p.key, "name": p.name, "voice": p.voice} for p in PERSONAS.values() if p.key != "editor"]

    @app.get("/api/trending")
    def trending(category: str = "", k: int = 8):
        hits = assistant.index.trending(category or None, k=max(1, min(k, 24)))
        return [{"article_id": int(r["article_id"]), "prod_name": r["prod_name"],
                 "product_type_name": r["product_type_name"], "colour_group_name": r["colour_group_name"],
                 "trend_score": round(float(r["trend_score"]), 3),
                 "trend_lift": round(float(r.get("trend_lift", 1.0) or 1.0), 2),
                 "image_url": f"/api/items/{int(r['article_id'])}/image"} for _, r in hits.iterrows()]

    @app.get("/api/items/{article_id}/image")
    def item_image(article_id: int):
        if article_id not in catalog.index:
            raise HTTPException(404, "No product with that id.")
        path = Path(str(catalog.loc[article_id, "image_path"]))
        if not path.exists():
            raise HTTPException(404, "Photo not available for this product.")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    @app.post("/api/recommend")
    async def recommend(image: UploadFile = File(...), request: str = Form(""), personas: str = Form("")):
        data = await read_image(image)
        take_slot()
        try:
            t0 = time.time()
            state = await run_in_threadpool(assistant.recommend, data, request[:300], parse_personas(personas))
            return serialize_result(state, time.time() - t0)
        finally:
            slots.release()

    @app.post("/api/recommend/stream")
    async def recommend_stream(image: UploadFile = File(...), request: str = Form(""), personas: str = Form("")):
        data = await read_image(image)
        take_slot()
        released = [False]

        def release():            # runs once, whether the stream finishes or the client leaves early
            if not released[0]:
                released[0] = True
                slots.release()

        def events():
            t0 = time.time()
            try:
                state = {}
                for node, update, state in assistant.stream(data, request[:300], parse_personas(personas)):
                    yield f"event: step\ndata: {json.dumps(step_summary(node, update), default=str)}\n\n"
                yield f"event: result\ndata: {json.dumps(serialize_result(state, time.time() - t0), default=str)}\n\n"
            except Exception as err:  # noqa: BLE001 - report to the UI instead of a dropped connection
                msg = f"The stylist hit an error: {err.__class__.__name__}: {str(err)[:200]}"
                yield f"event: error\ndata: {json.dumps({'message': msg})}\n\n"
            finally:
                release()

        # StreamingResponse runs this sync generator in a worker thread
        return StreamingResponse(events(), media_type="text/event-stream", background=BackgroundTask(release),
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if FRONTEND_DIR.exists():   # serve the same UI locally / through the tunnel
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app
