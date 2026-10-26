"""The LangGraph nodes. Each one is a plain function: state in -> partial state out.

    analyze_outfit      Gemini vision -> OutfitAnalysis (structured output) + CLIP vector of the photo
    agent               Gemini with tools bound; decides which searches / lookups to run
    tools               (LangGraph's prebuilt ToolNode executes the tool calls)
    collect_candidates  parses tool results out of the conversation into one candidate pool
    rank                Influencer Suggestion scorer, once per persona
    write_up            Gemini turns the ranked evidence into a friendly answer
    not_an_outfit       early exit when the photo is not clothing
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..config import settings
from ..rag import embeddings as emb
from ..rag.catalog_index import CatalogIndex
from ..ranking.influencer_score import diversify, explain_row, score_candidates
from ..ranking.personas import PERSONAS
from . import prompts
from .llm import b64_to_pil, message_text
from .schemas import OutfitAnalysis


class Nodes:
    def __init__(self, llm, tools: list, index: CatalogIndex,
                 encode_image: Callable | None = None):
        self.llm = llm
        self.llm_with_tools = llm.bind_tools(tools)
        self.index = index
        self.encode_image = encode_image or (lambda pil: emb.encode_images([pil])[0])

    # ------------------------------------------------------------------ 1. look at the photo
    def analyze_outfit(self, state: dict) -> dict:
        extra = f"The user adds: {state['user_request']}" if state.get("user_request") else ""
        msg = HumanMessage(content=[
            {"type": "text", "text": prompts.ANALYZE_PROMPT.format(extra=extra)},
            {"type": "image_url", "image_url": {"url": f"data:{state.get('mime_type', 'image/jpeg')};base64,{state['image_b64']}"}},
        ])
        analysis = self._structured(msg)
        vec = self.encode_image(b64_to_pil(state["image_b64"]))
        kickoff = prompts.AGENT_KICKOFF.format(outfit_json=analysis.model_dump_json(indent=1),
                                               request=state.get("user_request") or "none")
        return {
            "outfit": analysis.model_dump(),
            "outfit_embedding": [float(x) for x in np.asarray(vec)],
            "messages": [SystemMessage(content=prompts.AGENT_SYSTEM), HumanMessage(content=kickoff)],
            "steps": 0,
        }

    def _structured(self, msg) -> OutfitAnalysis:
        """Structured output, with a JSON-parsing fallback if the model/SDK refuses the schema."""
        try:
            out = self.llm.with_structured_output(OutfitAnalysis).invoke([msg])
            if isinstance(out, OutfitAnalysis):
                return out
            return OutfitAnalysis.model_validate(out)
        except Exception as err:  # noqa: BLE001
            print(f"[analyze] structured output failed ({err.__class__.__name__}), falling back to JSON")
            schema = json.dumps(OutfitAnalysis.model_json_schema()["properties"])
            raw = message_text(self.llm.invoke([msg, HumanMessage(content=f"Reply ONLY with JSON with these keys: {schema}")]))
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
            return OutfitAnalysis.model_validate_json(raw)

    @staticmethod
    def route_after_analyze(state: dict) -> str:
        return "agent" if state["outfit"].get("is_outfit", True) else "not_an_outfit"

    @staticmethod
    def not_an_outfit(state: dict) -> dict:
        return {"final_answer": "I couldn't spot an outfit in that photo. Try a picture that shows the clothes clearly "
                                "(worn or laid flat).", "ranked": [], "persona_picks": {}}

    # ------------------------------------------------------------------ 2. the tool-calling loop
    def agent(self, state: dict) -> dict:
        ai = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [ai], "steps": state.get("steps", 0) + 1}

    @staticmethod
    def route_after_agent(state: dict) -> str:
        last = state["messages"][-1]
        wants_tools = bool(getattr(last, "tool_calls", None))
        if wants_tools and state.get("steps", 0) < settings.max_agent_steps:
            return "tools"
        return "collect_candidates"

    # ------------------------------------------------------------------ 3. pool the evidence
    def collect_candidates(self, state: dict) -> dict:
        best: dict[int, float] = {}
        notes: list[str] = []
        for m in state.get("messages", []):
            if not isinstance(m, ToolMessage):
                continue
            try:
                payload = json.loads(message_text(m))
            except (json.JSONDecodeError, TypeError):
                continue
            for r in payload.get("results", []):
                aid = int(r["article_id"])
                best[aid] = max(best.get(aid, 0.0), float(r.get("match", 0.0)))
            for rule in payload.get("rules", []):
                notes.append(rule["text"])

        outfit = state["outfit"]
        # Safety net: if the LLM skipped searching, search each suggested category ourselves.
        if not best:
            colours = " ".join(outfit.get("colors", [])[:2])
            for cat in outfit.get("suggested_categories") or ["earrings", "bag", "necklace"]:
                q = f"{outfit.get('style', '')} {cat} to wear with {colours}".strip()
                for _, r in self.index.search_text(q, k=settings.candidates_per_search, category=cat).iterrows():
                    best[int(r["article_id"])] = max(best.get(int(r["article_id"]), 0.0), float(r["sim"]))
        # Always add items that *look* like they belong with the photo (image -> image search).
        for _, r in self.index.search_vector(np.array(state["outfit_embedding"]), k=10).iterrows():
            best.setdefault(int(r["article_id"]), float(r["sim"]))

        cands = [{"article_id": a, "retrieval_sim": s} for a, s in best.items()]
        return {"candidates": cands, "style_notes": list(dict.fromkeys(notes))}

    # ------------------------------------------------------------------ 4. Influencer Suggestion
    def rank(self, state: dict) -> dict:
        cands = pd.DataFrame(state["candidates"])
        rows = self.index.rows(cands["article_id"]).merge(cands, on="article_id")
        img_sim = self.index.similarity(np.array(state["outfit_embedding"]), rows["article_id"])
        # style = half "matches what the agent asked for", half "looks right next to the photo"
        rows["style_sim"] = 0.5 * rows["retrieval_sim"].to_numpy() + 0.5 * img_sim
        colours = state["outfit"].get("colors", [])
        worn = " ".join(state["outfit"].get("existing_accessories", [])).lower()

        def top(persona: str) -> list[dict]:
            scored = score_candidates(rows, colours, persona)
            if worn:   # don't suggest a second bag if they already carry one
                already = scored["product_type_name"].str.lower().map(lambda t: any(w in worn for w in t.split("/")))
                if (~already).sum() >= settings.final_top_k:
                    scored = scored[~already]
            picks = diversify(scored, settings.final_top_k)
            picks["evidence"] = picks.apply(explain_row, axis=1)
            keep = ["article_id", "prod_name", "product_type_name", "colour_group_name", "image_path",
                    "style_fit", "trend_fit", "color_fit", "persona_fit", "influencer_score", "evidence"]
            return json.loads(picks[keep].to_json(orient="records"))

        personas = state.get("personas") or []
        return {"ranked": top("editor"),
                "persona_picks": {p: top(p)[:2] for p in personas if p in PERSONAS and p != "editor"}}

    # ------------------------------------------------------------------ 5. explain it
    def write_up(self, state: dict) -> dict:
        o = state["outfit"]

        def fmt(items):
            return "\n".join(f"- {i['prod_name']} ({i['colour_group_name']} {i['product_type_name']}): {i['evidence']}"
                             for i in items) or "- (none)"

        personas_txt = "\n".join(
            f"[{PERSONAS[k].name} | voice: {PERSONAS[k].voice}]\n{fmt(v)}" for k, v in state.get("persona_picks", {}).items()
        ) or "(no personas requested)"
        prompt = prompts.WRITEUP_PROMPT.format(
            vibe=o.get("vibe", ""), garments=", ".join(o.get("garments", [])), colors=", ".join(o.get("colors", [])),
            style=o.get("style", ""), occasion=o.get("occasion", ""), request=state.get("user_request") or "none",
            rules="\n".join(f"- {n[:300]}" for n in state.get("style_notes", [])[:3]) or "- (none retrieved)",
            editor=fmt(state.get("ranked", [])), personas=personas_txt,
        )
        return {"final_answer": message_text(self.llm.invoke([HumanMessage(content=prompt)]))}
