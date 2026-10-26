"""Runs the real LangGraph graph end to end with a scripted fake LLM.

No API key needed: the fake model plays Gemini's part (structured analysis,
tool calls, final text), so this checks the wiring - routing, ToolNode,
candidate pooling, ranking, personas - not the model's taste.
"""
import base64
import io

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from PIL import Image  # noqa: E402

from fashion_assistant.agent.graph import build_graph  # noqa: E402
from fashion_assistant.agent.schemas import OutfitAnalysis  # noqa: E402

from conftest import fake_vec  # noqa: E402


class ScriptedGemini(BaseChatModel):
    is_outfit: bool = True
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def with_structured_output(self, schema, **kw):
        return RunnableLambda(lambda _: OutfitAnalysis(
            is_outfit=self.is_outfit, garments=["black slip dress"], colors=["black"], style="minimalist",
            occasion="date night", suggested_categories=["earrings", "bag"], vibe="Sleek black evening look."))

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.calls.append(messages)
        has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
        if "Outfit analysis" in str(messages[-1].content) and not has_tool_results:
            msg = AIMessage(content="", tool_calls=[
                {"name": "lookup_style_rules", "args": {"question": "jewellery for black dress"}, "id": "c1"},
                {"name": "search_accessories", "args": {"query": "gold drop earrings", "category": "earrings"}, "id": "c2"},
                {"name": "search_accessories", "args": {"query": "small black clutch", "category": "bag"}, "id": "c3"},
            ])
        elif has_tool_results and "Influencer Suggestion" not in str(messages[-1].content):
            msg = AIMessage(content="DONE")
        else:
            msg = AIMessage(content="## Influencer Suggestion\nWear the gold earrings.")
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeKB:
    def search(self, q, k=3):
        return [{"section": "Colour rules", "text": "Colour rules: neutral outfits take any colour.", "score": 0.9}]


def _img_b64():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "black").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_full_graph(index):
    llm = ScriptedGemini()
    graph = build_graph(llm, index, FakeKB(), encode_image=lambda pil: fake_vec("outfit"))
    out = graph.invoke({"image_b64": _img_b64(), "mime_type": "image/jpeg",
                        "user_request": "date night", "personas": ["minimalist", "streetwear"]})
    assert out["outfit"]["style"] == "minimalist"
    tool_names = [m.name for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_names.count("search_accessories") == 2 and "lookup_style_rules" in tool_names
    assert len(out["ranked"]) == 5
    assert set(out["persona_picks"]) == {"minimalist", "streetwear"}
    assert out["style_notes"] and "Influencer Suggestion" in out["final_answer"]
    scores = [r["influencer_score"] for r in out["ranked"]]
    assert all(0 <= s <= 1 for s in scores)


def test_not_an_outfit_exits_early(index):
    llm = ScriptedGemini(is_outfit=False)
    graph = build_graph(llm, index, FakeKB(), encode_image=lambda pil: fake_vec("x"))
    out = graph.invoke({"image_b64": _img_b64(), "personas": []})
    assert out["ranked"] == [] and "couldn't spot an outfit" in out["final_answer"]
    assert llm.calls == []          # never reached the agent


def test_mermaid_diagram(index):
    graph = build_graph(ScriptedGemini(), index, FakeKB(), encode_image=lambda pil: fake_vec("o"))
    m = graph.get_graph().draw_mermaid()
    for node in ["analyze_outfit", "agent", "tools", "collect_candidates", "rank", "write_up"]:
        assert node in m
